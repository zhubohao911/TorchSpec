# GPU & CUDA Knowledge — Supplementary Notes（中英对照）

> 说明：本文是 [`knowledge.zh-en.md`](knowledge.zh-en.md) 的**配套补充**，专门
> 把 colocate 文档中一笔带过的 GPU / CUDA 概念展开讲透。读完本文，你应该能
> 回答："为什么 MPS 必须用 daemon？"、"为什么 `cudaMemcpyDeviceToDevice`
> 几乎免费？"、"`expandable_segments` 到底改了什么？"。
>
> Audience: anyone who read `knowledge.zh-en.md` and felt the GPU/CUDA terms
> (SM, context, stream, MPS daemon, allocator, intra-device copy …) deserved
> more than a one-line gloss.

🇨🇳 **读者**：读完 [`knowledge.zh-en.md`](knowledge.zh-en.md) 后，对里面 GPU /
CUDA 相关的术语（SM、context、stream、MPS daemon、allocator、设备内拷贝……）
觉得"过得太快"的人。

> ⚠️ **Cross-check — updated 2026-05-21.** Two claims below are now
> wrong and flagged inline:
> 1. **"Intra-device NCCL P2P is cheap"** — there is no such thing. NCCL
>    refuses a communicator with two ranks on one physical GPU, so
>    same-GPU NCCL `send`/`recv` cannot run at all. The colocate
>    hidden-state transport is **CUDA IPC** (default; a real on-device
>    `cudaMemcpyDeviceToDevice` after a handle exchange) or **gloo
>    CPU-staging** (fallback). The "device-to-device copy is nearly free"
>    intuition is correct — it just is not reached *via NCCL*.
> 2. **"`expandable_segments:True` is mandatory for colocate"** — no
>    longer true. The default **CUDA IPC** transport needs plain
>    `cudaMalloc` memory and the colocate path **disables**
>    `expandable_segments`; only the gloo fallback wants it.
>
> The MPS, context, stream, allocator, and CUDA-VMM material is all still
> accurate. See [`transport_benchmark.md`](transport_benchmark.md) and
> [`implementation_log.md`](implementation_log.md) rounds 1 / 7 / 9.
>
> 🇨🇳 **交叉核对 —— 2026-05-21 更新。** 下文有两处现在是错的，已就地标注：
> ① **"同卡 NCCL P2P 几乎免费"** —— 根本不存在这回事。NCCL 拒绝"同一张
> 物理 GPU 上两个 rank"的 communicator，同卡 NCCL `send`/`recv` 压根跑
> 不起来。colocate 的 hidden-state 传输是 **CUDA IPC**（默认；句柄交换后
> 做一次真正的 `cudaMemcpyDeviceToDevice`）或 **gloo CPU 中转**（回退）。
> "设备内拷贝几乎免费"这个直觉本身没错 —— 只是不经由 NCCL 达成。
> ② **"`expandable_segments:True` 是 colocate 必选项"** —— 不再成立。
> 默认的 **CUDA IPC** 传输需要普通 `cudaMalloc` 显存，colocate 路径会
> **关闭** `expandable_segments`；只有 gloo 回退才需要它。
> MPS、context、stream、allocator、CUDA 虚拟内存等内容依然正确。

---

## 1. GPU hardware in 5 minutes
## 1. 5 分钟看懂 GPU 硬件

A modern NVIDIA GPU (H100, A100, …) is a hierarchy:

🇨🇳 现代 NVIDIA GPU（H100、A100 等）是一个层级结构：

```
┌──────────────────────── GPU (one PCIe device) ────────────────────────┐
│                                                                       │
│  ┌───────────────── HBM (VRAM, e.g. 80 GB on H100) ────────────────┐  │
│  │   one shared, high-bandwidth memory pool                        │  │
│  └────────────────────────────┬────────────────────────────────────┘  │
│                               │  ~3 TB/s                              │
│  ┌──────────┐  ┌──────────┐  ─┴─  ┌──────────┐  ┌──────────┐         │
│  │  SM 0    │  │  SM 1    │ ...   │  SM 131  │  │  SM 132  │  (H100) │
│  │ ┌──────┐ │  │ ┌──────┐ │       │ ┌──────┐ │  │ ┌──────┐ │         │
│  │ │warp 0│ │  │ │warp 0│ │       │ │warp 0│ │  │ │warp 0│ │         │
│  │ │warp 1│ │  │ │warp 1│ │       │ │warp 1│ │  │ │warp 1│ │         │
│  │ │ ...  │ │  │ │ ...  │ │       │ │ ...  │ │  │ │ ...  │ │         │
│  │ └──────┘ │  │ └──────┘ │       │ └──────┘ │  │ └──────┘ │         │
│  │  L1 / SMEM (~256 KB / SM)                                          │
│  └──────────┘  └──────────┘       └──────────┘  └──────────┘          │
│                                                                       │
│         shared L2 cache (~50 MB on H100)                              │
└───────────────────────────────────────────────────────────────────────┘
        │                                              │
        │ PCIe Gen5 ~64 GB/s                           │ NVLink ~900 GB/s
        ▼                                              ▼
      Host (CPU/RAM)                              peer GPUs
```

🇨🇳 上图：一块 GPU 由几十~一百多个 **SM（Streaming Multiprocessor，流式多处理器）**
组成（H100 有 132 个），共享一块 **HBM** 显存（H100 是 80 GB）和一块 L2
cache。每个 SM 内部还有 L1/共享内存。GPU 通过 **PCIe** 连主机 CPU/内存，通过
**NVLink** 直连同机其它 GPU。

Key bandwidths to internalise (H100 SXM):

🇨🇳 几个关键带宽数字（H100 SXM，记住能省很多猜测）：

| Path | Bandwidth | When you pay this |
|---|---|---|
| HBM ↔ SM | ~3 TB/s | Every tensor load/store |
| Intra-GPU L2 | ~12 TB/s | Cached reuse |
| NVLink (GPU↔GPU, same node) | ~900 GB/s | NCCL all-reduce within node |
| PCIe Gen5 (GPU↔CPU) | ~64 GB/s | Host↔device copies, pinned memory |
| Network (RDMA 400 Gb/s) | ~50 GB/s | NCCL cross-node, Mooncake |

🇨🇳 **看这张表你应该立刻明白**：colocate 让 hidden_state 走"同卡设备内拷贝"
（约 3 TB/s 显存带宽，本质是 HBM 内部移动），而 disaggregated 走的是网络
（50 GB/s），差了 **60 倍**。这是 colocate 性能优势的物理基础。

### What's a "kernel"
### 什么是 "kernel"

A **CUDA kernel** is a function written to run on the GPU. You launch it from
the host with a `<<<grid, block>>>` configuration. Each kernel launch:

🇨🇳 **CUDA kernel** 就是一个跑在 GPU 上的函数。主机端用 `<<<grid, block>>>`
配置启动它。每次 kernel 启动会：

1. Be **enqueued** onto a CUDA stream (more in §3).
2. Get scheduled onto some subset of SMs.
3. Execute in lockstep groups of 32 threads called **warps**.
4. Read/write HBM and exit.

🇨🇳 ① 被**排队**进某个 CUDA stream（详见 §3）；② 被调度到一部分 SM 上；
③ 以 32 线程为一组的 **warp** 单位齐步执行；④ 读写 HBM 后退出。

Important property: **kernel launches are asynchronous**. The CPU enqueues
them and moves on; the GPU runs them in stream order. This is why
`torch.cuda.synchronize()` exists — to force the CPU to wait.

🇨🇳 关键性质：**kernel 启动是异步的**。CPU 把 kernel 丢进队列就走，GPU 按
stream 顺序执行。所以才有 `torch.cuda.synchronize()` —— 强制 CPU 等 GPU。

---

## 2. CUDA contexts and `CUDA_VISIBLE_DEVICES`
## 2. CUDA context 和 `CUDA_VISIBLE_DEVICES`

### CUDA context
### CUDA context（CUDA 上下文）

A **CUDA context** is the GPU-side equivalent of a process: it owns
allocations, streams, module loads (compiled kernels), and a virtual address
space. By default:

🇨🇳 **CUDA context** 是 GPU 这一侧的"进程"概念：它拥有显存分配、stream、
加载的模块（已编译的 kernel）、一段虚拟地址空间。默认情况下：

- **One process = one CUDA context per GPU it uses.**
- The first CUDA call lazily creates the primary context (~200 MB overhead
  just for runtime, cuBLAS handles, etc.).
- Contexts are **independent**: process A's pointers are meaningless to
  process B, even on the same GPU.

🇨🇳 **每个进程对所用的每张 GPU 各自持有一个 CUDA context**。第一次 CUDA 调
用会懒加载创建主 context（光是 runtime、cuBLAS handle 这些就占约 200 MB）。
contexts 之间**互相独立**：进程 A 的指针在进程 B 看来毫无意义，哪怕在同一
张 GPU 上。

Without MPS, the GPU's hardware scheduler **time-slices** between contexts:
context A's kernels run for a slice, then context B's, then A's. Each switch
flushes pipelines and burns a few µs. That's why naive multi-process GPU
sharing is slow — not because of contention on the SMs, but because of
context-switch overhead.

🇨🇳 没有 MPS 时，GPU 的硬件调度器在 contexts 之间**时间片轮转**：A 的
kernel 跑一片，然后 B 的，然后再 A 的。每次切换都要 flush 流水线，烧几个
微秒。这就是为什么"多进程裸共享 GPU"慢——慢的不是 SM 抢资源，而是 context
切换本身的开销。

### `CUDA_VISIBLE_DEVICES`
### `CUDA_VISIBLE_DEVICES`（环境变量）

An env var that **filters and renumbers** the GPUs a process sees:

🇨🇳 一个**过滤并重新编号**进程能看到的 GPU 的环境变量：

```bash
# Physical GPUs on host: 0..7
CUDA_VISIBLE_DEVICES=3 python train.py
# Inside the process:
#   torch.cuda.device_count() == 1
#   torch.cuda.current_device() == 0   (renumbered!)
#   But it's actually physical GPU 3.
```

🇨🇳 关键点：值会**重新编号**。你在进程里看到的是 `cuda:0`，但实际指的是物
理卡 3。Ray 也是用它把"逻辑 GPU"绑定到物理卡上。

For colocate: Ray sets `CUDA_VISIBLE_DEVICES=<one physical id>` on **both**
the trainer and engine process for a given bundle. Both processes then think
they own `cuda:0`, but it's the same physical card. Without MPS they'd
time-slice; with MPS they share.

🇨🇳 **对于 colocate**：Ray 给同一个 bundle 上的 trainer 和 engine 进程都设
**同一个物理 id**。两个进程都以为自己独占 `cuda:0`，其实是同一张物理卡。
没 MPS 就时间片轮转；有 MPS 就并发共享。

---

## 3. CUDA streams
## 3. CUDA stream（CUDA 流）

A **stream** is an in-order queue of GPU work within a context. Two streams
in the same context can execute **concurrently** if they don't conflict.

🇨🇳 **stream** 是 context 内部一条按序执行的 GPU 工作队列。同一 context 里
**两条不冲突的 stream 可以并发执行**。

```
Stream A:  [kernel1] → [kernel2] → [memcpyD2D] → ...      (ordered)
Stream B:  [kernel3] → [allreduce] → ...                  (ordered)
                                  ↕  may overlap on SMs
```

Why this matters for colocate:

🇨🇳 为什么 colocate 关心 stream：

- PyTorch by default uses **one CUDA stream per device** (the "default
  stream"). All ops on that device serialise.
- NCCL collectives normally use their own internal stream — that's how
  all-reduce overlaps with compute.
- In our colocate transfer path, the trainer's `dist.recv` lands on **the
  same stream as the FSDP all-gather** unless we explicitly move it. They'd
  then serialise behind each other.

🇨🇳 ① PyTorch 默认每张卡用**一个 stream**（"default stream"），那张卡上所
有操作串行；② NCCL 集合通信通常用自己内部的 stream，这就是 all-reduce 能
和 compute 重叠的原理；③ 我们 colocate 里 trainer 的 `dist.recv` 默认会
落到**和 FSDP all-gather 同一条 stream**上，两者就会互相挤队列。

That's why §6 of the main doc says "put the transfer on a dedicated stream":

🇨🇳 这就是主文档 §6 强调"用独立 stream"的原因：

```python
transfer_stream = torch.cuda.Stream()
with torch.cuda.stream(transfer_stream):
    dist.recv(buf, src=engine_rank)
# buf's producer is now transfer_stream. If you use buf elsewhere, you must
# synchronise the consuming stream against transfer_stream:
torch.cuda.current_stream().wait_stream(transfer_stream)
```

🇨🇳 ⚠️ 用了 stream 之后要记得**做 stream 同步**：`buf` 在 `transfer_stream`
上产生，如果之后在 default stream 上用它，必须 `wait_stream` 一下，否则会
读到没写完的数据。

### Events
### Event（事件）

A `torch.cuda.Event` is a marker you record on stream A and query/wait from
stream B (or the CPU). Used to implement fine-grained sync, e.g. "the
allocator may not reuse this buffer until the kernel that consumed it has
finished." PyTorch's caching allocator uses events internally to track
**stream-safe reuse**.

🇨🇳 `torch.cuda.Event` 是放在 stream A 上、可以从 stream B（或 CPU）查询
/等待的标记，用于细粒度同步。PyTorch caching allocator 内部用 event 来追踪
"这块 buffer 在消费它的 kernel 跑完之前不能被复用"，从而做**stream-safe
的内存复用**。

---

## 4. CUDA memory model
## 4. CUDA 内存模型

### Allocation flavors
### 分配方式

| API | What you get | Pays |
|---|---|---|
| `cudaMalloc` | Pointer to HBM, fixed lifetime | Slow (~100 µs), syscall-like |
| `cudaMallocAsync` | Same, but pooled | Fast, stream-ordered |
| `cudaMallocHost` | Pinned host RAM | Slow alloc, fast H2D |
| `cuMemAddressReserve` + `cuMemCreate` + `cuMemMap` | Virtual range, then back it with physical pages | Most flexible; underlies `expandable_segments` |

🇨🇳 **`cudaMalloc`** 直接拿 HBM 指针，每次都要陷入驱动，约 100 微秒一次，
所以谁都不会裸用。**`cudaMallocAsync`** 是 CUDA 11+ 的池化版，按 stream 顺
序分配/释放，快得多。**`cudaMallocHost`** 是分配"锁页"的主机内存，H2D 拷
贝时不用走中转 buffer，能跑满 PCIe。**`cuMemAddressReserve` + `cuMemMap`**
是低层 API：先预留一段**虚拟地址**，再用物理页填充——这正是
`expandable_segments` 背后的机制。

### `cudaMemcpy` flavors
### `cudaMemcpy` 的几种方向

- `cudaMemcpyHostToDevice` (H2D) — pinned host → GPU, over PCIe (~50 GB/s).
- `cudaMemcpyDeviceToHost` (D2H) — GPU → pinned host, over PCIe.
- `cudaMemcpyDeviceToDevice` (D2D) — same GPU → same GPU, in HBM (~3 TB/s).
- `cudaMemcpyPeer` — GPU 0 → GPU 1, over NVLink (~900 GB/s) or PCIe.

🇨🇳 重点是 **D2D**：源和目的都在同一张卡的 HBM 内部，本质上是 GPU 内部一
次大显存搬运，跑显存带宽（H100 约 3 TB/s）。**colocate 下 NCCL P2P 退化为
的就是这个**——所以"几乎免费"。

### Why intra-device NCCL P2P is so cheap
### 为什么"同卡 NCCL P2P"几乎免费

> ⚠️ **This section is wrong — see the top banner.** NCCL does *not* have
> a same-GPU `send`/`recv` fast path; it rejects two ranks on one
> physical GPU outright. The colocate hidden-state transport reaches the
> on-device `cudaMemcpyDeviceToDevice` a different way — **CUDA IPC**
> (export a handle, map the peer's memory, one D→D copy) or **gloo
> CPU-staging**. The bandwidth intuition below is right; the "via NCCL"
> framing is not. Kept for the intuition; see
> [`transport_benchmark.md`](transport_benchmark.md).
> 🇨🇳 **本节有误 —— 见顶部说明。** NCCL 没有同卡 `send`/`recv` 快速路径，
> 它直接拒绝同卡两 rank。colocate 是通过 **CUDA IPC**（导出句柄、映射对端
> 显存、做一次 D→D 拷贝）或 **gloo CPU 中转**达成设备内拷贝的。下文的带宽
> 直觉没错，错的是"经由 NCCL"这个说法。保留以说明直觉。

When sender and receiver of `dist.send/recv` happen to be on the same
physical GPU (and in colocate, **they always are**), NCCL detects this and
takes a fast path:

🇨🇳 当 `dist.send/recv` 的发送方和接收方碰巧在**同一张物理 GPU 上**（在
colocate 里**永远如此**），NCCL 会检测到并走快速路径：

1. No ring buffer staging.
2. No PCIe traversal.
3. No network packets.
4. Just a `cudaMemcpyDeviceToDevice` from the sender's tensor to the
   receiver's tensor.

🇨🇳 ① 不走 ring buffer 中转；② 不走 PCIe；③ 不走网络；④ 直接一次
`cudaMemcpyDeviceToDevice` 把发送方的 tensor 拷到接收方的 tensor。

This is why the colocate timeline (main doc §8) treats hidden-state transfer
as essentially zero-cost.

🇨🇳 这就是主文档 §8 把 hidden-state 传输当成零成本的原因。

---

## 5. CUDA MPS deep dive
## 5. CUDA MPS 深入

The main doc explained *what* MPS is. Here's *how* it actually works.

🇨🇳 主文档讲了 MPS **是什么**，这里讲它**怎么工作**。

### The architecture
### 架构

```
                              ┌─────────────────────────────────┐
                              │   GPU (single device)           │
                              │  one merged CUDA context        │
                              └──────────────┬──────────────────┘
                                             │
                              ┌──────────────┴──────────────────┐
                              │   nvidia-cuda-mps-server        │
                              │   (one server per (uid, GPU))   │
                              └──────────────┬──────────────────┘
                                             │ Unix sockets in
                                             │ $CUDA_MPS_PIPE_DIRECTORY
                ┌────────────────────────────┼────────────────────────────┐
                │                            │                            │
        ┌───────┴────────┐         ┌─────────┴────────┐         ┌─────────┴────────┐
        │ client proc A  │         │ client proc B    │  ...    │ client proc N    │
        │ (trainer)      │         │ (engine)         │         │                  │
        └────────────────┘         └──────────────────┘         └──────────────────┘

      (the per-node nvidia-cuda-mps-control daemon spawns the server on demand)
```

🇨🇳 架构图说明：每台机有一个 `nvidia-cuda-mps-control` daemon（管理进程），
它在第一个客户端连上来时按需 fork 出 `nvidia-cuda-mps-server`（每个
(uid, GPU) 一个 server）。客户端通过 `$CUDA_MPS_PIPE_DIRECTORY` 下的 Unix
socket 连 server。server 把所有客户端的 CUDA 调用合并到**同一个 GPU
context** 里提交，于是不同进程的 kernel 可以在 SM 上交错运行。

### Why "merge into one context" is the magic
### 为什么"合并到一个 context"是 MPS 的精髓

Without MPS: each client has its **own** GPU context → hardware time-slices.

With MPS: each client's CUDA calls are **forwarded over the socket** to the
MPS server, which submits them all from a **single shared context**. From the
GPU's perspective, it's seeing one context with many streams — exactly the
same situation as a single multi-threaded process. Hyper-Q (NVIDIA's stream
parallelism feature) handles the rest.

🇨🇳 没 MPS：每个客户端有**自己的** GPU context → 硬件时间片切换。

🇨🇳 有 MPS：每个客户端的 CUDA 调用都被**通过 socket 转发**给 MPS server，
server 用**一个共享 context** 把它们全部提交。从 GPU 看，就是一个 context
带很多 stream——和一个多线程进程一模一样。然后 NVIDIA 的 Hyper-Q（多
stream 并行硬件特性）负责真正的并发执行。

### Costs and gotchas
### 代价和坑

| Concern | Details |
|---|---|
| **Latency** | Each CUDA call traverses a Unix socket. For tiny kernels (<10 µs) this can add ~5–10% overhead. For real workloads it's negligible. |
| **Single point of failure** | If the MPS server crashes (e.g. one client OOMs and corrupts state), **all clients on that GPU die**. With Volta+ (compute ≥7.0), each client has its own address space → one client's segfault no longer kills siblings. We're on H100 (compute 9.0), so we're fine, but log carefully. |
| **No memory isolation** | Already covered. The merged context means `cuMemGetInfo` returns total free across all clients combined. |
| **Per-client SM caps** | `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50` set in the client's env caps that client at 50% of SMs. Useful if engine starves trainer. |
| **Lifecycle** | The control daemon must start *before* any GPU app on that node, and stop *after* all of them. Order matters. |

🇨🇳 注意事项：
- **延迟**：每次 CUDA 调用要过 Unix socket。极小 kernel（<10 µs）可能多 5–10%
  开销，正常负载可忽略。
- **单点故障**：MPS server 崩了（比如某客户端 OOM 把状态搞坏），**同 GPU 上
  所有客户端都跟着死**。Volta 及以后（compute ≥7.0）每个客户端有独立地址空
  间，一个客户端 segfault 不会再连累兄弟。我们是 H100（compute 9.0）所以没
  问题，但日志要仔细看。
- **无内存隔离**：合并 context 意味着 `cuMemGetInfo` 返回的"空闲"是所有客户
  端加起来的总和。
- **每客户端 SM 上限**：在客户端环境里设 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50`
  可以把那个客户端封顶到 50% SM，engine 抢 trainer 时有用。
- **生命周期**：daemon 必须**先于**该机上任何 GPU app 启动，**后于**它们关
  闭。顺序错了 client 会连不上。

### Compute capability gotcha
### Compute capability 的坑

MPS behavior changed at Volta:

🇨🇳 MPS 行为在 Volta 那代变了：

- **Pre-Volta (Pascal and older)**: All clients share one address space →
  one segfault kills everyone, harder to debug.
- **Volta+ (V100, A100, H100)**: Each client gets its own virtual address
  space inside the shared context. Isolation per-client.

🇨🇳 **Volta 之前（Pascal 及更老）**：所有客户端共享一个地址空间，一个 segfault
全员崩溃。**Volta 之后**（V100/A100/H100）：每个客户端在共享 context 里仍有
独立虚拟地址空间，隔离更好。我们在 H100 上稳。

### Environment variables cheat sheet
### 环境变量速查

```bash
# Server-side (on the node, before starting):
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps    # where the sockets live
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log     # server logs
nvidia-cuda-mps-control -d                        # start daemon

# Client-side (in each Ray worker's env):
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
# Optional cap:
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50

# Shutdown:
echo quit | nvidia-cuda-mps-control
```

🇨🇳 服务端（节点上，开任何 GPU 应用前）设 `CUDA_MPS_PIPE_DIRECTORY` 和
`CUDA_MPS_LOG_DIRECTORY`，再 `nvidia-cuda-mps-control -d` 启动 daemon。客户
端（每个 Ray worker 的环境）至少设 `CUDA_MPS_PIPE_DIRECTORY`，可选设
`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`。关闭用 `echo quit | nvidia-cuda-mps-control`。

---

## 6. PyTorch CUDA caching allocator
## 6. PyTorch CUDA caching allocator（缓存分配器）

PyTorch doesn't call `cudaMalloc` for every `torch.empty()`. That would be
~100 µs per tensor — unusable. Instead it has its own allocator:

🇨🇳 PyTorch 不会每次 `torch.empty()` 都调 `cudaMalloc`，那样每个 tensor 要
100 微秒，不能用。它自己实现了一个分配器：

### The default behavior
### 默认行为

1. On first `torch.empty(size)`, allocator calls `cudaMalloc` for a **big
   segment** (e.g. 20 MB or 2 GB depending on requested size — there are two
   "pool" size classes).
2. Hands you a sub-slice of that segment.
3. On `del tensor`, **does not** call `cudaFree`. Marks the slice as free;
   keeps the segment.
4. Next allocation of similar size reuses the cached segment — fast.

🇨🇳 ① 第一次 `torch.empty(size)` 时，分配器调一次 `cudaMalloc` 拿一个**大
段**（20 MB 或 2 GB，两个 pool）；② 切一片返给你；③ `del tensor` 时**不会**
调 `cudaFree`，只标记这片空闲，整段留着；④ 下次差不多大小的分配就复用这段
缓存——快。

This is why `torch.cuda.memory_allocated()` (actually used) and
`torch.cuda.memory_reserved()` (held by the allocator) differ:

🇨🇳 这就是 `torch.cuda.memory_allocated()`（实际用的）和
`torch.cuda.memory_reserved()`（分配器握着的）经常不一样的原因：

```
reserved = sum of all segments the allocator has cudaMalloc'd
allocated = sum of sub-slices currently handed out to your code
fragmentation = reserved - allocated
```

🇨🇳 `reserved - allocated` 就是**碎片**。碎片越大，你"明明还有显存却 OOM"
的概率越高。

### The fragmentation problem under colocate
### Colocate 下的碎片问题

Imagine the trainer's allocator holds a 1 GB segment. Inside it: 200 MB used,
800 MB cached-free. Meanwhile the engine wants to allocate 500 MB. The engine
calls `cudaMalloc(500MB)`. CUDA driver says "no, we only have 200 MB
contiguous left" — even though the trainer's segment has 800 MB of free space
*inside* it. **OOM despite plenty of "logical" free memory.**

🇨🇳 想象：trainer 分配器握着一段 1 GB segment，里面 200 MB 在用、800 MB 缓
存空闲。这时 engine 想分 500 MB，调 `cudaMalloc(500MB)`。驱动说"对不起，连
续可用只剩 200 MB"——尽管 trainer 段**内部**有 800 MB 是空闲的。
**于是 OOM，明明逻辑上还有的是显存。**

This is **THE classic two-process-one-GPU bug**.

🇨🇳 这是**"两进程一张卡"的经典 bug**。

### `expandable_segments` to the rescue
### `expandable_segments` 救场

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

What changes:

🇨🇳 改了什么：

- Instead of `cudaMalloc(20MB)`, the allocator does `cuMemAddressReserve(20MB
  of virtual address space)` then `cuMemCreate(physical pages)` and `cuMemMap`
  them in.
- When a sub-slice is freed and the segment becomes mostly empty, the allocator
  can `cuMemUnmap` the physical pages and **return them to the driver**,
  while keeping the virtual address range reserved.
- The other process can now `cudaMalloc` from the freed physical pages.

🇨🇳 ① 分配器不再用 `cudaMalloc(20MB)`，而是 `cuMemAddressReserve` 预留 20
MB **虚拟地址空间**，再 `cuMemCreate` 拿物理页 + `cuMemMap` 映射进去；
② 当 segment 大部分空了，分配器可以 `cuMemUnmap` **把物理页还给驱动**，但
保留虚拟地址段；③ 另一个进程就能 `cudaMalloc` 拿到这些物理页。

The cost: a small constant overhead per allocation (~1 µs for the map call).
Worth it.

🇨🇳 代价：每次分配多约 1 微秒的 map 开销。值。

### Tuning extras
### 其它可调项

`PYTORCH_CUDA_ALLOC_CONF` accepts a comma-separated list. Useful keys:

🇨🇳 `PYTORCH_CUDA_ALLOC_CONF` 接受逗号分隔列表，常用：

```
expandable_segments:True           # 上面讲过
max_split_size_mb:512              # 段被切得过小时合并阈值
garbage_collection_threshold:0.8   # reserved/total 超过这个比例时 GC
```

For colocate, `expandable_segments:True` is the only one that's not optional.

🇨🇳 对 colocate 来说，**`expandable_segments:True` 是必选项**，其它按情况调。

> ⚠️ **Outdated (see top banner).** True only for the **gloo** fallback
> transport. The default **CUDA IPC** transport needs plain `cudaMalloc`
> memory — colocate sets `expandable_segments:False` for IPC actors — so
> it is *not* universally mandatory.
> 🇨🇳 **已过时（见顶部）**：这只对 **gloo** 回退传输成立。默认的 **CUDA IPC**
> 传输需要普通 `cudaMalloc` 显存，colocate 会给 IPC 进程设
> `expandable_segments:False`，所以它并非永远必选。

### `set_per_process_memory_fraction`
### 硬上限：`set_per_process_memory_fraction`

```python
torch.cuda.set_per_process_memory_fraction(0.45, device=0)
```

This installs a hard ceiling **inside the PyTorch allocator**: it will refuse
to call `cudaMalloc` beyond `0.45 * total_vram`. You get a clean PyTorch OOM,
not a system crash.

🇨🇳 这是在 **PyTorch 分配器内部**装一个硬上限：超过 `0.45 * total_vram` 它
就拒绝继续 `cudaMalloc`，抛出干净的 PyTorch OOM，而不是把整张卡搞挂。

Caveats:

🇨🇳 注意：

- It caps the **PyTorch allocator only**. NCCL workspaces, cuBLAS handles,
  CUDA runtime overhead are not counted.
- The "total" is the **physical** GPU's total, not what `cuMemGetInfo` says
  is free under MPS. So if the engine has already eaten half the GPU, your
  trainer setting `0.45` may still try to grow into the engine's territory →
  OOM at the driver layer. That's why initialisation order matters
  (`knowledge.zh-en.md` §7).
- Must be called **before** any allocation on that device, otherwise it's
  silently ignored for already-cached segments.

🇨🇳 ① 只管 PyTorch 分配器，**NCCL workspace / cuBLAS handle / CUDA runtime
开销不算**；② "total" 是**物理总量**，不是 MPS 下 `cuMemGetInfo` 报的空闲。
所以如果 engine 已经吃了半张卡，你 trainer 设 0.45 还是可能撞到 engine 的
地盘，在驱动层 OOM——这就是为什么"初始化顺序"很重要（主文档 §7）；③ 必须
在该设备**首次分配之前**调用，否则对已缓存段无效。

---

## 7. cuBLAS / cuDNN / NCCL workspaces — the "safety_pad" story
## 7. cuBLAS / cuDNN / NCCL workspace —— "safety_pad" 的故事

When you call `torch.matmul`, under the hood:

🇨🇳 你调 `torch.matmul` 时，背后发生：

1. PyTorch looks up an **algorithm** in cuBLAS for the (M, N, K, dtype) shape.
2. cuBLAS requests a **workspace** — temporary scratch memory for the algorithm
   (split-K reductions, im2col tiles, etc.).
3. The workspace is allocated **outside the PyTorch caching allocator**, via
   raw `cudaMalloc`. cuBLAS owns it.
4. Workspace can be megabytes to ~256 MB depending on shape.

🇨🇳 ① PyTorch 在 cuBLAS 里查（M, N, K, dtype）对应的**算法**；② cuBLAS 申
请一块 **workspace**（算法所需的临时草稿区，split-K、im2col tiles 等）；
③ workspace **不走 PyTorch 缓存分配器**，是裸 `cudaMalloc` 出来的，归
cuBLAS 管；④ workspace 大小从几 MB 到约 256 MB 不等。

Same story for cuDNN (convolutions) and NCCL (ring buffers, ~50–200 MB per
communicator).

🇨🇳 cuDNN（卷积）和 NCCL（ring buffer，每个 communicator 50–200 MB）也类似。

**Implications for colocate budgeting:**

🇨🇳 **对 colocate 的预算意义**：

- If you set `train_frac=0.5` and `infer_frac=0.5` summing to 1.0, you've
  left **zero room** for cuBLAS/cuDNN/NCCL workspaces. They'll allocate
  anyway (outside the PyTorch fraction), and you'll OOM at the driver.
- The recommended `safety_pad ≈ 0.10` is exactly to cover this.
- NCCL workspace is per-communicator. Union world + FSDP subgroup + Gloo
  group = 2–3 NCCL communicators = couple hundred MB.

🇨🇳 ① 如果你设 `train_frac=0.5 + infer_frac=0.5 = 1.0`，给 cuBLAS / cuDNN /
NCCL **一点空间都没留**。它们照样会分配（在 PyTorch fraction 之外），驱动层
OOM 等着你；② 推荐 `safety_pad ≈ 0.10` 就是覆盖这些；③ NCCL workspace 是
per-communicator 的，union world + FSDP 子组 + Gloo 组 = 2~3 个 NCCL
communicator = 几百 MB。

To probe actual workspace usage on your shapes:

🇨🇳 想看你这套 shape 实际吃了多少 workspace：

```python
free_before, total = torch.cuda.mem_get_info()
# ... run a step ...
free_after, _ = torch.cuda.mem_get_info()
# (free_before - free_after) - torch.cuda.memory_reserved()
#   ≈ memory used by non-allocator stuff (workspaces, runtime, etc.)
```

---

## 8. NCCL internals just enough to debug colocate
## 8. NCCL 内部机制（够调 colocate bug 用就行）

### What a NCCL communicator is
### NCCL communicator 是什么

A NCCL communicator is the runtime object behind a PyTorch `ProcessGroup`. It
owns:

🇨🇳 NCCL communicator 是 PyTorch `ProcessGroup` 背后的运行时对象。它持有：

- A list of (rank, GPU, host, NIC) tuples for every participant.
- **Topology graph** — discovered via `nvidia-smi topo`, NVLink probing, etc.
- A **ring** (or tree, or double-binary-tree) of ranks for collectives.
- **Channel buffers** in HBM for staging chunks of tensors.

🇨🇳 ① 所有参与者的 (rank, GPU, 主机, 网卡) 列表；② **拓扑图**，靠
`nvidia-smi topo` + NVLink 探测得到；③ 用于 collective 的 rank **环**（ring）
或树（tree、double-binary-tree）；④ 在 HBM 里的**通道缓冲**（channel buffer），
用于切块 staging。

When you `dist.init_process_group(backend="nccl")` with WORLD_SIZE=16, NCCL
builds **one communicator** that knows about all 16 ranks. When you then
`dist.new_group(ranks=[0..7])`, it builds **a second communicator** with
just those 8.

🇨🇳 `dist.init_process_group(backend="nccl")` WORLD_SIZE=16 时，NCCL 建**一
个** communicator，知道所有 16 个 rank。你再 `dist.new_group(ranks=[0..7])`
就建**第二个** communicator，只含那 8 个。

### Why every rank must call `new_group`
### 为什么 `new_group` 必须所有 rank 一起调

`new_group` is a **collective**: every rank in the parent world must call it
with the **same `ranks=` argument**, even ranks that won't be in the subgroup.
This is because NCCL does a `bootstrap_allgather` under the hood to exchange
"do I need to be part of this new communicator?" info.

🇨🇳 `new_group` 是 **collective**：父 world 里**每个 rank** 都必须用**相同
的 `ranks=` 参数**调用它，即使该 rank 不会进入这个子组。原因是 NCCL 底层要
做一次 `bootstrap_allgather` 交换"我要不要加入这个新 communicator"的信息。

If only some ranks call it → hang.

🇨🇳 只有部分 rank 调 → **死锁**。

### P2P vs collectives
### P2P 与 collective

- **Collectives** (`all_reduce`, `all_gather`, `reduce_scatter`, `broadcast`):
  every rank in the communicator participates, NCCL uses the ring/tree
  topology, chunks tensors into channels for pipelining.
- **P2P** (`send`, `recv`): just two ranks talk. NCCL picks the best path
  (NVLink > PCIe > network). For same-GPU pairs (colocate), it's a single
  `cudaMemcpyDeviceToDevice`.

🇨🇳 ① **Collective**（`all_reduce`、`all_gather` 等）：communicator 里每个
rank 参与，NCCL 走环/树拓扑，把 tensor 切片走 channel 流水线；② **P2P**
（`send`/`recv`）：只两个 rank 通信，NCCL 选最佳路径（NVLink > PCIe > 网络）；
同卡对（colocate 的情况）就是一次 `cudaMemcpyDeviceToDevice`。

### Debug tip: `NCCL_DEBUG=INFO`
### 调试小贴士：`NCCL_DEBUG=INFO`

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,P2P
```

On startup, NCCL prints which transport it picked between every (sender,
receiver) pair: `via P2P/IPC`, `via NET/Socket`, `via NVLS`, etc. For colocate
sanity, you want to see `via P2P/IPC` between paired ranks.

🇨🇳 启动时 NCCL 会打印每对 (发送方, 接收方) 选了哪种传输方式：`via P2P/IPC`、
`via NET/Socket`、`via NVLS` 等。colocate 健康检查时，你希望看到 paired ranks
之间是 **`via P2P/IPC`**。如果看到 `NET/Socket`，说明 colocate 没生效。

> ⚠️ **Outdated (see top banner).** In colocate there is **no** NCCL P2P
> between the paired engine/trainer ranks — the hidden-state transport is
> CUDA IPC / gloo, not NCCL `send`/`recv`. The `via P2P/IPC` check
> applies only to the separate-GPU Phase-3 dummy P2P test. `NCCL_DEBUG`
> is still useful for the union-world bootstrap and the FSDP collectives.
> 🇨🇳 **已过时（见顶部）**：colocate 里配对的引擎/训练 rank 之间**没有**
> NCCL P2P —— hidden-state 走 CUDA IPC / gloo。这个 `via P2P/IPC` 检查只
> 适用于分卡的 Phase-3 dummy P2P 测试；`NCCL_DEBUG` 对 union-world 启动和
> FSDP 集合通信仍然有用。

---

## 9. Putting it all together: the colocate "stack"
## 9. 把以上拼起来：colocate 的"技术栈"

```
┌────────────────────────────────────────────────────────────────────┐
│ Ray placement group (1 bundle = 1 physical GPU)                    │
│   ├── num_gpus=0.45 → CUDA_VISIBLE_DEVICES=3 → trainer process     │
│   └── num_gpus=0.45 → CUDA_VISIBLE_DEVICES=3 → engine  process     │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ CUDA MPS daemon on the node                                        │
│   merges both processes' CUDA contexts into one on GPU 3           │
│   so kernels concurrently submit, no time-slicing                  │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ PyTorch in each process                                            │
│   set_per_process_memory_fraction(0.45)  ← PyTorch allocator cap   │
│   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ← anti-frag     │
│   default stream + transfer_stream                                 │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ NCCL                                                               │
│   union world communicator (2N ranks, NCCL)                        │
│   FSDP DP subgroup (N ranks, NCCL)                                 │
│   meta subgroup (2N ranks, Gloo)                                   │
│   intra-GPU send/recv → cudaMemcpyDeviceToDevice                   │
└────────────────────────────────────────────────────────────────────┘
```

🇨🇳 整个 colocate "栈"自上而下：

🇨🇳 ① **Ray placement group**：一个 bundle 对一张物理卡，trainer 和 engine
两个进程各分到 0.45 num_gpus，Ray 自动设 `CUDA_VISIBLE_DEVICES` 指向同一张
物理卡；② **CUDA MPS daemon**：把两个进程的 CUDA context 合并成一个，避免
时间片轮转；③ **PyTorch**：`set_per_process_memory_fraction(0.45)` 给分配
器装硬上限；`expandable_segments:True` 防碎片；默认 stream 跑 FSDP，独立
`transfer_stream` 跑 P2P；④ **NCCL**：union world（2N ranks）+ FSDP DP 子
组（N ranks）+ Gloo 元数据组（2N ranks）；同卡 send/recv 退化为
`cudaMemcpyDeviceToDevice`。

Every layer in that stack solves a specific problem the layer above creates:

🇨🇳 这一栈里每一层都在解决上一层带来的问题：

| Problem | Solved by |
|---|---|
| Two processes on one GPU → context switch overhead | **MPS** |
| MPS doesn't isolate memory → OOM risk | **`set_per_process_memory_fraction`** |
| Concurrent alloc/free → fragmentation | **`expandable_segments`** |
| sglang computes its budget from "free" at start | **Init trainer first**, then engine |
| FSDP and P2P share default stream → serialise | **Dedicated `transfer_stream`** |
| Engine ranks accidentally pulled into FSDP collectives | **FSDP DP subgroup**, not union world |
| Need same-GPU hidden-state handoff to be cheap | ~~NCCL intra-device fast path~~ → **CUDA IPC** zero-copy handoff (default) / gloo CPU-staging (fallback). NCCL can't span one physical GPU — see top banner. |

🇨🇳 表格说明每一层各解决什么问题——出 bug 时按这张表逆向定位很快。

---

## 10. Glossary delta
## 10. 词汇表（本文新增）

| Term | One-liner |
|---|---|
| **SM** | Streaming Multiprocessor — a "core cluster" on the GPU. H100 has 132. |
| **HBM** | High-Bandwidth Memory — the GPU's main DRAM (e.g. 80 GB on H100). |
| **Warp** | A group of 32 threads that execute in lockstep on one SM. |
| **CUDA context** | The GPU-side equivalent of a process: owns allocations, streams, modules. |
| **CUDA stream** | An in-order queue of GPU work inside a context. Different streams may overlap. |
| **CUDA event** | A marker recorded on one stream, awaited by another (or by the CPU). |
| **Caching allocator** | PyTorch's wrapper around `cudaMalloc` that keeps a per-process pool of segments. |
| **Segment** | A `cudaMalloc`'d chunk the allocator owns and sub-slices to your tensors. |
| **Expandable segment** | A segment built on `cuMemAddressReserve` + `cuMemMap` whose physical pages can be returned without losing the virtual address range. |
| **Workspace** | Scratch memory cuBLAS/cuDNN/NCCL allocate outside the PyTorch allocator. The reason `safety_pad` exists. |
| **D2D copy** | `cudaMemcpyDeviceToDevice` — intra-GPU HBM-to-HBM move. ~3 TB/s on H100. |
| **NCCL communicator** | The runtime object behind a `ProcessGroup`; owns topology and channel buffers. |
| **`NCCL_DEBUG=INFO`** | Env var that makes NCCL print which transport each pair picked. First thing to check when colocate looks slow. |
| **`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`** | Per-client SM cap (e.g. `50` = max 50% of SMs). |
| **Compute capability** | The GPU architecture version (H100 = 9.0). Determines MPS isolation guarantees. |

🇨🇳 词汇表对应中文：

| 术语 | 一句话解释 |
|---|---|
| **SM**（Streaming Multiprocessor） | GPU 上的"核心簇"。H100 有 132 个。 |
| **HBM**（High-Bandwidth Memory） | GPU 主显存（H100 是 80 GB）。 |
| **Warp** | 32 线程齐步执行的一组，在一个 SM 上跑。 |
| **CUDA context** | GPU 这一侧的"进程"：持有分配、stream、加载的模块。 |
| **CUDA stream** | context 内部一条按序的 GPU 工作队列。不同 stream 可并发。 |
| **CUDA event** | 在一条 stream 上记录、可被另一条 stream 或 CPU 等待的标记。 |
| **Caching allocator**（缓存分配器） | PyTorch 包在 `cudaMalloc` 上的池化分配器。 |
| **Segment**（段） | 分配器 `cudaMalloc` 拿到的一大块，再切片返给 tensor。 |
| **Expandable segment** | 基于 `cuMemAddressReserve` + `cuMemMap` 的段，物理页可归还、虚拟地址保留。 |
| **Workspace** | cuBLAS / cuDNN / NCCL 在 PyTorch 分配器之外申请的草稿区。`safety_pad` 存在的原因。 |
| **D2D copy** | `cudaMemcpyDeviceToDevice`——同卡 HBM 内拷贝。H100 上约 3 TB/s。 |
| **NCCL communicator** | `ProcessGroup` 背后的运行时对象，持有拓扑和通道缓冲。 |
| **`NCCL_DEBUG=INFO`** | 让 NCCL 打印每对 rank 选了哪种传输。colocate 慢时第一步先看它。 |
| **`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`** | 每客户端的 SM 上限（如 `50` = 最多用 50% SM）。 |
| **Compute capability** | GPU 架构版本号（H100 = 9.0），决定 MPS 隔离强度。 |

---

## 11. Further reading
## 11. 进一步阅读

1. **NVIDIA MPS docs** — <https://docs.nvidia.com/deploy/mps/index.html>
   全文都值得一读，特别是 "Architectural Overview" 和 "Provisioning Sequence"。

2. **CUDA Programming Guide — Streams and Events**
   <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution>

3. **CUDA Virtual Memory Management** (the API behind expandable_segments)
   <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#virtual-memory-management>

4. **PyTorch CUDA Semantics**
   <https://pytorch.org/docs/stable/notes/cuda.html>
   尤其是 "Memory management" 一节。

5. **NCCL User Guide — Environment Variables**
   <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html>
   调 colocate 时常用：`NCCL_DEBUG`、`NCCL_P2P_LEVEL`、`NCCL_IB_DISABLE`。

6. **PyTorch CUDACachingAllocator source**
   `torch/csrc/cuda/CUDACachingAllocator.cpp`
   想真懂分配器、想懂 `expandable_segments` 改了什么，直接读源码最快。

7. **Back to** [`knowledge.zh-en.md`](knowledge.zh-en.md) — now the references
   to "MPS context"、"NCCL intra-device path"、"caching allocator
   fragmentation" should all click.

   🇨🇳 **回头再读** [`knowledge.zh-en.md`](knowledge.zh-en.md)——里面提到的
   "MPS context"、"NCCL 设备内路径"、"caching allocator 碎片" 这些点现在应该
   都串起来了。
