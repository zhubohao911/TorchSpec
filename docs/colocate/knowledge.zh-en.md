# Colocate Mode — Knowledge & Background （中英双语对照）

> 说明：本文是 [`knowledge.md`](knowledge.md) 的中英双语学习版。原文段落保留在前，中文翻译/解释紧跟其后（以 `🇨🇳` 引导）。代码块、表格、链接保持不变。

---

> Audience: anyone touching the colocate (training + inference on the same GPU) work
> for [Issue #81](https://github.com/lightseekorg/TorchSpec/issues/81).
>
> Goal: explain the *concepts* behind the design before we touch any code, so that
> when you read terms like "MPS", "share a bundle", "union NCCL world", you know
> exactly what is happening at the OS / driver / framework level.

🇨🇳 **读者**：所有要参与 "colocate（训练 + 推理放在同一张 GPU 上）" 工作的人，对应 [Issue #81](https://github.com/lightseekorg/TorchSpec/issues/81)。
🇨🇳 **目标**：在动代码之前，先把设计背后的*概念*讲清楚。这样当你看到 "MPS"、"share a bundle（共享一个 bundle）"、"union NCCL world（统一 NCCL 世界）" 这些词时，你能精确地知道它们在操作系统 / 驱动 / 框架层面到底发生了什么。

This document does **not** describe the implementation. See
[`implementation.md`](implementation.md) for the phased plan.

> ⚠️ **Cross-check — updated 2026-05-21.** This doc explains the *original*
> design, in which hidden states cross via **NCCL P2P, on-device,
> GPU-local `send`/`recv`**. That part did not survive contact with
> reality: NCCL hard-rejects a communicator with two ranks on one
> physical GPU (`ncclInvalidUsage`, "Duplicate GPU detected"), so
> same-GPU NCCL P2P is **impossible**. The shipped hidden-state transport
> is **CUDA IPC zero-copy (default)** with **gloo CPU-staging** as the
> opt-out fallback — both over a gloo `meta_group`. The union NCCL world,
> MPS, fractional bundles, and memory-cap concepts below are all still
> accurate. See [`implementation_log.md`](implementation_log.md) rounds 1
> / 7 / 9 and [`transport_benchmark.md`](transport_benchmark.md). The
> original NCCL-P2P text is kept for the design rationale and flagged
> inline.
>
> 🇨🇳 **交叉核对 —— 2026-05-21 更新。** 本文讲的是*最初*的设计：hidden
> states 通过 **NCCL P2P、设备内、GPU 本地的 `send`/`recv`** 传递。这部分
> 后来被证明行不通：NCCL 会硬拒绝"同一张物理 GPU 上有两个 rank"的
> communicator（报 `ncclInvalidUsage`、"Duplicate GPU detected"），所以
> 同卡 NCCL P2P **不可能实现**。实际落地的 hidden-state 传输是
> **CUDA IPC 零拷贝（默认）** + **gloo CPU 中转（可选回退）**，都走 gloo
> `meta_group`。下文关于 union NCCL world、MPS、分数 bundle、显存上限的
> 概念依然正确。详见 [`implementation_log.md`](implementation_log.md) 的
> round 1 / 7 / 9 与 [`transport_benchmark.md`](transport_benchmark.md)。
> 原 NCCL-P2P 文字保留以说明设计思路，并在文中就地标注。

🇨🇳 本文**不**讨论具体实现。分阶段实施方案见 [`implementation.md`](implementation.md)。

---

## 1. Where TorchSpec is today (the disaggregated baseline)
## 1. TorchSpec 现状（分离式 disaggregated 基线）

TorchSpec currently runs training and inference on **disjoint** GPUs and ships
hidden states between them through Mooncake (an RDMA / TCP KV store).

🇨🇳 当前 TorchSpec 把训练和推理跑在**互不相交**的 GPU 上，二者之间通过 Mooncake（一个基于 RDMA / TCP 的 KV 存储）来传递 hidden states（隐藏状态）。

```
2-node, 16-GPU example (today):

Node A (GPUs 0–7)        Node B (GPUs 0–7)
  Inference engines        Trainer ranks 0..7
  (sglang TP=8)            (FSDP-8)
        │
        │  hidden_states tensor
        ▼
   [Mooncake KV store]   ◀── network ──▶   trainer fetches by key
```

🇨🇳 上图：2 节点、16 卡的例子。Node A 跑推理引擎（sglang TP=8），Node B 跑训练（FSDP-8）。推理把 hidden_states 写入 Mooncake，训练通过 key 经网络拉回。

Concretely, each step looks like this:

🇨🇳 每一步具体长这样：

1. The **inference engine** (sglang Ray actor) runs the target model forward,
   gets `hidden_states ∈ [B, S, H]`, and writes it into Mooncake under some key.
   See [torchspec/inference/engine/sgl_engine.py](../../torchspec/inference/engine/sgl_engine.py)
   and [torchspec/transfer/mooncake/eagle_store.py](../../torchspec/transfer/mooncake/eagle_store.py).

   🇨🇳 **推理引擎**（sglang Ray actor）跑目标模型前向，得到 `hidden_states ∈ [B, S, H]`，并以某个 key 写入 Mooncake。

2. The engine returns just the **mooncake key** (a string) over Ray.

   🇨🇳 引擎仅通过 Ray 返回这个 **mooncake key**（一个字符串），不传大张量。

3. The `AsyncTrainingController` puts a `TrainSample(mooncake_key=..., shapes=..., dtypes=...)`
   onto a per-DP-rank Ray queue. See
   [torchspec/training/data_fetcher.py](../../torchspec/training/data_fetcher.py).

   🇨🇳 `AsyncTrainingController` 把 `TrainSample(mooncake_key=..., shapes=..., dtypes=...)` 放进每个 DP rank 各自的 Ray 队列里。

4. The **trainer** (`TrainerActor`) pulls a sample from its queue, calls
   `mooncake_store.get(key, shape, dtype, device=cuda)` to materialise the tensor
   on its GPU, and proceeds with FSDP forward/backward.

   🇨🇳 **训练 actor**（`TrainerActor`）从队列取样本，调 `mooncake_store.get(...)` 把张量物化到自己的 GPU 上，然后跑 FSDP 的前向/反向。

This is *async*: a background thread / `AsyncInferenceManager` keeps generating
ahead while the trainer is busy. There's a `SamplePool` capacity-based
backpressure to avoid filling Mooncake.

🇨🇳 这是**异步**流水：训练在忙的时候，后台线程 / `AsyncInferenceManager` 已经在提前生成新 batch。`SamplePool` 通过容量做反压（backpressure），防止把 Mooncake 撑爆。

### Why this is wasteful for some topologies
### 为什么这种拓扑在某些情况下浪费

For a 2-node / 16-GPU job:

🇨🇳 对一个 2 节点 / 16 卡的作业：

- We're forced to split, e.g. 8 train + 8 infer.

  🇨🇳 你被迫拆分资源，比如 8 卡训练 + 8 卡推理。

- Hidden states travel over **the network** (RDMA or TCP), even though the
  producer (engine TP rank 0 on node B GPU 0) and the consumer (trainer rank 0
  on node A GPU 0) could conceptually be the same physical device.

  🇨🇳 hidden states 走的是**网络**（RDMA 或 TCP），即使生产者（B 节点 GPU 0 上的引擎 TP rank 0）和消费者（A 节点 GPU 0 上的训练 rank 0）从概念上完全可以是同一个物理设备。

- We have a whole control-plane stack (`SamplePool`, Ray queues, mooncake master,
  retry loops) just to bridge that physical separation.

  🇨🇳 我们维护了整套控制面（`SamplePool`、Ray 队列、mooncake master、重试循环），只为弥合这种物理分离。

The **disaggregated** mode is still the right answer when training and inference
have very different scaling needs (e.g. 4 inference replicas feeding 32 trainer
ranks). But for the symmetric case — engine TP size == FSDP DP size — you can
do much better by putting them on the same GPU.

🇨🇳 当训练和推理的扩展需求差异很大（比如 4 个推理副本喂 32 个训练 rank）时，**分离式**仍然是对的答案。但**对称**场景下——引擎 TP size == FSDP DP size——把它们放在同一张 GPU 上能拿到大得多的收益。

---

## 2. What "colocate" actually means
## 2. "colocate" 到底是什么意思

**Colocate** = both the training process and the inference process are scheduled
onto the *same* physical GPUs at the same time.

🇨🇳 **Colocate（共置）** = 训练进程和推理进程被调度到*同一组*物理 GPU 上，同时运行。

```
2-node, 16-GPU example (colocate target):

Each GPU i (across both nodes):

    ┌──── GPU i (one physical device) ────┐
    │                                     │
    │   Process A: SglEngine TP rank i    │
    │   Process B: TrainerActor FSDP i    │
    │                                     │
    │   shared SMs (via CUDA MPS)         │
    │   shared VRAM (caps enforced soft)  │
    │                                     │
    └─────────────────────────────────────┘

    Engine rank i  ──── NCCL send (P2P, on-device) ────▶  Trainer rank i
```

🇨🇳 上图：colocate 目标拓扑。每张 GPU i 上同时有两个进程（SglEngine TP rank i 与 TrainerActor FSDP rank i），通过 CUDA MPS 共享 SM、共享 VRAM（软上限），通过 NCCL 点对点（P2P，设备内拷贝）传递 hidden_states。

So:

🇨🇳 也就是说：

- **Two OS processes** per GPU. Both have `CUDA_VISIBLE_DEVICES=i`.

  🇨🇳 每张 GPU 上有**两个 OS 进程**，两个进程的 `CUDA_VISIBLE_DEVICES` 都设为 i。

- **CUDA MPS** lets them concurrently submit kernels to the same GPU without
  context-switching overhead (more on this in §3).

  🇨🇳 **CUDA MPS** 让它们能并发地向同一张 GPU 提交 kernel，没有上下文切换开销（详见 §3）。

- The engine TP rank `i` and the trainer FSDP rank `i` are paired. Hidden states
  flow **GPU-local** between them via NCCL `send/recv`. No network, no Mooncake,
  no big payloads on Ray.

  🇨🇳 引擎 TP rank `i` 和训练 FSDP rank `i` 配对。hidden states 在它们之间走**GPU 本地**的 NCCL `send/recv`。不走网络、不经 Mooncake、Ray 上不传大块数据。

Two corollaries:

🇨🇳 由此推出两条推论：

- **Engine TP == FSDP world size.** Otherwise the 1:1 pairing doesn't make
  sense. (Multiple engines × the same TP can stack as `engine_count × TP = N`.)

  🇨🇳 **引擎 TP == FSDP world size**，否则 1:1 配对无意义。（多个引擎 × 相同 TP 可以拼成 `engine_count × TP = N`。）

- **Strictly serialised** within a step. The engine runs, then the trainer runs
  on the same GPU. No double-buffering, no pipeline overlap. Simpler control
  plane in exchange for a small (~10–20%) throughput hit vs. async.

  🇨🇳 一个 step 内**严格串行**：引擎跑完，训练再在同一张 GPU 上跑。没有双缓冲，没有 pipeline 重叠。控制面更简单，代价是相比异步模式有约 10–20% 的吞吐损失。

---

## 3. CUDA MPS — the "two processes share one GPU" enabler
## 3. CUDA MPS —— "两个进程共用一张 GPU" 的关键技术

### What it is
### 它是什么

**CUDA Multi-Process Service** is a NVIDIA daemon that lets multiple host
processes submit work to the same GPU **concurrently** (not just time-sliced).
Without MPS, the GPU runs one CUDA context at a time and round-robins between
processes — which is fine for throughput but adds a context-switch cost on
every kernel.

🇨🇳 **CUDA Multi-Process Service**（CUDA 多进程服务）是 NVIDIA 的一个守护进程（daemon），让多个宿主机进程能**并发地**（不是分时片）向同一张 GPU 提交任务。没有 MPS，GPU 一次只能跑一个 CUDA context，进程之间轮询 —— 吞吐没问题，但每个 kernel 都有上下文切换的开销。

With MPS:

🇨🇳 有了 MPS：

- One `nvidia-cuda-mps-control` daemon runs per GPU (or per node, supervising
  all GPUs).

  🇨🇳 每张 GPU（或每个节点统一管理所有 GPU）跑一个 `nvidia-cuda-mps-control` 守护进程。

- Client processes connect via Unix sockets at `CUDA_MPS_PIPE_DIRECTORY`.

  🇨🇳 客户端进程通过 `CUDA_MPS_PIPE_DIRECTORY` 下的 Unix socket 连接它。

- The MPS server merges their CUDA streams into one shared context, so kernels
  from different processes can interleave on the SMs.

  🇨🇳 MPS server 把多个进程的 CUDA stream 合并到一个共享 context，于是来自不同进程的 kernel 可以在 SM 上交错执行。

### Why we need it for colocate
### 为什么 colocate 需要它

- The engine and the trainer each have their own CUDA context (they're
  different processes). Without MPS they'd each get the GPU in turn → blocking.

  🇨🇳 引擎和训练是两个不同的进程，各自有独立的 CUDA context。没有 MPS 它们就要轮流占用 GPU → 互相阻塞。

- With MPS they can issue work concurrently. While the engine is doing target
  forward, the trainer's NCCL recv kernel is already queued and ready. While
  the trainer is doing fwd/bwd, the engine can prep its next batch.

  🇨🇳 有了 MPS 它们能并发提交任务：引擎在跑目标前向时，训练的 NCCL recv kernel 已经入队等待；训练在跑前向/反向时，引擎可以准备下一批数据。

### What MPS does *not* do
### MPS *不* 提供哪些能力

- **No memory isolation.** Both processes allocate from the same physical VRAM.
  If they both try to grow, you OOM. We have to enforce per-process caps in
  software (§7).

  🇨🇳 **不做显存隔离**。两个进程都从同一块物理 VRAM 分配。如果都想扩，就 OOM。必须靠软件层面给每个进程设上限（详见 §7）。

- **No fairness guarantees out of the box.** If one side dominates SM usage,
  the other slows down. There's an `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` env var
  you can use to cap per-process SM share (off by default; tuning knob).

  🇨🇳 **不保证开箱即用的公平性**。如果一方独占 SM，另一方就被拖慢。可以用 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 环境变量限定每个进程的 SM 份额（默认关闭，是个调优旋钮）。

- **MPS is per-node.** The daemon runs once per node and supervises all GPUs on
  it. Kubernetes/Ray needs to start it before any worker pod claims GPUs.

  🇨🇳 **MPS 是节点级别的**。每个节点跑一个 daemon，统管该节点上所有 GPU。Kubernetes/Ray 必须在 worker pod 占用 GPU 之前先启它。

### Mental model
### 心智模型

> MPS = "let two processes on the same GPU not have to take turns."
>
> That's it. Everything else (memory, scheduling fairness, lifecycle) is your
> problem.

🇨🇳 **一句话理解 MPS**：让同一张 GPU 上的两个进程"不用轮流来"。仅此而已。其他（显存、调度公平性、生命周期）都得你自己管。

### Operational notes
### 运维注意

- Start: `nvidia-cuda-mps-control -d` (one per node, before any GPU process).
- Set in client env: `CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps`,
  `CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log`.
- Stop: `echo quit | nvidia-cuda-mps-control`.
- Health check: `ls /tmp/nvidia-mps/control` and look for the socket.

🇨🇳 启动：`nvidia-cuda-mps-control -d`（每节点一个，在任何 GPU 进程之前）。客户端环境变量设 `CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps`、`CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log`。关闭：`echo quit | nvidia-cuda-mps-control`。健康检查：`ls /tmp/nvidia-mps/control` 看 socket 是否存在。

We'll wrap the start/stop in a Ray driver helper (see implementation doc Phase 1).

🇨🇳 我们会把启停逻辑封装到 Ray driver helper 里（见实现文档 Phase 1）。

---

## 4. Ray placement groups & bundles
## 4. Ray 的 placement group 和 bundle

This is where "training and inference actor share a bundle" comes from. Let's
unpack it.

🇨🇳 这一节解释 "training 和 inference actor 共享一个 bundle" 是什么意思。

### Bundles
### Bundle（资源捆）

A **bundle** in Ray is just a dict of resources Ray promises to reserve on a
single node. For TorchSpec a typical bundle is:

🇨🇳 Ray 里的 **bundle** 就是一个资源 dict，Ray 承诺在**单个节点**上预留这些资源。TorchSpec 的典型 bundle 是：

```python
{"GPU": 1, "CPU": 1}
```

A **placement group** (`PG`) is a list of bundles + a strategy:

🇨🇳 **placement group**（放置组，简称 `PG`）= 一组 bundle + 一种策略：

```python
bundles = [{"GPU": 1, "CPU": 1} for _ in range(N)]
pg = placement_group(bundles, strategy="PACK")
```

Strategies:
- `PACK`: try to put all bundles on as few nodes as possible.
- `SPREAD`: try to put each bundle on a different node.
- `STRICT_PACK` / `STRICT_SPREAD`: error if can't.

🇨🇳 策略：
- `PACK`：尽量把所有 bundle 塞进尽可能少的节点。
- `SPREAD`：尽量把每个 bundle 放到不同节点。
- `STRICT_PACK` / `STRICT_SPREAD`：做不到就直接报错。

When you create an actor, you tell Ray "schedule me onto bundle index `i` of
this PG":

🇨🇳 创建 actor 时，告诉 Ray "把我调度到这个 PG 的第 `i` 号 bundle 上"：

```python
SomeActor.options(
    num_gpus=1,
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=i,
    ),
).remote(...)
```

So a **bundle is essentially a logical "slot" on some GPU on some node**. The
PG locks N such slots, and you fill them with actors.

🇨🇳 所以一个 **bundle 本质上就是某节点某 GPU 上的一个逻辑"槽位"**。PG 把 N 个这样的槽位锁住，你往里塞 actor。

### How TorchSpec uses PGs today
### TorchSpec 当前怎么用 PG

See [torchspec/ray/placement_group.py](../../torchspec/ray/placement_group.py).

🇨🇳 详见 [torchspec/ray/placement_group.py](../../torchspec/ray/placement_group.py)。

- **Disaggregated (default):** one *unified* PG with `train_gpus + infer_gpus`
  bundles. The first `train_gpus` go to training actors, the rest go to engines.

  🇨🇳 **分离式（默认）**：一个*统一的* PG，包含 `train_gpus + infer_gpus` 个 bundle。前 `train_gpus` 个分给训练 actor，剩下的分给推理引擎。

- **`colocate=True` (existing partial):** a single PG with `max(train, infer)`
  bundles. Both `pgs["training"]` and `pgs["inference"]` point at this same PG —
  but actors today still claim a full `num_gpus=1` each, so you can't actually
  run two on the same bundle.

  🇨🇳 **`colocate=True`（现有的、不完整的实现）**：一个 PG，带 `max(train, infer)` 个 bundle。`pgs["training"]` 和 `pgs["inference"]` 指向同一个 PG —— 但当前每个 actor 仍声明 `num_gpus=1`，所以实际上还是不能让两个 actor 跑在同一个 bundle 里。

The existing colocate flag was meant for dev/debugging — share GPU across runs,
not run trainer+engine simultaneously.

🇨🇳 现有的 colocate 开关只是为了开发/调试用 —— 多次运行间共享 GPU，**并不是**让 trainer 和 engine 同时跑。

### What changes for "colocate trainer+engine on the same bundle"
### "trainer + engine 共用同一个 bundle" 需要改什么

Two things:

🇨🇳 两件事：

1. **Fractional `num_gpus`.** Each actor claims < 1.0 GPUs:
   ```python
   trainer_actor.options(num_gpus=0.45, ...)  # train_frac
   engine_actor.options(num_gpus=0.45, ...)   # infer_frac
   ```
   `0.45 + 0.45 < 1.0`, so Ray scheduler is happy putting **both** on the same
   bundle. Both processes see the **same physical GPU** (Ray sets
   `CUDA_VISIBLE_DEVICES` accordingly).

   🇨🇳 **小数 `num_gpus`**。每个 actor 申请不到 1.0 张 GPU：`0.45 + 0.45 < 1.0`，所以 Ray 调度器愿意把**两者**放进同一个 bundle。两个进程都看到**同一张物理 GPU**（Ray 会相应设置 `CUDA_VISIBLE_DEVICES`）。

2. **1:1 invariant.** We need engine TP rank `i` and trainer FSDP rank `i` to
   land on the same bundle. Today we *happen* to assign them in order; the
   colocate code has to **enforce** this rather than rely on coincidence.

   🇨🇳 **1:1 不变量（invariant）**：必须保证引擎 TP rank `i` 和训练 FSDP rank `i` 落到**同一个 bundle**。现在它们*碰巧*是按顺序分配的；colocate 代码必须**强制**这一条，而不是靠巧合。

So "training and inference share a bundle" literally means: the two Ray actors
are pinned to the same `(node, GPU)` slot, each consuming a fraction of it, and
both end up with `CUDA_VISIBLE_DEVICES=<that GPU>`.

🇨🇳 所以 "training 和 inference 共享一个 bundle" 字面意思就是：两个 Ray actor 被钉死在同一个 `(节点, GPU)` 槽位上，各占一部分，最终两者的 `CUDA_VISIBLE_DEVICES` 都指向同一张 GPU。

### The invariant in pictures
### 用图说明这个不变量

```
Bundle 0 → (node_A, gpu_0)
   ├── TrainerActor rank 0  (num_gpus=0.45)
   └── SglEngine    rank 0  (num_gpus=0.45)

Bundle 1 → (node_A, gpu_1)
   ├── TrainerActor rank 1
   └── SglEngine    rank 1

...

Bundle 15 → (node_B, gpu_7)
   ├── TrainerActor rank 15
   └── SglEngine    rank 15
```

The fact that both ranks see the same physical GPU is what makes NCCL P2P
between them an on-device copy.

🇨🇳 正因为两个 rank 看到同一张物理 GPU，它们之间的 NCCL P2P 才能退化成**设备内**拷贝。

---

## 5. NCCL P2P (`send` / `recv`)
## 5. NCCL 点对点（`send` / `recv`）

> ⚠️ **Superseded for the hidden-state plane (see top banner).** This
> section's premise — that the engine→trainer handoff is same-GPU NCCL
> `send`/`recv` "degenerating to an intra-device copy" — is wrong. NCCL
> refuses two ranks on one physical GPU. The shipped transport is CUDA
> IPC (default) / gloo CPU-staging (fallback). NCCL is still used for the
> **union-world bootstrap** and the trainer-only **FSDP collectives**
> described elsewhere — only the hidden-state P2P here is superseded.
> 🇨🇳 **本节关于 hidden-state 传输的前提已作废（见顶部说明）**：引擎→训练
> 的传递不是同卡 NCCL `send`/`recv`"退化为设备内拷贝" —— NCCL 不允许同卡
> 两 rank。实际传输是 CUDA IPC（默认）/ gloo 中转（回退）。NCCL 仍用于
> union-world 启动和 FSDP 集合通信，仅本节的 hidden-state P2P 作废。

NCCL is the GPU collective library. Most of TorchSpec's NCCL usage today is
**collectives**: all-reduce (FSDP grad sync), all-gather, reduce-scatter, etc.

🇨🇳 NCCL 是 GPU 集合通信库。TorchSpec 今天大部分 NCCL 用法都是**集合通信**：all-reduce（FSDP 梯度同步）、all-gather、reduce-scatter 等。

For colocate hidden-state transfer we want **point-to-point** instead.

🇨🇳 但 colocate 下传输 hidden states 我们要的是**点对点**。

### What `dist.send(tensor, dst)` does
### `dist.send(tensor, dst)` 在做什么

- Caller and receiver are both GPU ranks in the same NCCL process group.
- The sender posts a kernel that copies `tensor.data_ptr()` into the NCCL ring
  buffer, then onto the wire (or, in our case, into the receiver's memory).
- The receiver posts `dist.recv(out_tensor, src)` and NCCL drops the bytes
  there.

🇨🇳 调用方和接收方都是同一个 NCCL 进程组里的 GPU rank。发送方提交一个 kernel，把 `tensor.data_ptr()` 拷到 NCCL ring buffer，再发到对端（在我们的场景里就是直接落到接收方的内存）。接收方调 `dist.recv(out_tensor, src)`，NCCL 把字节落到那里。

When sender and receiver are on the **same physical GPU** (our colocate case),
NCCL uses CUDA's intra-device path (`cudaMemcpy` between two device buffers in
the same context view) — it never goes near PCIe / NVLink / network.

🇨🇳 当发送方和接收方在**同一张物理 GPU** 上（colocate 的场景），NCCL 走的是 CUDA 设备内路径（同一 context 视图下两块 device buffer 之间的 `cudaMemcpy`）—— 完全不碰 PCIe / NVLink / 网络。

### Why not reduce-scatter?
### 为什么不用 reduce-scatter？

The hidden states are already replicated across the engine's TP ranks (sglang
does an all-reduce at the TP boundary). So:

🇨🇳 hidden states 在引擎的 TP ranks 之间已经被复制了（sglang 在 TP 边界做了 all-reduce）。所以：

- Reduce-scatter would need a "reduce" step that collapses replicated copies
  → it'd actually just pick one and discard the rest, i.e. degenerate to
  scatter.

  🇨🇳 reduce-scatter 需要一个 "reduce" 步骤来合并这些复制副本 → 实际上就是挑一个、扔其他，退化成 scatter。

- A plain scatter still requires every rank to talk to every other rank.

  🇨🇳 朴素的 scatter 仍然要求每个 rank 都跟其他每个 rank 通信。

Local chunk + paired P2P is simpler and avoids patching sglang's TP boundary.

🇨🇳 "本地切块 + 配对 P2P" 更简单，并且不需要去改 sglang 的 TP 边界。

### Why a separate process group?
### 为什么需要一个独立的 process group？

PyTorch lets you create **subgroups** of the world (`dist.new_group(ranks=...)`).
Why bother?

🇨🇳 PyTorch 允许你从 world 里建**子组**（`dist.new_group(ranks=...)`）。为什么要这么折腾？

- The **FSDP DP group** must contain only trainer ranks. If you give FSDP the
  union world, it'll try to all-reduce gradients across engines too. Bad.

  🇨🇳 **FSDP DP group** 只能包含训练 rank。如果你把 union world 给 FSDP，它会试图把梯度 all-reduce 到引擎那边去 —— 灾难。

- The **CPU/Gloo group** is used for small metadata sync (step id, batch shape).
  You don't want that on NCCL because Gloo is faster for tiny CPU-side payloads.

  🇨🇳 **CPU/Gloo group** 用于同步小块元数据（step id、batch shape）。不要走 NCCL，因为 Gloo 在 CPU 侧小载荷更快。

For the actual hidden-state P2P, you can use the **global world** directly —
P2P between two specific ranks doesn't need a dedicated subgroup.

🇨🇳 真正的 hidden-state P2P 直接用 **global world** 就行 —— 两个特定 rank 间的 P2P 不需要专门的子组。

So we end up with three logical groups:

🇨🇳 最终我们有三个逻辑组：

| Group | Backend | Members | Used for |
|---|---|---|---|
| `world` (union) | NCCL | all `2N` ranks (N trainers + N engines) | P2P hidden-state transfer |
| `fsdp_dp` | NCCL | `N` trainer ranks only | FSDP grad/param collectives |
| `meta` | Gloo | all `2N` ranks (CPU) | step metadata broadcast |

🇨🇳 表格翻译：
- `world`（union 联合世界），NCCL 后端，全部 2N 个 rank（N 个训练 + N 个引擎），用于 P2P 传 hidden state。
- `fsdp_dp`，NCCL 后端，只含 N 个训练 rank，用于 FSDP 的梯度/参数集合通信。
- `meta`，Gloo 后端，全部 2N 个 rank（CPU 侧），用于广播 step 元数据。

---

## 6. PyTorch process groups: union world
## 6. PyTorch 进程组：union world（联合世界）

This is the bit that surprises people coming from "FSDP only" land.

🇨🇳 这一节会让"只玩 FSDP"的人觉得意外。

Today, `TrainerActor.init` calls `dist.init_process_group(backend="nccl")` with
`WORLD_SIZE = N` trainer ranks. That's the world; FSDP runs on it.

🇨🇳 现在，`TrainerActor.init` 调 `dist.init_process_group(backend="nccl")`，`WORLD_SIZE = N` 个训练 rank。这就是 world，FSDP 跑在它上面。

For colocate, we want **all `2*N` processes** (trainers + engines) in one NCCL
world, so they can `send/recv` directly.

🇨🇳 但对 colocate，我们要让**全部 `2*N` 个进程**（训练 + 引擎）都在同一个 NCCL world 里，这样它们才能直接 `send/recv`。

### Bootstrapping the union world
### 如何引导（bootstrap）这个 union world

1. The Ray driver picks one node and one port to be the **rendezvous point**
   (`MASTER_ADDR:MASTER_PORT`).

   🇨🇳 Ray driver 选一个节点和端口作为**会合点**（`MASTER_ADDR:MASTER_PORT`）。

2. Every actor (trainer + engine) sets these env vars before
   `init_process_group`:
   ```
   MASTER_ADDR=...
   MASTER_PORT=...
   WORLD_SIZE=2*N
   RANK=<unique 0..2N-1>
   ```

   🇨🇳 每个 actor（训练 + 引擎）在调 `init_process_group` 之前设上这些环境变量。

3. They all call `dist.init_process_group(backend="nccl", ...)` and PyTorch
   does the handshake.

   🇨🇳 所有 actor 一起调 `dist.init_process_group(backend="nccl", ...)`，由 PyTorch 完成握手。

The natural rank assignment: trainer ranks `0..N-1`, engine ranks `N..2N-1`.
That way `engine_rank_i = N + trainer_rank_i` for the colocated pair on GPU `i`.

🇨🇳 自然的 rank 分配：训练 rank 取 `0..N-1`，引擎 rank 取 `N..2N-1`。这样在 GPU `i` 上的 colocate 配对就有 `engine_rank_i = N + trainer_rank_i`。

### Subgroup construction
### 构造子组

After the union world is up, we run on every rank:

🇨🇳 union world 起好之后，每个 rank 上都跑：

```python
trainer_ranks = list(range(N))
fsdp_dp_group = dist.new_group(ranks=trainer_ranks, backend="nccl")
```

`new_group` is a **collective** — every rank in the world has to call it (with
the same `ranks=` argument), even those not in the subgroup.

🇨🇳 `new_group` 本身就是个**集合通信调用** —— world 里**每个 rank** 都必须调（带相同的 `ranks=`），哪怕它不在子组里。

The trainer then passes `fsdp_dp_group` to FSDP2's `fully_shard(...)`. From
FSDP's point of view, the world is just those N ranks — it never sees the
engine ranks.

🇨🇳 训练把 `fsdp_dp_group` 传给 FSDP2 的 `fully_shard(...)`。在 FSDP 看来，world 就是这 N 个 rank，它根本看不到引擎 rank。

### Subtlety: NCCL streams
### 微妙之处：NCCL stream

Both FSDP collectives and our P2P happen on the same NCCL underlying
communicator. If they share a CUDA stream, they serialise. To overlap, we put
the transfer P2P on a **dedicated CUDA stream**:

🇨🇳 FSDP 的集合通信和我们的 P2P 共用同一个底层 NCCL communicator。如果它们用同一个 CUDA stream，就会串行化。要想 overlap，把 transfer P2P 放到一个**独立的 CUDA stream** 上：

```python
transfer_stream = torch.cuda.Stream()
with torch.cuda.stream(transfer_stream):
    dist.recv(buf, src=engine_rank_i)
```

This is a small but important detail — without it, FSDP's all-gather and our
recv can serialise behind each other.

🇨🇳 这是个小但重要的细节 —— 不加这一手，FSDP 的 all-gather 和我们的 recv 会互相排队。

---

## 7. Memory isolation under MPS (the "soft caps" story)
## 7. MPS 下的显存隔离（"软上限"的故事）

MPS doesn't isolate VRAM. Both processes pull from the same `cudaMalloc` pool.
We need three layers of protection.

🇨🇳 MPS 不做 VRAM 隔离。两个进程都从同一个 `cudaMalloc` 池里拿。我们要做三层防护。

### Layer 1: Config-time budget
### 第一层：配置时的预算

```
train_frac + infer_frac + safety_pad <= 1.0
```

The `safety_pad ≈ 0.10` covers cuBLAS / cuDNN / NCCL workspaces, which both
processes implicitly use and aren't accounted for in the per-process fractions.

🇨🇳 `safety_pad ≈ 0.10` 用来覆盖 cuBLAS / cuDNN / NCCL 的工作区 —— 两个进程都会隐式占用，并没有被计入各自的 fraction 里。

For DFlash on H100: 0.45 / 0.45 is a reasonable starting point.

🇨🇳 H100 上跑 DFlash：0.45 / 0.45 是个合理起点。

### Layer 2: Per-process hard caps
### 第二层：每进程的硬上限

**Trainer side** — PyTorch caching allocator:

🇨🇳 **训练侧** —— PyTorch 缓存分配器：

```python
torch.cuda.set_per_process_memory_fraction(train_frac, device=local_gpu)
```

This is a *hard ceiling* enforced by PyTorch's `CUDACachingAllocator`. If the
trainer's allocator tries to grow past `train_frac × total_vram`, you get a
proper PyTorch OOM rather than a system-wide crash.

🇨🇳 这是 PyTorch `CUDACachingAllocator` 强制的**硬上限**。训练分配器一旦想超过 `train_frac × total_vram`，就抛规规矩矩的 PyTorch OOM，不会引发系统级崩溃。

**Engine side** — sglang's own knob:

🇨🇳 **引擎侧** —— sglang 自己的旋钮：

```python
sgl.Engine(..., mem_fraction_static=infer_frac)
```

But: **sglang computes its fraction off "free" memory at startup**, not total
memory. So if the trainer hasn't claimed its slice yet, sglang sees ~95% free
and over-allocates.

🇨🇳 但请注意：**sglang 启动时是基于"空闲显存"算 fraction**，而不是总显存。所以如果训练还没占住自己的份额，sglang 会看到约 95% 空闲，然后超分。

→ **Trainer must initialise first**, including a one-step warmup that brings
its allocator to peak. Then sglang starts and observes only `1 - train_frac`
free.

🇨🇳 → **必须让训练先初始化**，包括跑一步 warmup 把自己分配器顶到峰值。然后再启 sglang，它就只能看到 `1 - train_frac` 的空闲。

### Layer 3: Allocator hygiene
### 第三层：分配器卫生

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This tells PyTorch's allocator to use `cuMemAddressReserve` (virtual address
reservation) instead of fixed-size segments. Why we need it:

🇨🇳 这告诉 PyTorch 分配器用 `cuMemAddressReserve`（虚拟地址预留）而不是固定大小的段。为什么需要它：

- Concurrent alloc/free from two processes on the same GPU is a perfect
  fragmentation generator.

  🇨🇳 同一张 GPU 上两个进程并发地 alloc/free，是制造碎片的完美场景。

- Expandable segments mean PyTorch can release physical memory back to the
  driver without losing the virtual address range, so the *other* process can
  pick it up.

  🇨🇳 expandable segments 能让 PyTorch 把物理内存还给驱动，但保留虚拟地址范围，这样*另一个*进程就能接过去用。

Without this you'll see slowly growing peak VRAM until OOM around step 50–100.

🇨🇳 不加这个，你会看到峰值 VRAM 缓慢上涨，到第 50–100 步左右 OOM。

### Validation
### 验证方法

Run 1000 steps and check `torch.cuda.memory_stats()["allocated_bytes.all.peak"]`
on both processes after step 10. It should be flat. If it isn't, fragmentation
is winning.

🇨🇳 跑 1000 步，第 10 步后在两个进程上看 `torch.cuda.memory_stats()["allocated_bytes.all.peak"]`。它应该是平的。如果不平，碎片化正在赢。

---

## 8. The big picture: per-step timeline
## 8. 全景图：单步时间线

Here's what one training step looks like in colocate mode, end-to-end, on one
GPU:

🇨🇳 colocate 模式下，单 GPU 上一步训练端到端长这样：

```
time ─────────────────────────────────────────────────────────▶

[CPU/Gloo broadcast: step_id, B, S, loss_mask, input_ids]
       │
       ▼
ENGINE: target forward   ───▶  hidden = [B, S, H] on GPU
                                    │
                                    │ (still in engine process)
                                    │ chunk along batch:
                                    │   shard = hidden[i*B/TP : (i+1)*B/TP]
                                    │
                                    └── dist.send(shard, dst=trainer_rank) ──▶
                                                                            │
                                                                            ▼
TRAINER: dist.recv(buf, src=engine_rank)  ◀──────  (NCCL P2P, on-device copy)
       │
       ▼
TRAINER: fwd, bwd, opt step
       │
       ▼
[done; loop]
```

🇨🇳 流程：先用 Gloo CPU 组广播 step_id / B / S / loss_mask / input_ids → 引擎跑目标前向得到 `[B, S, H]` → 在引擎进程内按 batch 切分 → `dist.send` 把切片发给配对的 trainer rank → trainer `dist.recv` 收到（NCCL P2P，设备内拷贝）→ trainer 跑前向、反向、优化器 step → 进入下一步。

A few things to internalise:

🇨🇳 几个要点要内化：

- **The engine and trainer do not overlap.** While the engine is doing target
  forward, the trainer is idle (waiting on the metadata broadcast). While the
  trainer is doing fwd/bwd, the engine is idle (already finished its forward).
  This is a deliberate simplification vs. the async pipeline.

  🇨🇳 **引擎和训练并不重叠**。引擎在跑目标前向时，训练在空等元数据广播；训练在跑前向/反向时，引擎已经跑完空闲。这相对于异步流水是一个**有意为之**的简化。

- **The hidden-state copy is essentially free.** Same physical GPU, same
  context (under MPS), same VRAM pool. NCCL's intra-device path is a single
  `cudaMemcpyDeviceToDevice`.

  🇨🇳 **hidden-state 的拷贝几乎免费**。同物理 GPU、同 context（MPS 下）、同 VRAM 池。NCCL 设备内路径就是一次 `cudaMemcpyDeviceToDevice`。

- **MPS gives you nothing for free for *this* timeline** — there's no overlap
  by design. The reason MPS is needed is so the *transfer kernel itself* can be
  posted from the engine while the trainer's recv kernel is queued, without
  context switch overhead. Future async optimisations (next batch generation
  during current backward) would need MPS to actually overlap.

  🇨🇳 **就*这条*时间线本身而言 MPS 没给你白送任何收益** —— 设计上就没有重叠。需要 MPS 的真正原因是：让*传输 kernel 本身*能够从引擎那边提交，与训练侧已入队的 recv kernel 协作，省掉上下文切换开销。后续异步优化（在当前反向时生成下一批数据）才需要 MPS 来真正实现重叠。

---

## 9. Glossary
## 9. 术语表

| Term | One-liner |
|---|---|
| **Colocate** | Train + infer on the same physical GPU. |
| **Disaggregate** | Train + infer on disjoint GPUs (today's default). |
| **MPS** | NVIDIA daemon allowing concurrent kernels from multiple processes on one GPU. |
| **Bundle** | Ray's resource reservation slot (e.g. `{"GPU": 1, "CPU": 1}`) on a node. |
| **Placement group (PG)** | A list of bundles + a strategy (PACK/SPREAD). |
| **TP rank** | "Tensor parallel rank" within an inference engine. Engine 0 with TP=8 has TP ranks 0..7. |
| **DP rank** | "Data parallel rank" within FSDP. With FSDP-16, DP ranks are 0..15. |
| **Union world** | The single NCCL process group containing **both** trainer and engine ranks (`2*N` total). |
| **FSDP DP group** | NCCL subgroup with only the `N` trainer ranks; what FSDP collectives run on. |
| **Gloo group** | CPU process group used for small metadata broadcasts (step id, shapes). |
| **`mem_fraction_static`** | sglang's own VRAM cap, computed off *free* memory at engine startup. |
| **`set_per_process_memory_fraction`** | PyTorch caching allocator's hard cap. |
| **`expandable_segments`** | PyTorch alloc-conf flag that lets segments shrink/grow → less fragmentation under concurrent processes. |
| **Mooncake** | The current network KV store used to ship hidden states between trainer and engine in disaggregated mode. **Not used** in colocate. |

🇨🇳 术语中文对照：

| 术语 | 中文一句话解释 |
|---|---|
| **Colocate（共置）** | 训练 + 推理跑在同一张物理 GPU 上。 |
| **Disaggregate（分离）** | 训练 + 推理跑在互不相交的 GPU 上（当前默认）。 |
| **MPS** | NVIDIA 守护进程，允许多个进程的 kernel 在同一张 GPU 上并发执行。 |
| **Bundle** | Ray 在节点上预留的资源槽位（如 `{"GPU": 1, "CPU": 1}`）。 |
| **Placement group (PG)** | 一组 bundle + 一种策略（PACK / SPREAD）。 |
| **TP rank** | 推理引擎内的"张量并行 rank"。一个 TP=8 的引擎有 TP rank 0..7。 |
| **DP rank** | FSDP 内的"数据并行 rank"。FSDP-16 下 DP rank 是 0..15。 |
| **Union world（联合世界）** | 同时包含**训练和引擎** rank 的 NCCL 进程组（共 `2*N` 个 rank）。 |
| **FSDP DP group** | 只含 `N` 个训练 rank 的 NCCL 子组，FSDP 集合通信跑在它上面。 |
| **Gloo group** | CPU 端进程组，用于广播小块元数据（step id、形状）。 |
| **`mem_fraction_static`** | sglang 自己的 VRAM 上限，按引擎启动时的*空闲*显存计算。 |
| **`set_per_process_memory_fraction`** | PyTorch 缓存分配器的硬上限。 |
| **`expandable_segments`** | PyTorch 分配器配置项，让 segment 可伸缩 → 并发进程下减少碎片。 |
| **Mooncake** | 当前分离式模式下用于在训练和引擎间传 hidden state 的网络 KV 存储。**colocate 不用它**。 |

---

## 10. Recommended reading order before implementing
## 10. 动手实现前的推荐阅读顺序

1. **This document** end-to-end. Especially §3 (MPS), §4 (bundles), §6 (union world).

   🇨🇳 通读**本文**，特别是 §3（MPS）、§4（bundles）、§6（union world）。

2. Existing TorchSpec code:
   - [torchspec/ray/placement_group.py](../../torchspec/ray/placement_group.py) — read all of `create_placement_groups`.
   - [torchspec/ray/train_group.py](../../torchspec/ray/train_group.py) — `_allocate_gpus_for_training` (how a trainer actor claims its bundle today).
   - [torchspec/inference/factory.py](../../torchspec/inference/factory.py) — `_prepare_sgl_engines` (how an engine actor claims its bundle today).
   - [torchspec/training/trainer_actor.py](../../torchspec/training/trainer_actor.py) — `init` (how the NCCL world is set up today).

   🇨🇳 现有 TorchSpec 代码：通读 `create_placement_groups`；看训练 actor 当前如何申请 bundle（`_allocate_gpus_for_training`）；看引擎 actor 如何申请 bundle（`_prepare_sgl_engines`）；看 NCCL world 当前怎么搭起来（`TrainerActor.init`）。

3. PyTorch docs:
   - [`torch.distributed.new_group`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.new_group)
   - [`torch.cuda.set_per_process_memory_fraction`](https://pytorch.org/docs/stable/generated/torch.cuda.set_per_process_memory_fraction.html)
   - [Allocator config](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)

   🇨🇳 PyTorch 文档：`new_group`（建子组）、`set_per_process_memory_fraction`（硬上限）、Allocator 配置。

4. NVIDIA MPS overview: <https://docs.nvidia.com/deploy/mps/index.html>

   🇨🇳 NVIDIA MPS 概览。

5. sglang's `mem_fraction_static` source — search for it in the patched sglang
   in `patches/`.

   🇨🇳 看 sglang 中 `mem_fraction_static` 的源码 —— 在 `patches/` 下打过补丁的 sglang 里搜。

6. **Then** read [`implementation.md`](implementation.md) for the phased plan.

   🇨🇳 **最后**再读 [`implementation.md`](implementation.md) 看分阶段实施方案。
