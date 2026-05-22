# Colocate Mode — Knowledge & Background

> Audience: anyone touching the colocate (training + inference on the same GPU) work
> for [Issue #81](https://github.com/lightseekorg/TorchSpec/issues/81).
>
> Goal: explain the *concepts* behind the design before we touch any code, so that
> when you read terms like "MPS", "share a bundle", "union NCCL world", you know
> exactly what is happening at the OS / driver / framework level.

This document does **not** describe the implementation. See
[`implementation.md`](implementation.md) for the phased plan.

---

## 1. Where TorchSpec is today (the disaggregated baseline)

TorchSpec currently runs training and inference on **disjoint** GPUs and ships
hidden states between them through Mooncake (an RDMA / TCP KV store).

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

Concretely, each step looks like this:

1. The **inference engine** (sglang Ray actor) runs the target model forward,
   gets `hidden_states ∈ [B, S, H]`, and writes it into Mooncake under some key.
   See [torchspec/inference/engine/sgl_engine.py](../../torchspec/inference/engine/sgl_engine.py)
   and [torchspec/transfer/mooncake/eagle_store.py](../../torchspec/transfer/mooncake/eagle_store.py).
2. The engine returns just the **mooncake key** (a string) over Ray.
3. The `AsyncTrainingController` puts a `TrainSample(mooncake_key=..., shapes=..., dtypes=...)`
   onto a per-DP-rank Ray queue. See
   [torchspec/training/data_fetcher.py](../../torchspec/training/data_fetcher.py).
4. The **trainer** (`TrainerActor`) pulls a sample from its queue, calls
   `mooncake_store.get(key, shape, dtype, device=cuda)` to materialise the tensor
   on its GPU, and proceeds with FSDP forward/backward.

This is *async*: a background thread / `AsyncInferenceManager` keeps generating
ahead while the trainer is busy. There's a `SamplePool` capacity-based
backpressure to avoid filling Mooncake.

### Why this is wasteful for some topologies

For a 2-node / 16-GPU job:

- We're forced to split, e.g. 8 train + 8 infer.
- Hidden states travel over **the network** (RDMA or TCP), even though the
  producer (engine TP rank 0 on node B GPU 0) and the consumer (trainer rank 0
  on node A GPU 0) could conceptually be the same physical device.
- We have a whole control-plane stack (`SamplePool`, Ray queues, mooncake master,
  retry loops) just to bridge that physical separation.

The **disaggregated** mode is still the right answer when training and inference
have very different scaling needs (e.g. 4 inference replicas feeding 32 trainer
ranks). But for the symmetric case — engine TP size == FSDP DP size — you can
do much better by putting them on the same GPU.

---

## 2. What "colocate" actually means

**Colocate** = both the training process and the inference process are scheduled
onto the *same* physical GPUs at the same time.

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

So:

- **Two OS processes** per GPU. Both have `CUDA_VISIBLE_DEVICES=i`.
- **CUDA MPS** lets them concurrently submit kernels to the same GPU without
  context-switching overhead (more on this in §3).
- The engine TP rank `i` and the trainer FSDP rank `i` are paired. Hidden states
  flow **GPU-local** between them via NCCL `send/recv`. No network, no Mooncake,
  no big payloads on Ray.

Two corollaries:

- **Engine TP == FSDP world size.** Otherwise the 1:1 pairing doesn't make
  sense. (Multiple engines × the same TP can stack as `engine_count × TP = N`.)
- **Strictly serialised** within a step. The engine runs, then the trainer runs
  on the same GPU. No double-buffering, no pipeline overlap. Simpler control
  plane in exchange for a small (~10–20%) throughput hit vs. async.

---

## 3. CUDA MPS — the "two processes share one GPU" enabler

### What it is

**CUDA Multi-Process Service** is a NVIDIA daemon that lets multiple host
processes submit work to the same GPU **concurrently** (not just time-sliced).
Without MPS, the GPU runs one CUDA context at a time and round-robins between
processes — which is fine for throughput but adds a context-switch cost on
every kernel.

With MPS:

- One `nvidia-cuda-mps-control` daemon runs per GPU (or per node, supervising
  all GPUs).
- Client processes connect via Unix sockets at `CUDA_MPS_PIPE_DIRECTORY`.
- The MPS server merges their CUDA streams into one shared context, so kernels
  from different processes can interleave on the SMs.

### Why we need it for colocate

- The engine and the trainer each have their own CUDA context (they're
  different processes). Without MPS they'd each get the GPU in turn → blocking.
- With MPS they can issue work concurrently. While the engine is doing target
  forward, the trainer's NCCL recv kernel is already queued and ready. While
  the trainer is doing fwd/bwd, the engine can prep its next batch.

### What MPS does *not* do

- **No memory isolation.** Both processes allocate from the same physical VRAM.
  If they both try to grow, you OOM. We have to enforce per-process caps in
  software (§7).
- **No fairness guarantees out of the box.** If one side dominates SM usage,
  the other slows down. There's an `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` env var
  you can use to cap per-process SM share (off by default; tuning knob).
- **MPS is per-node.** The daemon runs once per node and supervises all GPUs on
  it. Kubernetes/Ray needs to start it before any worker pod claims GPUs.

### Mental model

> MPS = "let two processes on the same GPU not have to take turns."
>
> That's it. Everything else (memory, scheduling fairness, lifecycle) is your
> problem.

### Operational notes

- Start: `nvidia-cuda-mps-control -d` (one per node, before any GPU process).
- Set in client env: `CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps`,
  `CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log`.
- Stop: `echo quit | nvidia-cuda-mps-control`.
- Health check: `ls /tmp/nvidia-mps/control` and look for the socket.

We'll wrap the start/stop in a Ray driver helper (see implementation doc Phase 1).

---

## 4. Ray placement groups & bundles

This is where "training and inference actor share a bundle" comes from. Let's
unpack it.

### Bundles

A **bundle** in Ray is just a dict of resources Ray promises to reserve on a
single node. For TorchSpec a typical bundle is:

```python
{"GPU": 1, "CPU": 1}
```

A **placement group** (`PG`) is a list of bundles + a strategy:

```python
bundles = [{"GPU": 1, "CPU": 1} for _ in range(N)]
pg = placement_group(bundles, strategy="PACK")
```

Strategies:
- `PACK`: try to put all bundles on as few nodes as possible.
- `SPREAD`: try to put each bundle on a different node.
- `STRICT_PACK` / `STRICT_SPREAD`: error if can't.

When you create an actor, you tell Ray "schedule me onto bundle index `i` of
this PG":

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

### How TorchSpec uses PGs today

See [torchspec/ray/placement_group.py](../../torchspec/ray/placement_group.py).

- **Disaggregated (default):** one *unified* PG with `train_gpus + infer_gpus`
  bundles. The first `train_gpus` go to training actors, the rest go to engines.
- **`colocate=True` (existing partial):** a single PG with `max(train, infer)`
  bundles. Both `pgs["training"]` and `pgs["inference"]` point at this same PG —
  but actors today still claim a full `num_gpus=1` each, so you can't actually
  run two on the same bundle.

The existing colocate flag was meant for dev/debugging — share GPU across runs,
not run trainer+engine simultaneously.

### What changes for "colocate trainer+engine on the same bundle"

Two things:

1. **Fractional `num_gpus`.** Each actor claims < 1.0 GPUs:
   ```python
   trainer_actor.options(num_gpus=0.45, ...)  # train_frac
   engine_actor.options(num_gpus=0.45, ...)   # infer_frac
   ```
   `0.45 + 0.45 < 1.0`, so Ray scheduler is happy putting **both** on the same
   bundle. Both processes see the **same physical GPU** (Ray sets
   `CUDA_VISIBLE_DEVICES` accordingly).

2. **1:1 invariant.** We need engine TP rank `i` and trainer FSDP rank `i` to
   land on the same bundle. Today we *happen* to assign them in order; the
   colocate code has to **enforce** this rather than rely on coincidence.

So "training and inference share a bundle" literally means: the two Ray actors
are pinned to the same `(node, GPU)` slot, each consuming a fraction of it, and
both end up with `CUDA_VISIBLE_DEVICES=<that GPU>`.

### The invariant in pictures

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

---

## 5. NCCL P2P (`send` / `recv`)

NCCL is the GPU collective library. Most of TorchSpec's NCCL usage today is
**collectives**: all-reduce (FSDP grad sync), all-gather, reduce-scatter, etc.

For colocate hidden-state transfer we want **point-to-point** instead.

### What `dist.send(tensor, dst)` does

- Caller and receiver are both GPU ranks in the same NCCL process group.
- The sender posts a kernel that copies `tensor.data_ptr()` into the NCCL ring
  buffer, then onto the wire (or, in our case, into the receiver's memory).
- The receiver posts `dist.recv(out_tensor, src)` and NCCL drops the bytes
  there.

When sender and receiver are on the **same physical GPU** (our colocate case),
NCCL uses CUDA's intra-device path (`cudaMemcpy` between two device buffers in
the same context view) — it never goes near PCIe / NVLink / network.

### Why not reduce-scatter?

The hidden states are already replicated across the engine's TP ranks (sglang
does an all-reduce at the TP boundary). So:

- Reduce-scatter would need a "reduce" step that collapses replicated copies
  → it'd actually just pick one and discard the rest, i.e. degenerate to
  scatter.
- A plain scatter still requires every rank to talk to every other rank.

Local chunk + paired P2P is simpler and avoids patching sglang's TP boundary.

### Why a separate process group?

PyTorch lets you create **subgroups** of the world (`dist.new_group(ranks=...)`).
Why bother?

- The **FSDP DP group** must contain only trainer ranks. If you give FSDP the
  union world, it'll try to all-reduce gradients across engines too. Bad.
- The **CPU/Gloo group** is used for small metadata sync (step id, batch shape).
  You don't want that on NCCL because Gloo is faster for tiny CPU-side payloads.

For the actual hidden-state P2P, you can use the **global world** directly —
P2P between two specific ranks doesn't need a dedicated subgroup.

So we end up with three logical groups:

| Group | Backend | Members | Used for |
|---|---|---|---|
| `world` (union) | NCCL | all `2N` ranks (N trainers + N engines) | P2P hidden-state transfer |
| `fsdp_dp` | NCCL | `N` trainer ranks only | FSDP grad/param collectives |
| `meta` | Gloo | all `2N` ranks (CPU) | step metadata broadcast |

---

## 6. PyTorch process groups: union world

This is the bit that surprises people coming from "FSDP only" land.

Today, `TrainerActor.init` calls `dist.init_process_group(backend="nccl")` with
`WORLD_SIZE = N` trainer ranks. That's the world; FSDP runs on it.

For colocate, we want **all `2*N` processes** (trainers + engines) in one NCCL
world, so they can `send/recv` directly.

### Bootstrapping the union world

1. The Ray driver picks one node and one port to be the **rendezvous point**
   (`MASTER_ADDR:MASTER_PORT`).
2. Every actor (trainer + engine) sets these env vars before
   `init_process_group`:
   ```
   MASTER_ADDR=...
   MASTER_PORT=...
   WORLD_SIZE=2*N
   RANK=<unique 0..2N-1>
   ```
3. They all call `dist.init_process_group(backend="nccl", ...)` and PyTorch
   does the handshake.

The natural rank assignment: trainer ranks `0..N-1`, engine ranks `N..2N-1`.
That way `engine_rank_i = N + trainer_rank_i` for the colocated pair on GPU `i`.

### Subgroup construction

After the union world is up, we run on every rank:

```python
trainer_ranks = list(range(N))
fsdp_dp_group = dist.new_group(ranks=trainer_ranks, backend="nccl")
```

`new_group` is a **collective** — every rank in the world has to call it (with
the same `ranks=` argument), even those not in the subgroup.

The trainer then passes `fsdp_dp_group` to FSDP2's `fully_shard(...)`. From
FSDP's point of view, the world is just those N ranks — it never sees the
engine ranks.

### Subtlety: NCCL streams

Both FSDP collectives and our P2P happen on the same NCCL underlying
communicator. If they share a CUDA stream, they serialise. To overlap, we put
the transfer P2P on a **dedicated CUDA stream**:

```python
transfer_stream = torch.cuda.Stream()
with torch.cuda.stream(transfer_stream):
    dist.recv(buf, src=engine_rank_i)
```

This is a small but important detail — without it, FSDP's all-gather and our
recv can serialise behind each other.

---

## 7. Memory isolation under MPS (the "soft caps" story)

MPS doesn't isolate VRAM. Both processes pull from the same `cudaMalloc` pool.
We need three layers of protection.

### Layer 1: Config-time budget

```
train_frac + infer_frac + safety_pad <= 1.0
```

The `safety_pad ≈ 0.10` covers cuBLAS / cuDNN / NCCL workspaces, which both
processes implicitly use and aren't accounted for in the per-process fractions.

For DFlash on H100: 0.45 / 0.45 is a reasonable starting point.

### Layer 2: Per-process hard caps

**Trainer side** — PyTorch caching allocator:

```python
torch.cuda.set_per_process_memory_fraction(train_frac, device=local_gpu)
```

This is a *hard ceiling* enforced by PyTorch's `CUDACachingAllocator`. If the
trainer's allocator tries to grow past `train_frac × total_vram`, you get a
proper PyTorch OOM rather than a system-wide crash.

**Engine side** — sglang's own knob:

```python
sgl.Engine(..., mem_fraction_static=infer_frac)
```

But: **sglang computes its fraction off "free" memory at startup**, not total
memory. So if the trainer hasn't claimed its slice yet, sglang sees ~95% free
and over-allocates.

→ **Trainer must initialise first**, including a one-step warmup that brings
its allocator to peak. Then sglang starts and observes only `1 - train_frac`
free.

### Layer 3: Allocator hygiene

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This tells PyTorch's allocator to use `cuMemAddressReserve` (virtual address
reservation) instead of fixed-size segments. Why we need it:

- Concurrent alloc/free from two processes on the same GPU is a perfect
  fragmentation generator.
- Expandable segments mean PyTorch can release physical memory back to the
  driver without losing the virtual address range, so the *other* process can
  pick it up.

Without this you'll see slowly growing peak VRAM until OOM around step 50–100.

### Validation

Run 1000 steps and check `torch.cuda.memory_stats()["allocated_bytes.all.peak"]`
on both processes after step 10. It should be flat. If it isn't, fragmentation
is winning.

---

## 8. The big picture: per-step timeline

Here's what one training step looks like in colocate mode, end-to-end, on one
GPU:

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

A few things to internalise:

- **The engine and trainer do not overlap.** While the engine is doing target
  forward, the trainer is idle (waiting on the metadata broadcast). While the
  trainer is doing fwd/bwd, the engine is idle (already finished its forward).
  This is a deliberate simplification vs. the async pipeline.
- **The hidden-state copy is essentially free.** Same physical GPU, same
  context (under MPS), same VRAM pool. NCCL's intra-device path is a single
  `cudaMemcpyDeviceToDevice`.
- **MPS gives you nothing for free for *this* timeline** — there's no overlap
  by design. The reason MPS is needed is so the *transfer kernel itself* can be
  posted from the engine while the trainer's recv kernel is queued, without
  context switch overhead. Future async optimisations (next batch generation
  during current backward) would need MPS to actually overlap.

---

## 9. Glossary

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

---

## 10. Recommended reading order before implementing

1. **This document** end-to-end. Especially §3 (MPS), §4 (bundles), §6 (union world).
2. Existing TorchSpec code:
   - [torchspec/ray/placement_group.py](../../torchspec/ray/placement_group.py) — read all of `create_placement_groups`.
   - [torchspec/ray/train_group.py](../../torchspec/ray/train_group.py) — `_allocate_gpus_for_training` (how a trainer actor claims its bundle today).
   - [torchspec/inference/factory.py](../../torchspec/inference/factory.py) — `_prepare_sgl_engines` (how an engine actor claims its bundle today).
   - [torchspec/training/trainer_actor.py](../../torchspec/training/trainer_actor.py) — `init` (how the NCCL world is set up today).
3. PyTorch docs:
   - [`torch.distributed.new_group`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.new_group)
   - [`torch.cuda.set_per_process_memory_fraction`](https://pytorch.org/docs/stable/generated/torch.cuda.set_per_process_memory_fraction.html)
   - [Allocator config](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
4. NVIDIA MPS overview: <https://docs.nvidia.com/deploy/mps/index.html>
5. sglang's `mem_fraction_static` source — search for it in the patched sglang
   in `patches/`.
6. **Then** read [`implementation.md`](implementation.md) for the phased plan.
