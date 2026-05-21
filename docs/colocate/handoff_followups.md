# Colocate (PR #92 / issue #81) — leftover follow-ups (handoff)

> Handoff summary as of 2026-05-21. Self-contained — an agent picking this
> up should not need prior conversation context.

## Current state (grounding)

- **Branch:** `feature/colocate-training-inference`;
  **PR #92** (still `[WIP]` DRAFT) on `github.com/lightseekorg/TorchSpec`.
  Repo fork remote: `zhubohao911/TorchSpec`. The round-10
  transport-optimization work is merged in (`8905c55`); the PR
  description was rewritten concise — full detail preserved in
  `docs/colocate/pr92_detail.md`.
- **Transport:** CUDA IPC zero-copy is the **default**;
  `TORCHSPEC_COLOCATE_IPC=0` opts back to gloo CPU-staging. Three pieces:
  `e166c21` (non-destructive IPC capability probe — the old
  `reduce_tensor` probe wedged CUDA under MPS), `e62c941`
  (factory/train_group actively clear `expandable_segments` for IPC
  actors), and **round 10** (transport optimization investigated — no
  C++/CUDA/Triton kernel needed; `ipc-pipe` ack pipelining is a
  low-priority protocol-level 3.9× win, now wired into `cuda_ipc.py`
  behind the opt-in `TORCHSPEC_COLOCATE_IPC_PIPELINE` flag — see item 7,
  GPU-validated 2026-05-21).
- **Validated:** `run_smoke_host.sh --full` matrix is **green on 4×H100
  under IPC default** — 13 colocate tests pass (single-node). A
  3000-step 4-GPU multi-engine soak (round 10) ran clean. sglang patch:
  `v0.5.10.post1` is the default (`v0.5.8.post1` still selectable via
  `SGLANG_PATCH_VERSION`).
- **Docs of record:** `docs/colocate/implementation_log.md` (rounds
  1–10), `docs/colocate/transport_benchmark.md`,
  `docs/colocate/transport_optimization.md` (transport
  kernel-vs-protocol investigation + MPS-validated A/B),
  `docs/colocate/pr92_detail.md` (full PR narrative).
- **GPU access:** `runpodctl` is configured; SSH key
  `~/.runpod/ssh/runpodctl-ssh-key`; recipe = clone the branch +
  `bash scripts/colocate/run_smoke_host.sh --full`.

## Leftover items

| # | Item | Status | What "done" needs |
|---|---|---|---|
| 1 | **Multi-node 2-node run** | code-complete, untested | Run colocate on 2 nodes × 8 GPU. Code: `ensure_mps_on_all_nodes` (`torchspec/colocate/mps.py`), config `configs/colocate_qwen3_8b_2node.yaml`. Needs a 2-node rented cluster with cross-node networking. |
| 2 | **Large `engine_tp_size` (8-GPU TP per engine)** | validated only at `engine_tp_size=2` | Issue #81 scale-out wants 1 engine × 8-GPU TP. Rank math (`engine_global_rank`, `build_engine_tp_ranks`) + data plane (`colocate_loop.py` dispatch, `build_hidden_states_writer(tp_rank)`, `_send_hidden_states_to_nccl` in `colocate.patch`) handle any TP size but are only GPU-tested at tp=2 (`test_colocate_tp2.py`) + 2-engine fan-out (`test_colocate_multi_engine.py`). Needs an 8-GPU config + run. **Not currently in the PR's follow-up list.** |
| 3 | **`--stability` harness coverage** | ✅ **validated 2026-05-21** | `run_smoke_host.sh --stability` ran the 1000-step `test_stability.py` green on a 4×H100 pod (exit 0, ~321 s, peak-alloc flat ~25.75 GB). The `--stability` harness path is now exercised; combined with the pre-existing 3000-step soak (`transport_optimization.md` Part 5) the stability requirement is fully covered. |
| 4 | **1k-step convergence vs Mooncake** | ✅ **validated 2026-05-21** | `test_convergence_disagg_overlap` (`tests/colocate/test_convergence.py`) ran on a 4×H100 pod: colocate vs Mooncake-disagg loss curves over **1000 steps**, **mean rel. deviation 0.006 %, max 0.219 %** (2 % tolerance) — the colocate transport converges identically to the disaggregated baseline. Both training loops emit the env-gated `[loss_curve]` line the test parses; the disagg arm needs an importable Mooncake (`mooncake-transfer-engine==0.3.10.post1`). |
| 5 | **`pp_size > 1`** | open, **out-of-scope by agreement** | Pipeline parallelism — blocked by an explicit guard in `colocate.patch`. Listed for completeness; not planned. |
| 6 | **`grad_parity_smoke` (Qwen3-8B) confirm** | ✅ **confirmed 2026-05-21** | Passed in the `--full` matrix on a 4×H100 pod with `HF_TOKEN` set (15 passed, 0 failed — no HF-Hub 429). |
| 7 | **Productionize `ipc-pipe`** | ✅ **validated 2026-05-21 (one bug found + fixed)** | `ipc-pipe` is folded into `cuda_ipc.py` as `IpcPipelineTransport`, wired into `NcclHiddenStatesConnector` + `NcclMultiTensorFetcher` behind the opt-in `TORCHSPEC_COLOCATE_IPC_PIPELINE` flag (default off — plain `ipc_send`/`ipc_recv` unchanged). GPU run (`--full` with the flag): 12/13 colocate tests passed, but **`test_phase6_peak_alloc_flatness` OOM'd the memory-tight Qwen3-8B config** — the pool's variable-`seq_len` resize retired old buffers and *never freed them*, and the ×2 grow overshoot stacked on top of sglang's KV cache. **Fixed** in `cuda_ipc.py`: exact-size grow (no ×2 overshoot) + retired buffers freed one step later, the moment the trainer acks the resize step (it has re-opened the new handle by then). Re-test (`test_stability.py` with the flag) passed — peak-alloc flat ~25.75 GB, no OOM. Teardown-safe without a flush (engine needs none; trainer keeps ≤1 ack isend in flight), so **no sglang-patch change is needed**. **Low-priority / opt-in** — the transport is ~1 % of a colocate step. |

Items **1–4 are all facets of issue #81's "Scale-out" bullet** — the
feature is functionally complete and single-node-validated. With #3 and
#4 now validated (below), the **2-node (#1) and 8-GPU-TP (#2) scale-out
runs are the only remaining issue-#81 work**.

## One-pod batch validation — results (2026-05-21)

Items **3, 4, 6, 7** were validated in one session on a 4×H100
secure-cloud pod (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel`,
driver 580, ~1.6 h wall, ~$21):

| Item | Result |
|---|---|
| #6 `grad_parity_smoke` (Qwen3-8B) | ✅ **PASS** — in the `--full` matrix with `HF_TOKEN` (15 passed, 0 failed). |
| #3 `--stability` 1000-step | ✅ **PASS** — `run_smoke_host.sh --stability` exit 0 (~321 s), peak-alloc flat. |
| #7 `ipc-pipe` (`--full` + `IPC_PIPELINE=1`) | ✅ **PASS after fix** — 12/13 on the first pass; `test_phase6_peak_alloc_flatness` OOM'd the 8B config (pool retired-buffer leak + ×2 overshoot). Fixed in `cuda_ipc.py`; re-test passed, peak-alloc flat ~25.75 GB. |
| #4 convergence vs Mooncake (1000 steps) | ✅ **PASS** — colocate vs disagg loss curves overlap: mean 0.006 % / max 0.219 % deviation (2 % tol). |

Recipe to re-run — clone the branch, one 4×H100 secure pod (~$13/hr),
setup once then `--skip-setup`:

```bash
export HF_TOKEN=...                                 # #6 — avoids HF-Hub 429
bash scripts/colocate/run_smoke_host.sh --full      # setup + 13-test matrix (#6)

TORCHSPEC_COLOCATE_IPC_PIPELINE=1 \                 # #7 — pipelined transport
  bash scripts/colocate/run_smoke_host.sh --full --skip-setup

bash scripts/colocate/run_smoke_host.sh --stability --skip-setup   # #3

PHASE7_CONVERGE_STEPS=1000 \                        # #4 — target the node id
  bash scripts/colocate/run_smoke_host.sh --skip-setup \
  --tests=tests/colocate/test_convergence.py::test_convergence_disagg_overlap
```

Two gotchas learned this run: (a) target the **`::test_convergence_disagg_overlap`
node id** for #4 — running the whole `test_convergence.py` file also runs
the 8B `test_phase7_convergence_loss_decreases`, which `pytest -x` lets
block #4 on any failure; (b) the 8B runs leave large checkpoints in
`outputs/` — `rm -rf outputs/* /tmp/ray/*` between phases or the 200 GB
disk fills and Ray fails to acquire GPUs.

## PR #92 description

Rewritten concise (round 10): the ~270-line accumulated phase / round /
bug detail now lives in `docs/colocate/pr92_detail.md`, and the PR body
links there. The body's "Open follow-ups" line matches the table above
(`engine_tp_size=8` and `ipc-pipe` are both listed; the 1-step
Mooncake-disagg parity is *done* via `test_phase7_grad_parity_vs_disagg`,
so only the 1000-step convergence-curve comparison — item #4 — remains).

## Environment gotchas for the GPU work

- HF-Hub **429 rate-limits** unauthenticated Qwen3-8B fetches mid-`--full`;
  set `HF_TOKEN`, or pre-cache models + `HF_HUB_OFFLINE=1`.
- RunPod **community-cloud H100s are usually unavailable** — secure cloud
  (~$3.29/GPU/hr) works.
- This container type **blocks `py-spy`/ptrace**; for hung-process
  diagnosis use `faulthandler.dump_traceback_later` via a
  `sitecustomize.py`, not a SIGUSR1 handler.
