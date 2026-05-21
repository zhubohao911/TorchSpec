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
  low-priority protocol-level 3.9× win, **not** wired into `cuda_ipc.py`).
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
| 3 | **Literal 1000-step stability** | only 200 steps recorded green | `run_smoke_host.sh --stability` (sets 1000 steps) + nightly `.github/workflows/colocate-stability.yml` are wired, but no recorded 1000-step green result. Issue #81 asks for 1000 steps, no mem growth after step 10. ~$5–8 on 4×H100. |
| 4 | **1k-step convergence vs Mooncake** | not done | Issue #81 scale-out: run colocate and disaggregated/Mooncake for ~1000 steps on the same job, compare loss curves. `test_phase7_grad_parity_vs_disagg` covers *1-step* per-parameter parity; the *convergence-curve* comparison is separate and never run. |
| 5 | **`pp_size > 1`** | open, **out-of-scope by agreement** | Pipeline parallelism — blocked by an explicit guard in `colocate.patch`. Listed for completeness; not planned. |
| 6 | **`grad_parity_smoke` (Qwen3-8B) confirm** | minor / environment | This test didn't run in the IPC-default `--full` (HF-Hub `429` rate-limit on the unauthenticated Qwen3-8B fetch). Re-confirm with `HF_TOKEN` set, or `HF_HUB_OFFLINE=1` against a warm model cache. Not a colocate defect. |
| 7 | **Productionize `ipc-pipe`** | benchmark prototype only | Round 10's `ipc-pipe` (persistent send-buffer pool + one-step ack pipelining) is a 3.9× win on the engine-`send()` stall, MPS-validated, but lives only in `scripts/colocate/bench_transport.py`. **Low-priority** — the transport is ~1 % of a colocate step, not a step-time bottleneck. To ship: fold into `torchspec/colocate/cuda_ipc.py` behind a `TORCHSPEC_COLOCATE_IPC_PIPELINE` flag, with the `flush()`-at-loop-exit drain + variable-`seq_len` pool-resize handling from `transport_optimization.md` Opt 2. |

Items **1–4 are all facets of issue #81's "Scale-out" bullet** — the
feature is functionally complete and single-node-validated; the 2-node /
8-GPU-TP / 1k-step scale validation is the real remaining work.

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
