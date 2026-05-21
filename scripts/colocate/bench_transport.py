#!/usr/bin/env python3
# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Benchmark: gloo CPU-staged vs CUDA IPC zero-copy hidden-state transport.

The colocate hidden-state plane moves engine→trainer tensors between two
processes that share one physical GPU. Two transports exist:

  * **gloo CPU-staged** — engine ``D->H`` copy, gloo ship, trainer
    ``H->D`` copy. Two PCIe-class copies + a host memcpy per tensor.
  * **CUDA IPC** (the default) — engine exports a CUDA IPC handle, the
    trainer maps that memory and does a single on-device ``D->D`` copy.
    No host round-trip.

This script spins up **two processes on GPU 0** (the colocate topology),
forms a 2-rank gloo group, and times both transports across a payload
sweep plus a realistic Eagle3-shaped multi-tensor case. It reports
end-to-end latency (mean / p50 / p99), effective bandwidth, the
engine/trainer split, and a per-stage breakdown — then writes a
Markdown report.

It needs **1 GPU**. MPS is not required (CUDA IPC works process-to-
process regardless); run it under MPS for a fully faithful colocate
picture. Do **not** export ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments``
— CUDA IPC needs plain ``cudaMalloc`` memory (the script warns if it is
set).

Run on a GPU host (after ``run_smoke_host.sh`` has installed torchspec,
or any env with torch + this repo importable)::

    python scripts/colocate/bench_transport.py
    python scripts/colocate/bench_transport.py --iters 50 --sizes-mb 1,16,256
    python scripts/colocate/bench_transport.py --out colocate-transport-bench.md
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

# Make the repo importable when run as a plain script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

# Load torchspec/colocate/cuda_ipc.py directly by file path. cuda_ipc.py
# has no torchspec-internal imports, so loading it standalone avoids
# triggering torchspec's package __init__ chain (which pulls heavy model
# deps). The benchmark then runs on a bare torch install — no
# `pip install -e .` needed on the GPU host.
import importlib.util as _ilu

_CUDA_IPC_PATH = _REPO_ROOT / "torchspec" / "colocate" / "cuda_ipc.py"
_spec = _ilu.spec_from_file_location("colocate_cuda_ipc", _CUDA_IPC_PATH)
_cuda_ipc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cuda_ipc)
ipc_send, ipc_recv = _cuda_ipc.ipc_send, _cuda_ipc.ipc_recv

TRAINER_RANK = 0
ENGINE_RANK = 1

# IPC handshake tags — kept distinct from cuda_ipc.py's (7001-7003) and
# from the gloo per-tensor tags so nothing collides on the shared group.
_BREAKDOWN_TAG = 9100


# ---------------------------------------------------------------------------
# Transport implementations
# ---------------------------------------------------------------------------
# The gloo path is replicated inline here (it mirrors the gloo branch of
# NcclHiddenStatesConnector.send / NcclMultiTensorFetcher.recv_step) so the
# benchmark stays self-contained and does not drag in the ray-heavy engine
# package. The IPC path calls the real torchspec.colocate.cuda_ipc API.


def gloo_send(tensors: dict, dst: int, group) -> None:
    """Engine side, gloo: D->H copy each tensor, ship over gloo."""
    for tag, name in enumerate(sorted(tensors)):
        cpu_t = tensors[name].detach().to("cpu", copy=True).contiguous()
        dist.send(cpu_t, dst=dst, group=group, tag=tag)


def gloo_recv(specs: dict, src: int, device, group) -> dict:
    """Trainer side, gloo: recv into host buffer, H->D copy to device."""
    out = {}
    for tag, name in enumerate(sorted(specs)):
        shape, dtype = specs[name]
        buf = torch.empty(tuple(shape), dtype=dtype, device="cpu")
        dist.recv(buf, src=src, group=group, tag=tag)
        out[name] = buf.to(device)
    torch.cuda.synchronize()
    return out


# ipc_send / ipc_recv are imported from torchspec.colocate.cuda_ipc.


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------

def _single_tensor_spec(mb: float) -> dict:
    """One 2-D bf16 tensor of approximately ``mb`` megabytes."""
    cols = 4096
    rows = max(1, int(mb * 1024 * 1024) // (cols * 2))
    return {"hidden_states": ((rows, cols), torch.bfloat16)}


def _eagle3_spec(tokens: int, hidden: int) -> dict:
    """Realistic Eagle3-shaped multi-tensor hidden-state set."""
    return {
        "hidden_states": ((tokens, hidden), torch.bfloat16),
        "aux_hidden_states": ((tokens, 3 * hidden), torch.bfloat16),
        "last_hidden_states": ((tokens, hidden), torch.bfloat16),
    }


def _spec_bytes(spec: dict) -> int:
    total = 0
    for shape, dtype in spec.values():
        n = 1
        for d in shape:
            n *= d
        total += n * torch.empty(0, dtype=dtype).element_size()
    return total


def _make_payload(spec: dict, device, seed: int) -> dict:
    """Fresh CUDA tensors — fresh each iteration, like real hidden states
    (so CUDA IPC pays a real cudaIpcOpenMemHandle every step)."""
    g = torch.Generator(device=device).manual_seed(seed)
    out = {}
    for name, (shape, dtype) in spec.items():
        out[name] = torch.randn(tuple(shape), generator=g, device=device).to(dtype)
    torch.cuda.synchronize()
    return out


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _stats(samples_s: list) -> dict:
    """mean / p50 / p99 / min in milliseconds from a list of seconds."""
    ms = sorted(s * 1e3 for s in samples_s)
    n = len(ms)
    return {
        "mean": sum(ms) / n,
        "p50": ms[int(0.50 * (n - 1))],
        "p99": ms[int(0.99 * (n - 1))],
        "min": ms[0],
        "n": n,
    }


def _bench_transport(name, send_fn, recv_fn, spec, *, iters, warmup,
                     rank, device, group) -> dict:
    """Time one transport on one payload. Returns span + own-side stats.

    ``span`` is the barrier-to-barrier end-to-end transfer time (both
    ranks see the same window). ``own`` is this rank's own send/recv
    call duration — the engine/trainer split.
    """
    spans, own = [], []
    for i in range(warmup + iters):
        payload = _make_payload(spec, device, seed=i) if rank == ENGINE_RANK else None
        torch.cuda.synchronize()
        dist.barrier(group)
        t0 = time.perf_counter()
        if rank == ENGINE_RANK:
            ts = time.perf_counter()
            send_fn(payload, TRAINER_RANK, group)
            te = time.perf_counter()
        else:
            ts = time.perf_counter()
            recv_fn(spec, ENGINE_RANK, device, group)
            te = time.perf_counter()
        dist.barrier(group)
        t1 = time.perf_counter()
        if i >= warmup:
            spans.append(t1 - t0)
            own.append(te - ts)
    return {"transport": name, "span": spans, "own": own}


# ---------------------------------------------------------------------------
# Per-stage breakdown (instrumented replicas of each transport's stages)
# ---------------------------------------------------------------------------

def _breakdown(spec, *, iters, rank, device, group) -> dict:
    """Stage-by-stage anatomy of one payload, for both transports.

    The replicas below mirror NcclHiddenStatesConnector / cuda_ipc.py so
    each stage can be timed individually. Engine reports its stages,
    trainer reports its stages; merged on rank 0.
    """
    names = sorted(spec)
    acc: dict = {}

    def add(stage, dt):
        acc.setdefault(stage, []).append(dt)

    for i in range(iters):
        payload = _make_payload(spec, device, seed=1000 + i) if rank == ENGINE_RANK else None
        torch.cuda.synchronize()

        # ---- gloo ----
        dist.barrier(group)
        if rank == ENGINE_RANK:
            cpu_tensors = []
            t = time.perf_counter()
            for name in names:
                cpu_tensors.append(
                    payload[name].detach().to("cpu", copy=True).contiguous())
            add("gloo.engine D->H copy", time.perf_counter() - t)
            t = time.perf_counter()
            for tag, ct in enumerate(cpu_tensors):
                dist.send(ct, dst=TRAINER_RANK, group=group, tag=tag)
            add("gloo.engine gloo ship", time.perf_counter() - t)
        else:
            bufs = []
            t = time.perf_counter()
            for tag, name in enumerate(names):
                shape, dtype = spec[name]
                b = torch.empty(tuple(shape), dtype=dtype, device="cpu")
                dist.recv(b, src=ENGINE_RANK, group=group, tag=tag)
                bufs.append(b)
            add("gloo.trainer gloo recv", time.perf_counter() - t)
            t = time.perf_counter()
            dev = [b.to(device) for b in bufs]
            torch.cuda.synchronize()
            add("gloo.trainer H->D copy", time.perf_counter() - t)
            del dev

        # ---- CUDA IPC ----
        dist.barrier(group)
        if rank == ENGINE_RANK:
            t = time.perf_counter()
            args_list = []
            for name in names:
                tt = payload[name].detach()
                if not tt.is_contiguous():
                    tt = tt.contiguous()
                _fn, args = reduce_tensor(tt)
                args_list.append((name, args))
            add("ipc.engine handle export", time.perf_counter() - t)
            t = time.perf_counter()
            blob = pickle.dumps(args_list, protocol=pickle.HIGHEST_PROTOCOL)
            buf = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
            dist.send(torch.tensor([buf.numel()], dtype=torch.long),
                      dst=TRAINER_RANK, group=group, tag=_BREAKDOWN_TAG)
            dist.send(buf, dst=TRAINER_RANK, group=group, tag=_BREAKDOWN_TAG + 1)
            add("ipc.engine ship handles", time.perf_counter() - t)
            t = time.perf_counter()
            ack = torch.zeros(1, dtype=torch.uint8)
            dist.recv(ack, src=TRAINER_RANK, group=group, tag=_BREAKDOWN_TAG + 2)
            add("ipc.engine wait for ack", time.perf_counter() - t)
        else:
            length = torch.empty(1, dtype=torch.long)
            dist.recv(length, src=ENGINE_RANK, group=group, tag=_BREAKDOWN_TAG)
            rbuf = torch.empty(int(length.item()), dtype=torch.uint8)
            dist.recv(rbuf, src=ENGINE_RANK, group=group, tag=_BREAKDOWN_TAG + 1)
            payloads = pickle.loads(rbuf.numpy().tobytes())
            t = time.perf_counter()
            aliases = [rebuild_cuda_tensor(*args) for _name, args in payloads]
            add("ipc.trainer handle open", time.perf_counter() - t)
            t = time.perf_counter()
            cloned = [a.to(device, copy=True) for a in aliases]
            torch.cuda.synchronize()
            add("ipc.trainer D->D copy", time.perf_counter() - t)
            del aliases, cloned
            dist.send(torch.ones(1, dtype=torch.uint8),
                      dst=ENGINE_RANK, group=group, tag=_BREAKDOWN_TAG + 2)

    return {stage: sum(v) / len(v) * 1e3 for stage, v in acc.items()}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(rank, world_size, port, argsd, result_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.group.WORLD

    iters, warmup = argsd["iters"], argsd["warmup"]
    payloads = list(argsd["payloads"])  # [(label, spec)]

    results = []
    for label, spec in payloads:
        gloo = _bench_transport("gloo", gloo_send, gloo_recv, spec,
                                iters=iters, warmup=warmup, rank=rank,
                                device=device, group=group)
        ipc = _bench_transport("ipc", ipc_send, ipc_recv, spec,
                               iters=iters, warmup=warmup, rank=rank,
                               device=device, group=group)
        gathered = [None, None]
        dist.all_gather_object(gathered, {"gloo": gloo, "ipc": ipc})
        if rank == TRAINER_RANK:
            results.append((label, spec, gathered))

    # Stage breakdown on the largest payload only (it is the clearest).
    big_label, big_spec = max(payloads, key=lambda ls: _spec_bytes(ls[1]))
    bd = _breakdown(big_spec, iters=max(8, warmup), rank=rank,
                    device=device, group=group)
    bd_gathered = [None, None]
    dist.all_gather_object(bd_gathered, bd)

    if rank == TRAINER_RANK:
        merged_bd = {}
        for d in bd_gathered:
            merged_bd.update(d)
        report = _build_report(results, (big_label, big_spec, merged_bd),
                               iters=iters, warmup=warmup)
        Path(result_path).write_text(report)
        print(report)

    dist.barrier(group)
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_mb(nbytes: int) -> str:
    mb = nbytes / (1024 * 1024)
    return f"{mb:.2f} MB" if mb < 1 else f"{mb:.0f} MB"


def _build_report(results, breakdown, *, iters, warmup) -> str:
    try:
        gpu = torch.cuda.get_device_name(0)
    except Exception:
        gpu = "unknown"
    mps = "yes" if os.environ.get("CUDA_MPS_PIPE_DIRECTORY") else "no"

    L = []
    L.append("# Colocate hidden-state transport benchmark — gloo vs CUDA IPC")
    L.append("")
    L.append(f"- GPU: **{gpu}**  ·  torch {torch.__version__}  ·  "
             f"CUDA {torch.version.cuda}")
    L.append(f"- Host: {platform.platform()}  ·  MPS active: {mps}")
    L.append(f"- Method: 2 processes on GPU 0, 2-rank gloo group; "
             f"{warmup} warmup + {iters} measured iters; fresh payload "
             f"allocated every iter.")
    L.append("- Latency = barrier-to-barrier end-to-end transfer "
             "(engine send + trainer recv/copy).")
    L.append("")

    # Headline table.
    L.append("## End-to-end transfer latency")
    L.append("")
    L.append("| Payload | Size | gloo mean | gloo p99 | IPC mean | IPC p99 "
             "| gloo GB/s | IPC GB/s | **IPC speedup** |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for label, spec, gathered in results:
        nbytes = _spec_bytes(spec)
        gloo_spans = gathered[TRAINER_RANK]["gloo"]["span"]
        ipc_spans = gathered[TRAINER_RANK]["ipc"]["span"]
        g, i = _stats(gloo_spans), _stats(ipc_spans)
        g_bw = nbytes / (g["mean"] / 1e3) / 1e9
        i_bw = nbytes / (i["mean"] / 1e3) / 1e9
        speedup = g["mean"] / i["mean"]
        L.append(f"| {label} | {_fmt_mb(nbytes)} | {g['mean']:.3f} ms "
                 f"| {g['p99']:.3f} ms | {i['mean']:.3f} ms | {i['p99']:.3f} ms "
                 f"| {g_bw:.1f} | {i_bw:.1f} | **{speedup:.1f}×** |")
    L.append("")

    # Engine / trainer split.
    L.append("## Engine / trainer split (own-call duration, mean)")
    L.append("")
    L.append("| Payload | gloo engine send | gloo trainer recv "
             "| IPC engine send | IPC trainer recv |")
    L.append("|---|--:|--:|--:|--:|")
    for label, spec, gathered in results:
        ge = _stats(gathered[ENGINE_RANK]["gloo"]["own"])["mean"]
        gt = _stats(gathered[TRAINER_RANK]["gloo"]["own"])["mean"]
        ie = _stats(gathered[ENGINE_RANK]["ipc"]["own"])["mean"]
        it = _stats(gathered[TRAINER_RANK]["ipc"]["own"])["mean"]
        L.append(f"| {label} | {ge:.3f} ms | {gt:.3f} ms "
                 f"| {ie:.3f} ms | {it:.3f} ms |")
    L.append("")

    # Stage breakdown.
    big_label, big_spec, bd = breakdown
    L.append(f"## Per-stage breakdown — {big_label} "
             f"({_fmt_mb(_spec_bytes(big_spec))}, mean ms)")
    L.append("")
    L.append("| Stage | Time |")
    L.append("|---|--:|")
    for stage in sorted(bd):
        L.append(f"| `{stage}` | {bd[stage]:.3f} ms |")
    L.append("")
    L.append("> gloo pays two PCIe-class copies (D->H, H->D) + a host ship; "
             "CUDA IPC pays a tiny handle exchange + one on-device D->D copy. "
             "`cudaIpcOpenMemHandle` (`ipc.trainer handle open`) is a fixed "
             "per-step cost — it is re-paid every step because the engine "
             "reallocates hidden states each step.")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=30,
                    help="measured iterations per transport per payload")
    ap.add_argument("--warmup", type=int, default=5, help="warmup iterations")
    ap.add_argument("--sizes-mb", default="0.25,1,4,16,64,256",
                    help="comma-separated single-tensor payload sizes in MB")
    ap.add_argument("--tokens", type=int, default=4096,
                    help="Eagle3 multi-tensor case: number of tokens (B*S)")
    ap.add_argument("--hidden", type=int, default=4096,
                    help="Eagle3 multi-tensor case: hidden dim")
    ap.add_argument("--port", type=int, default=29555, help="rendezvous port")
    ap.add_argument("--out", default=str(_REPO_ROOT / "colocate-transport-bench.md"),
                    help="Markdown report output path")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: this benchmark needs a CUDA GPU.", file=sys.stderr)
        return 2
    for ev in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        if "expandable" in os.environ.get(ev, ""):
            print(f"WARNING: {ev}={os.environ[ev]!r} — CUDA IPC needs plain "
                  f"cudaMalloc memory and will fail. Unset it.", file=sys.stderr)

    payloads = [(f"single {s.strip()} MB", _single_tensor_spec(float(s)))
                for s in args.sizes_mb.split(",") if s.strip()]
    payloads.append((f"Eagle3 ({args.tokens}t × {args.hidden}h, 3 tensors)",
                     _eagle3_spec(args.tokens, args.hidden)))

    argsd = {"iters": args.iters, "warmup": args.warmup, "payloads": payloads}
    print(f"Benchmarking {len(payloads)} payloads, "
          f"{args.warmup}+{args.iters} iters each, on "
          f"{torch.cuda.get_device_name(0)} …\n")
    mp.spawn(_worker, args=(2, args.port, argsd, args.out), nprocs=2, join=True)
    print(f"\nReport written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
