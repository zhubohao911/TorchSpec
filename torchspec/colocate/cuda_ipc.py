# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""CUDA IPC zero-copy hidden-state transport for colocate mode.

This is the **default** colocate hidden-state transport. The engine
exports a CUDA IPC handle for each hidden-state tensor (via PyTorch's
``torch.multiprocessing`` reduction machinery), ships the small handle
blobs over the gloo channel, and the trainer maps the engine's GPU
memory directly and does a single on-device D->D copy into its own
buffer. No host round-trip.

The fallback is the gloo CPU-staged transport: the engine does a D->H
copy, ships the bytes over the gloo ``meta_group``, and the trainer
does an H->D copy — two PCIe-class copies per tensor per step. Both
processes share the *same physical GPU* under MPS, so that host
round-trip is pure overhead (the data never needs to leave the device)
— which is exactly what this IPC path eliminates.

Default & opt-out
-----------------
CUDA IPC is **on by default**, layered on top of ``transfer_mode=nccl``
(it replaces only the gloo transport, not the union-world bootstrap).
Set ``TORCHSPEC_COLOCATE_IPC=0`` to fall back to the gloo CPU-staged
transport. Both the engine connector and the trainer fetcher read the
*same* env var, so the two sides always agree on the transport without
a runtime negotiation message.

The ``expandable_segments`` conflict
------------------------------------
CUDA IPC has two memory-sharing paths. For plain ``cudaMalloc`` memory
it uses the classic ``cudaIpcGetMemHandle`` / ``cudaIpcOpenMemHandle``
handles, which work in any container. For the virtual-memory segments
produced by ``expandable_segments:True`` it instead passes the backing
fd between processes via the ``pidfd_getfd`` syscall — and that needs
``CAP_SYS_PTRACE``, which typical containers (RunPod, most Docker
hosts) do not grant, so ``rebuild_cuda_tensor`` fails with
``pidfd_getfd: Operation not permitted``.

Resolution: when ``TORCHSPEC_COLOCATE_IPC`` is opted in, the colocate
path (``ray/train_group.py``, ``inference/factory.py``) **does not**
inject ``expandable_segments`` into the trainer/engine actors, so IPC
stays on the capability-free classic-handle path. (IPC already avoids
the H<->D staging churn that ``expandable_segments`` was mitigating.)

:func:`probe_ipc_capability` still runs a ``reduce_tensor`` smoke check
at construction; the connector/fetcher **fail fast** with an actionable
message if IPC was requested but is unavailable, rather than silently
falling back (a one-sided fallback would desync the wire protocol).

Wire protocol
-------------
Per step, engine -> trainer over the gloo group:

  1. engine: ``send_object_list([[(name, ipc_args), ...]])`` — the
     pickled IPC handle blobs, in ``sorted(name)`` order.
  2. trainer: ``recv_object_list`` -> rebuild each tensor as an alias of
     the engine's memory -> ``.clone()`` into a trainer-owned buffer ->
     ``cuda.synchronize()``.
  3. trainer: send a 1-byte ack back.
  4. engine: block on the ack before returning from ``send`` — this
     keeps the engine's (sglang-owned) hidden-state tensors alive until
     the trainer has finished copying, exactly like the blocking gloo
     ``send`` it replaces.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

_IPC_ENV = "TORCHSPEC_COLOCATE_IPC"

# Env values that disable IPC and fall back to the gloo transport.
_IPC_DISABLE_VALUES = ("0", "false", "no", "off")

# Cached (ok, reason) from the one-time capability probe.
_probe_cache: Optional[Tuple[bool, str]] = None


def ipc_enabled() -> bool:
    """True iff the CUDA IPC zero-copy transport is selected.

    CUDA IPC is the **default** colocate hidden-state transport. Any
    value of ``TORCHSPEC_COLOCATE_IPC`` other than an explicit disable
    token (``0`` / ``false`` / ``no`` / ``off``) — including the var
    being unset — selects it. Set one of those tokens to fall back to
    the gloo CPU-staged transport.
    """
    return os.environ.get(_IPC_ENV, "").strip().lower() not in _IPC_DISABLE_VALUES


def probe_ipc_capability() -> Tuple[bool, str]:
    """Probe whether CUDA IPC can actually be used on this process.

    Returns ``(ok, reason)``. Cached after the first call. ``ok`` is
    False when CUDA is absent, or when ``reduce_tensor`` raises — most
    commonly because ``expandable_segments`` is active (its cuMemMap
    segments are not IPC-shareable).
    """
    global _probe_cache
    if _probe_cache is not None:
        return _probe_cache

    try:
        import torch

        if not torch.cuda.is_available():
            _probe_cache = (False, "CUDA not available")
            return _probe_cache

        from torch.multiprocessing.reductions import reduce_tensor

        scratch = torch.empty(8, dtype=torch.float32, device="cuda")
        # reduce_tensor -> (rebuild_fn, args); this is the call that
        # invokes the storage's _share_cuda_ and raises on expandable
        # segments / unsupported platforms.
        reduce_tensor(scratch)
        del scratch
        _probe_cache = (True, "ok")
    except Exception as e:  # pragma: no cover - needs a real GPU
        hint = ""
        if "expandable" in repr(e).lower() or "cuMemMap" in repr(e):
            hint = (
                " — likely PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
                "CUDA IPC needs plain cudaMalloc memory. Drop expandable_"
                "segments for the colocate run, or set TORCHSPEC_COLOCATE_IPC=0"
                " to use the gloo transport."
            )
        _probe_cache = (False, f"{e!r}{hint}")
    return _probe_cache


def ensure_ipc_usable() -> None:
    """Raise a clear error if IPC (the default transport) is not usable.

    Called once at connector/fetcher construction. Both sides run the
    same check on the same platform, so they fail (or pass) together.
    """
    ok, reason = probe_ipc_capability()
    if not ok:
        raise RuntimeError(
            f"CUDA IPC is the default colocate hidden-state transport but "
            f"is not usable on this host: {reason}. Set "
            f"TORCHSPEC_COLOCATE_IPC=0 to fall back to the gloo CPU-staged "
            f"transport."
        )


def _reset_probe_cache_for_test() -> None:
    """Test hook: clear the cached probe result."""
    global _probe_cache
    _probe_cache = None


# ---------------------------------------------------------------------------
# Wire protocol
# ---------------------------------------------------------------------------

# Distinct tags for the three point-to-point messages of one transfer.
# The payload is shipped as plain dist.send/recv of byte tensors — the
# same primitive the gloo CPU-staged path uses (proven on the union
# meta_group). The send_object_list / recv_object_list helpers were
# observed to deadlock on this group, so we pickle + frame the blob
# ourselves.
_IPC_LEN_TAG = 7001
_IPC_DATA_TAG = 7002
_IPC_ACK_TAG = 7003


def ipc_send(
    tensors: Dict[str, "torch.Tensor"],  # noqa: F821
    dst: int,
    group,
) -> None:
    """Engine side: ship hidden-state tensors to ``dst`` via CUDA IPC.

    Blocks until the trainer acks (i.e. has cloned the data), so the
    caller's tensors stay valid for the whole transfer — same contract
    as the blocking gloo ``dist.send`` this replaces.
    """
    import pickle

    import torch
    import torch.distributed as dist
    from torch.multiprocessing.reductions import reduce_tensor

    names = sorted(tensors.keys())
    keepalive = []  # hold contiguous copies alive until the ack
    payloads = []
    for name in names:
        t = tensors[name].detach()
        if t.device.type != "cuda":
            raise ValueError(
                f"cuda_ipc.ipc_send requires CUDA tensors; '{name}' is on "
                f"{t.device}"
            )
        if not t.is_contiguous():
            t = t.contiguous()
        keepalive.append(t)
        # reduce_tensor returns (rebuild_cuda_tensor, args); only the
        # args tuple needs to travel — the receiver knows the rebuild fn.
        _rebuild_fn, args = reduce_tensor(t)
        payloads.append((name, args))

    # Pickle the IPC-handle payloads and ship as a length-framed byte
    # tensor via plain dist.send (the gloo path's proven primitive).
    blob = bytearray(pickle.dumps(payloads, protocol=pickle.HIGHEST_PROTOCOL))
    buf = torch.frombuffer(blob, dtype=torch.uint8)
    length = torch.tensor([buf.numel()], dtype=torch.long)
    dist.send(length, dst=dst, group=group, tag=_IPC_LEN_TAG)
    dist.send(buf, dst=dst, group=group, tag=_IPC_DATA_TAG)

    # Block until the trainer has cloned the data out of our memory.
    ack = torch.zeros(1, dtype=torch.uint8)
    dist.recv(ack, src=dst, group=group, tag=_IPC_ACK_TAG)
    del keepalive, blob


def ipc_recv(
    tensor_specs: Dict[str, Tuple],
    src: int,
    device: "torch.device",  # noqa: F821
    group,
) -> Dict[str, "torch.Tensor"]:  # noqa: F821
    """Trainer side: receive hidden-state tensors from ``src`` via CUDA IPC.

    Maps the engine's GPU memory, copies (D->D, on-device) into
    trainer-owned buffers, then acks. ``tensor_specs`` is used only to
    validate the received key set — the shapes/dtypes ride along inside
    the IPC payload.
    """
    import pickle

    import torch
    import torch.distributed as dist
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    # Receive the length-framed pickled payload (mirrors ipc_send).
    length = torch.empty(1, dtype=torch.long)
    dist.recv(length, src=src, group=group, tag=_IPC_LEN_TAG)
    buf = torch.empty(int(length.item()), dtype=torch.uint8)
    dist.recv(buf, src=src, group=group, tag=_IPC_DATA_TAG)
    payloads = pickle.loads(buf.numpy().tobytes())
    if not isinstance(payloads, list):
        raise RuntimeError(
            f"cuda_ipc.ipc_recv: expected a list payload, got {type(payloads)}"
        )

    out: Dict[str, torch.Tensor] = {}
    aliases = []  # keep IPC aliases alive until the post-clone sync
    for name, args in payloads:
        alias = rebuild_cuda_tensor(*args)
        aliases.append(alias)
        # D->D copy into trainer-owned (normal) memory on `device`.
        out[name] = alias.to(device, copy=True)

    # The clones above are async on the current stream; finish them
    # before we drop the aliases and ack (after which the engine may
    # free its memory).
    torch.cuda.synchronize()
    del aliases

    expected = set(tensor_specs.keys())
    got = set(out.keys())
    if expected != got:
        raise RuntimeError(
            f"cuda_ipc.ipc_recv: key mismatch — expected {sorted(expected)}, "
            f"got {sorted(got)}"
        )

    ack = torch.ones(1, dtype=torch.uint8)
    dist.send(ack, dst=src, group=group, tag=_IPC_ACK_TAG)
    return out
