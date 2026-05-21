# Copyright (c) 2026 LightSeek Foundation
# MIT License

"""Unit tests for the CUDA IPC transport plumbing.

CUDA IPC is the default colocate hidden-state transport; these tests
run on a Mac dev box (no real CUDA) and exercise the env-var contract
(default-on, opt-out via TORCHSPEC_COLOCATE_IPC=0) and the fail-fast
guard — not the actual IPC handle exchange (which needs two processes
on one GPU and is covered by the colocate e2e tests).
"""

from __future__ import annotations

import os

import pytest

from torchspec.colocate import cuda_ipc


@pytest.fixture(autouse=True)
def _clean():
    saved = os.environ.get("TORCHSPEC_COLOCATE_IPC")
    cuda_ipc._reset_probe_cache_for_test()
    yield
    if saved is None:
        os.environ.pop("TORCHSPEC_COLOCATE_IPC", None)
    else:
        os.environ["TORCHSPEC_COLOCATE_IPC"] = saved
    cuda_ipc._reset_probe_cache_for_test()


@pytest.mark.parametrize(
    "value,expected",
    [
        # Default-on: any value that is not an explicit disable token
        # (including an empty string and unrecognised junk) enables IPC.
        ("1", True), ("true", True), ("YES", True),
        ("garbage", True), ("", True),
        ("0", False), ("false", False), ("no", False), ("OFF", False),
    ],
)
def test_ipc_enabled_env_toggle(value, expected):
    os.environ["TORCHSPEC_COLOCATE_IPC"] = value
    assert cuda_ipc.ipc_enabled() is expected


def test_ipc_enabled_unset_defaults_on():
    os.environ.pop("TORCHSPEC_COLOCATE_IPC", None)
    assert cuda_ipc.ipc_enabled() is True


def test_ensure_ipc_usable_raises_when_probe_fails(monkeypatch):
    monkeypatch.setattr(
        cuda_ipc, "probe_ipc_capability",
        lambda: (False, "expandable_segments active"),
    )
    with pytest.raises(RuntimeError, match="expandable_segments active"):
        cuda_ipc.ensure_ipc_usable()


def test_ensure_ipc_usable_passes_when_probe_ok(monkeypatch):
    monkeypatch.setattr(cuda_ipc, "probe_ipc_capability", lambda: (True, "ok"))
    cuda_ipc.ensure_ipc_usable()  # must not raise


def test_probe_cache_reset_hook():
    cuda_ipc._probe_cache = (True, "stale")
    cuda_ipc._reset_probe_cache_for_test()
    assert cuda_ipc._probe_cache is None
