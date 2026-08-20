"""The benchmark must refuse to run on CPU.

The round path has asserted this since the CPU-fallback hole. The BENCHMARK path never did, and on
2026-08-18 it spent 31 minutes and ~$3.60 on a rented H100 SXM at 1252% CPU and 0% GPU. The cause
was `CUDA error 802: system not yet initialized` — a board whose `nvidia-fabricmanager` would not
start, so the NVLink fabric was never up and `cudaGetDeviceCount()` failed. torch was the correct
2.11.0+cu128 and driver 570.148.08 supported it. Nothing about the install was wrong and nothing
said so.

It is not only a money failure: CPU and GPU do not produce identical logits, so a silent device
change is a fourth confound in a harness rebuilt to remove three.
"""
import sys
import types

import pytest

from bench.compare import _assert_gpu


def _torch(monkeypatch, available, name="NVIDIA H100 80GB HBM3"):
    m = types.ModuleType("torch")
    m.cuda = types.SimpleNamespace(is_available=lambda: available,
                                   get_device_name=lambda i: name)
    monkeypatch.setitem(sys.modules, "torch", m)


def test_refuses_when_torch_sees_no_cuda(monkeypatch):
    _torch(monkeypatch, False)
    monkeypatch.delenv("RALPH_ALLOW_CPU_BENCH", raising=False)
    with pytest.raises(SystemExit, match="refusing to benchmark"):
        _assert_gpu()


def test_passes_on_a_real_gpu(monkeypatch):
    _torch(monkeypatch, True)
    _assert_gpu()


def test_the_escape_hatch_is_explicit(monkeypatch):
    """Accepting CPU must be a deliberate act, not a default."""
    _torch(monkeypatch, False)
    monkeypatch.setenv("RALPH_ALLOW_CPU_BENCH", "1")
    _assert_gpu()


def test_a_missing_torch_is_not_a_refusal(monkeypatch):
    """GGUF-only runs go through llama.cpp; absent torch says nothing about the GPU, and failing
    here would refuse a run that never needed torch at all."""
    monkeypatch.delenv("RALPH_ALLOW_CPU_BENCH", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    _assert_gpu()
