"""A CUDA build that cannot START must be reported as cpu, not cuda.

`probe()` asked only whether llama.cpp was BUILT with CUDA — the capability symbol, or a *cuda*
shared object on disk. On 2026-08-18 live round 2 rented an H100 on scaleway/warsaw-poland-1 whose
driver was older than the runtime the wheel was compiled against. The build checks both passed, the
guard in `score_job` passed, ggml fell back to CPU, and 13 submissions were being scored at ~5x the
time on a GPU we were paying for — heading for a signed record asserting `NVIDIA H100 PCIe`.

The stderr below is copied verbatim from that round's log.
"""
from eval.gpu_check import _scan, probe

# verbatim, /var/log/ralph-validator.log, live round 2
REAL = ("ggml_cuda_init: failed to initialize CUDA: CUDA driver version is insufficient for "
        "CUDA runtime version")


def test_the_real_failure_line_is_detected():
    assert _scan(REAL) == REAL


def test_it_is_found_among_ordinary_load_noise():
    err = "\n".join([
        "llama_model_loader: loaded meta data with 26 key-value pairs",
        "ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no",
        REAL,
        "llama_model_load_internal: mem required  = 4096 MB",
    ])
    assert _scan(err) == REAL


def test_a_healthy_cuda_load_is_not_flagged():
    """The expensive mistake in the other direction: refusing a box that would have worked."""
    err = "\n".join([
        "ggml_cuda_init: found 1 CUDA devices:",
        "  Device 0: NVIDIA H100 PCIe, compute capability 9.0, VMM: yes",
        "llm_load_tensors: offloaded 37/37 layers to GPU",
    ])
    assert _scan(err) == ""


def test_empty_and_garbage_are_safe():
    for x in ("", None, "\n\n", "no cuda mentioned at all"):
        assert _scan(x) == ""


def _fake_cuda_build(tmp_path, monkeypatch):
    """A llama_cpp package that LOOKS like a correct CUDA build — which is exactly the situation:
    the wheel on that box was fine, and the box it landed on was not."""
    import sys
    import types

    (tmp_path / "libggml-cuda.so").write_bytes(b"")
    pkg = types.ModuleType("llama_cpp")
    pkg.__file__ = str(tmp_path / "__init__.py")
    # no `llama_supports_gpu_offload`: upstream dropped it, so real bindings fall through to the
    # on-disk check, and that is the path that wrongly said "cuda"
    inner = types.ModuleType("llama_cpp.llama_cpp")
    pkg.llama_cpp = inner
    monkeypatch.setitem(sys.modules, "llama_cpp", pkg)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", inner)


def test_probe_reports_cpu_when_the_runtime_will_not_start(tmp_path, monkeypatch):
    """The whole point: build present on disk, runtime dead -> cpu, with the cause carried."""
    _fake_cuda_build(tmp_path, monkeypatch)
    monkeypatch.setattr("eval.gpu_check._cuda_init_failure", lambda: REAL)
    ok, why = probe()
    assert ok is False
    assert "will not start here" in why and "insufficient" in why


def test_a_working_cuda_build_is_still_accepted(tmp_path, monkeypatch):
    """A dead probe must not refuse a working round. Same box, runtime fine -> still cuda."""
    _fake_cuda_build(tmp_path, monkeypatch)
    monkeypatch.setattr("eval.gpu_check._cuda_init_failure", lambda: "")
    ok, why = probe()
    assert ok is True
    assert "libggml-cuda.so" in why


def test_the_runtime_check_only_runs_for_a_cuda_build(tmp_path, monkeypatch):
    """No CUDA on disk is already conclusive — do not pay for a subprocess to confirm it."""
    import sys
    import types

    calls = []
    pkg = types.ModuleType("llama_cpp")
    pkg.__file__ = str(tmp_path / "__init__.py")
    pkg.llama_cpp = types.ModuleType("llama_cpp.llama_cpp")
    monkeypatch.setitem(sys.modules, "llama_cpp", pkg)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", pkg.llama_cpp)
    monkeypatch.setattr("eval.gpu_check._cuda_init_failure",
                        lambda: (calls.append(1), "")[1])
    ok, _why = probe()
    assert ok is False and calls == []
