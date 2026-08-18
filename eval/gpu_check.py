"""Does the installed llama.cpp actually have CUDA?

ONE implementation, used by the install guard and by the runner that reports the backend into the
record. They disagreed once and it cost a good round: the guard asked for
`llama_supports_gpu_offload`, that symbol is gone from recent bindings, the call raised, the shell's
`||` read the non-zero exit as "no GPU", and a correctly built 283 MB CUDA wheel was refused.

The lesson is not "use a different symbol" — it is that ONE probe returning false is not evidence.
An absent capability function and an absent capability look identical through `getattr`. So this
asks in two independent ways and only concludes CPU when both agree, because the two failure modes
are asymmetric: calling a CPU build a GPU build wastes a round's money, while calling a GPU build a
CPU build refuses a round that would have worked. The second is what actually happened.

    python -m eval.gpu_check          # exit 0 = CUDA, 9 = CPU-only, prints which and why
"""
from __future__ import annotations

import glob
import os
import sys


def _cuda_init_failure() -> str:
    """POSITIVE evidence that llama.cpp's CUDA backend cannot START on this box, or "".

    A THIRD failure mode the two checks below cannot see. Both of them ask whether llama.cpp was
    BUILT with CUDA. Neither asks whether CUDA still works here — and on 2026-08-18, on
    scaleway/warsaw-poland-1, it did not: the wheel was compiled against a newer CUDA runtime than
    the box's driver, so ggml printed

        ggml_cuda_init: failed to initialize CUDA: CUDA driver version is insufficient for CUDA
        runtime version

    and fell back to CPU. Every build-time signal still said "cuda", the guard in `score_job`
    passed, and live round 2 scored students on CPU at ~5x the time on an H100 it was paying for —
    heading for a signed, anchored record asserting `NVIDIA H100 PCIe`.

    IN A SUBPROCESS, because ggml emits this while the library loads: by the time anything in this
    process can capture stderr, an earlier `import llama_cpp` has already printed it and a fresh
    import is a silent no-op. A clean interpreter is the only way to be sure we are watching.

    ONLY POSITIVE EVIDENCE COUNTS — it returns "" on any internal trouble. The asymmetry this file
    already documents still holds in both directions: believing a CPU box is a GPU box wastes a
    round's money, and believing a GPU box is a CPU box refuses a round that would have worked. A
    probe that cannot run must not do the second one."""
    import subprocess

    code = (
        "try:\n"
        "    from llama_cpp import llama_cpp as c\n"
        "    f = getattr(c, 'llama_backend_init', None)\n"
        "    if f is not None: f()\n"
        "    g = getattr(c, 'ggml_backend_dev_count', None)\n"
        "    if g is not None: g()\n"
        "except Exception:\n"
        "    pass\n")
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=180)
        err = (r.stderr or b"").decode("utf-8", "replace")
    except Exception:
        return ""
    return _scan(err)


def _scan(err: str) -> str:
    """The failure line in ggml's stderr, or "". Split out so it can be tested against the bytes
    llama.cpp really emitted rather than a paraphrase of them — a fixture written from memory tests
    the guess, not the parser."""
    for line in (err or "").splitlines():
        low = line.lower()
        if "failed to initialize cuda" in low or ("driver version" in low and "insufficient" in low):
            return line.strip()
    return ""


def probe() -> tuple[bool, str]:
    """(has_cuda, why). Never raises — a probe that can throw is a probe that fails closed by
    accident, which is exactly the bug this file exists to fix."""
    reasons = []

    # 1. llama.cpp's own capability call, WHEN THE BINDING STILL EXPOSES IT. Upstream dropped
    #    `llama_supports_gpu_offload` in favour of the ggml backend registry, so its absence says
    #    nothing about the build.
    try:
        from llama_cpp import llama_cpp as _c
        fn = getattr(_c, "llama_supports_gpu_offload", None)
        if fn is None:
            reasons.append("llama_supports_gpu_offload: absent from this binding")
        else:
            try:
                if bool(fn()):
                    dead = _cuda_init_failure()
                    if dead:
                        return False, f"built with CUDA but it will not start here: {dead}"
                    return True, "llama_supports_gpu_offload() is true"
                reasons.append("llama_supports_gpu_offload() is false")
            except Exception as e:
                reasons.append(f"llama_supports_gpu_offload() raised {type(e).__name__}")
    except Exception as e:
        reasons.append(f"llama_cpp will not import ({type(e).__name__})")

    # 2. THE BYTES ON DISK. A CUDA build links a CUDA ggml backend; a CPU wheel does not, and the
    #    size difference is ~283 MB against ~20 MB. This one cannot be deprecated out from under us.
    try:
        import llama_cpp
        d = os.path.dirname(os.path.abspath(llama_cpp.__file__))
        hits = []
        for pat in ("lib/*cuda*", "*cuda*", "lib/*.so*"):
            for path in glob.glob(os.path.join(d, pat)):
                base = os.path.basename(path).lower()
                if "cuda" in base:
                    hits.append(base)
        if hits:
            dead = _cuda_init_failure()
            if dead:
                return False, f"CUDA backend on disk but it will not start here: {dead}"
            return True, f"CUDA backend present on disk: {sorted(set(hits))[:3]}"
        reasons.append("no *cuda* shared object beside llama_cpp")
    except Exception as e:
        reasons.append(f"could not inspect the package ({type(e).__name__})")

    return False, "; ".join(reasons)


def main() -> int:
    ok, why = probe()
    sys.stdout.write(f"llama.cpp backend: {'cuda' if ok else 'cpu'} ({why})\n")
    return 0 if ok else 9


if __name__ == "__main__":
    raise SystemExit(main())
