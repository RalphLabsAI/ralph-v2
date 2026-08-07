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
