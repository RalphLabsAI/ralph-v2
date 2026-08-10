"""Rent a GPU, run bench/compare.py on it, bring the results back, destroy the box.

Deliberately separate from `run_remote_round`. A benchmark is not a round: nothing is signed,
nothing is anchored, no miner's emission depends on it, and it must never be able to disturb the
scoring path. It shares only the provider and the teardown discipline — rent, run, and destroy in a
`finally` that also survives SIGTERM, because the failure that matters here is the same one that
matters there: a box nobody destroys bills until somebody notices.

    python -m bench.run_remote --limit 200
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from eval.orchestrator import (_SSH_OPTS, _VENV_CMD, GpuSpec, RemoteRoundError,
                               ShadeformProvider, _teardown_on_signal, _torch_index,
                               gpu_devices)

NAME_PREFIX = "ralph-bench-"


def _ssh_argv(inst, spec):
    """THE IDENTITY AND THE PORT, which `_SSH_OPTS` does not carry. Leaving them off cost a rented
    H100 and three minutes: every command came back `Permission denied (publickey)` because the
    agent has no key for a box created seconds earlier. `eval.orchestrator._ssh` has always built
    it this way; this file just did not copy the whole line."""
    return ["ssh", "-i", spec.ssh_key, "-p", str(inst.ssh_port), *_SSH_OPTS]


def _sh(inst, spec, cmd, timeout=None):
    return subprocess.run([*_ssh_argv(inst, spec), f"{inst.ssh_user}@{inst.ip}", cmd],
                          timeout=timeout)


def _stream(inst, spec, cmd, timeout=None):
    full = [*_ssh_argv(inst, spec), f"{inst.ssh_user}@{inst.ip}", cmd]
    p = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        sys.stdout.write("  | " + line)
        sys.stdout.flush()
    return p.wait()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--only", default="")
    ap.add_argument("--max-hours", type=float, default=5.0)
    ap.add_argument("--out", default="bench-results.json")
    a = ap.parse_args(argv)

    spec = GpuSpec(max_hours=a.max_hours,
                   require_gpu=os.environ.get("RALPH_REQUIRE_GPU", ""),
                   max_price_per_hour=float(os.environ.get("RALPH_MAX_GPU_PRICE", "4.50")),
                   exclude_regions=tuple(x.strip() for x in
                                         os.environ.get("RALPH_GPU_EXCLUDE", "").split(",")
                                         if x.strip()))
    prov = ShadeformProvider()

    # A RENTAL CAN BE "READY" WITH NO GPU. Seen 2026-08-10: nvidia modules loaded, nvidia-smi
    # answering "No devices were found", zero NVIDIA devices on the PCI bus. One ssh catches it;
    # the alternative is discovering it after a fifteen-minute install. Retry into a DIFFERENT
    # (cloud, region) rather than the same broken one, which is what `exclude` is for.
    tried: tuple = ()
    inst = None
    for attempt in range(1, 4):
        cand = prov.rent(spec, f"{NAME_PREFIX}{int(time.time()) % 100000}", exclude=tried)
        print(f"rented {cand.id} {cand.instance_type} @ ${cand.price_per_hour:.2f}/hr "
              f"({cand.cloud}/{cand.region})")
        try:
            prov.wait_ready(cand, out=sys.stdout)
            devs = gpu_devices(cand, spec)
        except Exception as e:
            print(f"  not usable: {type(e).__name__}: {e}")
            devs = []
        if devs:
            print(f"  gpus: {', '.join(devs)}")
            inst = cand
            break
        print(f"  NO GPU on {cand.cloud}/{cand.region} — destroying and trying elsewhere "
              f"(attempt {attempt}/3)")
        tried += ((cand.cloud, cand.region),)
        try:
            prov.destroy(cand)
        except Exception as e:
            print(f"  !! could not destroy {cand.id}: {e}")
    if inst is None:
        print("no rental with a working GPU after 3 attempts")
        return 2
    t0 = time.time()
    rc = 1
    try:
        with _teardown_on_signal():
            print(f"ready at {inst.ip}")
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _sh(inst, spec, "mkdir -p ~/ralph-v2", timeout=120)
            subprocess.run(["rsync", "-az",
                            "-e", " ".join(_ssh_argv(inst, spec)),
                            "--exclude", ".venv/", "--exclude", "__pycache__/",
                            "--exclude", "wallets/", "--exclude", ".env",
                            f"{here}/eval", f"{here}/bench",
                            f"{inst.ssh_user}@{inst.ip}:~/ralph-v2/"], check=True, timeout=900)

            # ASK THE BOX WHICH CUDA ITS DRIVER SPEAKS. pip's default torch is built for a newer
            # CUDA than these images ship, and the mismatch does not raise — `is_available()` just
            # returns False and everything silently runs on CPU. Same trap the round hit.
            cuda = subprocess.run(
                [*_ssh_argv(inst, spec), f"{inst.ssh_user}@{inst.ip}",
                 # RAW STRING. Without it `\1` is a control character, sed matches nothing, the
                 # CUDA version reads empty, and we install pip's default torch — which is the
                 # silent-CPU trap this line exists to avoid.
                 r"nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -1"],
                capture_output=True, text=True, timeout=180).stdout.strip()
            idx = _torch_index(cuda)
            print(f"driver speaks CUDA {cuda or '?'} -> torch wheel {idx or 'pypi default'}")
            torch_cmd = (f".venv/bin/pip install --index-url {idx} torch && " if idx
                         else ".venv/bin/pip install torch && ")

            install = (
                "cd ~/ralph-v2 && export PIP_RETRIES=10 PIP_TIMEOUT=60 "
                "PIP_DISABLE_PIP_VERSION_CHECK=1; " + _VENV_CMD + torch_cmd +
                ".venv/bin/pip install --no-cache-dir transformers safetensors datasets "
                "huggingface_hub accelerate && "
                "(sudo -n apt-get update -q >/dev/null 2>&1 || true); "
                "(sudo -n apt-get install -y -q build-essential g++ >/dev/null 2>&1 || true); "
                "NVCC=$(command -v nvcc || ls -1 /usr/local/cuda*/bin/nvcc 2>/dev/null | head -1); "
                "if [ -n \"$NVCC\" ]; then export CUDACXX=$NVCC; "
                "export PATH=$(dirname $NVCC):$PATH; "
                "CMAKE_ARGS='-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=90' "
                ".venv/bin/pip install --no-cache-dir --no-binary llama-cpp-python "
                "llama-cpp-python || .venv/bin/pip install --no-cache-dir llama-cpp-python; "
                "else .venv/bin/pip install --no-cache-dir llama-cpp-python; fi && "
                ".venv/bin/python -m eval.gpu_check || echo 'WARNING: llama.cpp is CPU-only, "
                "the GGUF models will be slow'")
            print("installing…")
            if _stream(inst, spec, install, timeout=5400) != 0:
                raise RuntimeError("the remote install failed")

            only = f" --only {a.only}" if a.only else ""
            print("benchmarking…")
            rc = _stream(inst, spec, f"cd ~/ralph-v2 && .venv/bin/python -m bench.compare "
                               f"--limit {a.limit} --max-new {a.max_new}{only} "
                               f"--out {a.out} 2>&1", timeout=int(a.max_hours * 3600))
            subprocess.run(["scp", "-i", spec.ssh_key, "-P", str(inst.ssh_port), *_SSH_OPTS,
                            f"{inst.ssh_user}@{inst.ip}:~/ralph-v2/{a.out}", a.out], timeout=300)
            print(f"\nresults -> {a.out}")
    finally:
        try:
            prov.destroy(inst)
            print(f"destroyed {inst.id} after {(time.time() - t0) / 60:.1f} min "
                  f"(~${inst.price_per_hour * (time.time() - t0) / 3600:.2f})")
        except Exception as e:
            print(f"!! COULD NOT DESTROY {inst.id}: {e} — kill it in the console")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
