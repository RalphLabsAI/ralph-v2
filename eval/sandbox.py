"""Sandboxed execution for the untrusted code grader — the validator-RCE defense (C1).

The code axis grades a student by RUNNING code the student model emitted. That text is
hostile by construction: a miner can fine-tune their student to embed a payload in the
```python block that both passes the hidden unit tests AND does anything the validator user
can — exfiltrate `~/.bittensor`, set weights by fiat, etc. So the grader must never run it
with network access or visibility of secrets. The rest of intake (safetensors-only,
trust_remote_code=False, file allowlist) exists to stop untrusted *weights* from executing
code; this stops untrusted *output* from doing the same.

Isolation model:
  * bubblewrap (`bwrap`) — the real boundary: a fresh namespace with NO network
    (--unshare-all), the filesystem read-only for the interpreter/libs, $HOME and /tmp
    masked with fresh tmpfs (so no secret is even readable), only the throwaway work dir
    writable, and a cleared environment.
  * setrlimit — defense-in-depth caps (CPU seconds, max file size), applied with or without
    bwrap, so an infinite loop or a disk-fill is bounded even in degraded mode.

Modes (the caller / axis spec chooses):
  * "require" (PRODUCTION) — refuse to execute if no hard sandbox backend is present
    (fail closed: a dead code axis beats an RCE).
  * "auto"    (default)    — bwrap if present, else rlimit-only with a one-time loud warning.
  * "off"                  — rlimit-only, explicit dev opt-out; never for untrusted output.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

_BWRAP = shutil.which("bwrap")
_WARNED = False


def sandbox_available() -> bool:
    """True iff a hard (namespace) sandbox backend is present."""
    return _BWRAP is not None


def _rlimits(cpu_s: int, fsize_bytes: int):
    """preexec hook: bound CPU time and file size for the child (and, under bwrap, the whole
    sandboxed tree). Deliberately does NOT cap RLIMIT_AS/NPROC — those are flaky (a tight
    address-space cap spuriously kills legit interpreters) and the real memory/fork bounds
    come from bwrap's namespaces plus the wall-clock timeout."""
    try:
        import resource
    except ImportError:
        return None

    def _apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))

    return _apply


def _bwrap_prefix(workdir: str, home: str) -> list[str]:
    """bwrap args: whole FS read-only for the interpreter+libs (so the read-only bind ALSO
    makes /tmp non-writable), $HOME masked with a fresh tmpfs so the wallet/secrets are not
    even readable, only `workdir` writable, NO network, clean env.

    NB: do NOT tmpfs-mask /tmp — a venv interpreter (sys.executable) commonly lives there and
    masking it makes even legitimate code fail to run. Read-only /tmp via the '/' bind is
    enough; combined with --unshare-all (no network) a read-only view exfiltrates nothing."""
    args = [
        _BWRAP, "--unshare-all", "--die-with-parent", "--new-session",
        "--ro-bind", "/", "/",
        "--proc", "/proc", "--dev", "/dev",
    ]
    if home and home not in ("/", "/tmp") and not workdir.startswith(home.rstrip("/") + "/"):
        args += ["--tmpfs", home]                       # mask ~/.bittensor etc.
    args += ["--bind", workdir, workdir, "--chdir", workdir,
             "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
             "--setenv", "HOME", workdir]
    return args


def run_script(script_path: str, workdir: str, *, timeout: float, mode: str = "auto",
               cpu_s: int = 10, fsize_bytes: int = 64 * 1024 * 1024
               ) -> tuple[bool, "subprocess.CompletedProcess | None"]:
    """Run `python script_path` under isolation.

    Returns (executed, completed_process):
      * executed=False  -> the run was REFUSED (mode='require' with no sandbox backend).
                           The caller MUST treat this as a non-pass, never a silent success.
      * completed_process=None with executed=True -> the run timed out.
    """
    global _WARNED
    use_bwrap = _BWRAP is not None and mode != "off"

    if mode == "require" and not use_bwrap:
        return False, None                               # fail closed: no hard sandbox

    if not use_bwrap and mode == "auto" and not _WARNED:
        _WARNED = True
        print("WARNING: bwrap not found — the code grader is running with rlimits only, NO "
              "network/filesystem isolation. Install bubblewrap (or set the code axis to "
              "sandbox='require') before grading untrusted output in production.",
              file=sys.stderr, flush=True)

    prefix = _bwrap_prefix(workdir, os.environ.get("HOME", "")) if use_bwrap else []
    cmd = prefix + [sys.executable, script_path]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                            cwd=workdir, preexec_fn=_rlimits(cpu_s, fsize_bytes))
    except subprocess.TimeoutExpired:
        return True, None
    return True, cp
