"""The heartbeat that makes "went quiet" mean "is stuck".

The orchestrator now tears a rental down when the remote scorer says nothing for twenty minutes
(`orchestrator.SILENCE_S`). That rule is only correct if a WORKING scorer talks, and nothing on the
scoring path did: between `parent qwen3-8b…` and a finished record there was a 16 GB checkpoint
download, a 60 GB artifact ceiling, nine batched generations and two observer passes per sample,
none of which printed anything. A silence timeout over that is not a safety net — it is a way to
kill healthy rounds at the twenty-minute mark, which is worse than the hang it replaces because it
fails on the rounds that were going to succeed.

WHERE THE TICKS GO. Only at boundaries that can plausibly outlast the silence budget on their own:
per stage in `score_job`, per file and before hashing in `fetch`, per checkpoint load and per
generation batch in `runners`, per sample in `observer_round`. Anything faster than that is already
covered by the stage line bracketing it, and a tick per KL position would be noise that costs more
than it reports.

TO STDERR, RATE-LIMITED, AND NEVER FATAL. stdout carries the scorer's result line; diagnostics
belong on the other stream, and the orchestrator merges the two anyway. The rate limit is GLOBAL
rather than per stage because the consumer is a silence detector — it cares that something was
said, not what. And a heartbeat that could raise would be a scoring failure caused by logging, so
this swallows its own errors.
"""
from __future__ import annotations

import sys
import time

# Well under the twenty-minute budget, well over "every loop iteration". The margin is deliberate:
# a round that ticks once every 20 s has sixty chances to prove it is alive before it is killed.
MIN_INTERVAL = 20.0

_T0 = time.monotonic()
_last = 0.0


def tick(stage: str, detail: str = "", force: bool = False) -> None:
    """One heartbeat. `force` for stage boundaries — they are rare and each one is worth a line."""
    global _last
    now = time.monotonic()
    if not force and now - _last < MIN_INTERVAL:
        return
    _last = now
    try:
        sys.stderr.write(f"[progress +{now - _T0:6.0f}s] {stage}"
                         f"{' ' + detail if detail else ''}\n")
        sys.stderr.flush()
    except Exception:
        pass
