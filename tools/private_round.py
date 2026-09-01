"""Score the current field on a box we already own, and publish nothing.

    python -m tools.private_round            # scores, writes the record to disk, exits

WHAT THIS IS FOR. Seeing what a round WOULD decide without putting it on the trail: no HF write, no
anchor, no weight vector, no crown mirror. The record is built, signed and gated exactly as in
production — `LocalSink` just puts it in a directory instead of a dataset repo.

WHAT IT IS NOT. It is not the round. The exam is drawn from the block at run time, so a real round
later draws a different one, and measured exam-to-exam swing is ~0.05 — the whole dethrone margin.
Read the SHAPE (who passes intake, who gates, who is near a king), never the exact number.

THE BOX IS NOT OURS TO DESTROY. `destroy()` is a no-op here on purpose: the orchestrator tears down
rentals in a `finally`, and pointing that at a machine somebody else is using would be the most
expensive possible bug. Nothing in this file stops, reboots or cleans the host.
"""
from __future__ import annotations

import os
import sys

from eval.orchestrator import Instance
from eval.run_orchestrated import Config, run

HOST = os.environ.get("RALPH_BYO_HOST", "160.202.129.159")
USER = os.environ.get("RALPH_BYO_USER", "root")
# NOT EVERY BORROWED BOX LISTENS ON 22. A RunPod pod publishes ssh on a mapped high port, and a
# hard-coded 22 sends every command of the round to whatever answers there instead.
PORT = int(os.environ.get("RALPH_BYO_PORT", "22"))


class ByoProvider:
    """A provider that hands back a machine we already have.

    The orchestrator's rent-retry loop, its region exclusions and its price cap all exist to pick a
    rental; none of them apply. `wait_ready` returns immediately because the host is already up,
    and `destroy` deliberately does nothing."""

    def rent(self, spec, name: str, exclude: tuple = ()) -> Instance:
        return Instance(id=f"byo-{HOST}", ip=HOST, ssh_user=USER, ssh_port=PORT,
                        cloud="byo", region="local", instance_type="H100",
                        price_per_hour=0.0, status="active")

    def wait_ready(self, inst: Instance, timeout_s: float = 900, out=None) -> Instance:
        inst.status = "active"
        return inst

    def destroy(self, inst: Instance) -> None:
        return None


def main() -> int:
    for k, v in (("RALPH_SET_WEIGHTS", "0"), ("RALPH_PUBLISH_CROWNS", "0")):
        os.environ[k] = v                       # belt: `run` also refuses to write without --live
    if not os.environ.get("RALPH_RECORDS_DIR"):
        sys.stdout.write("refusing to run: set RALPH_RECORDS_DIR, or the record publishes to HF\n")
        return 2
    cfg = Config.from_env()
    cfg.live = False                            # no anchor, no weights, ever, from this entrypoint
    cfg.publish_crowns = False
    sys.stdout.write(f"  private round on {USER}@{HOST}:{PORT}"
                     f" (CUDA_VISIBLE_DEVICES={os.environ.get('RALPH_REMOTE_ENV', 'unset')})\n")
    return run(cfg, provider=ByoProvider())


if __name__ == "__main__":
    raise SystemExit(main())
