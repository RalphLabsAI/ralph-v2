"""What did the round actually decide? Read-only, from the PUBLISHED record.

Deliberately reads the published artifact rather than any local state. The record is the thing a
miner or an auditor can fetch and check for themselves, so a summary built from anything else is a
summary of a different object than the one the subnet is accountable for — and the operator's local
files are exactly the half they could have edited.

    python -m eval.show_round               # the newest published round
    python -m eval.show_round --round 1
    python -m eval.show_round --json        # the same numbers, machine-readable
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def load(round_no: int | None, repo: str):
    """Fetch through the SAME path `verify_window` uses: the index names the blob, the sink serves
    it, and the digest is re-checked here rather than assumed. A summary that skipped the digest
    would happily render a record that no longer matches what was anchored."""
    from .publish import HFSink, RecordPublisher, roundtrip_reason

    # THE SAME high-water-mark file the round uses, not a throwaway publisher. It is what makes a
    # failed index read raise instead of rendering as "no rounds yet" — which is precisely the
    # false answer this tool would otherwise give at the worst moment, right after a round lands.
    # Read-only: nothing here publishes, so the file is never written.
    work_dir = os.environ.get("RALPH_WORK_DIR", "/workspace/ralph-v2-work")
    hwm = os.path.join(work_dir, f"publish-hwm-{repo.replace('/', '_')}.json")
    pub = RecordPublisher(HFSink(repo), window=8, state_path=hwm)
    idx = pub.load_index()
    rounds = idx.get("rounds") or []
    if not rounds:
        raise SystemExit(f"no rounds published in {repo} yet")
    if round_no is None:
        round_no = max(int(r.get("round", 0)) for r in rounds)
    entry = next((r for r in rounds if int(r.get("round", -1)) == int(round_no)), None)
    if entry is None:
        raise SystemExit(f"round {round_no} is not in the index of {repo} "
                         f"(have: {sorted(int(r.get('round', -1)) for r in rounds)})")
    blob = pub.sink.get(entry["name"])
    if blob is None:
        raise SystemExit(f"round {round_no} is indexed as {entry['name']} but is not fetchable")
    # THE REAL VERIFIER, not a reimplementation of it. My first version compared
    # `sha256(blob)` to the index digest and rejected the very first honest record this subnet
    # ever published — the digest is over `canonical()`, so two different serialisations of the
    # same record are the same record, and `roundtrip_reason`'s docstring says so by name. It also
    # checks the round number and the pinned signature, which a bare digest comparison misses:
    # canonical() excludes the signature, so a record can be re-signed with the digest intact.
    ok, reason = roundtrip_reason(blob, entry.get("sha256", ""),
                                  expect_round=int(round_no),
                                  expect_signature=entry.get("signature", ""))
    if not ok:
        raise SystemExit(f"round {round_no} failed verification: {reason}")
    return round_no, json.loads(blob.decode()), idx


def render(round_no, rec, repo, w=sys.stdout.write):
    subs = list(getattr(rec, "submissions", None) or rec.get("submissions", []))
    events = list(getattr(rec, "events", None) or rec.get("events", []))
    weights = dict(getattr(rec, "weights", None) or rec.get("weights", {}))
    noise = dict(getattr(rec, "noise", None) or rec.get("noise", {}))

    w(f"\nround {round_no}   records -> {repo}\n")
    sig = getattr(rec, "signer", "") or (rec.get("signer", "") if isinstance(rec, dict) else "")
    w(f"  signed by : {sig or '(unsigned)'}\n")
    if noise:
        w(f"  noise floor: {noise}\n")

    # RETENTION IS THE PRODUCT. Sorted by tier then score so the crown line below is checkable by
    # eye against the table rather than taken on trust.
    w("\n  miner            tier      retention   lower bound  gates\n")
    w("  " + "-" * 62 + "\n")
    for s in sorted(subs, key=lambda x: (_get(x, "tier"), -_num(_get(x, "retention")))):
        hk = str(_get(s, "miner"))[:14]
        gates = "ok" if _get(s, "gates_ok") else "REJECTED"
        w(f"  {hk:<16} {str(_get(s, 'tier')):<9} {_fmt(_get(s, 'retention')):>9}  "
          f"{_fmt(_get(s, 'retention_lb')):>11}  {gates}\n")
        for r in (_get(s, "reasons") or []):
            w(f"      - {r}\n")

    if events:
        w("\n  crown events\n")
        for e in events:
            w(f"    {json.dumps(e, sort_keys=True)}\n")

    w("\n  weights\n")
    if not weights:
        w("    (none set)\n")
    else:
        total = sum(float(v) for v in weights.values()) or 1.0
        for hk, v in sorted(weights.items(), key=lambda kv: -float(kv[1])):
            w(f"    {hk[:16]:<18} {_fmt(v)}   ({100 * float(v) / total:.1f}%)\n")
    w("\n")


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _num(x):
    try:
        return float(x)
    except Exception:
        return float("-inf")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--repo", default=os.environ.get("RALPH_HF_REPO",
                                                     "RalphLabsAI/ralph-v2-rounds"))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    round_no, rec, _idx = load(a.round, a.repo)
    if a.json:
        body = rec if isinstance(rec, dict) else json.loads(rec.canonical())
        print(json.dumps(body, indent=2, sort_keys=True))
        return 0
    render(round_no, rec, a.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
