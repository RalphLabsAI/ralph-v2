"""Finish a round whose scoring completed but whose tail did not.

Round 2 scored, was audited, and was then REFUSED by a bug in the audit itself (a hold round bound
no kings). The GPU is gone; the record and pool are on disk. This runs exactly the tail
`run_orchestrated` would have run — audit, sign, publish, anchor — and nothing else. It never
re-scores, never rents, and never sets weights.

    python finish_round.py --round 2            # show what would happen, touch nothing
    python finish_round.py --round 2 --commit   # sign, publish, anchor
"""
import argparse
import json
import os
import sys

from eval.chain_bittensor import BittensorChainIO
from eval.publish import HFSink, RecordPublisher, publish_and_gate
from eval.rerun import FAIL, audit_loaded, load_pool, record_from_blob
from eval.signing import Ed25519Signer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--commit", action="store_true")
    a = ap.parse_args()

    work = os.environ.get("RALPH_WORK_DIR", "/workspace/ralph-v2-work")
    wd = os.path.join(work, f"round-{a.round}")
    repo = os.environ.get("RALPH_HF_REPO", "RalphLabsAI/ralph-v2-rounds")

    rec = record_from_blob(open(os.path.join(wd, "record.json"), "rb").read())
    pool_blob = open(os.path.join(wd, "pool.jsonl"), "rb").read()
    pool = load_pool(os.path.join(wd, "pool.jsonl"))

    print(f"round {rec.round}: {len(rec.submissions)} submissions, {len(rec.events)} events")
    print(f"  weights    : {json.dumps(rec.weights)}")
    print(f"  already signed: {bool(rec.signature)}")
    print(f"  prev_anchor: {(rec.prev_anchor or '(none)')[:24]}…")

    # 1. THE SAME AUDIT the orchestrator runs before its key touches anything. Signature checks are
    #    excluded here and only here: the record is unsigned BY DESIGN at this point.
    audit = audit_loaded(rec, pool=pool)
    bad = [c for c in audit.checks if c.status == FAIL and not c.name.startswith("signature")]
    for c in bad:
        print(f"  REJECT [{c.level}] {c.name}: {c.detail}")
    if bad:
        print(f"\n{len(bad)} blocking failure(s) — NOT signing")
        return 1
    print(f"  audited    : {len(audit.checks)} checks, 0 blocking failures")

    hwm = os.path.join(work, f"publish-hwm-{repo.replace('/', '_')}.json")
    pub = RecordPublisher(HFSink(repo), window=8, state_path=hwm)

    # 2. THE CHAIN IT MUST HANG FROM. A record whose prev_anchor does not match the published head
    #    is a fork, not a continuation.
    head = pub.head_anchor()
    print(f"  trail head : {(head or '(none)')[:24]}…")
    if (rec.prev_anchor or "") != (head or ""):
        print(f"\nprev_anchor does not match the published head — this record does not continue "
              f"the trail. Refusing.")
        return 2
    print("  chains on  : OK")

    if not a.commit:
        print("\nDRY — nothing written. Re-run with --commit to sign, publish and anchor.")
        return 0

    seed = os.environ["RALPH_RECORD_SEED"]
    rec.sign(Ed25519Signer(seed=bytes.fromhex(seed)[:32]))
    print(f"  signed by  : {rec.signer[:16]}…")

    chain = BittensorChainIO(netuid=int(os.environ.get("RALPH_NETUID", "40")),
                             network=os.environ.get("RALPH_NETWORK", "finney"),
                             wallet_name=os.environ.get("RALPH_WALLET", "default"),
                             hotkey_name=os.environ.get("RALPH_HOTKEY", "default"),
                             read_only=False)

    pub.publish_pool(pool_blob, digest=(rec.manifest or {}).get("pool_sha256", ""))
    rep = publish_and_gate(pub, rec, anchor_fn=chain.publish_record,
                           head_anchor_fn=chain.head_anchor, allow_unanchored=False)
    print(f"  published  : {rep.published.uri if rep.published else None}")
    print(f"  anchored   : {rep.anchor_verified}   reasons={rep.reasons or '-'}")
    if not rep.ok:
        print("  WITHHELD — publishing did not verify")
        return 3
    # WEIGHTS ARE DELIBERATELY NOT SET. Shakedown: the trail is the product, the payout is not.
    print("  weights    : NOT SET (shakedown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
