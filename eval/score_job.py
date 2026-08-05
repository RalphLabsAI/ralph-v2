"""What runs ON THE RENTED GPU. Scores one round and writes an UNSIGNED record.

THIS BOX HOLDS NO KEYS, and that is the whole reason the validator is split in two. The June 2026
compromise of a persistent GPU box gave an attacker root for seven hours and with it the signing
keys — they rigged a crowning and wiped the logs. A rented box is worse, not better, on that axis:
it is somebody else's hardware, briefly. So the wallet, the record seed and the write token all stay
on the CPU orchestrator, and what comes back from here is an unsigned record plus the pool it was
scored against. The orchestrator re-derives it before signing.

WHAT THIS MEANS FOR TRUST. The orchestrator does NOT take these numbers on faith just because it
rented the box. It re-runs L0 and L1 against the returned record before its key ever touches it —
the same audit an outsider runs, pointed at our own scorer. So the failure this design has to
survive is "the rented box returns a plausible lie", and the answer is that a lie has to survive
the operator's own audit before it can be signed, and L3 afterwards.

    python -m eval.score_job job.json out/          # reads a job spec, writes record.json + pool.jsonl
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict


def score(job: dict, out_dir: str) -> dict:
    """Run the round described by `job`. Returns a small summary; writes the artifacts."""
    from .bitrate import TIERS
    from .economics import RegistrationLedger
    from .fetch import resolver
    from .gates import TierBudget
    from .koth import Tier, Tournament
    from .parent import PARENTS
    from .pool import build_pool, check_balance, dump_pool
    from .runners import HFRunner, student_runner
    from .validator_observer_loop import CommittedSubmission, run_observer_round

    os.makedirs(out_dir, exist_ok=True)
    spec = PARENTS[job["parent_key"]]

    print(f"  pool ({job['pool_size']} trajectories)…", flush=True)
    pool, pspec = build_pool(int(job["pool_size"]))
    ok, why = check_balance(pool)
    if not ok:
        raise SystemExit(f"pool refused: {why}")

    print(f"  parent {spec.name}…", flush=True)
    parent = HFRunner(spec.name)
    observers = {o: HFRunner(o) for o in job["observers"]}

    by_name = {t.name: t for t in TIERS}
    budgets = {n: TierBudget(name=n, max_params=int(spec.weight_params * (1 + spec.tol)),
                             max_effective_bits=by_name[n].max_container_bits,
                             bit_tier=by_name[n], parent=spec)
               for n in job["tiers"]}
    tiers = [Tier(n, max_params=budgets[n].max_params, weight=1.0 / len(job["tiers"]))
             for n in job["tiers"]]

    # Artifacts land HERE, on the rented disk, not on the orchestrator: a miner controls the size
    # and the CPU box also holds the signing key.
    fetch_log: list = []
    reveals = {c["hotkey"]: {"content_hash": c.get("revealed_hash", ""), "salt": c.get("salt", "")}
               for c in job["committed"]}
    fetch_dir_for = resolver(os.path.join(out_dir, "artifacts"), reveals=reveals, log=fetch_log)

    committed, skipped = [], []
    for c in job["committed"]:
        d = fetch_dir_for(c["hotkey"], c.get("artifact_uri", ""))
        if not d:
            skipped.append((c["hotkey"], "artifact could not be fetched"))
            continue
        committed.append(CommittedSubmission(
            hotkey=c["hotkey"], coldkey=c.get("coldkey", ""), tier=c["tier"], ckpt_dir=d,
            declared_compute_h100h=float(c.get("declared_compute_h100h", 0.0)),
            bond_posted=float(c.get("bond_posted", 0.0)),
            # dispatch by FORMAT: GGUF is the only artifact that can pass the bit tiers,
            # and hardcoding the safetensors loader here made every one of them unscoreable
            make_runner=(lambda cd=d: student_runner(cd)),
            revealed_hash=c.get("revealed_hash", ""), salt=c.get("salt", ""),
            committed_value=c.get("committed_value", ""), artifact_uri=c.get("artifact_uri", "")))

    # THE THRONE IS INHERITED, NOT RE-OPENED. A fresh Tournament takes koth's open-throne branch
    # every round and crowns max(retention) outright, which makes the dethrone margin, the paired
    # bootstrap and the anti-copy guarantee unreachable — a leaderboard that resets hourly wearing
    # the name of a king of the hill. The lineage is derived from the published trail by the
    # orchestrator (eval/lineage.py) and supplied here, and the king's artifact is refetched so the
    # incumbent is RE-SCORED on this round's items rather than defended on last round's number.
    from .lineage import Reign
    tournament = Tournament(tiers, margin=float(job.get("margin", 0.05)))
    registry = {}
    for tier, k in (job.get("kings") or {}).items():
        r = Reign(**k)
        tournament.kings[tier] = r.as_king()
        kd = fetch_dir_for(f"king:{tier}", r.artifact_uri)
        if kd:
            try:
                registry[r.model_id] = student_runner(kd)
            except Exception as e:
                # koth emits a "king unavailable" hold for a missing registry entry, which keeps the
                # throne without re-scoring it. That is the right conservative default and it is
                # recorded, not silent.
                skipped.append((f"king:{tier}", f"incumbent not loadable: {type(e).__name__}: {e}"))
        else:
            skipped.append((f"king:{tier}", f"incumbent artifact not fetchable: {r.artifact_uri}"))

    out = run_observer_round(
        int(job["round"]), job["commit_root"], job["round_nonce"], committed, pool, parent,
        observers, tiers, budgets, tournament,
        RegistrationLedger(), registry, parent_id=spec.name,
        prev_anchor=job.get("prev_anchor", ""),
        signer=None,                       # UNSIGNED, deliberately: the key is not on this box
        n_items=int(job["n_items"]), corpus_spec=pspec.as_corpus_spec())

    with open(os.path.join(out_dir, "record.json"), "w") as fh:
        json.dump(asdict(out.record) if out.record else None, fh)
    with open(os.path.join(out_dir, "pool.jsonl"), "wb") as fh:
        fh.write(dump_pool(pool))
    summary = {"round": int(job["round"]), "accepted": list(out.accepted),
               "kings_before": {t: k.model_id for t, k in (out.kings_before or {}).items()},
               "rejected": [[h, r] for h, r in out.rejected], "skipped": skipped,
               "identity": out.identity, "observer": out.observer,
               "events": out.events, "weights": out.weights,
               "fetch": [list(map(str, e)) for e in fetch_log],
               "gpu": _gpu_name()}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def _gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except Exception:
        return ""


def main(argv: list) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    job = json.loads(open(argv[0]).read())
    s = score(job, argv[1])
    print(json.dumps({k: s[k] for k in ("round", "accepted", "identity", "observer", "gpu")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
