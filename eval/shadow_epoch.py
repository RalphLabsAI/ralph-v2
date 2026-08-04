"""Shadow epoch on real models — the full v2 pipeline, end to end, on a GPU.

gpu_run.py validates the scoring CORE (run_round). This validates EVERYTHING a live
shadow launch touches, by driving `run_v2_epoch`:

  real HF checkpoints on disk  (stand in for fetched miner submissions)
    -> intake gates on real safetensors headers (params/bits/tier, forbidden files)
    -> SafeStudentRunner untrusted load (trust_remote_code=False, safetensors-only)
    -> batched scoring vs pinned GLM + grounded judge, normalized against base
    -> KOTH crown/dethrone + economics bond settle
    -> FakeChain write-back (set_king/set_weights) + signed round record

Shadow = no emission, no real chain: FakeChain records the writes so we can assert the
validator DECIDED correctly on real inference. Flip to the real chain (chain.ChainIO) by
swapping FakeChain for eval.chain_bittensor.BittensorChainIO once the box is up.

    python -m eval.shadow_epoch 2>&1 | tee shadow_epoch.log

Env: RALPH_TEACHER, RALPH_JUDGE, RALPH_BASE, RALPH_STUDENTS (comma HF ids),
RALPH_PILE (swe|teacher, default teacher), RALPH_NPOINTS, RALPH_NROLLOUTS,
RALPH_TIER_MAXP (tier param cap), RALPH_BATCH.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .chain import Commitment, run_v2_epoch
from .economics import RegistrationLedger
from .gates import TierBudget
from .koth import Tier, Tournament
from .round_record import RoundRecord
from .runners import HFRunner, SafeStudentRunner
from .trajectory import GroundedJudge

TEACHER = os.environ.get("RALPH_TEACHER", "Qwen/Qwen2.5-7B-Instruct")
JUDGE = os.environ.get("RALPH_JUDGE", "Qwen/Qwen2.5-7B-Instruct")
BASE = os.environ.get("RALPH_BASE", "Qwen/Qwen2.5-0.5B-Instruct")
STUDENTS = os.environ.get("RALPH_STUDENTS",
                          "Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-1.5B-Instruct").split(",")
PILE = os.environ.get("RALPH_PILE", "teacher")
NPOINTS = int(os.environ.get("RALPH_NPOINTS", "48"))
NROLLOUTS = int(os.environ.get("RALPH_NROLLOUTS", "12"))
TIER_MAXP = int(os.environ.get("RALPH_TIER_MAXP", "4000000000"))   # sub-4B tier for this rehearsal
BATCH = int(os.environ.get("RALPH_BATCH", "8"))


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


class FakeChain:
    """Records chain writes so a shadow run can assert the validator's decisions."""

    def __init__(self, commits):
        self._c = commits
        self.kings, self.weights, self.records, self.blacklisted = {}, None, [], []

    def current_block(self):
        return 1000

    def block_hash(self, b):
        return hashlib.sha256(f"block{b}".encode()).hexdigest()

    def read_commitments(self, lo, hi):
        return self._c

    def commit_root(self, lo, hi):
        h = hashlib.sha256()
        for c in self._c:
            h.update(c.hotkey.encode())
        return h.hexdigest()

    def set_weights(self, w):
        self.weights = w
        return True

    def set_king(self, tier, hk, mid):
        self.kings[tier] = (hk, mid)

    def get_king(self, tier):
        return self.kings.get(tier)

    def blacklist(self, hk, reason):
        self.blacklisted.append((hk, reason))

    def publish_record(self, record: RoundRecord):
        self.records.append(record)


def _fetch(model_id: str) -> str:
    """Download a real model to a local dir so it can be treated as a fetched submission."""
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])


def _load_pile(teacher):
    if PILE == "swe":
        from .pile import load_swe_agent_trajectories
        exp = load_swe_agent_trajectories(limit=NROLLOUTS)
        log(f"loaded {len(exp)} SWE-agent rollouts")
        return exp
    from .rollouts_gen import generate_experience
    log("generating experience pile with the teacher...")
    exp = generate_experience(teacher, n=NROLLOUTS)
    log(f"  {len(exp)} rollouts, steps: {[len(r.steps) for r in exp]}")
    return exp


def main() -> int:
    log(f"teacher={TEACHER} judge={JUDGE} base={BASE} students={STUDENTS} tier_maxp={TIER_MAXP:,}")
    teacher = HFRunner(TEACHER, name="teacher", batch_size=BATCH)
    judge_model = HFRunner(JUDGE, name="judge", batch_size=BATCH)
    base = HFRunner(BASE, name="base", batch_size=BATCH)
    judge = GroundedJudge(judge_model)

    exp = _load_pile(teacher)
    if len(exp) < 3:
        log("too few rollouts — aborting")
        return 1

    # fetch students to local dirs; they enter as UNTRUSTED fetched checkpoints
    log("fetching student checkpoints...")
    commits = []
    for i, mid in enumerate(STUDENTS):
        d = _fetch(mid.strip())
        commits.append(Commitment(hotkey=f"miner{i}", coldkey=f"ck{i}", tier="rehearsal",
                                   ckpt_dir=d, declared_compute_h100h=200.0, bond_posted=0.0))
        log(f"  miner{i} <- {mid} @ {d}")

    tiers = [Tier("rehearsal", max_params=TIER_MAXP, weight=1.0)]
    budgets = {"rehearsal": TierBudget("rehearsal", TIER_MAXP, max_effective_bits=16.5)}
    tour = Tournament(tiers, margin=0.03)
    ledger = RegistrationLedger(per_coldkey_round_cap=3, base_bond=1.0)
    registry = {}

    def make_safe_runner(ckpt_dir: str):
        return SafeStudentRunner(ckpt_dir, batch_size=BATCH)

    chain = FakeChain(commits)
    log(f"running full v2 epoch on {NPOINTS} fresh points...")
    res = run_v2_epoch(chain, 1, exp, teacher, base, judge, tiers, budgets, tour, ledger,
                       registry, make_safe_runner=make_safe_runner, n_points=NPOINTS,
                       self_frac=0.25)

    out = res.outcome
    report = {
        "teacher": TEACHER, "judge": JUDGE, "base": BASE, "pile": PILE,
        "n_rollouts": len(exp),
        "accepted": out.accepted,
        "rejected": [(h, r) for h, r in out.rejected],
        "chain_set_king": chain.kings,
        "chain_set_weights": chain.weights,
        "record_published": len(chain.records) == 1,
        "record_sha256": res.record_sha256,
        "retentions": {s.model_id: round(s.retention, 4) for s in (out.record.submissions if out.record else [])},
    }
    print("\n===== SHADOW EPOCH RESULT =====")
    print(json.dumps(report, indent=2))

    # invariants a real shadow launch must hold
    assert chain.weights is not None, "weights never set"
    assert len(chain.records) == 1, "round record not published"
    log("OK: intake -> safe load -> score -> crown -> weights -> record, on real models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
