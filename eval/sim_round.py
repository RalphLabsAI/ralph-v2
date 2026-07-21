"""Full subnet loop, simulated end-to-end on CPU.

Runs several rounds with real KOTH dynamics and asserts the properties the design
rests on:
  * an honest challenger crowns an open throne
  * a COPY of the king ties and does NOT dethrone (no margin cleared)
  * a genuinely better student DOES dethrone, past the margin
  * weights follow the crown

    python -m eval.sim_round
"""
from __future__ import annotations

import random

from .koth import Submission, Tier, Tournament
from .round_engine import run_round
from .trajectory import Rollout

VOCAB = "alpha beta gamma delta epsilon zeta eta theta iota kappa lam mu nu xi omi".split()


def make_experience(n: int, length: int, seed: int) -> list[Rollout]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        steps = [" ".join(rng.sample(VOCAB, 5)) for _ in range(length)]
        out.append(Rollout(id=f"exp{i}", context="TASK", steps=steps, success=True))
    return out


class Model:
    """A student whose step matches GLM's with probability `fidelity`.

    DETERMINISTIC given the state: the per-step coin and the deviation are seeded from
    hash(prefix, model_seed). So two models with the same seed are the SAME function —
    an exact copy reproduces byte-identical steps on identical points, which makes its
    paired difference against the original exactly zero (a true tie), not a lucky
    divergence. Reproducibility is also what scoring requires (greedy, re-runnable)."""
    def __init__(self, name, fidelity, seed):
        self.name = name
        self.fidelity = fidelity
        self.seed = seed

    @staticmethod
    def _glm_step(prefix):
        # stable hash (Python's hash() is per-process randomized -> flaky sim)
        import hashlib
        last = prefix.strip().split("\n")[-1]
        h = int(hashlib.sha256(last.encode()).hexdigest()[:8], 16) % len(VOCAB)
        return " ".join(VOCAB[(h + j) % len(VOCAB)] for j in range(5))

    def generate(self, prompts, max_new_tokens=256):
        out = []
        for p in prompts:
            last = p.strip().split("\n")[-1]
            r = random.Random(f"{self.seed}|{last}")   # deterministic in (model, state)
            if r.random() < self.fidelity:
                out.append(self._glm_step(p))
            else:
                out.append(" ".join(r.sample(VOCAB, 5)))
        return out


def main() -> int:
    exp = make_experience(40, 12, seed=1)
    glm = Model("glm", 1.0, seed=0)
    base = Model("base", 0.35, seed=42)
    judge = __import__("eval.trajectory", fromlist=["SimJudge"]).SimJudge()

    tiers = [Tier("sub-3B", max_params=3_000_000_000, weight=1.0)]
    tour = Tournament(tiers, margin=0.03)

    # models keyed by model_id, so the reigning king stays re-scorable across rounds
    runners = {
        "m_weak": Model("weak", 0.55, seed=11),
        "m_good": Model("good", 0.78, seed=12),
        "m_copy": Model("copy", 0.78, seed=12),   # same seed as m_good == a true copy
        "m_better": Model("better", 0.95, seed=14),
    }

    def sub(mid, miner, params=2_000_000_000, compute=200.0):
        return Submission(miner=miner, tier="sub-3B", model_id=mid, params=params, compute_h100h=compute)

    rounds = [
        # round 1: weak + good compete for the open throne -> good should crown
        [sub("m_weak", "alice"), sub("m_good", "bob")],
        # round 2: a COPY of the king arrives -> must NOT dethrone (ties)
        [sub("m_copy", "eve"), sub("m_good", "bob")],
        # round 3: a genuinely better student -> should dethrone past the margin
        [sub("m_better", "carol"), sub("m_good", "bob")],
    ]

    print(f"tier sub-3B, dethrone margin {tour.margin}\n")
    ok = True
    registry: dict = {}   # the validator's persistent store of every king's checkpoint
    for i, subs in enumerate(rounds, start=1):
        submissions = [(s, runners[s.model_id]) for s in subs]
        # self_frac=0: self_state is only meaningful on genuinely multi-turn agentic
        # piles (a re-prompted single-response state restarts). This synthetic hash-based
        # demo exercises the KOTH crown/hold/dethrone dynamics on teacher_state only.
        res = run_round(i, commit_seed=1000 + i, experience=exp, glm=glm, base=base,
                        judge=judge, tiers=tiers, tournament=tour, submissions=submissions,
                        registry=registry, n_points=240, self_frac=0.0)
        ev = res.events[0]
        king = tour.kings["sub-3B"]
        line = (f"round {i}: {ev['action']:<9} king={king.model_id:<9} "
                f"miner={king.miner:<6} reign={king.reign}")
        if "margin_lcb" in ev:
            line += f"  margin_lcb={ev['margin_lcb']:+.3f}"
        print(line)
        print(f"          retentions: " + ", ".join(
            f"{mid.replace('m_',''):>7}={s.retention:.3f}" for mid, s in res.scored.items()))
        print(f"          weights: {res.weights}\n")

        if i == 1 and ev["action"] != "crown":
            ok = False; print("  FAIL: open throne should crown the best challenger")
        if i == 2 and ev["action"] == "dethrone":
            ok = False; print("  FAIL: a copy of the king dethroned it (margin breached)")
        if i == 3 and ev["action"] != "dethrone":
            ok = False; print("  FAIL: a genuinely better student failed to dethrone")

    print("PASS: crown -> copy holds -> better dethrones -> weights track the crown"
          if ok else "FAIL: KOTH dynamics incorrect")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
