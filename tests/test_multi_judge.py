"""Every judge scores every round; the crown metric is the mean over judges of each judge's
worst-slice score; a long reign halves both margins.

The record comes from the REAL round loop (run_v2_observer_epoch) and the verdict from the REAL
audit (eval.rerun.audit) — a hand-built multi-judge fixture would encode my assumptions about the
layout instead of testing them."""
from __future__ import annotations

import json
import random
import struct
import tempfile
import types
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from eval.chain import Commitment, run_v2_observer_epoch
from eval.economics import RegistrationLedger
from eval.gates import TierBudget
from eval.identity import commit_value, content_hash
from eval.koth import (DETHRONE_MARGIN, PERSIST_MARGIN, REIGN_DECAY_AFTER, REIGN_DECAY_FACTOR,
                       SOFTMIN_P, Scored, Tier, Tournament, _soft_min, floor_regression_slices,
                       judge_of, margins_for_reign, softmin_lcb_diff)
from eval.rerun import audit, record_from_blob
from eval.shadow_axis_epoch import FakeChain
from eval.signing import Ed25519Signer
from eval.steps import Trajectory


def _d3(x, y, z):
    return {"a": x, "b": y, "c": z}


class Obs:
    """A judge whose reading of the parent's step is `sharp`, so two judges score one student
    differently and the mean is a real combination rather than a no-op."""

    def __init__(self, sharp: float):
        self.sharp = sharp

    def generate(self, prompts, max_new_tokens=128):
        return ["cont cont"] * len(prompts)

    def distributions(self, prefix, continuation):
        tail = prefix.rstrip()[-1:]
        if tail == "X":
            r = (1.0 - self.sharp) / 2
            return [_d3(self.sharp, r, r)] * 6
        if tail == "W":
            return [_d3(0.55, 0.25, 0.20)] * 6
        return [_d3(0.34, 0.33, 0.33)] * 6


class Step:
    def __init__(self, tok):
        self.tok = tok

    def generate(self, prompts, max_new_tokens=256):
        return [self.tok] * len(prompts)


POOL = [Trajectory(id=f"t{i}", source="glaive_r1", prefix=f"p{i}", step="s", index=0)
        for i in range(400)]


def _epoch(dd, observers, student="W"):
    hdr = {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
    hb = json.dumps(hdr).encode()
    with open(Path(dd) / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + b"\0" * 16)
    (Path(dd) / "config.json").write_text('{"hidden_size":8}')
    h, salt = content_hash(dd), "s0"
    tiers = [Tier("t", 10 ** 12, 1.0)]
    return run_v2_observer_epoch(
        FakeChain([Commitment("hot0", "cold0", "t", dd, 1.0, revealed_hash=h, salt=salt,
                              committed_value=commit_value(h, salt), artifact_uri=f"file://{dd}")]),
        1, POOL, Step("X"), observers, tiers,
        {"t": TierBudget(name="t", max_params=10 ** 12, max_effective_bits=32.0)},
        Tournament(tiers, margin=DETHRONE_MARGIN), RegistrationLedger(), {},
        make_safe_runner=lambda cd: Step(student),
        signer=Ed25519Signer(seed=b"z" * 32), n_items=20,
        corpus_spec="glaive_r1@rev=abc123|dedup=none|order=stream")


def _write(dd, rec):
    rp, pp = str(Path(dd) / "record.json"), str(Path(dd) / "pool.jsonl")
    Path(rp).write_text(json.dumps(asdict(rec)))
    with open(pp, "w") as fh:
        for t in POOL:
            fh.write(json.dumps(asdict(t)) + "\n")
    return rp, pp


def test_every_judge_scores_and_the_crown_metric_is_their_mean():
    obs = {"judge-a": Obs(0.8), "judge-b": Obs(0.6)}
    with tempfile.TemporaryDirectory() as dd:
        rec = _epoch(dd, obs).outcome.record
        man = rec.manifest
        assert man["observers_scored"] == ["judge-a", "judge-b"]
        assert man["observer"] == "all" and man["judge_aggregate"] == "mean_of_worst_slice"
        items = [p["rollout_id"] for p in rec.points if p["observer"] == "judge-a"]
        assert [p["observer"] for p in rec.points] == ["judge-a"] * len(items) + ["judge-b"] * len(items)
        assert [p["rollout_id"] for p in rec.points if p["observer"] == "judge-b"] == items
        sub = [s for s in rec.submissions if s.role == "challenger"][0]
        assert len(sub.steps) == len(sub.effects) == len(rec.points)

        per_judge: dict = {}
        for k, v in sub.slices.items():
            per_judge.setdefault(judge_of(k), []).append(sum(v) / len(v))
        scores = {j: min(v) for j, v in per_judge.items()}
        assert set(scores) == {"obs=judge-a", "obs=judge-b"}
        assert abs(scores["obs=judge-a"] - scores["obs=judge-b"]) > 1e-3, "judges must disagree here"
        assert abs(sub.retention - sum(scores.values()) / 2) <= 1e-6

        # the real audit passes it, once per judge
        rp, pp = _write(dd, rec)
        for name in ("judge-a", "judge-b"):
            a = audit(rp, pp, obs[name], observer_name=name)
            assert not a.failed, [(c.name, c.detail) for c in a.failed]
            names = {c.name for c in a.checks if c.status == "PASS"}
            assert {"every judge read every item", "every candidate judge scored the round",
                    "audited with one of the round's judges"} <= names, names


def test_different_answers_under_different_judges_are_caught():
    obs = {"judge-a": Obs(0.8), "judge-b": Obs(0.6)}
    with tempfile.TemporaryDirectory() as dd:
        rec = _epoch(dd, obs).outcome.record
        raw = json.loads(json.dumps(asdict(rec)))
        sub = [s for s in raw["submissions"] if s["role"] == "challenger"][0]
        n = len(sub["steps"]) // 2
        sub["steps"][n] = sub["steps"][n] + " (a different answer for judge-b)"
        r2 = record_from_blob(json.dumps(raw).encode())
        r2.signature = r2.signer = r2.sig_scheme = ""
        r2.sign(Ed25519Signer(seed=b"z" * 32))
        rp, pp = _write(dd, r2)
        a = audit(rp, pp)
        assert any(c.name.startswith("one answer per item across judges") for c in a.failed), \
            [(c.name, c.detail) for c in a.failed]


def test_one_judge_in_the_pool_is_the_old_single_judge_round():
    with tempfile.TemporaryDirectory() as dd:
        rec = _epoch(dd, {"only": Obs(0.8)}).outcome.record
        man = rec.manifest
        assert man["observer"] == "only" and man["observers_scored"] == ["only"]
        assert man["judge_aggregate"] == "single"
        assert not any("observer" in p for p in rec.points), "single-judge points keep the old shape"
        rp, pp = _write(dd, rec)
        a = audit(rp, pp, Obs(0.8), observer_name="only")
        assert not a.failed, [(c.name, c.detail) for c in a.failed]


def test_floor_rule_compares_within_a_judge():
    A, A2 = "obs=a|lang=en|depth=deep", "obs=a|lang=hi|depth=deep"
    B, B2 = "obs=b|lang=en|depth=deep", "obs=b|lang=hi|depth=deep"
    king = {A: [0.30] * 10, A2: [0.40] * 10, B: [0.20] * 10, B2: [0.25] * 10}
    # judge a's en/deep falls to 0.25: above the king's floor ACROSS judges (0.20) but under
    # judge a's own floor (0.30) — per-judge comparison has to catch it
    worse = {A: [0.25] * 10, A2: [0.45] * 10, B: [0.30] * 10, B2: [0.30] * 10}
    assert floor_regression_slices(worse, king) == A
    fine = {A: [0.31] * 10, A2: [0.45] * 10, B: [0.21] * 10, B2: [0.30] * 10}
    assert floor_regression_slices(fine, king) is None


def _old_softmin_lcb(a, b, z_reps=2000, alpha=0.05, seed=0, p=SOFTMIN_P):
    """The single-judge statistic exactly as it was published before this change."""
    shared = [ax for ax in sorted(a) if ax in b and a[ax] and len(a[ax]) == len(b[ax])]
    rng, diffs = random.Random(seed), []
    for _ in range(z_reps):
        ar, br = [], []
        for ax in shared:
            av, bv = a[ax], b[ax]
            n = len(av)
            idx = [rng.randrange(n) for _ in range(n)]
            ar.append(sum(av[i] for i in idx) / n)
            br.append(sum(bv[i] for i in idx) / n)
        diffs.append(_soft_min(ar, p) - _soft_min(br, p))
    diffs.sort()
    return diffs[int(alpha * z_reps)]


def _sc(mid, per_axis, reign_retention=None):
    sub = types.SimpleNamespace(model_id=mid, miner="5" + mid, tier="t")
    ret = min(sum(v) / len(v) for v in per_axis.values())
    return Scored(sub=sub, retention=ret, retention_lb=1.0, per_point=[], gates_ok=True,
                  per_axis=per_axis)


def test_single_judge_bound_is_byte_identical_to_the_published_statistic():
    rng = random.Random(7)
    axes = [f"obs=o|lang={l}|depth={d}" for l in ("en", "hi") for d in ("deep", "shallow")]
    a = {ax: [rng.random() * 0.5 for _ in range(29)] for ax in axes}
    b = {ax: [rng.random() * 0.5 for _ in range(29)] for ax in axes}
    assert softmin_lcb_diff(_sc("a", a), _sc("b", b)) == _old_softmin_lcb(a, b)


AXES = [f"obs=o|lang={l}|depth={d}" for l in ("en", "hi", "zh") for d in ("deep", "shallow")]


def test_a_long_reign_halves_both_margins():
    assert margins_for_reign(REIGN_DECAY_AFTER - 1) == (DETHRONE_MARGIN, PERSIST_MARGIN)
    assert margins_for_reign(REIGN_DECAY_AFTER) == (DETHRONE_MARGIN * REIGN_DECAY_FACTOR,
                                                    PERSIST_MARGIN * REIGN_DECAY_FACTOR)
    # the same uniform +0.012 lead holds against a young king and dethrones a stale one
    for reign, want in ((REIGN_DECAY_AFTER - 1, "hold"), (REIGN_DECAY_AFTER, "dethrone")):
        t = Tournament([Tier("t", 10 ** 12, 1.0)])
        king = _sc("king", {ax: [0.300] * 29 for ax in AXES})
        t.kings["t"] = types.SimpleNamespace(miner="5king", model_id="king",
                                             retention=king.retention, reign=reign)
        e = t.consider("t", [_sc("ch", {ax: [0.312] * 29 for ax in AXES})], king)
        assert e["action"] == want, e
        assert e["king_reign"] == reign
        assert e["margin_applied"] == round(margins_for_reign(reign)[0], 6)


def test_a_byte_copy_never_dethrones_even_a_stale_king():
    t = Tournament([Tier("t", 10 ** 12, 1.0)])
    king = _sc("king", {ax: [0.30 + 0.001 * i for i in range(29)] for ax in AXES})
    t.kings["t"] = types.SimpleNamespace(miner="5king", model_id="king",
                                         retention=king.retention, reign=REIGN_DECAY_AFTER + 5)
    copy = _sc("copy", deepcopy(king.per_axis))
    for _ in range(4):
        e = t.consider("t", [copy], king)
        t.round += 1
        assert e["action"] == "hold" and e["lead"] == 0.0, e
