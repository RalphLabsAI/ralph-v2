"""The exam's allocation rule is part of the exam, so the record names it.

The remainder of the stratified draw was shared in proportion to stratum size until 2026-08-31
(9bec1ed) and levelled up to the smallest stratum after. Both are deterministic in the nonce and
they draw different exams, so an audit that knew only the current rule re-derived a different
selection for rounds 1-4 and published "the operator, not the nonce, chose the exam" — a signed
accusation over a rule change. Records now carry `selection_rule`; a record that names none is
placed by its shape, and only the legacy shape (no drop accounting either) may match either rule.

Both halves run the REAL code: the record comes from run_v2_observer_epoch, the verdict from
eval.rerun.audit."""
from __future__ import annotations

import functools
import json
import struct
import tempfile
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

import eval.validator_observer_loop as vol
from eval.chain import Commitment, run_v2_observer_epoch
from eval.economics import RegistrationLedger
from eval.gates import TierBudget
from eval.identity import commit_value, content_hash
from eval.koth import Tier, Tournament
from eval.observer_round import (LEGACY_SELECTION_RULE, SELECTION_RULE, SELECTION_RULES,
                                 select_trajectories)
from eval.rerun import audit, record_from_blob
from eval.shadow_axis_epoch import FakeChain
from eval.signing import Ed25519Signer
from eval.steps import Trajectory

N_ITEMS = 48


def _d3(x, y, z):
    return {"a": x, "b": y, "c": z}


class Obs:
    def generate(self, prompts, max_new_tokens=128):
        return ["cont cont"] * len(prompts)

    def distributions(self, prefix, continuation):
        if prefix.rstrip().endswith("X"):
            return [_d3(0.8, 0.1, 0.1)] * 6
        return [_d3(0.34, 0.33, 0.33)] * 6


class Step:
    def __init__(self, tok):
        self.tok = tok

    def generate(self, prompts, max_new_tokens=256):
        return [self.tok for _ in prompts]


def _pool():
    """Four (language x depth) strata of UNEQUAL size, so the remainder split matters."""
    mk = lambda pre, src, idx, n: [Trajectory(id=f"{pre}{i}", source=src, prefix=f"{pre}{i}",
                                              step="s", index=idx) for i in range(n)]
    return (mk("en", "glaive_r1", 0, 300) + mk("hi", "samvaad_hi", 0, 60)
            + mk("ed", "glaive_r1", 3, 100) + mk("hd", "samvaad_hi", 3, 40))


def _epoch(dd, pool):
    hdr = {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
    hb = json.dumps(hdr).encode()
    with open(Path(dd) / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + b"\0" * 16)
    (Path(dd) / "config.json").write_text('{"hidden_size":8}')
    h, salt = content_hash(dd), "s0"
    tiers = [Tier("t", 10 ** 12, 1.0)]
    return run_v2_observer_epoch(
        FakeChain([Commitment("hot0", "cold0", "t", dd, 1.0, revealed_hash=h, salt=salt,
                              committed_value=commit_value(h, salt),
                              artifact_uri=f"file://{dd}")]),
        1, pool, Step("X"), {"kimi": Obs(), "qwen": Obs()}, tiers,
        {"t": TierBudget(name="t", max_params=10 ** 12, max_effective_bits=32.0)},
        Tournament(tiers), RegistrationLedger(), {},
        make_safe_runner=lambda cd: Step("X"),
        signer=Ed25519Signer(seed=b"z" * 32), n_items=N_ITEMS,
        corpus_spec="glaive_r1@rev=abc123|dedup=none|order=stream")


def _write(dd, rec, pool):
    rec_path, pool_path = str(Path(dd) / "record.json"), str(Path(dd) / "pool.jsonl")
    Path(rec_path).write_text(json.dumps(asdict(rec)))
    with open(pool_path, "w") as fh:
        for t in pool:
            fh.write(json.dumps(asdict(t)) + "\n")
    return rec_path, pool_path


def _rigged(dd, rec_path, pool_path, mutate):
    raw = deepcopy(json.loads(Path(rec_path).read_text()))
    mutate(raw)
    r2 = record_from_blob(json.dumps(raw).encode())
    r2.signature = r2.signer = r2.sig_scheme = ""
    r2.sign(Ed25519Signer(seed=b"z" * 32))
    p = Path(dd) / "rigged.json"
    p.write_text(json.dumps(asdict(r2)))
    return audit(str(p), pool_path)


def _selection(a):
    return next(c for c in a.checks if c.name == "item selection derives from the nonce")


def test_the_two_rules_draw_different_exams_from_the_same_entropy():
    pool = _pool()
    cur, ic = select_trajectories(pool, "root", "nonce", N_ITEMS)
    leg, il = select_trajectories(pool, "root", "nonce", N_ITEMS, rule=LEGACY_SELECTION_RULE)
    assert len(ic) == len(il) == N_ITEMS and ic != il, "the remainder split must matter here"
    assert select_trajectories(pool, "root", "nonce", N_ITEMS)[1] == ic, "deterministic"
    assert select_trajectories(pool, "root", "nonce", N_ITEMS,
                               rule=LEGACY_SELECTION_RULE)[1] == il
    with pytest.raises(ValueError):
        select_trajectories(pool, "root", "nonce", N_ITEMS, rule="uniform-v9")
    assert SELECTION_RULE in SELECTION_RULES and LEGACY_SELECTION_RULE in SELECTION_RULES


def test_a_record_names_its_rule_and_the_audit_reruns_that_rule():
    pool = _pool()
    with tempfile.TemporaryDirectory() as dd:
        rec = _epoch(dd, pool).outcome.record
        assert rec.manifest["selection_rule"] == SELECTION_RULE
        rec_path, pool_path = _write(dd, rec, pool)

        a = audit(rec_path, pool_path)
        assert not a.failed, [(c.name, c.detail) for c in a.failed]
        assert "named by the record" in _selection(a).detail

        # an unknown rule is a FAIL, never a guess
        a = _rigged(dd, rec_path, pool_path,
                    lambda r: r["manifest"].update(selection_rule="uniform-v9"))
        assert _selection(a).status == "FAIL" and "does not know" in _selection(a).detail

        # naming the wrong known rule re-derives a different exam
        a = _rigged(dd, rec_path, pool_path,
                    lambda r: r["manifest"].update(selection_rule=LEGACY_SELECTION_RULE))
        assert _selection(a).status == "FAIL"
        assert "the operator, not the nonce, chose the exam" in _selection(a).detail


def test_a_legacy_record_is_audited_under_the_rule_that_drew_it(monkeypatch):
    """Rounds 1-4: drawn under the proportional split, no `selection_rule`, no `exam_dropped`."""
    pool = _pool()
    monkeypatch.setattr(vol, "select_trajectories",
                        functools.partial(select_trajectories, rule=LEGACY_SELECTION_RULE))
    with tempfile.TemporaryDirectory() as dd:
        rec = _epoch(dd, pool).outcome.record
        rec_path, pool_path = _write(dd, rec, pool)
        _, current = select_trajectories(pool, rec.commit_root, rec.round_nonce, N_ITEMS)
        assert list(rec.manifest["item_indices"]) != list(current), "fixture drew the legacy exam"

        def legacy_shape(r):
            r["manifest"].pop("selection_rule")
            r["manifest"].pop("exam_dropped")

        a = _rigged(dd, rec_path, pool_path, legacy_shape)
        assert not a.failed, [(c.name, c.detail) for c in a.failed]
        sel = _selection(a)
        assert sel.status == "PASS" and "legacy record" in sel.detail
        assert LEGACY_SELECTION_RULE in sel.detail
        assert next(c for c in a.checks
                    if c.name == "every drawn item was scored or dropped for a stated reason"
                    ).status == "PASS"

        # a record that already accounts its drops postdates the rule change: no fallback
        a = _rigged(dd, rec_path, pool_path, lambda r: r["manifest"].pop("selection_rule"))
        assert _selection(a).status == "FAIL" and "current rule assumed" not in _selection(a).detail

        # the tag as the loop wrote it names the CURRENT rule, which did not draw this exam
        a = _rigged(dd, rec_path, pool_path, lambda r: None)
        assert _selection(a).status == "FAIL"
        assert f"tried ['{SELECTION_RULE}']" in _selection(a).detail

        # and naming the legacy rule outright is the honest, passing record
        a = _rigged(dd, rec_path, pool_path,
                    lambda r: r["manifest"].update(selection_rule=LEGACY_SELECTION_RULE))
        assert not a.failed, [(c.name, c.detail) for c in a.failed]
