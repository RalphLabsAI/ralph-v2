"""A drawn item the round could not use must be ACCOUNTED, not silently absent.

build_shared drops a sample for miner-independent reasons (empty parent step, silent observer,
parent effect under the floor) — decided before any miner runs. Round 5 (2026-08-31) drew 144
items, found one unusable, honestly scored the other 143, and the audit refused to sign its own
record: "the exam was pruned after it was drawn". The record now names each drop with its reason,
and the audit accepts absence exactly there — while still refusing absences it cannot explain,
items both scored and dropped, and drops large enough to replace the exam.

Both halves run the REAL code: the record comes from run_v2_observer_epoch, the verdict from
eval.rerun.audit — a hand-built fixture here would be the inverse of what the writer emits."""
from __future__ import annotations

import json
import struct
import tempfile
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from eval.chain import Commitment, run_v2_observer_epoch
from eval.economics import RegistrationLedger
from eval.gates import TierBudget
from eval.identity import commit_value, content_hash
from eval.koth import Tier, Tournament
from eval.rerun import audit, record_from_blob
from eval.shadow_axis_epoch import FakeChain
from eval.signing import Ed25519Signer
from eval.steps import Trajectory


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
    """Parent that goes silent on chosen prefixes — the real "unusable" trigger."""

    def __init__(self, tok, silent_on=()):
        self.tok, self.silent = tok, set(silent_on)

    def generate(self, prompts, max_new_tokens=256):
        return ["" if p in self.silent else self.tok for p in prompts]


def _epoch(dd, pool, parent):
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
        1, pool, parent, {"kimi": Obs(), "qwen": Obs()}, tiers,
        {"t": TierBudget(name="t", max_params=10 ** 12, max_effective_bits=32.0)},
        Tournament(tiers, margin=0.03), RegistrationLedger(), {},
        make_safe_runner=lambda cd: Step("X"),
        signer=Ed25519Signer(seed=b"z" * 32), n_items=20,
        corpus_spec="glaive_r1@rev=abc123|dedup=none|order=stream")


def test_a_recorded_drop_passes_and_an_unexplained_absence_fails():
    pool = [Trajectory(id=f"t{i}", source="glaive_r1", prefix=f"p{i}", step="s", index=0)
            for i in range(400)]

    with tempfile.TemporaryDirectory() as dd:
        # First pass: learn which items the nonce draws, then silence the parent on one of them.
        scout = _epoch(dd, pool, Step("X"))
        drawn = list(scout.outcome.record.manifest["item_indices"])
        victim = pool[drawn[0]]

    with tempfile.TemporaryDirectory() as dd:
        res = _epoch(dd, pool, Step("X", silent_on={victim.prefix}))
        rec = res.outcome.record
        man = rec.manifest
        assert man["exam_dropped"] == [{"id": victim.id, "reason": "parent produced an empty step"}]
        assert victim.id not in {p.get("rollout_id") for p in rec.points}

        rec_path, pool_path = str(Path(dd) / "record.json"), str(Path(dd) / "pool.jsonl")
        Path(rec_path).write_text(json.dumps(asdict(rec)))
        with open(pool_path, "w") as fh:
            for t in pool:
                fh.write(json.dumps(asdict(t)) + "\n")

        # the honest record — one drawn item dropped, named, reasoned — passes L0+L1
        a = audit(rec_path, pool_path)
        assert not a.failed, [(c.name, c.detail) for c in a.failed]
        assert any(c.name == "every drawn item was scored or dropped for a stated reason"
                   and c.status == "PASS" for c in a.checks)

        sig = Ed25519Signer(seed=b"z" * 32)

        def rigged(mutate):
            raw = deepcopy(json.loads(Path(rec_path).read_text()))
            mutate(raw)
            r2 = record_from_blob(json.dumps(raw).encode())
            r2.signature = r2.signer = r2.sig_scheme = ""
            r2.sign(sig)
            p = Path(dd) / "rigged.json"
            p.write_text(json.dumps(asdict(r2)))
            return audit(str(p), pool_path)

        # strip the accounting: same absence, no stated reason -> pruning
        a = rigged(lambda r: r["manifest"].update(exam_dropped=[]))
        assert any("no recorded reason" in c.detail for c in a.failed), \
            [(c.name, c.detail) for c in a.failed]

        # a drop list that contradicts the points is not describing this record
        a = rigged(lambda r: r["manifest"].update(exam_dropped=[
            {"id": victim.id, "reason": "parent produced an empty step"},
            {"id": r["points"][0]["rollout_id"], "reason": "parent produced an empty step"}]))
        assert any(c.name == "no item is both scored and dropped" for c in a.failed), \
            [(c.name, c.detail) for c in a.failed]

        # and drops cannot swallow the exam, whatever the reasons say
        def swallow(r):
            keep = {p["rollout_id"] for p in r["points"][:2]}
            gone = [p["rollout_id"] for p in r["points"] if p["rollout_id"] not in keep]
            r["points"] = [p for p in r["points"] if p["rollout_id"] in keep]
            r["manifest"]["exam_dropped"] = (r["manifest"]["exam_dropped"] +
                                             [{"id": i, "reason": "parent produced an empty step"}
                                              for i in gone])
        a = rigged(swallow)
        assert any(c.name == "the drops are a sliver of the exam" for c in a.failed), \
            [(c.name, c.detail) for c in a.failed]
