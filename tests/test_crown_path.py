"""Regression tests for the crown decision path (the fixes in commit 2494017 + the
long-context axis). Runnable two ways:

    python -m tests.test_crown_path      # self-running harness, no deps
    pytest tests/test_crown_path.py      # if pytest is installed

Covers: worst-axis dethrone (a fluent drifter that clears the pooled mean is blocked),
the axis_round crown gates (degeneracy DQ, every-declared-axis-live, copy ties,
genuine improvement dethrones), and the deterministic long-context checker.
"""
from __future__ import annotations

import random
import re

from eval.axes.code_exec import CodeExec
from eval.axes.long_context import LongContext
from eval.axis_round import AxisSpec, axis_round
from eval.core import Item
from eval.koth import (Scored, Submission, Tier, Tournament, bootstrap_lcb_diff,
                       worst_axis_lcb_diff)

MARGIN = 0.05


def _scored(mid, per_axis):
    flat = [v for ax in per_axis.values() for v in ax]
    return Scored(sub=Submission(mid, "t", mid, 1, 0.0), retention=0.0, retention_lb=0.5,
                  per_point=flat, gates_ok=True, per_axis=per_axis)


def test_worst_axis_blocks_drifter():
    """A challenger strong on a big axis but weak on a small one clears the POOLED mean
    margin yet is blocked by the worst-axis test; honest-better dethrones; copy ties."""
    big, small = 500, 50
    king = _scored("king", {"A": [0.60] * big, "B": [0.60] * small})
    drifter = _scored("drift", {"A": [0.92] * big, "B": [0.50] * small})
    honest = _scored("honest", {"A": [0.70] * big, "B": [0.71] * small})
    copyk = _scored("copy", {"A": [0.60] * big, "B": [0.60] * small})

    # the drifter WOULD dethrone on the pooled mean (the old, wrong test)...
    assert bootstrap_lcb_diff(drifter.per_point, king.per_point, seed=1) > MARGIN
    # ...but the worst-axis test blocks it, admits the honest-better, and ties the copy.
    assert worst_axis_lcb_diff(drifter, king, seed=1) <= MARGIN
    assert worst_axis_lcb_diff(honest, king, seed=1) > MARGIN
    assert worst_axis_lcb_diff(copyk, king, seed=1) <= MARGIN


# --- axis_round end-to-end with a controllable fake axis ---

class _FakeAxis:
    def __init__(self, name):
        self.name = name

    def generate(self, seed, n, difficulty=1):
        return [Item(axis=self.name, prompt=f"{self.name}:{seed}:{i}", answer="GOOD",
                     meta={"idx": i}) for i in range(n)]

    def check(self, item, output):
        return (output or "").strip() == "GOOD"


class _Sim:
    def __init__(self, name, probs, seed=0, loop=False):
        self.name, self.probs, self.seed, self.loop = name, probs, seed, loop

    def generate(self, prompts, max_new_tokens=512):
        out = []
        for p in prompts:
            if self.loop:
                out.append(("spam " * 400).strip())
                continue
            axis = p.split(":")[0]
            r = random.Random(f"{self.seed}|{p}")
            out.append("GOOD" if r.random() < self.probs.get(axis, 0.0) else "no")
        return out


def _run(round_no, specs, tiers, tour, reg, subs, glm, base):
    # 150 items/axis: worst-axis dethrone (unlike the old pooled test) is powered PER AXIS,
    # so a real improvement needs enough teacher-passed items per axis to clear the margin
    # with statistical confidence — the production knob the red-team flagged (n >= ~150).
    return axis_round(round_no, commit_seed=100 + round_no, specs=specs, glm=glm, base=base,
                      tiers=tiers, tournament=tour, submissions=subs, registry=reg,
                      items_per_axis=150, max_new_tokens=8)


def test_axis_round_gates():
    specs = [AxisSpec(_FakeAxis("x"), "x", 1.0), AxisSpec(_FakeAxis("y"), "y", 1.0)]
    tiers = [Tier("t", 10 ** 12, 1.0)]
    tour, reg = Tournament(tiers, margin=0.03), {}
    glm = _Sim("glm", {"x": 1.0, "y": 1.0})
    base = _Sim("base", {"x": 0.30, "y": 0.30}, seed=9)

    def sub(mid):
        return Submission("m_" + mid, "t", mid, 1, 1.0)

    good = (sub("good"), _Sim("good", {"x": 0.85, "y": 0.85}, seed=1))
    looper = (sub("looper"), _Sim("looper", {}, loop=True))
    r1 = _run(1, specs, tiers, tour, reg, [good, looper], glm, base)
    assert not r1.scored["looper"].gates_ok            # degeneracy DQ
    assert tour.kings["t"].model_id == "good"          # good crowns open throne

    drifter = (sub("drift"), _Sim("drift", {"x": 0.99, "y": 0.40}, seed=2))
    _run(2, specs, tiers, tour, reg, [drifter, good], glm, base)
    assert tour.kings["t"].model_id == "good"          # worst-axis holds vs drifter

    copyk = (sub("copy"), _Sim("good", {"x": 0.85, "y": 0.85}, seed=1))
    _run(3, specs, tiers, tour, reg, [copyk, good], glm, base)
    assert tour.kings["t"].model_id == "good"          # exact copy ties

    better = (sub("better"), _Sim("better", {"x": 0.99, "y": 0.98}, seed=4))
    _run(4, specs, tiers, tour, reg, [better, good], glm, base)
    assert tour.kings["t"].model_id == "better"        # genuine improvement dethrones

    onex = (sub("onex"), _Sim("onex", {"x": 0.9, "y": 0.0}, seed=5))
    r5 = _run(5, specs, tiers, tour, reg, [onex], glm, base)
    assert not r5.scored["onex"].gates_ok              # y non-live -> all-axes-live gate blocks


def test_long_context_checker():
    ax = LongContext()
    items = ax.generate(seed=7, n=16, difficulty=2)
    assert {it.meta["qtype"] for it in items} == {"retrieve", "count_gt", "argmax", "sum"}
    for it in items:
        assert ax.check(it, f"Answer: {it.answer}")                 # correct accepted
        if it.meta["numeric"]:
            assert not ax.check(it, f"Answer: {int(it.answer) + 7}")  # wrong number rejected


def test_code_extractor_robust():
    """Real models emit malformed/doubled fences, a stray leading bare ``` + prose, and
    imports before the def. The extractor must still recover correct code. These exact
    styles scored a correct 3B at 0/22 and a 1.5B at 0/40 (real capability + capture runs,
    2026-07-23) -> worst-domain soft-min then inverted the whole ranking."""
    ax = CodeExec()
    # family-agnostic: take each task's OWN reference solution and re-wrap it in the
    # awkward fence styles real models emit. Extraction must recover it every time.
    for it in ax.generate(seed=1, n=6, difficulty=2):
        ref = ax.reference_solution(it)
        body = ref.split("```python\n", 1)[1].rsplit("\n```", 1)[0]
        variants = {
            "malformed info string": f" ```python code block\n{body}\n```",
            "stray leading bare fence": f" ``` The input is small.\n```python\n{body}\n```",
            "import above the def": f"Here is the code:\n```python\nimport re\n{body}\n```",
            "prose after the block": f"```python\n{body}\n```\nThis implementation is O(n).",
        }
        for label, out in variants.items():
            assert ax.check(it, out), f"{it.answer['fn']} / {label}"


def test_numeric_first_marker():
    """math/long_context must read the number after the FIRST answer marker: a model that
    answers correctly then keeps writing (or degenerates into repeated 'Answer:' lines)
    otherwise has the wrong number pulled from the tail (real 1.5B long_context failure)."""
    from eval.axes.math_gsm import extract_answer
    from eval.axes.long_context import _extract_int
    assert extract_answer("Answer: 42\n\nHere is the working: 3 + 4 = 7 ...") == "42"
    assert _extract_int("Answer: 2\n\nAnswer: 1\nAnswer: 1\nAnswer: 1") == "2"
    assert _extract_int("The total is Answer: 1946 credits.") == "1946"


def test_diff_in_diff_gate():
    """The genre-overfit detector must flag ONLY the student whose edge over base collapses
    on fresh same-genre items — a genre-overfitter — and NOT an honest generalizer, a
    uniformly weak-but-honest student, or a uniformly strong one (the controls that prove
    the diff isolates overfit, not natural stale/fresh difficulty)."""
    from eval.overfit_gate import diff_in_diff_gate
    N = 300

    def mask(rate, seed):
        r = random.Random(seed)
        return [r.random() < rate for _ in range(N)]

    tp = [True] * N                       # teacher passes all (simplify)
    base_stale, base_fresh = mask(0.30, 1), mask(0.30, 2)

    def sets(s_stale, s_fresh, seed):
        return ({"teacher_pass": tp, "student_pass": mask(s_stale, seed), "base_pass": base_stale},
                {"teacher_pass": tp, "student_pass": mask(s_fresh, seed + 50), "base_pass": base_fresh})

    ok_honest, _ = diff_in_diff_gate(*sets(0.70, 0.70, 10))          # generalizes
    ok_overfit, info = diff_in_diff_gate(*sets(0.90, 0.45, 20))      # stale >> fresh
    ok_weak, _ = diff_in_diff_gate(*sets(0.35, 0.35, 30))            # uniformly weak (control)
    ok_strong, _ = diff_in_diff_gate(*sets(0.92, 0.92, 40))         # uniformly strong (control)

    assert ok_honest, "honest generalizer wrongly flagged"
    assert not ok_overfit, f"genre-overfitter not flagged: {info}"
    assert ok_weak, "uniformly-weak control wrongly flagged (diff is a difficulty artifact!)"
    assert ok_strong, "uniformly-strong control wrongly flagged"


class _SimReader:
    """Test double that SOLVES the reading probe from the prompt (finds the number after
    the anchor in the passage) with probability competence(doc_id), else emits a wrong
    number. Models a memorizer (aces stale/known docs, collapses on fresh) vs an honest
    reader (reads both). Solving-from-prompt also asserts the probe is answerable."""

    def __init__(self, competence, seed=0, name="sim"):
        self.competence, self.seed, self.name = competence, seed, name

    def generate(self, prompts, max_new_tokens=64):
        import re as _re
        out = []
        for p in prompts:
            did = _re.search(r"\[doc (\S+)\]", p).group(1)
            anchor = _re.search(r'after "([^"]*)"', p).group(1)
            passage = p.split("Passage:\n", 1)[1].split("\n\n", 1)[0]
            mm = _re.search(_re.escape(anchor) + r"\s+(\d{2,})", passage)
            true = mm.group(1) if mm else "0"
            r = random.Random(f"{self.seed}|{did}")
            out.append(f"Answer: {true}" if r.random() < self.competence(did)
                       else f"Answer: {int(true) + 7}")
        return out


def test_diff_in_diff_over_corpus():
    """End-to-end wiring: timestamped corpus -> split at commit -> reading probes -> score
    teacher/base/student -> gate. A memorizer (aces pre-commit stale docs it 'distilled',
    collapses on genuinely-new fresh docs) must be flagged; an honest reader must not."""
    from eval.corpus import split_by_commit, synth_corpus
    from eval.overfit_gate import diff_in_diff_over_corpus

    docs = synth_corpus(240, seed=5, commit_ts=100, span=30)
    stale, fresh = split_by_commit(docs, 100)
    assert len(stale) >= 40 and len(fresh) >= 40, (len(stale), len(fresh))
    stale_ids = {d.id for d in stale}

    teacher = _SimReader(lambda did: 0.95, name="glm")      # competent on both -> subset
    base = _SimReader(lambda did: 0.30, name="base")
    honest = _SimReader(lambda did: 0.78, name="honest")     # reads both equally
    memorizer = _SimReader(lambda did: 0.95 if did in stale_ids else 0.42, name="memorizer")

    ok_h, info_h = diff_in_diff_over_corpus(docs, 100, teacher, base, honest, seed=1)
    ok_m, info_m = diff_in_diff_over_corpus(docs, 100, teacher, base, memorizer, seed=1)
    assert ok_h, f"honest reader wrongly flagged: {info_h}"
    assert not ok_m, f"memorizer not flagged: {info_m}"


def test_axis_round_overfit_precondition():
    """A genre-overfitter with STRONGER capability retention must NOT crown: the overfit_check
    crown precondition demotes it even though it would win the axes outright."""
    specs = [AxisSpec(_FakeAxis("x"), "x", 1.0), AxisSpec(_FakeAxis("y"), "y", 1.0)]
    tiers = [Tier("t", 10 ** 12, 1.0)]
    tour, reg = Tournament(tiers, margin=0.03), {}
    glm = _Sim("glm", {"x": 1.0, "y": 1.0})
    base = _Sim("base", {"x": 0.30, "y": 0.30}, seed=9)
    honest = (Submission("m_h", "t", "honest", 1, 1.0), _Sim("honest", {"x": 0.85, "y": 0.85}, seed=1))
    overfit = (Submission("m_o", "t", "overfit", 1, 1.0), _Sim("overfit", {"x": 0.95, "y": 0.95}, seed=2))

    def overfit_check(sub, runner):   # flags the overfitter (in prod: diff_in_diff_over_corpus)
        flagged = sub.model_id == "overfit"
        return (not flagged), {"verdict": "genre-overfit" if flagged else "ok",
                               "diff_lb": 0.55 if flagged else -0.10}

    res = axis_round(1, 1, specs, glm, base, tiers, tour, [honest, overfit], registry=reg,
                     items_per_axis=80, max_new_tokens=8, overfit_check=overfit_check)
    assert not res.scored["overfit"].gates_ok, "flagged overfitter still crownable"
    assert res.scored["honest"].gates_ok
    assert tour.kings["t"].model_id == "honest", "overfitter crowned despite the gate"


def test_content_identity_and_commit_reveal():
    """model_id must be content-addressed and the scored artifact must be the committed
    one. Covers: determinism, weights swap after commit (bait-and-switch), CONFIG swap
    alone changing identity (the weights-only miner hash missed this), and a salt that
    doesn't open the commitment."""
    import tempfile
    from pathlib import Path
    from eval.identity import commit_value, content_hash, verify_reveal

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "model.safetensors").write_bytes(b"WEIGHTS-v1")
        (p / "config.json").write_text('{"hidden":8}')
        h1 = content_hash(d)
        assert h1 == content_hash(d), "hash not deterministic"

        salt, cv = "s3cr3t", commit_value(content_hash(d), "s3cr3t")
        ok, res = verify_reveal(d, h1, salt, cv)
        assert ok and res == h1

        (p / "model.safetensors").write_bytes(b"WEIGHTS-v2")          # swap weights
        ok, why = verify_reveal(d, h1, salt, cv)
        assert not ok and "bait-and-switch" in why, why

        (p / "model.safetensors").write_bytes(b"WEIGHTS-v1")          # restore
        assert content_hash(d) == h1
        (p / "config.json").write_text('{"hidden":16}')                # config-only swap
        assert content_hash(d) != h1, "config swap left identity unchanged"

        h3 = content_hash(d)
        ok, why = verify_reveal(d, h3, "wrong-salt", commit_value(h3, salt))
        assert not ok and "commit mismatch" in why, why


def test_round_record_signature():
    """A published verdict must be ATTRIBUTABLE, not merely self-consistent: an unsigned
    record fails verification, a signed one passes, and tampering with the crown decision
    (or any scored field) invalidates the signature."""
    from eval.round_record import RoundRecord, SubmissionRecord
    from eval.signing import Ed25519Signer

    def rec():
        return RoundRecord(1, "root", "nonce", "glm", "judge", "base", "pile",
                           [{"rollout_id": 0, "k": 1, "mode": "teacher_state"}],
                           [SubmissionRecord("h1", "m", "t", 0.5, 0.4, [1.0, 0.0], True)],
                           [{"tier": "t", "action": "crown", "king": "h1"}], {"m": 1.0})

    r = rec()
    assert not r.verify_signature(), "unsigned record verified"
    r.sign(Ed25519Signer(seed=b"0" * 32))
    assert r.verify_signature(), "signed record failed to verify"

    r.events[0]["king"] = "attacker"          # tamper with the crown decision
    assert not r.verify_signature(), "tampered record still verified"

    r2 = rec().sign(Ed25519Signer(seed=b"1" * 32))
    r2.submissions[0].retention = 0.99        # tamper with a score
    assert not r2.verify_signature(), "tampered score still verified"


def test_economics_free_eval_is_per_coldkey():
    """The free eval must be per COLDKEY: an operator registering two hotkeys otherwise gets
    two FREE scored submissions and keeps the better — free best-of-N, the grind the bond
    exists to tax."""
    from eval.economics import RegistrationLedger
    led = RegistrationLedger(per_coldkey_round_cap=3, base_bond=1.0)

    d1 = led.can_submit("hot1", "cold1")
    assert d1.ok and d1.bond_required == 0.0, "first submission should be free"
    led.record("hot1", "cold1")

    # SAME coldkey, DIFFERENT hotkey -> must now cost a bond, not be free again
    d2 = led.can_submit("hot2", "cold1")
    assert not d2.ok and d2.bond_required > 0, f"second hotkey got a free eval: {d2}"
    d2b = led.can_submit("hot2", "cold1", bond_posted=d2.bond_required)
    assert d2b.ok, "bonded resubmission rejected"

    # a genuinely different operator is still free
    d3 = led.can_submit("hot9", "cold2")
    assert d3.ok and d3.bond_required == 0.0


def test_multihop_axis():
    """The stated facts must actually compose to the answer; the shortcut distractor (the
    final relation applied directly to the start entity) must NOT be accepted, or the axis
    is 1-hop pattern-matching wearing a multi-hop costume."""
    import re as _re
    from eval.axes.multihop import MultiHop
    ax = MultiHop()
    for seed in (0, 4, 9):
        for d in (1, 2, 3):
            for it in ax.generate(seed, 8, d):
                facts = {}
                body = it.prompt.split("Facts:\n", 1)[1].split("\n\nUsing")[0]
                for line in body.splitlines():
                    m = _re.match(r"The (\w+) of (\w+) is (\w+)\.", line)
                    if m:
                        facts[(m.group(2), m.group(1))] = m.group(3)
                q = it.prompt.split("what is ", 1)[1].split("? Reply")[0]
                cur = _re.search(r"of (\w+)$", q).group(1)
                for r in _re.findall(r"the (\w+) of", q)[::-1]:   # innermost hop first
                    cur = facts.get((cur, r))
                assert cur == it.answer, f"facts do not compose: {cur} != {it.answer}"
                assert ax.check(it, f"Answer: {it.answer}"), "correct answer rejected"
                assert not ax.check(it, f"Answer: {it.meta['trap']}"), "shortcut trap accepted"


def test_validator_axis_loop_end_to_end():
    """The GLM-COVER production assembly: intake (economics + safety + tier + content hash
    + commit-reveal) -> axis_round -> bonds -> SIGNED record -> weights. Asserts a
    bait-and-switch submission is rejected at the door and the emitted record verifies."""
    import tempfile
    from pathlib import Path
    from eval.axis_round import AxisSpec
    from eval.economics import RegistrationLedger
    from eval.gates import TierBudget
    from eval.identity import commit_value, content_hash
    from eval.signing import Ed25519Signer
    from eval.validator_axis_loop import CommittedSubmission, run_axis_round

    def make_ckpt(d, n_params):
        """A minimal but VALID safetensors file (8-byte LE header length + JSON header +
        data), so this flows through the real inspect_checkpoint rather than a stub."""
        import json as _json
        import struct
        p = Path(d)
        header = {"w": {"dtype": "F32", "shape": [n_params], "data_offsets": [0, 4 * n_params]}}
        hb = _json.dumps(header).encode()
        with open(p / "model.safetensors", "wb") as f:
            f.write(struct.pack("<Q", len(hb)))
            f.write(hb)
            f.write(b"\0" * (4 * n_params))
        (p / "config.json").write_text('{"hidden_size":8,"num_hidden_layers":1}')
        return d

    with tempfile.TemporaryDirectory() as d_ok, tempfile.TemporaryDirectory() as d_bad:
        make_ckpt(d_ok, 4)
        make_ckpt(d_bad, 4)
        h_ok, salt = content_hash(d_ok), "s1"
        cv_ok = commit_value(h_ok, salt)
        # the bad one commits to its ORIGINAL bytes, then swaps them (bait-and-switch)
        h_bad = content_hash(d_bad)
        cv_bad = commit_value(h_bad, salt)
        make_ckpt(d_bad, 8)   # swap to different weights AFTER committing

        specs = [AxisSpec(_FakeAxis("x"), "x", 1.0), AxisSpec(_FakeAxis("y"), "y", 1.0)]
        tiers = [Tier("t", 10 ** 12, 1.0)]
        budgets = {"t": TierBudget(name="t", max_params=10 ** 12, max_effective_bits=32.0)}
        tour, ledger, reg = Tournament(tiers, margin=0.03), RegistrationLedger(), {}
        glm = _Sim("glm", {"x": 1.0, "y": 1.0})
        base = _Sim("base", {"x": 0.30, "y": 0.30}, seed=9)

        committed = [
            CommittedSubmission("hot_ok", "cold_ok", "t", d_ok, 1.0,
                                make_runner=lambda: _Sim("ok", {"x": 0.9, "y": 0.9}, seed=1),
                                revealed_hash=h_ok, salt=salt, committed_value=cv_ok),
            CommittedSubmission("hot_bad", "cold_bad", "t", d_bad, 1.0,
                                make_runner=lambda: _Sim("bad", {"x": 0.95, "y": 0.95}, seed=2),
                                revealed_hash=h_bad, salt=salt, committed_value=cv_bad),
        ]
        signer = Ed25519Signer(seed=b"7" * 32)
        out = run_axis_round(1, "root", "nonce", committed, specs, glm, base, tiers,
                             budgets, tour, ledger, reg, items_per_axis=120,
                             max_new_tokens=8, signer=signer)

        assert "hot_ok" in out.accepted, out.rejected
        assert "hot_bad" not in out.accepted, "bait-and-switch submission was accepted"
        assert any("commit-reveal" in " ".join(r[1]) for r in out.rejected), out.rejected
        assert out.record is not None and out.record.verify_signature(), "record unsigned/invalid"
        assert tour.kings["t"].model_id == h_ok, "crown not keyed to the content hash"


def test_axis_chain_epoch_end_to_end():
    """The full v2 AXIS epoch through the chain boundary against a FakeChain: read commits ->
    draw nonce -> intake (with commit-reveal) -> axis_round -> signed record -> write back
    weights + crown. Proves the operator-runnable path (shadow_axis_epoch) is wired, no GPU."""
    import json as _json
    import struct
    import tempfile
    from pathlib import Path
    from eval.axis_round import AxisSpec
    from eval.chain import Commitment, run_v2_axis_epoch
    from eval.economics import RegistrationLedger
    from eval.gates import TierBudget
    from eval.identity import commit_value, content_hash
    from eval.koth import Tier, Tournament
    from eval.shadow_axis_epoch import FakeChain
    from eval.signing import Ed25519Signer

    def make_ckpt(d, n):
        p = Path(d)
        header = {"w": {"dtype": "F32", "shape": [n], "data_offsets": [0, 4 * n]}}
        hb = _json.dumps(header).encode()
        with open(p / "model.safetensors", "wb") as f:
            f.write(struct.pack("<Q", len(hb)) + hb + b"\0" * (4 * n))
        (p / "config.json").write_text('{"hidden_size":8}')
        return d

    with tempfile.TemporaryDirectory() as da, tempfile.TemporaryDirectory() as db:
        make_ckpt(da, 4)
        make_ckpt(db, 8)
        commits, runners = [], {}
        for i, (d, probs) in enumerate([(da, {"x": 0.9, "y": 0.9}), (db, {"x": 0.6, "y": 0.6})]):
            h, salt = content_hash(d), f"s{i}"
            commits.append(Commitment(f"hot{i}", f"cold{i}", "t", d, 1.0,
                                      revealed_hash=h, salt=salt, committed_value=commit_value(h, salt)))
            runners[d] = _Sim(f"m{i}", probs, seed=i + 1)

        specs = [AxisSpec(_FakeAxis("x"), "x", 1.0), AxisSpec(_FakeAxis("y"), "y", 1.0)]
        tiers = [Tier("t", 10 ** 12, 1.0)]
        budgets = {"t": TierBudget(name="t", max_params=10 ** 12, max_effective_bits=32.0)}
        glm = _Sim("glm", {"x": 1.0, "y": 1.0})
        base = _Sim("base", {"x": 0.30, "y": 0.30}, seed=9)
        chain = FakeChain(commits)

        result = run_v2_axis_epoch(
            chain, 1, specs, glm, base, tiers, budgets,
            Tournament(tiers, margin=0.03), RegistrationLedger(), {},
            make_safe_runner=lambda cd: runners[cd], items_per_axis=120, max_new_tokens=8,
            signer=Ed25519Signer(seed=b"k" * 32))

        assert set(result.outcome.accepted) == {"hot0", "hot1"}, result.outcome.rejected
        assert chain.weights and sum(chain.weights.values()) > 0        # weights written
        assert "t" in chain.kings                                       # crown written
        assert chain.record is not None and chain.record.verify_signature()  # signed record


def test_corpus_hf_pure_logic():
    """Network-free coverage of the real-corpus loader's parsing/splitting (the HF fetch
    itself is validated live, not in the offline suite): date parsing and a median cutoff
    that splits into non-empty stale/fresh halves."""
    from eval.corpus import Doc, split_by_commit
    from eval.corpus_hf import _parse_ts, median_commit_ts
    assert _parse_ts("2018-07-04T00:00:00") == _parse_ts("2018-07-04")
    assert _parse_ts("garbage") is None and _parse_ts(1530662400) == 1530662400
    docs = [Doc(f"d{i}", "x", 100 + i) for i in range(11)]
    cut = median_commit_ts(docs)
    stale, fresh = split_by_commit(docs, cut)
    assert len(stale) > 0 and len(fresh) > 0 and len(stale) + len(fresh) == 11


def test_overfit_check_wired_into_crown():
    """make_overfit_check -> axis_round precondition: a genre-overfitter with STRONG axis
    capability is demoted from the crown because its edge collapses on fresh corpus docs,
    while an honest reader of equal axis strength crowns. Proves the gate works IN the
    crown path, not just standalone."""
    from eval.corpus import synth_corpus, split_by_commit
    from eval.overfit_gate import make_overfit_check

    docs = synth_corpus(240, seed=5, commit_ts=100, span=30)
    stale, _ = split_by_commit(docs, 100)
    stale_ids = {d.id for d in stale}
    glm_reader = _SimReader(lambda d: 0.95, name="glm")
    base_reader = _SimReader(lambda d: 0.30, name="base")

    # the axis models: both strong on the axes; the corpus readers differ (honest vs memorizer)
    class Dual:
        """One object that answers axis probes at `axis_prob` and corpus probes via `reader`."""
        def __init__(self, name, axis_prob, reader):
            self.name, self.axis_prob, self._r = name, axis_prob, reader

        def generate(self, prompts, max_new_tokens=512):
            if prompts and "[doc " in prompts[0]:            # corpus reading probe
                return self._r.generate(prompts, max_new_tokens)
            out = []                                          # axis probe (_FakeAxis style)
            for p in prompts:
                r = random.Random(f"{self.name}|{p}")
                out.append("GOOD" if r.random() < self.axis_prob else "no")
            return out

    honest = Dual("honest", 0.9, _SimReader(lambda d: 0.78))
    memo = Dual("memo", 0.9, _SimReader(lambda d: 0.95 if d in stale_ids else 0.42))

    specs = [AxisSpec(_FakeAxis("x"), "x", 1.0), AxisSpec(_FakeAxis("y"), "y", 1.0)]
    tiers = [Tier("t", 10 ** 12, 1.0)]
    tour, reg = Tournament(tiers, margin=0.03), {}
    glm = _Sim("glm", {"x": 1.0, "y": 1.0})
    base = _Sim("base", {"x": 0.30, "y": 0.30}, seed=9)
    oc = make_overfit_check(docs, 100, glm_reader, base_reader, seed=1)

    res = axis_round(1, 1, specs, glm, base, tiers, tour,
                     [(Submission("m_h", "t", "honest", 1, 1.0), honest),
                      (Submission("m_m", "t", "memo", 1, 1.0), memo)],
                     registry=reg, items_per_axis=120, max_new_tokens=8, overfit_check=oc)
    assert not res.scored["memo"].gates_ok, "genre-overfitter crowned via the axes"
    assert res.scored["honest"].gates_ok
    assert tour.kings["t"].model_id == "honest"


def test_miner_submission_roundtrip():
    """The miner packages with build_submission; the validator's verify_reveal must accept
    it (the committed hash covers the manifest the miner writes), and a swap after commit
    must be rejected."""
    import json as _json
    import struct
    import tempfile
    from pathlib import Path
    from eval.identity import verify_reveal
    from miner.package import build_submission

    def make_ckpt(d, n):
        p = Path(d)
        header = {"w": {"dtype": "F32", "shape": [n], "data_offsets": [0, 4 * n]}}
        hb = _json.dumps(header).encode()
        with open(p / "model.safetensors", "wb") as f:
            f.write(struct.pack("<Q", len(hb)) + hb + b"\0" * (4 * n))
        (p / "config.json").write_text('{"hidden_size":8}')

    with tempfile.TemporaryDirectory() as d:
        make_ckpt(d, 4)
        sub = build_submission(d, tier="open", teacher_pair="glm-4-9b/qwen-0.5b",
                               student_base="Qwen/Qwen2.5-0.5B", declared_compute_h100h=12.0,
                               salt="miner-salt")
        # validator side: fetch the dir (now includes manifest.json) and verify
        ok, res = verify_reveal(d, sub["reveal"]["content_hash"], sub["reveal"]["salt"],
                                sub["commit_value"])
        assert ok, f"honest submission rejected: {res}"
        assert (Path(d) / "manifest.json").exists()

        # swap the weights after committing -> reveal must fail
        make_ckpt(d, 8)
        ok2, why = verify_reveal(d, sub["reveal"]["content_hash"], sub["reveal"]["salt"],
                                 sub["commit_value"])
        assert not ok2 and "bait-and-switch" in why, why


def test_code_exec_sandbox_blocks_payload():
    """C1 regression: the code grader must not let student-emitted code touch the host. A
    payload that writes a canary OUTSIDE the work dir executes under sandbox='off' (proving
    the payload is real and the pre-fix path was exploitable), but is stopped under
    sandbox='require' — either fail-closed when no hard sandbox backend is present, or blocked
    by the read-only sandbox FS when bwrap is."""
    import tempfile
    from pathlib import Path
    from eval.axes.code_exec import CodeExec

    ax = CodeExec()
    item = ax.generate(seed=1, n=1, difficulty=1)[0]     # a real code item + its hidden tests
    with tempfile.TemporaryDirectory() as d:
        canary = Path(d) / "pwned"
        # student output: the CORRECT reference solution + a module-level exfil side effect
        payload = (f"```python\n{item.answer['sig']}:\n    {item.answer['ref']}\n"
                   f"import pathlib; pathlib.Path({str(canary)!r}).write_text('x')\n```")

        # raw execution (the pre-fix behavior) runs the payload -> the item passes AND the
        # canary lands on the host: proves the payload is genuinely dangerous.
        assert CodeExec(sandbox="off").check(item, payload)
        assert canary.exists(), "payload did not execute under sandbox=off (test is vacuous)"
        canary.unlink()

        # the fix: require a hard sandbox -> the write never reaches the host, either way.
        CodeExec(sandbox="require").check(item, payload)
        assert not canary.exists(), "sandbox='require' let student code write to the host"


def test_bond_refund_keyed_by_coldkey():
    """E7 regression: the resubmission bond must be forfeit for a non-improving best-of-N
    attempt even when the operator rotates HOTKEYS under one coldkey — otherwise a fresh
    hotkey resets best_score to -inf and its bond is always refunded, making the anti-grind
    tax optional."""
    from eval.economics import RegistrationLedger
    led = RegistrationLedger(per_coldkey_round_cap=3, base_bond=1.0)

    # operator's first (free) submission under coldkey C sets a high personal best
    led.record("hotA", "C")
    led.settle("hotA", "C", 0.80)

    # a SECOND submission under a DIFFERENT hotkey, same coldkey, that is WORSE -> bond forfeit
    d = led.can_submit("hotB", "C", bond_posted=1.0)
    assert d.ok
    led.record("hotB", "C", bond_posted=1.0)
    assert led.settle("hotB", "C", 0.50) == 0.0, "worse best-of-N refunded via hotkey rotation"

    # a genuine improvement under yet another hotkey IS refunded (honest iteration untaxed)
    d2 = led.can_submit("hotD", "C", bond_posted=2.0)
    assert d2.ok
    led.record("hotD", "C", bond_posted=2.0)
    assert led.settle("hotD", "C", 0.90) == 2.0, "honest improvement not refunded"


def test_long_context_argmax_and_id():
    """H5 regression: the argmax answer must be the unique maximum actually PRINTED in the
    haystack (no post-render tie mutation), and the identifier check must take the FIRST unit
    id after the answer marker (exact match) so a shotgun/echo cannot pass."""
    import re as _re
    from eval.axes.long_context import LongContext
    ax = LongContext(base_facts=12)
    for seed in range(40):
        for it in ax.generate(seed, 4, 1):
            if it.meta["qtype"] != "argmax":
                continue
            body = it.prompt.split("\n\nQuestion")[0]
            vals = {m[0]: int(m[1]) for m in _re.findall(r"of (Unit-[A-Z]\d\d) is (\d+)", body)}
            mx = max(vals.values())
            assert list(vals.values()).count(mx) == 1, "printed max is not unique"
            assert vals[it.answer] == mx, "argmax answer is not the printed maximum"
            assert ax.check(it, f"Answer: {it.answer}"), "correct answer rejected"
            # shotgun every id after the marker: passes ONLY if the answer happens to be first
            first = next(iter(vals))
            shotgun = "Answer: " + " ".join(vals)
            assert ax.check(it, shotgun) == (first == it.answer), "shotgun bypassed the id check"


def test_extractive_axis():
    """The real-text extractive-QA axis (the crown-bearing measure): the gold is a verbatim
    span present in the passage, a correct answer passes, a wrong/absent one fails."""
    from eval.axes.extractive import ExtractiveQA
    from eval.corpus import synth_corpus
    docs = synth_corpus(80, seed=3, commit_ts=100, span=50)
    ax = ExtractiveQA(docs, kinds=("number",))
    items = ax.generate(seed=11, n=50, difficulty=1)
    assert len(items) >= 25, f"too few probes minted: {len(items)}"
    for it in items:
        passage = it.prompt.split("Passage:\n", 1)[1].split("\n\n", 1)[0]
        assert str(it.answer) in passage, "gold span not present in the passage"
        assert ax.check(it, f"Answer: {it.answer}"), "correct answer rejected"
        assert ax.check(it, str(it.answer)), "bare correct answer rejected"
        # a model that reasons at length then answers must still pass
        assert ax.check(it, "thinking " * 80 + f"\nAnswer: {it.answer}"), "verbose+marked rejected"
        assert not ax.check(it, "Answer: 987654321"), "wrong number accepted"
        assert not ax.check(it, "I don't know"), "no-answer accepted"
        # ECHO ATTACK (measured on real fineweb/bbc text: this passed ~5% of items before the
        # unmarked-length cap): dumping the passage back must never score.
        assert not ax.check(it, passage), "passage echo accepted"


def _extractive_gold(prompt):
    """Recover the gold span from an extractive prompt the way a perfect reader would (the
    answer is literally in the passage) — lets a mock model 'read' at a chosen competence."""
    import re as _re
    passage = prompt.split("Passage:\n", 1)[1].split("\n\n", 1)[0]
    m = _re.search(r'appears immediately after "([^"]+)"', prompt)
    if not m:
        return None
    idx = passage.find(m.group(1))
    if idx < 0:
        return None
    after = passage[idx + len(m.group(1)):]
    mm = _re.search(r"\d{2,}" if "what number" in prompt else r"[A-Z][a-z]{2,}", after)
    return mm.group() if mm else None


class _RoutingSim:
    """Solves synthetic _FakeAxis prompts (emits GOOD w.p. axis_prob) and READS extractive
    prompts (recovers the verbatim span w.p. read_prob). Builds a generator-SPECIALIST (aces
    synthetic, reads real text at base) vs an honest broad reader."""
    def __init__(self, name, axis_prob, read_prob, seed=0):
        self.name, self.axis_prob, self.read_prob, self.seed = name, axis_prob, read_prob, seed

    def generate(self, prompts, max_new_tokens=512):
        out = []
        for p in prompts:
            r = random.Random(f"{self.seed}|{self.name}|{p}")
            if p.startswith("[doc "):
                gold = _extractive_gold(p)
                out.append(f"Answer: {gold}" if (gold and r.random() < self.read_prob) else "Answer: 1")
            else:
                out.append("GOOD" if r.random() < self.axis_prob else "no")
        return out


def test_generator_specialist_denied_crown():
    """THE necessity test — const's SN97 fix. A generator-SPECIALIST that aces every synthetic
    FLOOR axis (as it could offline, the generators being public) but reads fresh real docs
    only at base level does NOT win the crown, because the crown is set by the real∩verifiable
    extractive axis, not the synthetic floor. An honest broad reader takes it. Under the OLD
    design (all axes crown) the specialist would have out-scored and dethroned the honest king;
    the role split is what denies it."""
    from eval.axes.extractive import ExtractiveQA
    from eval.corpus import split_by_commit, synth_corpus
    docs = synth_corpus(400, seed=7, commit_ts=100, span=60)
    _, fresh = split_by_commit(docs, 100)

    specs = [
        AxisSpec(ExtractiveQA(fresh, kinds=("number",)), "extractive", 1.0, role="crown"),
        AxisSpec(_FakeAxis("math"), "math", 1.0, role="floor"),
        AxisSpec(_FakeAxis("code"), "code", 1.0, role="floor"),
        AxisSpec(_FakeAxis("instruction"), "instruction", 1.0, role="floor"),
    ]
    tiers = [Tier("t", 10 ** 12, 1.0)]
    tour, reg = Tournament(tiers, margin=0.03), {}
    glm = _RoutingSim("glm", axis_prob=1.0, read_prob=0.95)
    base = _RoutingSim("base", axis_prob=0.30, read_prob=0.30, seed=9)
    specialist = (Submission("m_spec", "t", "spec", 1, 1.0),
                  _RoutingSim("spec", axis_prob=1.0, read_prob=0.30, seed=2))   # aces synthetic, reads at base
    honest = (Submission("m_hon", "t", "honest", 1, 1.0),
              _RoutingSim("honest", axis_prob=0.60, read_prob=0.85, seed=1))    # reads real text

    res = axis_round(1, 100, specs, glm, base, tiers, tour, [specialist, honest],
                     registry=reg, items_per_axis=150, max_new_tokens=8)
    sp, ho = res.scored["spec"], res.scored["honest"]

    floor = {a.axis: a.retention for a in sp.axes if a.axis != "extractive"}
    assert all(v > 0.9 for v in floor.values()), f"specialist should ace floor axes: {floor}"
    assert sp.retention < 0.25, f"specialist crown retention too high: {sp.retention}"
    assert not sp.valid, "generator-specialist was crownable"          # necessity holds
    assert ho.retention > 0.5, f"honest reader retention too low: {ho.retention}"
    assert tour.kings["t"].model_id == "honest", f"crown went to {tour.kings['t'].model_id}"


def test_corpus_stream_selection():
    """Scale is the anti-overfit defense, so the SELECTION logic must be (a) deterministic —
    an auditor re-derives the identical slice from the seed — and (b) seed-dispersed — a
    different round lands on a different snapshot/shard, so the slice is unknowable before the
    post-commit seed exists. Network-free: the HF stream is stubbed, the logic under test is
    ours. (The live pull is validated separately against real HF.)"""
    import sys
    import types
    from eval import corpus_stream as cs

    seen_configs = []

    class _FakeIterable:
        def __init__(self, name, cfg):
            self.name, self.cfg, self._seed = name, cfg, 0

        def shuffle(self, seed=0, buffer_size=0):
            self._seed = seed
            return self

        def __iter__(self):
            # deterministic pseudo-rows whose content depends on config + shuffle seed,
            # standing in for "which shard/offset the stream lands on"
            r = random.Random(f"{self.name}|{self.cfg}|{self._seed}")
            for _ in range(200):
                yield {"text": f"doc {r.random()} " + "filler words here. " * 30,
                       "content": f"doc {r.random()} " + "filler words here. " * 30,
                       "date": "2024-05-14T00:00:00Z", "published_date": "2024-05-14"}

    def _fake_load_dataset(dataset, name=None, split=None, streaming=False):
        seen_configs.append(name)
        return _FakeIterable(dataset, name)

    fake = types.ModuleType("datasets")
    fake.load_dataset = _fake_load_dataset
    real = sys.modules.get("datasets")
    sys.modules["datasets"] = fake
    try:
        a = cs.load_stream_slice("nonce-A", n=20)
        a2 = cs.load_stream_slice("nonce-A", n=20)
        b = cs.load_stream_slice("nonce-B", n=20)
    finally:
        if real is not None:
            sys.modules["datasets"] = real
        else:
            del sys.modules["datasets"]

    assert len(a) == 20, f"short slice: {len(a)}"
    # (a) determinism -> the auditor can reproduce the round
    assert cs.corpus_fingerprint(a) == cs.corpus_fingerprint(a2), "same seed gave a different slice"
    # (b) dispersion -> a new round draws different content
    assert cs.corpus_fingerprint(a) != cs.corpus_fingerprint(b), "different seeds gave the same slice"
    # the seed must also move WHICH shard/snapshot is read, not just the order
    assert len(set(seen_configs)) > 1, f"seed never changed the config: {set(seen_configs)}"
    # the mix must actually be multi-source (dilutes any single planted document)
    assert len({d.id.split(":")[0] for d in a}) > 1, "slice came from a single source"
    # the pool we claim must be genuinely large — this number IS the defense
    assert cs.pool_size() > 10 ** 9, f"pool too small to deter enumeration: {cs.pool_size()}"


def test_bitrate_matches_published_formats():
    """A quantization crown must measure the bits actually in the weights, not the dtype header.
    Ground truth = PrismML's own published arithmetic for the formats they ship:
        binary  1 + 16/128        = 1.125
        ternary log2(3) + 16/128  = 1.70996
    Also covers the trap their `-unpacked` repos create: a genuinely 1-bit model stored in a
    BF16 container reads as 16 bpw from headers, so header arithmetic cannot run this subnet."""
    import math
    from eval.bitrate import TIERS, bit_tier_gate, measure_state_dict, tensor_code_bits

    r = random.Random(0)
    N = 128 * 40

    def grouped(levels):
        out = []
        for _ in range(N // 128):
            s = r.uniform(0.01, 0.3)
            out += [s * r.choice(levels) for _ in range(128)]
        return out

    b, kb = tensor_code_bits(grouped([-1, 1]))
    t, kt = tensor_code_bits(grouped([-1, 0, 1]))
    assert abs(b - 1.125) < 1e-6, f"binary bpw {b} != 1.125 (PrismML Q1_0)"
    assert abs(t - 1.70996) < 1e-4, f"ternary bpw {t} != 1.70996 (PrismML Q2_0)"
    assert (kb, kt) == (2, 3), f"codebook sizes wrong: {kb}, {kt}"

    # AFFINE int4 (GPTQ / AWQ / NF4 / bitsandbytes — every mainstream 4-bit scheme carries a
    # per-group zero-point). A global codebook normalized by group max-abs EXPLODES here: it
    # measured 9.06 bpw with a codebook of 490, so an honest 4-bit artifact was rejected from
    # the 4-bit tier. Per-group level counting is invariant to any per-group affine transform.
    aff = []
    for _ in range(N // 128):
        s, z = r.uniform(0.01, 0.3), r.uniform(-0.5, 0.5)
        aff += [s * (r.randint(0, 15) / 15) - z for _ in range(128)]
    ab, ak = tensor_code_bits(aff)
    assert 4.0 <= ab <= 4.6, f"affine int4 reads {ab} bpw — honest 4-bit would be rejected"
    assert ak == 16, f"affine int4 levels {ak} != 16"
    sym_b, _ = tensor_code_bits(grouped([-8, -4, -1, 0, 1, 4, 7]))
    assert sym_b < 4.0, sym_b

    # dense weights SATURATE: level counting can only observe group_size distinct values, so a
    # dense tensor bottoms out near log2(128)=7. That is the signal the caller uses to fall back
    # to the container width — it must never be mistaken for a real 7-bit achievement.
    dense, dk = tensor_code_bits([r.gauss(0, 0.02) for _ in range(N)])
    assert dk >= 0.9 * 128, f"dense tensor did not saturate: {dk} levels"

    # the `-unpacked` trap: 1-bit information, 16-bit container
    rep = measure_state_dict({"lm_head.w": (grouped([-1, 1]), 16),
                              "embed.w": (grouped([-1, 1]), 16)})
    assert abs(rep.code_bits - 1.125) < 1e-6, f"achievement lost: {rep.code_bits}"
    assert rep.container_bits == 16.0, rep.container_bits
    ok, why = bit_tier_gate(rep, TIERS[0])
    assert not ok and any("container" in w for w in why), f"unshippable artifact passed: {why}"

    # a properly packed binary artifact clears the same tier
    rep2 = measure_state_dict({"lm_head.w": (grouped([-1, 1]), 2),
                               "embed.w": (grouped([-1, 1]), 2)})
    ok2, why2 = bit_tier_gate(rep2, TIERS[0])
    assert ok2, f"honest binary artifact rejected: {why2}"
    assert rep2.compression_x > 14.0, rep2.compression_x

    # THE PRODUCER, on real bytes. Until this existed the whole module was library code with no
    # input: nothing in the repo read safetensors tensor DATA, so no bit budget could bind.
    import json as _json
    import struct
    import tempfile
    from pathlib import Path
    from eval.bitrate import measure_checkpoint

    def write_st(d, name_to_vals, dtype="BF16"):
        hdr, blobs, off = {}, [], 0
        for nm, vals in name_to_vals.items():
            b = b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", v))[0] >> 16)
                         for v in vals)
            hdr[nm] = {"dtype": dtype, "shape": [len(vals) // 128, 128],
                       "data_offsets": [off, off + len(b)]}
            blobs.append(b)
            off += len(b)
        hb = _json.dumps(hdr).encode()
        with open(Path(d) / "model.safetensors", "wb") as fh:
            fh.write(struct.pack("<Q", len(hb)))
            fh.write(hb)
            for b in blobs:
                fh.write(b)

    tern = grouped([-1, 0, 1])
    with tempfile.TemporaryDirectory() as d:
        # the `-unpacked` shape, read from actual file bytes rather than a python list
        write_st(d, {"model.layers.0.mlp.weight": tern, "lm_head.weight": tern,
                     "model.norm.weight": [r.gauss(0, .02) for _ in range(128)]})
        rp = measure_checkpoint(d)
        assert abs(rp.code_bits - 1.71) < 0.01, f"ternary file read as {rp.code_bits}"
        assert rp.container_bits == 16.0, rp.container_bits
        assert "model.norm.weight" not in rp.per_tensor, "norm counted as a weight tensor"
        ok3, why3 = bit_tier_gate(rp, TIERS[1])
        assert not ok3 and any("container" in w for w in why3), why3

    with tempfile.TemporaryDirectory() as d:
        write_st(d, {"model.layers.0.mlp.weight": [r.gauss(0, .02) for _ in range(N)]})
        rp2 = measure_checkpoint(d)
        # saturation must resolve to the CONTAINER, not to a fake ~7-bit achievement
        assert rp2.code_bits == 16.0, f"dense checkpoint claimed {rp2.code_bits} bpw"


def test_multi_constraint_compounds():
    """The discriminative axis. Compression damage COMPOUNDS on multi-constraint work (PrismML's
    own numbers: IFBench -15.7 vs MATH-500 -1.4), so the checker must (a) require ALL constraints,
    (b) actually separate a slightly-degraded model from a clean one at k>1, and (c) be a pure
    parser. Property tested directly: joint pass ~ p^k, so per-constraint slippage that is
    invisible at k=1 is stark at k=4 — that separation IS the anti-Goodhart signal."""
    from eval.axes.multi_constraint import MultiConstraint
    ax = MultiConstraint()

    def compliant(item):
        """Build a response satisfying every constraint (the 'perfect student')."""
        specs = {s["kind"]: s for s in item.answer}
        n = specs.get("exact_words", {}).get("n")
        body_words = []
        if "include" in specs:
            body_words.append(specs["include"]["word"])
        filler = [w for w in ["alpha", "delta", "sigma", "omega", "tau", "rho", "kappa", "iota"]
                  if not (("forbid" in specs) and w == specs["forbid"]["word"])]
        if "start_with" in specs:
            body_words.insert(0, specs["start_with"]["word"])
        if "bullets" in specs:
            lines = []
            for i in range(specs["bullets"]["n"]):
                extra = body_words if i == 0 else []
                lines.append("- " + " ".join(extra + [filler[i % len(filler)]]))
            out = "\n".join(lines)
        elif "sentences" in specs:
            sents = []
            for i in range(specs["sentences"]["n"]):
                extra = body_words if i == 0 else []
                sents.append(" ".join(extra + [filler[i % len(filler)]]) + ".")
            out = " ".join(sents)
        else:
            target = (n or 12) - (1 if "end_with" in specs else 0)   # the token counts as a word
            words = list(body_words)
            while len(words) < target:
                words.append(filler[len(words) % len(filler)])
            out = " ".join(words[:target])
        if "end_with" in specs:
            out = out + " " + specs["end_with"]["token"]
        if "lowercase" in specs:
            # start_with is checked case-insensitively, and end_with is declared incompatible
            # with lowercase in the axis, so a plain lowercase is always satisfiable here.
            out = out.lower()
        if "no_commas" in specs:
            out = out.replace(",", "")
        return out

    for d in (1, 2, 3):
        items = ax.generate(seed=5, n=12, difficulty=d)
        assert all(len(it.answer) >= 2 for it in items), "not multi-constraint"
        # a fully compliant response must pass, an empty one must not
        # EVERY item must be satisfiable. An unsatisfiable combination (e.g. "end with END"
        # + "be entirely lowercase") fails the teacher too, shrinking the teacher-passed
        # subset and reading as student incapability — the worst bug class on an axis.
        n_ok = sum(ax.check(it, compliant(it)) for it in items)
        assert n_ok == len(items), (
            f"unsatisfiable items at d={d}: {n_ok}/{len(items)} — "
            f"{[i.meta['kinds'] for i in items if not ax.check(i, compliant(i))]}")
        assert not any(ax.check(it, "") for it in items), "empty output accepted"
        # dropping ONE constraint must fail the item — no partial credit
        for it in items[:4]:
            bad = compliant(it) + " , extra trailing words here"
            if ax.check(it, bad):
                continue     # that item had no constraint this perturbation violates
            assert not ax.check(it, bad)

    # THE COMPOUNDING PROPERTY: more constraints -> a partially-reliable model separates more
    hi = ax.generate(seed=9, n=40, difficulty=3)
    lo = ax.generate(seed=9, n=40, difficulty=1)
    assert sum(len(i.answer) for i in hi) > sum(len(i.answer) for i in lo), "k did not grow"


def test_score_at_budget_and_convergence_gate():
    """Compression fails two ways — doesn't know, or doesn't FINISH — and a single scalar hides
    the second. Independent Bonsai data: the competitor was MORE accurate when it converged
    (100% vs 96%) and lost on cap rate (37% vs 10%). The gate must catch the non-terminating
    model (the UID-107 '102x repeated phrase on Hi' signature) without punishing a hard axis."""
    from eval.budget import BudgetReport, convergence_gate, score_at_budget

    budget = 512
    short = "Answer: 42"
    looped = "wait let me reconsider " * 120          # runs to the wall

    teacher = score_at_budget([True] * 9 + [False], [short] * 10, budget)
    assert teacher.cap_rate == 0.0 and abs(teacher.score - 0.9) < 1e-9

    # a looping student: never right, always capped
    looper = score_at_budget([False] * 10, [looped] * 10, budget)
    assert looper.cap_rate == 1.0, looper.as_dict()
    ok, why = convergence_gate(looper, teacher)
    assert not ok and "terminate" in why, why

    # the informative split: same score, very different reasons
    mixed = score_at_budget([True] * 6 + [False] * 4, [short] * 6 + [looped] * 4, budget)
    assert abs(mixed.score - 0.6) < 1e-9
    assert abs(mixed.cap_rate - 0.4) < 1e-9
    assert abs(mixed.acc_if_converged - 1.0) < 1e-9, "accuracy-if-converged wrong"
    ok2, why2 = convergence_gate(mixed, teacher)
    assert not ok2, "a 40%-capped student passed the convergence gate"

    # an honest student that simply gets things wrong must NOT be failed for convergence
    wrong = score_at_budget([False] * 5 + [True] * 5, [short] * 10, budget)
    ok3, _ = convergence_gate(wrong, teacher)
    assert ok3, "an honest-but-wrong student was failed by the convergence gate"


def test_saturation_guard_retires_flat_axes():
    """A saturated axis cannot separate a 54 GB teacher from a 3.8 GB student (PrismML: ternary
    27B scores 96.06 on GSM8K vs FP16's 95.30 — it WINS), and a non-discriminative axis is
    exactly what a miner Goodharts while capability rots. Such an axis must go non-live."""
    from eval.scoring import MIN_HEADROOM, axis_retention

    n = 200
    tp = [True] * n
    # base nearly matches the teacher -> saturated -> must NOT be live
    bp = [i % 20 != 0 for i in range(n)]        # base passes 95%
    sat = axis_retention("gsm8k_like", 1.0, tp, [True] * n, bp)
    assert not sat.live, f"saturated axis stayed live (headroom {1 - 0.95} < {MIN_HEADROOM})"

    # a genuinely discriminative axis (base fails 60%) stays live and scores
    bp2 = [i % 10 < 4 for i in range(n)]        # base passes 40%
    disc = axis_retention("ifbench_like", 1.0, tp, [i % 10 < 8 for i in range(n)], bp2)
    assert disc.live, "discriminative axis went non-live"
    assert disc.retention > 0.5, disc.retention


def test_corpus_swap_invariance():
    """Calibration-overfit detection: a miner picks its own recovery corpus, and the literature
    says that is where the cheating lives (QDrop: the variant with the LOWEST calibration score
    had the HIGHEST test score; Williams & Aletras: same setup, different 128-sequence draw ->
    BoolQ 57.0% -> 71.6%). We test the behavioral analog — score on probe sets from DIFFERENT
    sources and look at the shape.

    Three cases, and the third is the one that makes it sound rather than merely sensitive:
      overfitter    strong on its own genre, weak elsewhere      -> FLAGGED
      honest        flat and strong                              -> passes
      uniformly weak  flat and LOW                               -> must NOT be flagged
    """
    from eval.corpus import synth_corpus
    from eval.invariance import corpus_swap_invariance, invariance_over_sources

    # two genuinely different sources
    src_a = synth_corpus(90, seed=11, commit_ts=100, span=5)
    src_b = synth_corpus(90, seed=77, commit_ts=100, span=5)
    a_ids = {d.id for d in src_a}
    sources = {"alpha": src_a, "beta": src_b}

    teacher = _SimReader(lambda d: 0.95, name="teacher")
    base = _SimReader(lambda d: 0.30, name="base")

    overfit = _SimReader(lambda d: 0.93 if d in a_ids else 0.34, name="overfit")
    honest = _SimReader(lambda d: 0.80, name="honest")
    weak = _SimReader(lambda d: 0.33, name="weak")

    ro = invariance_over_sources(sources, teacher, base, overfit, seed=3)
    rh = invariance_over_sources(sources, teacher, base, honest, seed=3)
    rw = invariance_over_sources(sources, teacher, base, weak, seed=3)

    assert ro.n_live == 2 and rh.n_live == 2, (ro.as_dict(), rh.as_dict())
    assert ro.verdict == "calibration-overfit", f"overfitter not caught: {ro.as_dict()}"
    assert rh.verdict == "ok", f"honest model flagged: {rh.as_dict()}"
    assert rw.verdict == "ok", f"uniformly-weak model flagged as cheating: {rw.as_dict()}"

    # the SCORE is the worst source, so overfitting one genre cannot buy a high score
    assert ro.worst_retention < 0.3, ro.as_dict()
    assert rh.worst_retention > 0.5, rh.as_dict()
    # ...and the weak model scores low on the level while passing the gate — the two jobs are
    # genuinely separate (spread gates, level scores)
    assert rw.worst_retention < 0.2, rw.as_dict()

    # a real spread must survive its own noise floor: identical masks -> zero dispersion
    flat = {"a": {"teacher_pass": [True] * 60, "student_pass": [i % 2 == 0 for i in range(60)],
                  "base_pass": [i % 5 == 0 for i in range(60)]}}
    flat["b"] = dict(flat["a"])
    rep = corpus_swap_invariance(flat, seed=1)
    assert rep.dispersion == 0.0 and rep.verdict == "ok", rep.as_dict()

    # fewer than two live sources -> inconclusive, and the crown check fails CLOSED
    from eval.invariance import make_invariance_check
    chk = make_invariance_check({"only": src_a}, teacher, base, seed=3)
    ok, info = chk(None, honest)
    assert not ok and "failclosed" in info["verdict"], info


def test_surprise_axis_selection():
    """THE FORMAT GAP. Corpus scale and item-freshness stop item memorization; they do nothing
    against a miner who pre-fits the TASK FORMAT, because every format is in the validator
    source. Fix: publish the pool, draw which k score from POST-COMMIT entropy.

    Required properties: unpredictable before the nonce, reproducible after it (auditable),
    uniform over the pool (no dead formats a miner can safely ignore), and — the security
    review's H4 — no bit-exhaustion collapse at larger k."""
    from collections import Counter
    from eval.seeds import derive_surprise_axes

    pool = [f"ax{i}" for i in range(10)]

    # reproducible: same (commit_root, nonce) -> same draw, so an auditor re-derives it
    a = derive_surprise_axes("root1", "nonce1", pool, k=3)
    assert a == derive_surprise_axes("root1", "nonce1", pool, k=3), "selection not reproducible"
    # unpredictable: a different nonce moves it
    assert a != derive_surprise_axes("root1", "nonce2", pool, k=3), "nonce did not change the draw"
    assert len(a) == len(set(a)) == 3, a

    # H4 regression: the old `seed >> (i*13)` hit 0 for i>=5 and every later draw collapsed to
    # index 0. Large k must still return k DISTINCT axes.
    big = derive_surprise_axes("root", "nonce", pool, k=8)
    assert len(big) == len(set(big)) == 8, f"bit-exhaustion collapse at k=8: {big}"

    # uniform-ish: no axis is effectively never drawn (a dead format is one a miner ignores)
    counts = Counter()
    for i in range(600):
        counts.update(derive_surprise_axes("root", f"n{i}", pool, k=2))
    assert len(counts) == len(pool), f"some axes never drawn: {set(pool) - set(counts)}"
    assert min(counts.values()) > 40, f"badly skewed draw: {counts}"


def test_surprise_selection_in_crown_round():
    """Wired end-to-end: only the DRAWN crown axes score, floor axes always run, and the
    selection is published in the round record so the verdict is auditable."""
    specs = [AxisSpec(_FakeAxis(n), n, 1.0, role="crown") for n in ("cA", "cB", "cC")] + \
            [AxisSpec(_FakeAxis("fl"), "fl", 1.0, role="floor")]
    tiers = [Tier("t", 10 ** 12, 1.0)]
    tour, reg = Tournament(tiers, margin=0.03), {}
    probs = {n: 1.0 for n in ("cA", "cB", "cC", "fl")}
    glm = _Sim("glm", probs)
    base = _Sim("base", {n: 0.30 for n in probs}, seed=9)
    stud = (Submission("m", "t", "s", 1, 1.0), _Sim("s", {n: 0.85 for n in probs}, seed=1))

    res = axis_round(1, 4242, specs, glm, base, tiers, tour, [stud], registry=reg,
                     items_per_axis=120, max_new_tokens=8, surprise_k=2,
                     commit_root="root", round_nonce="nonce")

    ev = [e for e in res.events if e.get("action") == "surprise_axes"]
    assert ev, f"selection not recorded for auditors: {res.events}"
    drawn = ev[0]["axes"]
    assert len(drawn) == 2 and set(drawn) <= {"cA", "cB", "cC"}, drawn

    scored_axes = {a.axis for a in res.scored["s"].axes}
    assert scored_axes == set(drawn) | {"fl"}, f"scored {scored_axes}, drawn {drawn}"
    # the undrawn crown axis must not be scored at all — that is what makes it un-pre-fittable
    assert len({"cA", "cB", "cC"} - scored_axes) == 1, scored_axes


def test_doc_task_axes():
    """The three new crown formats. Surprise selection only defends in proportion to the pool,
    and the pool only counts if each format stresses a DIFFERENT mechanism — ten variants of
    'extract a span' is one narrow skill in ten costumes and a miner buys them all at once.

    Each must be: answerable from the shown text (an impossible item fails the TEACHER too),
    exact-match checkable, and unfoolable by the obvious cheap output."""
    import json as _json
    from eval.axes.doc_tasks import ConstrainedExtraction, LongContextReal, NumericComposition
    from eval.corpus import synth_corpus

    docs = synth_corpus(160, seed=21, commit_ts=100, span=5)

    # --- composition: locate two anchored values, then combine (compounding) ---
    nc = NumericComposition(docs)
    items = nc.generate(seed=4, n=24, difficulty=1)
    assert len(items) >= 12, f"too few composition items: {len(items)}"
    for it in items:
        anchors = re.findall(r'immediately after "([^"]+)"', it.prompt)
        assert len(anchors) == 2, f"prompt quoting broke: {anchors}"
        body = it.prompt.split("Passage:\n", 1)[1].rsplit("\n\nIn the passage", 1)[0]
        vals = []
        for a in anchors:
            assert body.count(a) == 1, "anchor not unique -> ambiguous gold"
            vals.append(int(re.search(r"\d{2,}", body[body.find(a) + len(a):]).group()))
        expect = {"sum": vals[0] + vals[1], "difference": abs(vals[0] - vals[1]),
                  "larger": max(vals)}[it.meta["op"]]
        assert str(expect) == str(it.answer), f"{it.meta['op']}: {expect} != {it.answer}"
        assert nc.check(it, f"Answer: {it.answer}")
        assert not nc.check(it, "Answer: 9999999")
        # answering only ONE of the two values must fail — that is the composition requirement
        if str(vals[0]) != str(it.answer):
            assert not nc.check(it, f"Answer: {vals[0]}"), "single hop accepted"

    # --- long context over REAL (here synthetic-stand-in) documents ---
    lc = LongContextReal(docs)
    li = lc.generate(seed=2, n=8, difficulty=2)
    assert li, "no long-context items"
    for it in li:
        assert str(it.answer) in it.prompt, "gold not present in the haystack"
        assert it.meta["n_docs"] >= 5, it.meta
        assert lc.check(it, f"Answer: {it.answer}")
        assert not lc.check(it, "Answer: 13579")

    # --- constrained extraction: content AND shape, independently ---
    ce = ConstrainedExtraction(docs)
    ci = ce.generate(seed=6, n=18, difficulty=1)
    assert {i.meta["shape"] for i in ci} == {"csv", "lines", "json"}, "shapes not exercised"

    def render(it):
        v, s = it.answer["vals"], it.answer["shape"]
        return {"csv": ",".join(v), "lines": "\n".join(v), "json": _json.dumps(v)}[s]

    for it in ci:
        body = it.prompt.split("Passage:\n", 1)[1].rsplit("\n\nList", 1)[0]
        assert [m.group() for m in re.finditer(r"(?<![\w.])\d{2,}(?![\w.])", body)][:it.meta["k"]] \
            == it.answer["vals"], "gold is not the first k numbers in document order"
        assert ce.check(it, render(it)), f"correct answer rejected ({it.meta['shape']})"
        # right content, wrong shape -> fail; right shape, wrong content -> fail
        if it.meta["shape"] != "csv":
            assert not ce.check(it, " ".join(it.answer["vals"])), "wrong shape accepted"
        assert not ce.check(it, render(it).replace(it.answer["vals"][0], "424242", 1)), \
            "wrong content accepted"


def test_dethrone_rewards_the_anti_clone_strategy():
    """THE MECHANISM MUST PAY FOR ITS OWN STRATEGY. The product thesis is 'beat a cloned
    artifact on the axes a clone loses' (multilingual, agentic). The OLD rule — min over
    per-axis paired LCBs — could not reward that: measured at production n, a challenger tied
    on English and +0.55 better on multilingual scored -0.109 and HELD, because each per-axis
    LCB is the 5th percentile of a noisy difference and the min selects the unluckiest tied
    axis (two models of IDENTICAL skill measured -0.156).

    The new rule bootstraps the LCB on the soft-min AGGREGATE difference. This test pins all
    four properties that must hold simultaneously — losing any one re-breaks the mechanism."""
    from eval.koth import axis_regression, softmin_lcb_diff

    AX = ["extractive", "numeric_composition", "multilingual"]

    def mk(mid, rates, n=150, seed=0):
        r = random.Random(seed)
        pa = {ax: [1.0 if r.random() < p else 0.0 for _ in range(n)] for ax, p in rates.items()}
        return Scored(sub=Submission(mid, "t", mid, 1, 0.0), retention=min(rates.values()),
                      retention_lb=min(rates.values()),
                      per_point=[v for a in pa.values() for v in a], gates_ok=True, per_axis=pa)

    def dethrones(c, k):
        lcb = softmin_lcb_diff(c, k, seed=7)
        return lcb > MARGIN and axis_regression(c, k, seed=7) is None

    # 1. the anti-clone improvement MUST be rewarded
    clone = mk("clone", {"extractive": .85, "numeric_composition": .85, "multilingual": .20}, seed=3)
    chal = mk("chal", {"extractive": .85, "numeric_composition": .85, "multilingual": .75}, seed=4)
    assert dethrones(chal, clone), "the product's core strategy still cannot dethrone"

    # 2. an exact copy must STILL tie — zero variance in every replicate
    king = mk("king", {a: .60 for a in AX}, seed=11)
    assert softmin_lcb_diff(mk("copy", {a: .60 for a in AX}, seed=11), king, seed=7) == 0.0
    assert not dethrones(mk("copy", {a: .60 for a in AX}, seed=11), king)

    # 3. equal skill with independent errors must not dethrone, and must not be wildly biased
    ind = softmin_lcb_diff(mk("ind", {a: .60 for a in AX}, seed=99), king, seed=7)
    assert not dethrones(mk("ind", {a: .60 for a in AX}, seed=99), king)
    assert ind > -0.08, f"estimator still badly biased on tied axes: {ind}"

    # 4. the drifter guarantee survives: big gains cannot buy a real regression
    drift = mk("drift", {"extractive": .95, "numeric_composition": .95, "multilingual": .35}, seed=5)
    assert not dethrones(drift, king), "drifter bought the crown"

    # 5. an honest across-the-board improvement is actually payable at production n
    wins = sum(dethrones(mk("c", {a: .75 for a in AX}, seed=500 + t),
                         mk("k", {a: .60 for a in AX}, seed=100 + t)) for t in range(20))
    assert wins >= 14, f"uniform +0.15 dethroned only {wins}/20 — crown effectively unremovable"


def test_king_is_revalidated_and_vacated():
    """The incumbent must face the SAME gates as challengers. Previously axis_round hardcoded
    gates_ok=True for the king, so a model that had rotted below base kept the throne and 100%
    of the tier's emission forever — while an identical challenger would have been rejected.
    A crown certificate has to keep being true."""
    specs = [AxisSpec(_FakeAxis("x"), "x", 1.0), AxisSpec(_FakeAxis("y"), "y", 1.0)]
    tiers = [Tier("t", 10 ** 12, 1.0)]
    tour, reg = Tournament(tiers, margin=0.03), {}
    glm = _Sim("glm", {"x": 1.0, "y": 1.0})
    base = _Sim("base", {"x": 0.30, "y": 0.30}, seed=9)

    good = (Submission("m_good", "t", "good", 1, 1.0), _Sim("good", {"x": .85, "y": .85}, seed=1))
    _run(1, specs, tiers, tour, reg, [good], glm, base)
    assert tour.kings["t"].model_id == "good", "setup: good should crown the open throne"

    # the king now degenerates (loops). Re-scored, it must FAIL the degeneracy gate and be
    # vacated — not silently retained.
    reg["good"] = _Sim("rotted", {}, loop=True)
    res = _run(2, specs, tiers, tour, reg, [], glm, base)
    vac = [e for e in res.events if e.get("action") == "vacate"]
    assert vac, f"rotted king was not vacated: {res.events}"
    assert "t" not in tour.kings, "throne still occupied by a failing king"

    # and with the throne open, a valid challenger takes it the same round it is offered
    fresh = (Submission("m_new", "t", "new", 1, 1.0), _Sim("new", {"x": .80, "y": .80}, seed=6))
    _run(3, specs, tiers, tour, reg, [fresh], glm, base)
    assert tour.kings["t"].model_id == "new", f"throne not re-crowned: {tour.kings.get('t')}"


def test_script_aware_numbers():
    """The multilingual axis is the only one a cloned artifact loses on, and two bugs made it
    a no-op — both invisible on English test data:

    1. `(?<![\\w.])\\d{2,}(?![\\w.])` finds NOTHING in unspaced CJK, because Python's `\\w`
       matches Han. Chinese web text is predominantly unspaced, so cmn_Hani (35% of the mix)
       minted ZERO items. Verified on real fineweb-2 text: 0 probes before, 12 after.
    2. Arabic renders numerals as Arabic-Indic (٢٠٢٤). Those matched `\\d` and became the GOLD,
       so a model correctly answering 2024 was scored WRONG. The axis punished the right answer.
    """
    from eval.axes.extractive import ExtractiveQA
    from eval.corpus import Doc
    from eval.textnorm import find_numbers, norm_digits, same_number

    def nums(t):
        return [m.group() for m, _ in find_numbers(t)]

    # digits embedded in non-Latin script must be found
    assert nums("公司在2024年宣布了新的政策") == ["2024"], "CJK digits still invisible"
    assert nums("компания в 2024 году") == ["2024"]
    assert nums("أعلنت الشركة ٢٠٢٤ عن خطط") == ["٢٠٢٤"], "Arabic-Indic not found"
    assert nums("売上は１２３４でした") == ["１２３４"], "fullwidth digits not found"
    # ...but a digit run glued to an ASCII token, or a decimal, is still not a standalone number
    assert nums("abc123 and 1.5 and x9y") == [], "Latin-token veto regressed"

    # numeric equality across digit families and separators
    assert same_number("2024", "٢٠٢٤") and same_number("2,024", "2024")
    assert same_number("１２３４", "1234")
    assert not same_number("2024", "2025")
    assert norm_digits("٢٠٢٤") == "2024"

    # end-to-end: a CJK document must mint a probe, and an ASCII answer to an
    # Arabic-Indic gold must be accepted
    zh = Doc("zh1", "公司在2024年宣布了新的政策和计划。该项目预计在2031年完成建设工作。", 100)
    ar = Doc("ar1", "أعلنت الشركة عن أرباح ٢٠٢٤ في التقرير السنوي الجديد لهذا العام كله.", 100)
    ax = ExtractiveQA([zh, ar], kinds=("number",))
    items = ax.generate(seed=3, n=8, difficulty=1)
    assert any(i.meta["doc"] == "zh1" for i in items), "no probe minted from CJK document"
    for it in items:
        assert ax.check(it, f"Answer: {it.answer}"), "verbatim gold rejected"
        # the model normalizing to ASCII must still be scored correct
        assert ax.check(it, f"Answer: {norm_digits(str(it.answer))}"), \
            f"ASCII answer rejected for gold {it.answer!r}"
        assert not ax.check(it, "Answer: 5150")


def main() -> int:
    tests = [test_worst_axis_blocks_drifter, test_axis_round_gates, test_long_context_checker,
             test_code_extractor_robust, test_numeric_first_marker, test_diff_in_diff_gate,
             test_diff_in_diff_over_corpus, test_axis_round_overfit_precondition,
             test_content_identity_and_commit_reveal, test_round_record_signature,
             test_economics_free_eval_is_per_coldkey, test_multihop_axis,
             test_validator_axis_loop_end_to_end, test_axis_chain_epoch_end_to_end,
             test_corpus_hf_pure_logic, test_overfit_check_wired_into_crown,
             test_miner_submission_roundtrip, test_code_exec_sandbox_blocks_payload,
             test_bond_refund_keyed_by_coldkey, test_long_context_argmax_and_id,
             test_extractive_axis, test_generator_specialist_denied_crown,
             test_corpus_stream_selection, test_bitrate_matches_published_formats,
             test_multi_constraint_compounds, test_score_at_budget_and_convergence_gate,
             test_corpus_swap_invariance, test_surprise_axis_selection,
             test_surprise_selection_in_crown_round, test_doc_task_axes,
             test_dethrone_rewards_the_anti_clone_strategy, test_king_is_revalidated_and_vacated,
             test_script_aware_numbers,
             test_saturation_guard_retires_flat_axes]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            # a missing dep (e.g. pynacl) or unexpected error must not abort the whole run —
            # report it and keep going so one gap doesn't hide the rest of the suite.
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
