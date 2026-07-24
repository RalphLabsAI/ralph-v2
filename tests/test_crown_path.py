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


def main() -> int:
    tests = [test_worst_axis_blocks_drifter, test_axis_round_gates, test_long_context_checker,
             test_code_extractor_robust, test_numeric_first_marker, test_diff_in_diff_gate,
             test_diff_in_diff_over_corpus, test_axis_round_overfit_precondition,
             test_content_identity_and_commit_reveal, test_round_record_signature,
             test_economics_free_eval_is_per_coldkey, test_multihop_axis,
             test_validator_axis_loop_end_to_end, test_axis_chain_epoch_end_to_end,
             test_corpus_hf_pure_logic, test_overfit_check_wired_into_crown,
             test_miner_submission_roundtrip]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
