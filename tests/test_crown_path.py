"""Regression tests for the crown decision path (the fixes in commit 2494017 + the
long-context axis). Runnable two ways:

    python -m tests.test_crown_path      # self-running harness, no deps
    pytest tests/test_crown_path.py      # if pytest is installed

Covers: worst-axis dethrone (a fluent drifter that clears the pooled mean is blocked),
the axis_round crown gates (degeneracy DQ, every-declared-axis-live, copy ties,
genuine improvement dethrones), and the deterministic long-context checker.
"""
from __future__ import annotations

import math
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


def test_the_exam_is_clamped_not_filtered():
    """THE EXAM IS BOUNDED BY THE WEAKEST ADMISSIBLE RUNTIME — and by CLAMPING, never by dropping.

    GGUF is the only format that can pass the bit tiers; it runs under llama.cpp, whose window
    RAISES rather than truncates, and a real round died mid-scoring on
    `Requested tokens (5990) exceed context window of 4096`.

    The first fix filtered oversized items out by CHARACTER length. Measured on the real pool that
    deleted 267 of 900 items and the ENTIRE en/deep slice — retention is aggregated worst-slice
    over (observer x language x depth), and that slice is exactly where a cloned artifact is
    weakest. A character bound is also non-monotonic in tokens: it binds hardest where characters
    are least dense, i.e. on English, and barely at all on Chinese.

    So: clamp every prefix, keep every item and every stratum."""
    from eval.pool import MAX_PREFIX_TOKENS, PREFIX_TRUNCATION_SIDE, clamp_prefixes
    from eval.runners import GGUFStudentRunner
    from eval.steps import Trajectory

    assert MAX_PREFIX_TOKENS + 256 <= GGUFStudentRunner.N_CTX, \
        "a clamped prefix plus a full-length step must still fit the student's window"
    # AND THE OBSERVER'S, WHICH IS TIGHTER. Two of the three observers stop at 4096 positions, and
    # the observer is fed prefix + step + continuation. Overflow does not raise there — a RoPE
    # model extrapolates and returns degraded distributions straight into the KL.
    TIGHTEST_OBSERVER_WINDOW = 4096
    assert MAX_PREFIX_TOKENS + 256 + 128 <= TIGHTEST_OBSERVER_WINDOW, \
        "the exam overflows the smallest observer in the pool; it would degrade silently"
    assert PREFIX_TRUNCATION_SIDE == "left", \
        "these prefixes end where the step begins; right-truncation deletes what predicts it"

    class _Tok:
        truncation_side = "right"

        def __call__(self, text, **kw):
            return {"input_ids": list(range(len(text)))}

        def decode(self, ids, **kw):
            return "x" * len(ids)

    long_t = Trajectory(id="a", prefix="y" * 9000, source="en_x", index=0, meta={})
    short_t = Trajectory(id="b", prefix="short", source="zh_x", index=4, meta={})
    out = clamp_prefixes([long_t, short_t], max_tokens=MAX_PREFIX_TOKENS, tok=_Tok())

    assert len(out) == 2, "clamping must never drop an item — that is what deleted a whole slice"
    assert len(long_t.prefix) == MAX_PREFIX_TOKENS
    assert short_t.prefix == "short", "a prefix under the ceiling must be untouched"


def test_the_incumbent_cannot_swap_its_bytes_between_rounds():
    """THE KING CONTROLS ITS OWN REPO. A challenger's artifact is pinned to the content hash it
    revealed at commit time, but the incumbent used to be refetched with `expect_hash=""` — from a
    URI whose repo the king owns. So the crown holder could force-push a model fitted to this
    round's items over the same ref and have it scored AS the incumbent, while the published record
    still named the ORIGINAL model_id. That single move defeats the dethrone margin (the paired
    comparison runs against bytes nobody published), the anti-copy guarantee, and the L2/L3 re-run.

    `Reign.model_id` is the recorded content hash, so pinning to it makes the swap self-refusing —
    and a mutable `@main` ref becomes harmless."""
    import tempfile
    from pathlib import Path

    import eval.fetch as F

    swapped = b"weights fitted to this round's exam"
    recorded_model_id = "a" * 64          # what the trail says the crown is

    def _try(reveals):
        with tempfile.TemporaryDirectory() as d:
            orig = F.fetch
            F.fetch = lambda uri, root, expect_hash="", **kw: orig(
                uri, root, expect_hash=expect_hash,
                lister=lambda repo, rev: [("model.gguf", len(swapped))],
                downloader=lambda repo, rev, name, out: Path(out).write_bytes(swapped))
            try:
                return F.resolver(d, reveals=reveals, log=[])("king:ternary",
                                                              "hf://kingrepo/m@main")
            finally:
                F.fetch = orig

    assert _try({}) != "", "precondition: an unpinned incumbent fetch does succeed"
    assert _try({"king:ternary": {"content_hash": recorded_model_id}}) == "", \
        "the king swapped its bytes and was scored anyway"


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


def test_gguf_intake():
    """The subnet's deliverable is a checkpoint that RUNS on a phone, and the format that ships
    is GGUF (PrismML's Bonsai releases are Q1_0 / Q2_0). Intake was safetensors-only, so the only
    sub-2-bit form that survived was their BF16 `-unpacked` container — the free clone vector.
    The subnet could not receive its own product.

    Header-only parse, no native parser. GGUF also gives EXACT bits/weight, because each ggml
    type has a fixed block size and bytes-per-block — no sampling or level counting needed."""
    import struct
    import tempfile
    from pathlib import Path
    from eval.gates import inspect_checkpoint
    from eval.gguf import GGML_TYPES, read_gguf, type_bits
    from eval.identity import HASHED_SUFFIXES, content_hash

    # published block layouts must give the published widths
    assert type_bits(8) == 8.5, "Q8_0 != 8.5 bpw"
    assert abs(type_bits(34) - 1.6875) < 1e-9, "TQ1_0 != 1.6875 bpw"
    assert abs(type_bits(35) - 2.0625) < 1e-9, "TQ2_0 != 2.0625 bpw"
    assert type_bits(1) == 16.0 and type_bits(0) == 32.0
    assert type_bits(9999) is None, "unknown ggml type must not be guessed"

    # a weight format that is allowed but NOT hashed would unbind weights from commit-reveal
    # and make a post-commit swap free — the two lists must move together.
    assert ".gguf" in HASHED_SUFFIXES, "gguf allowed at intake but excluded from the identity hash"

    def write_gguf(path, tensors, arch="qwen2"):
        """Minimal valid GGUF: magic, version, counts, one metadata string, tensor table."""
        out = bytearray(b"GGUF" + struct.pack("<I", 3))
        out += struct.pack("<QQ", len(tensors), 1)
        k = b"general.architecture"
        out += struct.pack("<Q", len(k)) + k + struct.pack("<I", 8)
        v = arch.encode()
        out += struct.pack("<Q", len(v)) + v
        for name, (dims, tt) in tensors.items():
            nb = name.encode()
            out += struct.pack("<Q", len(nb)) + nb
            out += struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
            out += struct.pack("<I", tt) + struct.pack("<Q", 0)
        Path(path).write_bytes(bytes(out))

    with tempfile.TemporaryDirectory() as d:
        # ternary body (TQ2_0) with an fp16 output head — the mixed-precision shape real
        # quantized artifacts have, and the reason an advertised label understates true bpw.
        write_gguf(Path(d) / "model.gguf", {
            "blk.0.attn_q.weight": ((4096, 4096), 35),
            "blk.0.ffn_down.weight": ((4096, 4096), 35),
            "output.weight": ((4096, 4096), 1),
            "blk.0.attn_norm.weight": ((4096,), 0),      # 1-D: the negligible tail, excluded
        })
        gi = read_gguf(Path(d) / "model.gguf")
        assert gi.ok, gi.reasons
        assert gi.arch == "qwen2" and gi.n_tensors == 4
        # (2 x 1.6875... no: TQ2_0 = 2.0625) two ternary tensors + one fp16 head
        expect = (2 * 2.0625 + 16.0) / 3
        assert abs(gi.bits_per_weight - expect) < 1e-3, f"{gi.bits_per_weight} != {expect}"

        (Path(d) / "config.json").write_text('{"hidden_size":4096}')
        insp = inspect_checkpoint(d)
        assert insp.ok, insp.reasons
        assert insp.params == 3 * 4096 * 4096, insp.params
        assert abs(insp.effective_bits_per_param - expect) < 1e-3
        # the artifact must be covered by the identity hash
        assert content_hash(d)

    # an unknown ggml type is a HARD REJECT, never a guessed width
    with tempfile.TemporaryDirectory() as d:
        write_gguf(Path(d) / "model.gguf", {"blk.0.attn_q.weight": ((256, 256), 250)})
        gi = read_gguf(Path(d) / "model.gguf")
        assert not gi.ok and any("unknown ggml type" in r for r in gi.reasons), gi.reasons

    # a truncated / non-GGUF file must fail closed rather than raise
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "model.gguf").write_bytes(b"NOTGGUF" + b"\x00" * 32)
        assert not read_gguf(Path(d) / "model.gguf").ok


def test_pinned_parent():
    """The task is "compress THIS model, architecture unchanged" — Bonsai's actual shape. Pinning
    the parent is what makes the score well-defined:

      * scoring against the MINER'S DECLARED parent invites weak-parent farming (declare a
        lobotomised parent, retention -> 1.0; the degenerate limit is parent == student);
      * scoring absolute capability rewards the strongest artifact from anywhere, which is the
        clone attack in a tie;
      * scoring against a PINNED parent is identical for every miner in the tier and the miner
        does not choose the denominator.

    Header-only, so it costs nothing. It is an ADMISSION gate, not provenance — nothing here
    proves descent, and no behavioural test can."""
    import struct
    import tempfile
    from pathlib import Path
    from eval.gates import TierBudget
    from eval.intake import intake
    from eval.parent import PARENTS, ParentSpec, parent_compat

    def write_gguf(path, params_per_tensor, n_tensors, arch, ggml_type=35):
        out = bytearray(b"GGUF" + struct.pack("<I", 3))
        out += struct.pack("<QQ", n_tensors, 1)
        k = b"general.architecture"
        out += struct.pack("<Q", len(k)) + k + struct.pack("<I", 8)
        v = arch.encode()
        out += struct.pack("<Q", len(v)) + v
        side = int(params_per_tensor ** 0.5)
        for i in range(n_tensors):
            nb = f"blk.{i}.attn_q.weight".encode()
            out += struct.pack("<Q", len(nb)) + nb
            out += struct.pack("<I", 2) + struct.pack("<QQ", side, side)
            out += struct.pack("<I", ggml_type) + struct.pack("<Q", 0)
        Path(path).write_bytes(bytes(out))

    spec = ParentSpec(name="test/parent-8b", arch="qwen2", weight_params=4 * 1024 * 1024)

    # the parent's own compressed forms are admitted — architecture is unchanged, only the
    # storage width differs, so the element count is identical across quantizations
    for tt in (35, 8, 1):                      # TQ2_0 (ternary), Q8_0, F16
        with tempfile.TemporaryDirectory() as d:
            write_gguf(Path(d) / "m.gguf", 1024 * 1024, 4, "qwen2", tt)
            c = parent_compat(d, spec)
            assert c.ok, f"honest compression of the pinned parent rejected (type {tt}): {c.reasons}"
            assert c.measured_params == 4 * 1024 * 1024

    # a FOREIGN artifact is refused, on size and on architecture
    with tempfile.TemporaryDirectory() as d:
        write_gguf(Path(d) / "m.gguf", 1024 * 1024, 1, "llama")
        c = parent_compat(d, spec)
        assert not c.ok
        assert any("architecture" in r for r in c.reasons), c.reasons
        assert any("params" in r for r in c.reasons), c.reasons

    # and the gate is actually enforced by the front door, not merely available
    with tempfile.TemporaryDirectory() as d:
        write_gguf(Path(d) / "m.gguf", 1024 * 1024, 1, "llama")
        t = TierBudget(name="t", max_params=10 ** 12, max_effective_bits=32.0, parent=spec)
        dec = intake(d, t)
        assert not dec.accepted, "foreign artifact accepted into a pinned-parent tier"
        t_open = TierBudget(name="t", max_params=10 ** 12, max_effective_bits=32.0)
        assert intake(d, t_open).accepted, "tier without a pinned parent should still accept"

    # the shipped registry entry must be a real measurement, not a placeholder
    q = PARENTS["qwen2.5-0.5b-instruct"]
    assert q.weight_params == 630_095_872 and q.arch == "qwen2"


def test_observer_kl_step_scoring():
    """Observer-KL: two steps are equivalent when they move an INDEPENDENT observer into the
    same predictive state. Scored on downstream EFFECT, never on wording — which is why the
    style-mimicry that beat SN97's KL crown has no channel here.

    Written to break it, not to confirm it. Every case below is a way a miner could try to win
    without carrying the teacher's information."""
    from eval.observer_kl import (MIN_TEACHER_EFFECT, StepEffect, kl, sample_score,
                                  score_miner, step_effect)

    def dist(**kw):
        return dict(kw)

    # --- the KL primitive on sparse top-k supports ---
    assert kl(dist(a=1.0), dist(a=1.0)) == 0.0, "identical distributions must score 0"
    assert kl(dist(a=0.9, b=0.1), dist(a=0.1, b=0.9)) > 0.5
    # a token one side proposes and the other never did must cost something FINITE
    only = kl(dist(z=1.0), dist(a=1.0))
    assert math.isfinite(only) and only > 5.0, only

    N = 12
    teacher = [dist(a=0.7, b=0.2, c=0.1)] * N          # teacher moves the observer here
    base = [dist(a=0.34, b=0.33, c=0.33)] * N          # unconditioned: near-uniform

    # 1. HONEST: a differently-worded step with the same effect scores top marks. This is the
    #    whole point — no style channel, so paraphrase costs nothing.
    same = step_effect(teacher, list(teacher), base)
    assert not same.discarded and same.s == 0.0 and same.magnitude_gap == 0.0
    assert sample_score(same) > 0.99, sample_score(same)

    # 2. INERT step (const's step 10): miner changed nothing, observer stays at baseline.
    #    Must be punished by the magnitude term even though it "disagrees" little.
    inert = step_effect(teacher, list(base), base)
    assert not inert.discarded, "an inert MINER step must still be scored, not discarded"
    assert inert.d_miner < 1e-9 and inert.d_teacher > MIN_TEACHER_EFFECT
    # normalised against the teacher's own effect, an inert step lands at ~exp(-1)exp(-1);
    # unnormalised it scored 0.56, i.e. doing nothing was worth over half marks.
    assert sample_score(inert) < 0.2, f"inert step scored {sample_score(inert)}"

    # 3. WRONG DIRECTION (const's step 12): moved the observer as much, but elsewhere.
    wrong = [dist(a=0.1, b=0.2, c=0.7)] * N
    wd = step_effect(teacher, wrong, base)
    assert wd.d_miner > MIN_TEACHER_EFFECT, "control: the wrong step does move the observer"
    assert wd.magnitude_gap < 0.35, "control: comparable magnitude, so only direction differs"
    assert sample_score(wd) < sample_score(same), "wrong direction must score below honest"
    assert sample_score(wd) < 0.5, sample_score(wd)

    # 4. DISCARD IS NOT MINER-CONTROLLABLE. When the TEACHER's step is inert the sample carries
    #    no signal and is dropped — but that decision must never depend on the miner, or a miner
    #    could bury its hard samples by emitting bland steps.
    flat = [dist(a=0.34, b=0.33, c=0.33)] * N
    for miner_side in (flat, wrong, teacher):
        d = step_effect(flat, list(miner_side), base)
        assert d.discarded and "teacher effect" in d.reason, d.as_dict()
    # ...and conversely a live teacher sample is NEVER discarded regardless of what the miner did
    for miner_side in (flat, wrong, teacher):
        assert not step_effect(teacher, list(miner_side), base).discarded

    # 3b. SCALE INVARIANCE: the same relative behaviour on a high-effect trajectory and a
    #     low-effect one must score the same, or the metric mostly measures which trajectories
    #     were drawn rather than what the miner did.
    big_t = [dist(a=0.97, b=0.02, c=0.01)] * N
    big_inert = step_effect(big_t, list(base), base)
    assert abs(sample_score(big_inert) - sample_score(inert)) < 0.05, \
        f"not scale-invariant: {sample_score(big_inert)} vs {sample_score(inert)}"

    # --- aggregation ---
    def samples(eff, n=10, key="obs=kimi|lang=en|len=2"):
        return [(key, eff) for _ in range(n)]

    honest = score_miner(samples(same))
    assert honest.score > 0.99 and honest.n_discarded == 0, honest.as_dict()

    # 5. WORST-SLICE, not mean: acing one slice must not pay for a bad one.
    mixed = samples(same, 10, "obs=kimi|lang=en|len=2") + samples(wd, 10, "obs=qwen|lang=ru|len=4")
    m = score_miner(mixed)
    assert len(m.per_slice) == 2
    assert abs(m.score - min(m.per_slice.values())) < 1e-9, "aggregation is not worst-slice"
    assert m.score < honest.score

    # 6. INERT MINER fails liveness outright — otherwise "never move the observer" is a strategy
    #    that scores well on similarity (two inert steps disagree very little).
    dead = score_miner(samples(inert, 10))
    assert dead.score == 0.0 and any("inert" in r for r in dead.reasons), dead.as_dict()

    # 7. a thin slice must not decide the crown on noise
    thin = score_miner([("obs=kimi|lang=en|len=2", same)] * 3)
    assert thin.score == 0.0 and any("slice" in r for r in thin.reasons), thin.as_dict()


def test_step_extraction():
    """Observer-KL needs (prefix K, step K->K+1) pairs, which raw web text cannot give — fineweb
    has no step boundaries. Each source therefore carries a BOUNDARY RULE, and the two ways this
    silently produces garbage are pinned here:

      * a wrong field name or delimiter yields an EMPTY corpus, indistinguishable from a quiet
        source (a <think> parser returns zero on OpenThoughts-114k, which uses
        <|begin_of_thought|>);
      * row count != unique content. SWE-ZERO advertises 12.29M rows; sampling 120 rows across
        six offsets spanning the whole set returned EIGHT distinct instance_id — ~100 rollouts
        per pull request. Uniform sampling makes 122,908 memorized tasks look like a 12M corpus,
        which would quietly falsify the entire "scale is the anti-overfit lever" argument.
    """
    from eval.steps import (TRAJECTORY_SOURCES, StepSpec, dedup_rows, extract_steps,
                            pool_rows, verify_source)

    # --- message boundaries: a step is one assistant turn, boundary is a list index ---
    spec = StepSpec(kind="message", msg_keys=("role", "content"))
    row = {"messages": [{"role": "user", "content": "fix the bug"},
                        {"role": "assistant", "content": "THOUGHT: look at the repo"},
                        {"role": "user", "content": "ok"},
                        {"role": "assistant", "content": "THOUGHT: patch it"}]}
    st = extract_steps(row, spec, "messages", "src", "r0")
    assert len(st) == 2, st
    assert st[0].step == "THOUGHT: look at the repo"
    assert "fix the bug" in st[0].prefix
    # the prefix must NOT contain the step it is being asked to predict
    for s in st:
        assert s.step not in s.prefix, "step leaked into its own prefix"
    # a leading assistant turn has no prefix and must be skipped, not emitted with an empty K
    lead = extract_steps({"messages": [{"role": "assistant", "content": "hi"}]}, spec,
                         "messages", "s", "r")
    assert lead == [], lead

    # --- tool_call boundaries ---
    tc = StepSpec(kind="tool_call")
    row2 = {"gen": "reason\n<tool_call>\nprint(1)\n</tool_call>\n```output\n1\n```\nmore\n"
                   "<tool_call>\nprint(2)\n</tool_call>"}
    st2 = extract_steps(row2, tc, "gen", "s", "r")
    assert len(st2) == 2 and st2[0].step.startswith("<tool_call>")
    assert "print(2)" not in st2[0].prefix, "later step visible in an earlier prefix"

    # --- para boundaries inside <think>, and the WRONG-DELIMITER case ---
    pa = StepSpec(kind="para")
    row3 = {"response": "<think>\nfirst block\n\nsecond block\n\nthird block\n</think>\nanswer"}
    st3 = extract_steps(row3, pa, "response", "s", "r")
    assert [s.step for s in st3] == ["second block", "third block"], [s.step for s in st3]
    # OpenThoughts-114k's delimiter: a <think> parser must return NOTHING rather than guess
    assert extract_steps({"response": "<|begin_of_thought|>a\n\nb<|end_of_thought|>"},
                         pa, "response", "s", "r") == []

    # --- verify_source must FAIL LOUDLY on a source that yields nothing ---
    ok, msg = verify_source("glaive_r1", [{"response": "no think tags here"}])
    assert not ok and "extracted 0 steps" in msg, msg
    ok2, _ = verify_source("glaive_r1", [{"response": "<think>a\n\nb\n\nc</think>"}])
    assert ok2

    # --- dedup: the check that keeps the scale claim honest ---
    fake = [{"instance_id": f"task{i // 100}", "messages": []} for i in range(1000)]
    kept = list(dedup_rows(fake, "instance_id", 5))
    assert len(kept) == 50, len(kept)
    assert len({r["instance_id"] for r in kept}) == 10
    assert TRAJECTORY_SOURCES["swe_zero"]["max_per_key"] == 5, "SWE-ZERO dedup cap removed"
    assert TRAJECTORY_SOURCES["swe_zero"]["dedup_key"] == "instance_id"

    # every source must declare a boundary rule and a real licence
    for name, s in TRAJECTORY_SOURCES.items():
        assert isinstance(s["step"], StepSpec), name
        assert s["license"], f"{name} has no licence — a validator cannot legally pull it"
        assert s["pool_rows"] > 0, name
    assert pool_rows() > 10_000_000, pool_rows()


def test_observer_round():
    """The round: four of the five rollouts are miner-INDEPENDENT and must be computed once,
    so an extra miner costs one generation plus one forward pass. Also pins the properties that
    make the round fair — the observer is drawn post-commit, and the discard decision is made
    before any miner is involved."""
    from eval.observer_round import (build_shared, pick_observer, score_submission)
    from eval.steps import Trajectory

    calls = {"parent": 0, "obs_dist": 0, "obs_gen": 0, "miner": 0}

    def d(a, b, c):
        return {"a": a, "b": b, "c": c}

    class StubObserver:
        """Predicts from whatever step is glued to the prefix: 'X' -> peaked on a, 'Y' -> on c,
        nothing -> flat. That makes 'same effect' and 'different effect' controllable."""
        def generate(self, prompts, max_new_tokens=128):
            calls["obs_gen"] += 1
            return ["cont cont cont"] * len(prompts)

        def distributions(self, prefix, continuation):
            calls["obs_dist"] += 1
            if prefix.rstrip().endswith("X"):
                return [d(0.8, 0.1, 0.1)] * 6
            if prefix.rstrip().endswith("Y"):
                return [d(0.1, 0.1, 0.8)] * 6
            return [d(0.34, 0.33, 0.33)] * 6      # unconditioned baseline

    class StubStepper:
        def __init__(self, tok, who):
            self.tok, self.who = tok, who

        def generate(self, prompts, max_new_tokens=256):
            calls[self.who] += 1
            return [self.tok] * len(prompts)

    # 12 samples at one depth so a single slice clears min_per_slice=8; the thin-slice guard
    # is exercised separately below.
    trajs = [Trajectory(id=f"t{i}", source="glaive_r1", prefix=f"prefix {i}", step="ref",
                        index=0) for i in range(12)]
    obs = StubObserver()
    parent = StubStepper("X", "parent")

    shared = build_shared(trajs, parent, obs, observer_name="qwen")
    assert len(shared) == 12 and all(s.usable for s in shared), [s.reason for s in shared]
    # ONE batched parent call for all trajectories, not one per trajectory
    assert calls["parent"] == 1, calls
    parent_dist_calls = calls["obs_dist"]

    # a miner reproducing the parent's effect scores ~1; a miner moving the observer elsewhere
    # scores far lower — and neither is compared on WORDING, only on effect
    good = score_submission(shared, StubStepper("X", "miner"), obs)
    bad = score_submission(shared, StubStepper("Y", "miner"), obs)
    assert good.score > 0.99, good.as_dict()
    assert bad.score < 0.2, bad.as_dict()
    assert good.n_scored == 12 and good.n_discarded == 0

    # PER-MINER COST: one generation call, and one observer pass per usable sample. The parent
    # rollouts, C, P_G and P_0 are NOT recomputed.
    before = calls["obs_dist"]
    _ = score_submission(shared, StubStepper("X", "miner"), obs)
    assert calls["obs_dist"] - before == 12, "shared rollouts were recomputed per miner"
    assert calls["parent"] == 1, "parent re-rolled for a later miner"

    # an EMPTY miner step is inert, and must be SCORED as such rather than discarded —
    # otherwise saying nothing is a way to dodge the hard samples
    empty = score_submission(shared, StubStepper("", "miner"), obs)
    assert empty.n_discarded == 0, empty.as_dict()
    assert empty.score == 0.0 and any("inert" in r for r in empty.reasons), empty.as_dict()

    # DISCARD IS DECIDED BEFORE ANY MINER RUNS: if the PARENT does not move the observer, the
    # sample is unusable for everyone, and no miner can influence that
    flat_parent = build_shared(trajs, StubStepper("", "parent"), obs, "qwen")
    assert all(not s.usable for s in flat_parent), [s.reason for s in flat_parent]

    # OBSERVER DRAWN POST-COMMIT: reproducible for an auditor, unpredictable before the nonce
    pool = ["kimi", "qwen", "llama", "mistral"]
    assert pick_observer("root", "n1", pool) == pick_observer("root", "n1", pool)
    picks = {pick_observer("root", f"n{i}", pool) for i in range(40)}
    assert len(picks) > 1, "observer never changes with the nonce"
    assert picks <= set(pool)

    # a slice too thin to trust must not decide the crown
    thin = score_submission(shared[:4], StubStepper("X", "miner"), obs)
    assert thin.score == 0.0 and any("slice" in r for r in thin.reasons), thin.as_dict()


def test_determinism_gate():
    """Observer-KL is derived from logits, and logit arithmetic is not bit-stable — batch
    composition, kernel choice and dtype all move low-order bits, and a KL turns that into what
    looks like a real score difference. SN97 measured 2-5pp of run-to-run variance on IFEval and
    then published 0.42-1.2pp margins as findings; a crown decided inside the noise band is a
    lottery a miner wins by resubmitting.

    So the validator measures its own noise floor and refuses to crown inside it."""
    from eval.determinism import crownable, margin_floor, measure_noise

    class Stable:
        def distributions(self, prefix, continuation):
            return [{1: 0.7, 2: 0.2, 3: 0.1}] * 5

    class Jittery:
        """Same input, slightly different answer each call — the real failure mode."""
        def __init__(self, eps):
            self.eps, self.n = eps, 0

        def distributions(self, prefix, continuation):
            self.n += 1
            d = self.eps * (self.n % 3)
            return [{1: 0.7 + d, 2: 0.2 - d, 3: 0.1}] * 5

    class ShapeShifter:
        def __init__(self):
            self.n = 0

        def distributions(self, prefix, continuation):
            self.n += 1
            return [{1: 0.7, 2: 0.3}] * (5 if self.n % 2 else 4)

    stable = measure_noise(Stable(), "K", "C", repeats=4)
    assert stable.identical and stable.max_kl == 0.0, stable.as_dict()
    assert not stable.reasons, stable.reasons

    # jitter sized like the real thing: SN97 measured 2-5 PERCENTAGE POINTS of run-to-run
    # variance, not 1e-6. At realistic magnitudes the noise-scaled floor dominates the absolute
    # one, which is the regime the gate exists for.
    jit = measure_noise(Jittery(0.06), "K", "C", repeats=4)
    assert not jit.identical and jit.max_kl > 0.0, jit.as_dict()
    assert any("non-deterministic" in r for r in jit.reasons)
    assert jit.max_prob_delta > 0.0

    # a run that returns a different NUMBER of positions is not reproducible at all
    shifty = measure_noise(ShapeShifter(), "K", "C", repeats=3)
    assert any("position count differs" in r for r in shifty.reasons), shifty.as_dict()

    # the floor scales with measured noise and never collapses to zero
    assert margin_floor(stable) > 0.0, "a perfect run must still not license a 0-margin crown"
    assert margin_floor(jit) > margin_floor(stable), (margin_floor(jit), margin_floor(stable))

    # THE GATE: a margin inside the noise band is not crownable; a clear one is
    ok, why = crownable(margin=jit.max_kl * 0.5, noise=jit)
    assert not ok and "noise floor" in why, why
    ok2, why2 = crownable(margin=jit.max_kl * 10, noise=jit)
    assert ok2, why2
    # and on a perfectly stable observer a real margin still passes
    ok3, why3 = crownable(margin=0.05, noise=stable)
    assert ok3, why3


def test_observer_epoch_end_to_end():
    """The full OBSERVER-KL epoch through the chain boundary: read commits -> draw nonce ->
    intake (economics, safety, bit budget, pinned parent, commit-reveal) -> observer drawn from
    the nonce -> shared rollouts -> per-miner scoring -> KOTH with the king re-gated -> noise
    gate -> signed record -> weights on chain.

    This is the test that says the pivot is WIRED rather than sitting beside the old path. Two
    substrates is how the predecessor ended up fixing one while shipping the other, and how our
    own overfit gate and surprise-axis selection became dead code."""
    import json as _json
    import struct
    import tempfile
    from pathlib import Path
    from eval.chain import Commitment, run_v2_observer_epoch
    from eval.economics import RegistrationLedger
    from eval.gates import TierBudget
    from eval.identity import commit_value, content_hash
    from eval.koth import Tier, Tournament
    from eval.shadow_axis_epoch import FakeChain
    from eval.signing import Ed25519Signer
    from eval.steps import Trajectory

    def make_ckpt(d, n):
        p = Path(d)
        hdr = {"w": {"dtype": "F32", "shape": [n], "data_offsets": [0, 4 * n]}}
        hb = _json.dumps(hdr).encode()
        with open(p / "model.safetensors", "wb") as f:
            f.write(struct.pack("<Q", len(hb)) + hb + b"\0" * (4 * n))
        (p / "config.json").write_text('{"hidden_size":8}')

    def d3(a, b, c):
        return {"a": a, "b": b, "c": c}

    class Obs:
        def generate(self, prompts, max_new_tokens=128):
            return ["cont cont cont"] * len(prompts)

        def distributions(self, prefix, continuation):
            if prefix.rstrip().endswith("X"):
                return [d3(0.8, 0.1, 0.1)] * 6
            if prefix.rstrip().endswith("Y"):
                return [d3(0.1, 0.1, 0.8)] * 6
            return [d3(0.34, 0.33, 0.33)] * 6

    class Step:
        def __init__(self, tok):
            self.tok = tok

        def generate(self, prompts, max_new_tokens=256):
            return [self.tok] * len(prompts)

    with tempfile.TemporaryDirectory() as da, tempfile.TemporaryDirectory() as db:
        # DIFFERENT sizes: identical bytes would content-address to the SAME model_id, which
        # is correct behaviour (an exact copy is the same artifact) but collapses the two
        # submissions into one and makes the test vacuous.
        make_ckpt(da, 4)
        make_ckpt(db, 8)
        commits, runners = [], {}
        for i, (dd, tok) in enumerate([(da, "X"), (db, "Y")]):
            h, salt = content_hash(dd), f"s{i}"
            commits.append(Commitment(f"hot{i}", f"cold{i}", "t", dd, 1.0,
                                      revealed_hash=h, salt=salt,
                                      committed_value=commit_value(h, salt)))
            runners[dd] = Step(tok)                    # hot0 matches the parent, hot1 does not

        trajs = [Trajectory(id=f"t{i}", source="glaive_r1", prefix=f"p{i}", step="ref", index=0)
                 for i in range(12)]
        tiers = [Tier("t", 10 ** 12, 1.0)]
        budgets = {"t": TierBudget(name="t", max_params=10 ** 12, max_effective_bits=32.0)}
        chain = FakeChain(commits)
        tour = Tournament(tiers, margin=0.03)

        res = run_v2_observer_epoch(
            chain, 1, trajs, Step("X"), {"kimi": Obs(), "qwen": Obs()},
            tiers, budgets, tour, RegistrationLedger(), {},
            make_safe_runner=lambda cd: runners[cd],
            signer=Ed25519Signer(seed=b"o" * 32))

        assert set(res.outcome.accepted) == {"hot0", "hot1"}, res.outcome.rejected
        assert res.outcome.observer in {"kimi", "qwen"}, res.outcome.observer
        # the miner that reproduces the parent's EFFECT wins; the one that moves the observer
        # elsewhere does not — and neither was compared on wording
        h0 = content_hash(da)
        assert tour.kings["t"].model_id == h0, (tour.kings, res.outcome.scores)
        assert res.outcome.scores[h0]["score"] > 0.9, res.outcome.scores
        assert res.outcome.scores[content_hash(db)]["score"] < 0.3
        # the validator measured and published its own noise floor
        assert res.outcome.noise and "max_kl" in res.outcome.noise, res.outcome.noise
        # chain write-back actually happened, and the record is signed
        assert chain.weights and sum(chain.weights.values()) > 0
        assert "t" in chain.kings
        assert chain.record is not None and chain.record.verify_signature()


def test_artifact_manifest():
    """A crown published a rolling content_hash and nothing else: an opaque digest with no
    locator and no file list, so nobody could fetch the king or prove the published file set was
    the scored one. For a subnet whose deliverable IS the artifact, that is the difference
    between a leaderboard and a product.

    Also closes a real hole. gates only rejects a file whose suffix is non-empty and not
    allowlisted, and identity only hashes KNOWN suffixes — so a SUFFIX-LESS file passed
    inspection AND escaped the hash entirely. Uninspected, unhashed bytes could ride along."""
    import json as _json
    import struct
    import tempfile
    from pathlib import Path
    from eval.artifact import build_manifest, verify_manifest
    from eval.identity import content_hash

    def make(d, n=4):
        p = Path(d)
        hdr = {"w": {"dtype": "F32", "shape": [n], "data_offsets": [0, 4 * n]}}
        hb = _json.dumps(hdr).encode()
        with open(p / "model.safetensors", "wb") as f:
            f.write(struct.pack("<Q", len(hb)) + hb + b"\0" * (4 * n))
        (p / "config.json").write_text('{"hidden_size":8}')

    with tempfile.TemporaryDirectory() as d:
        make(d)
        m = build_manifest(d, artifact_uri="hf://acme/bonsai-1b@abc123")
        assert m.artifact_uri.startswith("hf://"), m.artifact_uri
        assert m.root == content_hash(d), "manifest root must equal the committed identity"
        assert {f.path for f in m.files} == {"model.safetensors", "config.json"}
        assert all(f.size > 0 and len(f.sha256) == 64 for f in m.files)
        assert m.total_bytes > 0
        ok, why = verify_manifest(d, m)
        assert ok, why
        # the manifest is serialisable, so it can be published in the round record
        assert _json.loads(m.to_json())["root"] == m.root

        # ANY drift is a rejection: changed bytes...
        (Path(d) / "config.json").write_text('{"hidden_size":9}')
        ok2, why2 = verify_manifest(d, m)
        assert not ok2 and any("sha256 mismatch" in r for r in why2), why2
        (Path(d) / "config.json").write_text('{"hidden_size":8}')
        assert verify_manifest(d, m)[0]

        # ...a missing file...
        (Path(d) / "config.json").unlink()
        ok3, why3 = verify_manifest(d, m)
        assert not ok3 and any("missing file" in r for r in why3), why3
        (Path(d) / "config.json").write_text('{"hidden_size":8}')

        # ...an undeclared extra weight file...
        (Path(d) / "extra.safetensors").write_bytes(b"\x00" * 16)
        ok4, why4 = verify_manifest(d, m)
        assert not ok4 and any("undeclared file" in r for r in why4), why4
        (Path(d) / "extra.safetensors").unlink()

        # ...and THE STOWAWAY: a suffix-less file, which slips past both the gates allowlist
        # and the identity hash, so the root still matches and only the manifest catches it.
        (Path(d) / "payload").write_bytes(b"uninspected bytes")
        assert content_hash(d) == m.root, "control: the rolling hash does NOT see this file"
        ok5, why5 = verify_manifest(d, m)
        assert not ok5 and any("uninspected bytes" in r for r in why5), why5


def test_miner_submit_and_chain_adapter():
    """The two things that were missing for a miner to actually submit: somewhere to publish a
    commitment, and a validator that reads the real chain instead of FakeChain.

    Pins the bug this flow shipped with first: the state file holding the SALT was written INSIDE
    the checkpoint dir, where `.json` is hashed — so it changed the content hash after it was
    computed (breaking commit-reveal, measured) AND would have published the salt with the
    model."""
    import json as _json
    import struct
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    from eval.chain_bittensor import BittensorChainIO, build_commitment_envelope
    from eval.gates import TierBudget
    from eval.intake import intake

    def gguf(path, n=3, tt=35):
        out = bytearray(b"GGUF" + struct.pack("<I", 3)) + struct.pack("<QQ", n, 1)
        k = b"general.architecture"
        out += struct.pack("<Q", len(k)) + k + struct.pack("<I", 8)
        v = b"qwen2"
        out += struct.pack("<Q", len(v)) + v
        for i in range(n):
            nb = f"blk.{i}.attn_q.weight".encode()
            out += struct.pack("<Q", len(nb)) + nb
            out += struct.pack("<I", 2) + struct.pack("<QQ", 512, 512)
            out += struct.pack("<I", tt) + struct.pack("<Q", 0)
        Path(path).write_bytes(bytes(out))

    with tempfile.TemporaryDirectory() as base:
        d = Path(base) / "my-model"
        d.mkdir()
        gguf(d / "model.gguf")
        (d / "config.json").write_text('{"hidden_size":512}')
        r = subprocess.run([sys.executable, "-m", "miner.submit", "commit", "--ckpt", str(d),
                            "--tier", "sub4", "--uri", "hf://acme/demo@v1", "--dry-run"],
                           capture_output=True, text=True, cwd=str(Path(__file__).parent.parent))
        assert r.returncode == 0, r.stdout + r.stderr

        # the salt must NOT be inside the directory the miner publishes
        assert sorted(p.name for p in d.iterdir()) == ["config.json", "model.gguf"], \
            "submission state (with the salt) leaked into the published checkpoint dir"
        st = _json.loads((Path(base) / "my-model.ralph-submission.json").read_text())

        # and the commitment must still verify against the artifact — writing state must not
        # have perturbed the hash
        dec = intake(str(d), TierBudget(name="sub4", max_params=10 ** 12,
                                        max_effective_bits=3.0),
                     revealed_hash=st["content_hash"], salt=st["salt"],
                     committed_value=st["commit_value"])
        assert dec.accepted, dec.reasons

        # the validator adapter parses the miner's envelope, and refuses anything else. The
        # subtensor and metagraph are injected rather than wrapped: since v1 was retired this talks
        # to bittensor directly, so the seam that has to stay testable is the SDK boundary.
        class _MG:
            hotkeys = ["hkA", "hkB", "hkC"]
            coldkeys = ["ckA", "ckB", "ckC"]

        class _Sub:
            def get_commitment(self, netuid, uid):
                return {0: st["envelope"], 1: "legacy-v1-handshake",
                        2: '{"v":9,"tier":"x"}'}[uid]

            # commitments_map is preferred; give it the write heights so the ordering check runs
            class _Substrate:
                def query_map(self, module, storage_function, params):
                    def wrap(raw, blk):
                        return {"block": blk, "info": {"fields": [
                            {"Raw64": "0x" + raw.encode().hex()}]}}
                    return [("hkA", wrap(st["envelope"], 100)),
                            ("hkB", wrap("legacy-v1-handshake", 100)),
                            ("hkC", wrap('{"v":9,"tier":"x"}', 100))]
            substrate = _Substrate()

            def metagraph(self, netuid):
                return _MG()

            def get_current_block(self):
                return 1234

            def get_block_hash(self, b):
                return f"0x{b}"

        io = BittensorChainIO(subtensor=_Sub(), netuid=40,
                              fetch_dir_for=lambda hk, uri: str(d) if hk == "hkA" else "",
                              reveals={"hkA": {"content_hash": st["content_hash"],
                                               "salt": st["salt"]}})
        cs = io.read_commitments(1100, 1234)
        assert len(cs) == 1 and cs[0].hotkey == "hkA", cs

        # SEALED BEFORE THE NONCE, OR NOT SCORED. One slot holds cv, ch and salt in a single
        # write, so a miner who waits for the nonce block can see which items and observer it
        # draws, fit to that exam, and write all three together — verify_reveal passes trivially
        # because they were computed together. max_block IS the nonce block, so a commitment at
        # or after it was not sealed first. Reported by a miner, 2026-08-05.
        late = BittensorChainIO(subtensor=_Sub(), netuid=40,
                                fetch_dir_for=lambda hk, uri: str(d),
                                reveals={"hkA": {"content_hash": st["content_hash"],
                                                 "salt": st["salt"]}})
        assert late.read_commitments(0, 100) == [], "a commitment at the nonce block must not score"
        assert any("not sealed before" in w for _, w in late.skipped), late.skipped
        # ...and one written before it still scores
        early = BittensorChainIO(subtensor=_Sub(), netuid=40,
                                 fetch_dir_for=lambda hk, uri: str(d),
                                 reveals={"hkA": {"content_hash": st["content_hash"],
                                                  "salt": st["salt"]}})
        assert len(early.read_commitments(0, 101)) == 1
        assert cs[0].tier == "sub4" and cs[0].artifact_uri == "hf://acme/demo@v1"
        # a v1 handshake and an unknown version are SKIPPED with reasons, never guessed at
        assert len(io.skipped) == 2, io.skipped
        assert any("not JSON" in w for _, w in io.skipped)
        assert any("not a v2" in w for _, w in io.skipped)
        # COMMIT_ROOT MUST BIND THE COHORT. It is a digest over what miners wrote ON CHAIN, so it
        # cannot depend on whether this box happened to download their bytes — the CPU orchestrator
        # deliberately never fetches, and with the local requirement applied here every submission
        # was dropped and commit_root returned the digest of nothing for every round, attesting to
        # no cohort at all. That is the one property the value exists to provide.
        root_live = io.commit_root(1100, 1234)
        assert len(root_live) == 64

        class _Empty(_Sub):
            def get_commitment(self, netuid, uid):
                return None

            class _NoSubstrate:
                def query_map(self, module, storage_function, params):
                    return []
            substrate = _NoSubstrate()

        bare = BittensorChainIO(subtensor=_Empty(), netuid=40)
        assert bare.commit_root(1100, 1234) != root_live, \
            "commit_root is identical with and without submissions — it binds nothing"
        # and it must not depend on the fetcher, which is what made it constant
        nofetch = BittensorChainIO(subtensor=_Sub(), netuid=40, fetch_dir_for=None)
        assert nofetch.commit_root(1100, 1234) == root_live, \
            "commit_root changed when no artifact was fetched — on-chain commitments are the input"

        # WRITES ARE OFF BY DEFAULT — nothing may touch a live signer by accident
        assert io.set_weights({"hkA": 1.0}) is False
        io.set_king("sub4", "hkA", "mid")
        # the map query degrades to the per-uid path against a stub with no `substrate`, and says
        # so in `log` rather than in `skipped` — skipped is the miner-facing column
        assert [e[0] for e in io.log if e[0] != "commitments_map"] == \
            ["set_weights", "set_king"], io.log
        assert io.current_block() == 1234 and io.block_hash(7) == "0x7"
        # v2 has no on-chain king store on purpose: the crown lives in the signed record, so a
        # second source of truth that could disagree with it is not created
        assert io.get_king("ternary-4b") is None

    env = build_commitment_envelope("sub4", "cv", "hf://x@1")
    assert _json.loads(env)["v"] == 2 and len(env) < 300, env


def test_nonce_selects_items_and_record_is_rerunnable():
    """TWO properties that decide whether "one validator" is a trust assumption or a checkable
    convenience.

    1. THE OPERATOR MUST NOT CHOOSE THE EXAM. run_observer_round used to take the scored
       trajectory LIST from its caller, so the operator picked the items and no record showed it
       — the same trust a single-validator subnet has, but invisible. Selection now derives from
       commit_root+round_nonce and the indices are signed into the record.
    2. THE RECORD MUST BE SUFFICIENT TO RE-RUN. v1's auditor diverged on 37 of 40 real reports
       because the report omitted state the scorer carried. So the record freezes the parent step
       and continuation C, pins the stack, and carries the noise floor the crown was gated on."""
    from eval.observer_round import select_trajectories
    from eval.steps import Trajectory

    pool = [Trajectory(id=f"t{i}", source="glaive_r1", prefix=f"p{i}", step="s", index=0)
            for i in range(400)]

    # reproducible for an auditor, and it MOVES with the nonce
    a, ia = select_trajectories(pool, "root", "nonceA", 24)
    a2, ia2 = select_trajectories(pool, "root", "nonceA", 24)
    b, ib = select_trajectories(pool, "root", "nonceB", 24)
    assert ia == ia2 and [t.id for t in a] == [t.id for t in a2], "selection not reproducible"
    assert ia != ib, "nonce did not change which items are scored"
    assert len(ia) == len(set(ia)) == 24
    assert all(pool[i].id == t.id for i, t in zip(ia, a)), "indices do not match the selection"
    # spread across the pool, not a prefix an operator could arrange
    assert max(ia) > 200, ia

    # ---- the record, from a real round ----
    import json as _json
    import struct
    import tempfile
    from pathlib import Path
    from eval.chain import Commitment, run_v2_observer_epoch
    from eval.economics import RegistrationLedger
    from eval.gates import TierBudget
    from eval.identity import commit_value, content_hash
    from eval.koth import Tier, Tournament
    from eval.shadow_axis_epoch import FakeChain
    from eval.signing import Ed25519Signer

    def mk(d, n):
        hdr = {"w": {"dtype": "F32", "shape": [n], "data_offsets": [0, 4 * n]}}
        hb = _json.dumps(hdr).encode()
        with open(Path(d) / "model.safetensors", "wb") as f:
            f.write(struct.pack("<Q", len(hb)) + hb + b"\0" * (4 * n))
        (Path(d) / "config.json").write_text('{"hidden_size":8}')

    def d3(x, y, z):
        return {"a": x, "b": y, "c": z}

    class Obs:
        def generate(self, prompts, max_new_tokens=128):
            return ["cont cont"] * len(prompts)

        def distributions(self, prefix, continuation):
            if prefix.rstrip().endswith("X"):
                return [d3(0.8, 0.1, 0.1)] * 6
            return [d3(0.34, 0.33, 0.33)] * 6

    class Step:
        def __init__(self, tok):
            self.tok = tok

        def generate(self, prompts, max_new_tokens=256):
            return [self.tok] * len(prompts)

    with tempfile.TemporaryDirectory() as dd:
        mk(dd, 4)
        h, salt = content_hash(dd), "s0"
        commits = [Commitment("hot0", "cold0", "t", dd, 1.0, revealed_hash=h, salt=salt,
                              committed_value=commit_value(h, salt))]
        tiers = [Tier("t", 10 ** 12, 1.0)]
        chain = FakeChain(commits)
        res = run_v2_observer_epoch(
            chain, 1, pool, Step("X"), {"kimi": Obs(), "qwen": Obs()}, tiers,
            {"t": TierBudget(name="t", max_params=10 ** 12, max_effective_bits=32.0)},
            Tournament(tiers, margin=0.03), RegistrationLedger(), {},
            make_safe_runner=lambda cd: Step("X"),
            signer=Ed25519Signer(seed=b"z" * 32), n_items=20,
            corpus_spec="glaive_r1@rev=abc123|dedup=none|order=stream")

        rec = res.outcome.record
        assert rec is not None and rec.verify_signature(), "record unsigned or invalid"
        m = rec.manifest
        # the exam is pinned: which corpus, which ordering, which items, which observer
        assert m["corpus_spec"] == "glaive_r1@rev=abc123|dedup=none|order=stream"
        assert m["item_indices"] == res.outcome.item_indices and len(m["item_indices"]) == 20
        assert m["observer"] in m["observer_pool"] and len(m["observer_pool"]) == 2
        assert m["frozen_rollouts"] is True
        assert "torch" in m["versions"] or "topk" in m["versions"], m["versions"]

        # the NOISE the crown was gated on is inside the SIGNED body, and sets the tolerance
        assert rec.noise and "max_kl" in rec.noise, rec.noise
        assert rec.reproduction_tolerance <= 0.02, rec.reproduction_tolerance
        assert "noise" in rec.canonical() and "manifest" in rec.canonical()

        # every point carries the FROZEN parent step + continuation, so a re-run is a pure
        # forward pass and never has to reproduce batched greedy generation
        assert rec.points, rec.points
        for p in rec.points:
            assert p["parent_step"] and p["continuation"], p
            assert len(p["prefix_sha256"]) == 64
        # ...and tampering with the frozen text breaks the signature
        rec.points[0]["continuation"] = "tampered"
        assert not rec.verify_signature(), "frozen rollout text is not covered by the signature"


def test_rerun_audits_and_catches_a_rigged_record():
    """A record that cannot be RE-RUN is a press release. This exercises all three audit levels
    on a real round, then rigs the record four ways and requires each to be caught.

    The specific bar: the single-validator subnet we studied publishes every prompt, verdict and
    score, and is STILL unfalsifiable where it matters, because its verdicts come from an unpinned
    LLM with allow_fallbacks and no seed. An outsider can prove they summed wrong; nobody can prove
    they graded wrong. So L0 must catch a fabricated aggregate over honest measurements, and L2
    must be able to re-derive the measurements themselves."""
    import json as _json
    import struct
    import tempfile
    from copy import deepcopy
    from pathlib import Path
    from eval.chain import Commitment, run_v2_observer_epoch
    from eval.economics import RegistrationLedger
    from eval.gates import TierBudget
    from eval.identity import commit_value, content_hash
    from eval.koth import Tier, Tournament
    from eval.rerun import audit, load_record
    from eval.shadow_axis_epoch import FakeChain
    from eval.signing import Ed25519Signer
    from eval.steps import Trajectory

    pool = [Trajectory(id=f"t{i}", source="glaive_r1", prefix=f"p{i}", step="s", index=0)
            for i in range(400)]

    def d3(x, y, z):
        return {"a": x, "b": y, "c": z}

    class Obs:
        def generate(self, prompts, max_new_tokens=128):
            return ["cont cont"] * len(prompts)

        def distributions(self, prefix, continuation):
            # three response levels, so the round can contain a weak-but-LIVE challenger. A step
            # that moves the observer nowhere is inert and fails the liveness floor outright, so
            # an all-or-nothing stub cannot produce a dethrone to audit.
            tail = prefix.rstrip()[-1:]
            if tail == "X":
                return [d3(0.8, 0.1, 0.1)] * 6        # what the parent does
            if tail == "W":
                return [d3(0.55, 0.25, 0.20)] * 6     # same direction, ~a third of the effect
            return [d3(0.34, 0.33, 0.33)] * 6         # unconditioned

    class Step:
        def __init__(self, tok):
            self.tok = tok

        def generate(self, prompts, max_new_tokens=256):
            return [self.tok] * len(prompts)

    with tempfile.TemporaryDirectory() as dd:
        hdr = {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
        hb = _json.dumps(hdr).encode()
        with open(Path(dd) / "model.safetensors", "wb") as f:
            f.write(struct.pack("<Q", len(hb)) + hb + b"\0" * 16)
        (Path(dd) / "config.json").write_text('{"hidden_size":8}')
        h, salt = content_hash(dd), "s0"
        tiers = [Tier("t", 10 ** 12, 1.0)]
        spec = "glaive_r1@rev=abc123|dedup=none|order=stream"
        res = run_v2_observer_epoch(
            FakeChain([Commitment("hot0", "cold0", "t", dd, 1.0, revealed_hash=h, salt=salt,
                                  committed_value=commit_value(h, salt),
                                  artifact_uri=f"file://{dd}")]),
            1, pool, Step("X"), {"kimi": Obs(), "qwen": Obs()}, tiers,
            {"t": TierBudget(name="t", max_params=10 ** 12, max_effective_bits=32.0)},
            Tournament(tiers, margin=0.03), RegistrationLedger(), {},
            make_safe_runner=lambda cd: Step("X"),
            signer=Ed25519Signer(seed=b"z" * 32), n_items=20, corpus_spec=spec)

        from dataclasses import asdict
        rec_path = str(Path(dd) / "record.json")
        pool_path = str(Path(dd) / "pool.jsonl")
        Path(rec_path).write_text(_json.dumps(asdict(res.outcome.record)))
        with open(pool_path, "w") as fh:
            for t in pool:
                fh.write(_json.dumps(asdict(t)) + "\n")

        # a real auditor loads the model the manifest NAMES and re-runs with that. Passing the
        # name is what lets L2 refuse to certify a round it re-ran with some other model.
        obs_name = res.outcome.record.manifest["observer"]

        # L3 loads the actual checkpoint. Honest case: the record's frozen steps ARE what the
        # model emits, so re-generating reproduces them byte for byte.
        honest_runner = lambda mid, uri: Step("X")

        # ---- 1. the honest record reproduces at ALL FOUR levels
        a = audit(rec_path, pool_path, Obs(), observer_name=obs_name,
                  make_runner=honest_runner)
        assert a.exit_code == 0, [(c.level, c.name, c.status, c.detail) for c in a.checks
                                 if c.status != "PASS"]
        levels = {c.level for c in a.checks if c.status == "PASS"}
        assert levels == {"L0", "L1", "L2", "L3"}, levels

        # ---- 1b. THE FORGED-STEPS ATTACK, which every level below L3 is blind to by construction.
        #          The miner's steps are frozen into the record by the same operator who signs it,
        #          so the operator can write ideal steps and L0/L1/L2 will faithfully confirm that
        #          those strings produce those numbers. Only running the checkpoint binds the record
        #          to the model.
        forged_provenance = audit(rec_path, pool_path, Obs(), observer_name=obs_name,
                                  make_runner=lambda mid, uri: Step("W"))   # the REAL model is worse
        assert not any(c.status == "FAIL" and c.level in ("L0", "L1", "L2")
                       for c in forged_provenance.checks), \
            "L0-L2 should be blind to forged steps — that is precisely why L3 exists"
        assert any(c.level == "L3" and c.status == "FAIL"
                   for c in forged_provenance.checks), \
            [(c.name, c.detail) for c in forged_provenance.checks if c.level == "L3"]
        assert forged_provenance.exit_code == 1

        # ---- 1c. AN ARTIFACT NOBODY CAN FETCH IS NOT A PASS. If the auditor cannot obtain the
        #          bytes, that submission is unchecked, and the audit has to say INCOMPLETE rather
        #          than green.
        unavailable = audit(rec_path, pool_path, Obs(), observer_name=obs_name,
                            make_runner=lambda mid, uri: None)
        assert unavailable.exit_code == 2, unavailable.verdict
        assert any(c.level == "L3" and c.status == "SKIP" for c in unavailable.checks)

        # ---- 2. running only the cheap half must NOT report success. v1's auditor exited 0 for
        #         weeks while diverging on 37 of 40 reports; "no contradiction found" is not
        #         "verified", and the exit code has to say so.
        cheap = audit(rec_path)
        assert not cheap.failed, [c.name for c in cheap.failed]
        assert cheap.exit_code == 2 and cheap.verdict == "INCOMPLETE", cheap.verdict
        assert any(c.level == "L2" for c in cheap.skipped)

        def rigged(mutate, resign=True):
            """The adversary is the OPERATOR, who holds the signing key — so a rigged record is
            re-signed and the signature check passes. Any audit that only verifies signatures is
            therefore useless against the party it needs to constrain; that is the whole reason L0
            recomputes and L2 re-derives."""
            raw = _json.loads(Path(rec_path).read_text())
            mutate(raw)
            p2 = str(Path(dd) / "rigged.json")
            Path(p2).write_text(_json.dumps(raw))
            if resign:
                r2 = load_record(p2)
                r2.signature = r2.signer = r2.sig_scheme = ""
                r2.sign(Ed25519Signer(seed=b"z" * 32))
                Path(p2).write_text(_json.dumps(asdict(r2)))
            return audit(p2, pool_path, Obs(), observer_name=obs_name,
                         make_runner=honest_runner)

        # ---- 3. FABRICATED AGGREGATE over honest measurements. Every published KL is real; only
        #         the headline score is inflated. This is the fraud a verdict-publishing subnet
        #         can catch — and the floor, not the ceiling, of what this record supports.
        def inflate(raw):
            raw["submissions"][0]["retention"] = 0.99
            raw["submissions"][0]["retention_lb"] = 0.99
        r = rigged(inflate)
        assert not any(c.name.startswith("signature") and c.status == "FAIL" for c in r.checks), \
            "the rigged record is validly signed — signatures cannot catch the operator"
        assert any(c.name.startswith("recompute score") and c.status == "FAIL"
                   for c in r.checks), [c.name for c in r.failed]
        assert r.exit_code == 1 and r.verdict == "DIVERGED"

        # ---- 4. RIGGED MEASUREMENTS. Now the arithmetic is self-consistent — the operator edited
        #         the raw (s, d_G, d_A) AND the score that follows from them. L0 cannot see this;
        #         only recomputing the observer's distributions from the frozen text can. This is
        #         the level a judge-based subnet structurally cannot have.
        def fake_effects(raw):
            sub = raw["submissions"][0]
            key = sub["effects"][0][0]
            sub["effects"] = [[key, 0.0, 9.0, 9.0] for _ in sub["effects"]]
            sub["slices"] = {key: [1.0] * len(sub["effects"])}
            sub["retention"] = sub["retention_lb"] = 1.0
        r = rigged(fake_effects)
        assert not any(c.level == "L0" and c.status == "FAIL" for c in r.checks), \
            "L0 should be blind here — that is why L2 exists"
        assert any(c.level == "L2" and c.status == "FAIL" for c in r.checks), \
            [c.name for c in r.checks if c.level == "L2"]

        # ---- 5. THE OPERATOR CHOSE THE EXAM. Claim a different item set than the nonce implies.
        def swap_items(raw):
            raw["manifest"]["item_indices"] = list(range(20))
        r = rigged(swap_items)
        assert any(c.name.startswith("item selection") and c.status == "FAIL" for c in r.checks)

        # ---- 6. UNSIGNED. An unsigned record is self-consistent and mintable by anyone.
        r = rigged(lambda raw: raw.update(signature="", signer=""), resign=False)
        assert any(c.name.startswith("signature") and c.status == "FAIL" for c in r.checks)

        # ---- 7. A REAL DETHRONE, so the margin recompute is actually exercised rather than
        #         skipped. The dethrone margin is the number that moves emission, and it is a
        #         PAIRED statistic — recomputable only because the incumbent's re-score is
        #         published too. An unexercised check is not a check.
        def ck(name, n):
            q = Path(dd) / name
            q.mkdir()
            hb2 = _json.dumps({"w": {"dtype": "F32", "shape": [n],
                                     "data_offsets": [0, 4 * n]}}).encode()
            (q / "model.safetensors").write_bytes(struct.pack("<Q", len(hb2)) + hb2 + b"\0" * 4 * n)
            (q / "config.json").write_text('{"hidden_size":8}')
            return str(q)

        weak, strong = ck("weak", 4), ck("strong", 6)
        runners = {weak: Step("W"), strong: Step("X")}   # W moves the observer partway, X matches
        obs2 = {"kimi": Obs(), "qwen": Obs()}
        tour, ledger, reg = Tournament(tiers, margin=0.03), RegistrationLedger(), {}
        budgets = {"t": TierBudget(name="t", max_params=10 ** 12, max_effective_bits=32.0)}
        sig = Ed25519Signer(seed=b"z" * 32)

        def commit(hot, cold, cd):
            hh, ss = content_hash(cd), "s" + hot
            return Commitment(hot, cold, "t", cd, 1.0, revealed_hash=hh, salt=ss,
                              committed_value=commit_value(hh, ss),
                              artifact_uri=f"file://{cd}")

        chain = FakeChain([commit("hot1", "cold1", weak)])
        r1 = run_v2_observer_epoch(chain, 1, pool, Step("X"), obs2, tiers, budgets, tour, ledger,
                                   reg, make_safe_runner=lambda cd: runners[cd], signer=sig,
                                   n_items=20, corpus_spec=spec)
        assert any(e.get("action") == "crown" for e in r1.outcome.events), r1.outcome.events

        chain._commits = [commit("hot2", "cold2", strong)]
        r2 = run_v2_observer_epoch(chain, 2, pool, Step("X"), obs2, tiers, budgets, tour, ledger,
                                   reg, make_safe_runner=lambda cd: runners[cd], signer=sig,
                                   n_items=20, corpus_spec=spec)
        dth = [e for e in r2.outcome.events if e.get("action") == "dethrone"]
        assert dth, r2.outcome.events
        roles = {sub.role for sub in r2.outcome.record.submissions}
        assert roles == {"challenger", "incumbent"}, roles

        p2 = str(Path(dd) / "dethrone.json")
        Path(p2).write_text(_json.dumps(asdict(r2.outcome.record)))
        a2 = audit(p2, pool_path, Obs(),
                   observer_name=r2.outcome.record.manifest["observer"],
                   make_runner=lambda mid, uri: Step("X") if mid == content_hash(strong)
                   else Step("W"))
        assert a2.exit_code == 0, [(c.name, c.detail) for c in a2.checks if c.status != "PASS"]
        assert any(c.name.startswith("dethrone margin") and c.status == "PASS"
                   for c in a2.checks), [c.name for c in a2.checks]
        assert any(c.name.startswith("margin clears noise floor") and c.status == "PASS"
                   for c in a2.checks)

        # a crown claimed on a margin the round never measured
        def fake_margin(raw):
            for e in raw["events"]:
                if e.get("action") == "dethrone":
                    e["margin_lcb"] = 0.99
        raw2 = _json.loads(Path(p2).read_text())
        fake_margin(raw2)
        p3 = str(Path(dd) / "dethrone_rigged.json")
        Path(p3).write_text(_json.dumps(raw2))
        rr = load_record(p3)
        rr.signature = rr.signer = rr.sig_scheme = ""
        rr.sign(sig)
        Path(p3).write_text(_json.dumps(asdict(rr)))
        a3 = audit(p3, pool_path, Obs(),
                   observer_name=r2.outcome.record.manifest["observer"],
                   make_runner=lambda mid, uri: Step("X") if mid == content_hash(strong)
                   else Step("W"))
        assert any(c.name.startswith("dethrone margin") and c.status == "FAIL"
                   for c in a3.checks), [c.name for c in a3.failed]

        # ---- 6b. AUDITING WITH THE WRONG MODEL. The auditor picks the observer on the command
        #          line, and nothing used to check it against the round's. A green L2 then proved
        #          only "some model reproduces these numbers", which is not the claim being made.
        a_wrong = audit(rec_path, pool_path, Obs(), observer_name="some-other-model",
                        make_runner=honest_runner)
        assert any(c.name.startswith("audited with the round's observer") and c.status == "FAIL"
                   for c in a_wrong.checks)
        assert a_wrong.exit_code == 1

        # ---- 6c. THE OPERATOR PICKS THE GRADER. The observer is drawn from the nonce; a manifest
        #          naming a different one out of the pool means the choice was made by hand.
        other = [o for o in res.outcome.record.manifest["observer_pool"] if o != obs_name][0]
        r_obs = rigged(lambda raw: raw["manifest"].update(observer=other))
        assert any(c.name.startswith("observer derives from the nonce") and c.status == "FAIL"
                   for c in r_obs.checks), [c.name for c in r_obs.checks if c.level == "L2"]


def test_publisher_is_fail_closed():
    """The audit tool is useless if nothing publishes records — and v1 proves the failure is
    silent. There, scoring ran fine for weeks while the audit trail rotted: the publisher was
    fail-OPEN, the on-chain anchor went 22 days stale, and king.json's lineage regressed from 50
    entries to 6 because a cold-started process overwrote a fuller index with its own empty state.

    So the ordering is inverted (publish -> verify -> heartbeat -> set_weights) and every step
    below is a way that inversion can be defeated."""
    import json as _json
    import struct
    import tempfile
    from pathlib import Path
    from eval.chain import Commitment, run_v2_observer_epoch
    from eval.economics import RegistrationLedger
    from eval.gates import TierBudget
    from eval.identity import commit_value, content_hash
    from eval.koth import Tier, Tournament
    from dataclasses import asdict
    from eval.publish import (INDEX, LocalSink, PublishError, RecordPublisher, publish_and_gate,
                              record_name)
    from eval.shadow_axis_epoch import FakeChain
    from eval.signing import Ed25519Signer
    from eval.steps import Trajectory

    pool = [Trajectory(id=f"t{i}", source="glaive_r1", prefix=f"p{i}", step="s", index=0)
            for i in range(200)]
    d3 = lambda x, y, z: {"a": x, "b": y, "c": z}

    class Obs:
        def generate(self, prompts, max_new_tokens=128):
            return ["cont cont"] * len(prompts)

        def distributions(self, prefix, continuation):
            t = prefix.rstrip()[-1:]
            return [d3(0.8, 0.1, 0.1)] * 6 if t == "X" else \
                [d3(0.55, 0.25, 0.20)] * 6 if t == "W" else [d3(0.34, 0.33, 0.33)] * 6

    class Step:
        def __init__(self, tok):
            self.tok = tok

        def generate(self, prompts, max_new_tokens=256):
            return [self.tok] * len(prompts)

    class AnchoringChain(FakeChain):
        """ONE commitment slot, like the real thing. That constraint is the whole reason the anchor
        is a hash chain: committing a bare per-round digest would anchor only the newest round and
        leave every earlier one checkable against nothing but the operator's own index."""

        def __init__(self, commits):
            super().__init__(commits)
            self.anchors = {}
            self.slot = ""          # the single on-chain value
            self.weight_calls = 0

        def publish_record(self, record):
            from eval.publish import anchor_of
            self.slot = anchor_of(getattr(record, "prev_anchor", ""), record.sha256())
            self.anchors[record.round] = record.sha256()

        def head_anchor(self):
            return self.slot

        def record_anchors(self):
            return dict(self.anchors)

        def set_weights(self, w):
            self.weight_calls += 1
            return super().set_weights(w)

    with tempfile.TemporaryDirectory() as dd:
        def ck(name, n):
            q = Path(dd) / name
            q.mkdir()
            hb = _json.dumps({"w": {"dtype": "F32", "shape": [n],
                                    "data_offsets": [0, 4 * n]}}).encode()
            (q / "model.safetensors").write_bytes(struct.pack("<Q", len(hb)) + hb + b"\0" * 4 * n)
            (q / "config.json").write_text('{"hidden_size":8}')
            return str(q)

        weak, strong = ck("weak", 4), ck("strong", 6)
        runners = {weak: Step("W"), strong: Step("X")}
        tiers = [Tier("t", 10 ** 12, 1.0)]
        obs = {"kimi": Obs(), "qwen": Obs()}
        bud = {"t": TierBudget(name="t", max_params=10 ** 12, max_effective_bits=32.0)}
        sig = Ed25519Signer(seed=b"z" * 32)

        def cm(hot, cd):
            hh, ss = content_hash(cd), "s" + hot
            return Commitment(hot, "c" + hot, "t", cd, 1.0, revealed_hash=hh, salt=ss,
                              committed_value=commit_value(hh, ss),
                              artifact_uri=f"file://{cd}")

        def epoch(chain, rnd, cd, pub, tour, reg, led, **kw):
            chain._commits = [cm(f"hot{rnd}", cd)]
            return run_v2_observer_epoch(
                chain, rnd, pool, Step("X"), obs, tiers, bud, tour, led, reg,
                make_safe_runner=lambda c: runners[c], signer=sig, n_items=20,
                corpus_spec="glaive_r1@rev=abc|order=stream", publisher=pub, **kw)

        # ---- 1. HAPPY PATH: record is fetchable, indexed, anchored, and weights got set
        root = str(Path(dd) / "sink1")
        pubr = RecordPublisher(LocalSink(root), window=8,
                              state_path=str(Path(dd) / "hwm1.json"))
        chain = AnchoringChain([])
        tour, reg, led = Tournament(tiers, margin=0.03), {}, RegistrationLedger()
        r1 = epoch(chain, 1, weak, pubr, tour, reg, led)
        assert r1.publish.ok, r1.publish
        assert r1.weights_set and chain.weight_calls == 1
        idx = _json.loads((Path(root) / INDEX).read_text())
        assert [e["round"] for e in idx["rounds"]] == [1] and idx["head"] == 1
        assert chain.anchors[1] == r1.record_sha256
        # the published bytes are the record an auditor would re-run
        got = _json.loads((Path(root) / idx["rounds"][0]["name"]).read_text())
        assert got["manifest"]["item_indices"] == r1.outcome.item_indices

        # ---- 2. WRITE THAT SILENTLY DROPS. The sink says success and stores nothing — the
        #         object-store failure that made v1's index claim more rounds than it served.
        class BlackHole(LocalSink):
            def put(self, name, blob):
                return f"file://{name}"          # accepted, never stored

        try:
            RecordPublisher(BlackHole(str(Path(dd) / "sink2")), allow_no_state=True).publish(r1.outcome.record)
            raise AssertionError("a write that stored nothing was accepted")
        except PublishError as e:
            assert "cannot be fetched back" in str(e), e

        # ---- 3. READ-BACK MISMATCH (truncation / partial write)
        class Truncating(LocalSink):
            def get(self, name):
                b = super().get(name)
                return b[:-5] if b and name.startswith("rounds/") else b

        try:
            RecordPublisher(Truncating(str(Path(dd) / "sink3")), allow_no_state=True).publish(r1.outcome.record)
            raise AssertionError("a truncated read-back was accepted")
        except PublishError as e:
            assert "reads back different" in str(e), e

        # ---- 4. PUBLISH-THEN-DELETE. The obvious tamper: score honestly, publish, then remove
        #         the round an auditor is objecting to. Only a WINDOW re-check catches it, and the
        #         penalty has to land on the CURRENT round because that is the only leverage.
        (Path(root) / idx["rounds"][0]["name"]).unlink()
        before = chain.weight_calls
        r2 = epoch(chain, 2, strong, pubr, tour, reg, led)
        assert not r2.publish.ok, r2.publish
        assert [x["round"] for x in r2.publish.stale_rounds] == [1], r2.publish.stale_rounds
        assert not r2.weights_set and chain.weight_calls == before, "paid out on a deleted history"
        assert r2.withheld and r2.withheld["action"] == "withhold_weights", r2.withheld
        # THE WITHHOLD MUST NOT TOUCH THE SIGNED RECORD. The reason used to be appended to
        # outcome.events, which the record holds BY REFERENCE — so recording why we withheld
        # invalidated the signature of bytes that were already published, and made that round
        # permanently unrepublishable.
        assert r2.outcome.record.verify_signature(), "withholding broke the published signature"
        assert not any(e.get("action") == "withhold_weights"
                       for e in r2.outcome.record.events)
        # THE WITHHELD CROWN MUST NOT SURVIVE. tournament.consider() mutates kings in place during
        # scoring, so a bare `return` left the withheld round's king in memory and the NEXT round
        # wrote it on chain and paid it — the gate would withhold, then pay anyway.
        assert tour.kings.get("t") is None or \
            tour.kings["t"].model_id != content_hash(strong), "withheld crown survived rollback"
        before_k = dict(chain.kings)
        r3 = epoch(chain, 3, strong, pubr, tour, reg, led)
        assert not r3.weights_set and chain.weight_calls == before, "paid out after a withhold"
        assert chain.kings == before_k, "wrote the withheld crown on chain a round later"

        # ---- 5. ANCHOR MISMATCH. The chain is the authority; a published record that disagrees
        #         with what was anchored is a swap.
        root5 = str(Path(dd) / "sink5")
        p5 = RecordPublisher(LocalSink(root5), window=8, allow_no_state=True)
        p5.publish(r1.outcome.record)
        stale, checked, n_anch = p5.verify_window(1, {1: "0" * 64})
        assert checked == [1] and stale and "anchor" in stale[0]["why"], (stale, checked)
        stale, _, n_anch = p5.verify_window(1, {1: r1.outcome.record.sha256()})
        assert not stale and n_anch == 1

        # ---- 6. HISTORY REWRITE. Re-publishing a round with different content would let an
        #         operator replace the very round being challenged.
        rec_b = r1.outcome.record
        import copy as _copy
        forged = _copy.deepcopy(rec_b)
        forged.weights = {"someone_else": 1.0}
        forged.signature = forged.signer = forged.sig_scheme = ""
        forged.sign(sig)
        try:
            p5.publish(forged)
            raise AssertionError("a round was silently rewritten")
        except PublishError as e:
            # two independent guards catch this now: the signed prev_anchor no longer matches the
            # live head, and the index refuses a changed sha for an existing round. Either is a
            # correct refusal; asserting one exact message would make the test brittle about which
            # fires first.
            assert ("refusing to replace" in str(e)
                    or "refusing to graft it onto a different chain" in str(e)), e

        # ---- 7. THE V1 LINEAGE BUG ITSELF. A transient index read failure must not be read as
        #         "no history yet" — that is how 50 entries became 6.
        class FlakyIndex(LocalSink):
            blind = False

            def get(self, name):
                if name == INDEX and self.blind:
                    return None                  # read failed, index is actually still there
                return super().get(name)

        root7 = str(Path(dd) / "sink7")
        fl = FlakyIndex(root7)
        p7 = RecordPublisher(fl, state_path=str(Path(dd) / "hwm7.json"))
        p7.publish(r1.outcome.record)
        assert len(_json.loads((Path(root7) / INDEX).read_text())["rounds"]) == 1
        fl.blind = True
        try:
            p7.publish(forged)
            raise AssertionError("a failed index read was treated as an empty history")
        except PublishError as e:
            assert "v1 lineage bug" in str(e), e
        # and the high-water mark survives a restart, so a cold start cannot shrink it either
        p7b = RecordPublisher(fl, state_path=str(Path(dd) / "hwm7.json"))
        assert p7b._hwm == 1
        try:
            p7b.publish(forged)
            raise AssertionError("a cold-started process shrank the history")
        except PublishError as e:
            assert "v1 lineage bug" in str(e), e

        # ---- 8. UNSIGNED RECORDS NEVER ENTER THE HISTORY. Publishing one buys no
        #         accountability, so it is refused before it can be paid out.
        unsigned = _copy.deepcopy(r1.outcome.record)
        unsigned.signature = unsigned.signer = unsigned.sig_scheme = ""
        rep = publish_and_gate(RecordPublisher(LocalSink(str(Path(dd) / "sink8")), allow_no_state=True),
                              unsigned)
        assert not rep.ok and "unsigned" in rep.reasons[0]
        assert rep.published is None

        # ---- 9. A MISCONFIGURED PRODUCTION BOX CANNOT RUN UNAUDITED
        import os as _os
        _os.environ["RALPH_REQUIRE_PUBLISH"] = "1"
        try:
            epoch(AnchoringChain([]), 3, strong, None, Tournament(tiers, margin=0.03), {},
                  RegistrationLedger())
            raise AssertionError("ran a round with no publisher while RALPH_REQUIRE_PUBLISH=1")
        except PublishError as e:
            assert "no publisher" in str(e), e
        finally:
            _os.environ.pop("RALPH_REQUIRE_PUBLISH", None)

        # ---- 9b. THE ANCHOR IS THE ONLY CHECK THE OPERATOR CANNOT SATISFY ALONE, so a chain that
        #          does not expose it must withhold rather than degrade quietly. The first version
        #          read it with getattr(chain, "record_anchors", None); no adapter implemented it,
        #          the comparison was skipped in silence, and the heartbeat compared the operator's
        #          bytes to the operator's index.
        p9 = RecordPublisher(LocalSink(str(Path(dd) / "sink9b")),
                             state_path=str(Path(dd) / "hwm9b.json"))
        rep9 = publish_and_gate(p9, r1.outcome.record, head_anchor_fn=None)
        assert not rep9.ok and "no head_anchor" in rep9.reasons[0], rep9
        # and a chain whose anchor disagrees with the recomputed head withholds too
        rep9b = publish_and_gate(
            RecordPublisher(LocalSink(str(Path(dd) / "sink9c")),
                            state_path=str(Path(dd) / "hwm9c.json")),
            r1.outcome.record, head_anchor_fn=lambda: "f" * 64)
        assert not rep9b.ok and "does not match" in " ".join(rep9b.reasons), rep9b

        # ---- 9c. HASH CHAIN. One commitment slot has to commit to the WHOLE history, or deleting
        #          an old record breaks nothing. Deleting round 1's entry must change the head.
        from eval.publish import anchor_of, verify_history
        rootc = str(Path(dd) / "sinkc")
        pc = RecordPublisher(LocalSink(rootc), state_path=str(Path(dd) / "hwmc.json"))
        chc = AnchoringChain([])
        tc, rc, lc = Tournament(tiers, margin=0.03), {}, RegistrationLedger()
        recs = [epoch(chc, n, strong if n > 1 else weak, pc, tc, rc, lc) for n in (1, 2, 3)]
        assert all(e.publish.ok and e.weights_set for e in recs), [e.publish for e in recs]
        h = verify_history(pc, chc.record_anchors, head_anchor_fn=chc.head_anchor)
        assert h.ok and h.head_checked and not h.chain_breaks, h
        # every record's prev_anchor is the previous round's anchor — the link is SIGNED, so it
        # cannot be back-filled after the fact
        idxc = _json.loads((Path(rootc) / INDEX).read_text())["rounds"]
        for prev, cur in zip(idxc, idxc[1:]):
            assert cur["prev_anchor"] == prev["anchor"]
            assert cur["anchor"] == anchor_of(prev["anchor"], cur["sha256"])
        # excise the middle round from the index -> the recomputed head no longer matches the chain
        tampered = [e for e in idxc if e["round"] != 2]
        (Path(rootc) / INDEX).write_text(_json.dumps({"rounds": tampered,
                                                      "head": tampered[-1]["round"]}))
        h2 = verify_history(RecordPublisher(LocalSink(rootc), allow_no_state=True),
                            chc.record_anchors, head_anchor_fn=chc.head_anchor)
        assert not h2.ok and (h2.chain_breaks or h2.gaps), h2
        assert not h2.head_checked, "a truncated history still matched the on-chain head"

        # ---- 9d. ROUND ALIASING. Checking only the digest let one round's blob be pointed at from
        #          another round's index slot: the entry says round 7, the bytes are round 3's
        #          record, the sha matches those bytes, every check passed, round 7 was erased.
        rootd = str(Path(dd) / "sinkd")
        pd = RecordPublisher(LocalSink(rootd), state_path=str(Path(dd) / "hwmd.json"))
        chd = AnchoringChain([])
        td, rd, ld = Tournament(tiers, margin=0.03), {}, RegistrationLedger()
        e1 = epoch(chd, 1, weak, pd, td, rd, ld)
        e2 = epoch(chd, 2, strong, pd, td, rd, ld)
        idxd = _json.loads((Path(rootd) / INDEX).read_text())
        r2e = [e for e in idxd["rounds"] if e["round"] == 2][0]
        r1e = [e for e in idxd["rounds"] if e["round"] == 1][0]
        r2e.update(name=r1e["name"], sha256=r1e["sha256"], signature=r1e["signature"])
        (Path(rootd) / INDEX).write_text(_json.dumps(idxd))
        stale, checked, _ = RecordPublisher(LocalSink(rootd), allow_no_state=True).verify_window(2)
        assert any(x["round"] == 2 for x in stale), (stale, checked)

        # ---- 9e. RE-SIGNING A PUBLISHED RECORD. canonical() excludes the signature fields, so the
        #          digest and the anchor both survive a change of signer — attribution has to be
        #          pinned in the index or a record can be de-attributed after publication.
        roote = str(Path(dd) / "sinke")
        pe = RecordPublisher(LocalSink(roote), state_path=str(Path(dd) / "hwme.json"))
        pe.publish(r1.outcome.record)
        idxe = _json.loads((Path(roote) / INDEX).read_text())
        nm = idxe["rounds"][0]["name"]
        resigned = _copy.deepcopy(r1.outcome.record)
        resigned.signature = resigned.signer = resigned.sig_scheme = ""
        resigned.sign(Ed25519Signer(seed=b"q" * 32))          # a DIFFERENT key
        assert resigned.sha256() == r1.outcome.record.sha256(), "digest should be signature-blind"
        (Path(roote) / nm).write_text(_json.dumps(asdict(resigned)))
        stale, _, _ = RecordPublisher(LocalSink(roote), allow_no_state=True).verify_window(1)
        assert stale and "signature" in stale[0]["why"], stale

        # ---- 9f. RALPH_REQUIRE_PUBLISH IS A FLOOR, NOT A DEFAULT. It used to only fill in when
        #          require_publish was None, so one explicit require_publish=False anywhere in the
        #          call path defeated the single switch an operator sets to force the gate on.
        _os2 = __import__("os")
        _os2.environ["RALPH_REQUIRE_PUBLISH"] = "1"
        try:
            epoch(AnchoringChain([]), 5, strong, None, Tournament(tiers, margin=0.03), {},
                  RegistrationLedger(), require_publish=False)
            raise AssertionError("require_publish=False defeated RALPH_REQUIRE_PUBLISH=1")
        except PublishError:
            pass
        finally:
            _os2.environ.pop("RALPH_REQUIRE_PUBLISH", None)

        # ---- 9g. THE NEVER-SHRINK GUARD MUST NOT BE OPT-IN. state_path defaulted to "", so an
        #          ordinary construction left the high-water mark at 0 and a cold start plus one
        #          transient index read reproduced v1's lineage bug with no malice required.
        try:
            RecordPublisher(LocalSink(str(Path(dd) / "sinkf")))
            raise AssertionError("built a publisher with the never-shrink guard silently disabled")
        except PublishError as e:
            assert "state_path is required" in str(e), e

        # ---- 10. GAPS IN THE TRAIL. A round that was scored, paid out and never published has
        #          no record to audit, so no per-record check can see it. Only walking the index
        #          finds it — and that is the shape v1 had: 57 crowns in the report repo against
        #          6 lineage entries, with nothing comparing the two.
        from eval.publish import verify_history
        root_g = str(Path(dd) / "sinkg")
        pg = RecordPublisher(LocalSink(root_g), state_path=str(Path(dd) / "hwmg.json"))
        chg = AnchoringChain([])
        tg, rg, lg = Tournament(tiers, margin=0.03), {}, RegistrationLedger()
        for rnd in (1, 2, 4):                       # round 3 never published
            e = epoch(chg, rnd, strong if rnd > 1 else weak, pg, tg, rg, lg)
            assert e.publish.ok, (rnd, e.publish)
        h = verify_history(pg, chg.record_anchors)
        assert h.n_rounds == 3 and h.head == 4
        assert h.gaps == [3], h.gaps
        assert not h.broken and not h.mismatched and not h.unanchored
        assert not h.ok, "a hole in the trail must not read as complete"

        # ---- 11. RECOVERY. Once publishing works again the round pays out normally — the gate
        #          holds emission, it does not brick the subnet.
        root10 = str(Path(dd) / "sink10")
        p10 = RecordPublisher(LocalSink(root10), window=8, state_path=str(Path(dd) / "hwm10.json"))
        ch10 = AnchoringChain([])
        t10, r10, l10 = Tournament(tiers, margin=0.03), {}, RegistrationLedger()
        e10 = epoch(ch10, 4, strong, p10, t10, r10, l10)
        assert e10.publish.ok and e10.weights_set and ch10.weight_calls == 1
        assert e10.publish.anchors_checked == 1, e10.publish


def test_identity_canary_catches_a_nondeterministic_box():
    """The parent scored against itself must be EXACTLY 1.0, and a validator that cannot reproduce
    that has no business ranking anyone else.

    This gate exists because of a measured result, not a hypothetical. On an H100 PCIe the noise
    probe reported a clean floor while the parent scored 0.8754 against itself — generate() was
    nondeterministic above ~64 new tokens on that box (deterministic at 32/64, not at 128/256,
    which is a KV-length kernel heuristic switching to a split-K reduction with atomics). An A100
    and an L40S returned exactly 1.0 on the same code. So determinism is a property of the hardware
    and kernel selection, and a box can be silently unfit while every other gate passes."""
    from eval.determinism import identity_check
    from eval.observer_round import build_shared
    from eval.steps import Trajectory

    d3 = lambda x, y, z: {"a": x, "b": y, "c": z}

    class Obs:
        def generate(self, prompts, max_new_tokens=128):
            return ["cont cont"] * len(prompts)

        def distributions(self, prefix, continuation):
            return [d3(0.8, 0.1, 0.1)] * 6 if prefix.rstrip().endswith("X") \
                else [d3(0.34, 0.33, 0.33)] * 6

    class Steady:
        def generate(self, prompts, max_new_tokens=256):
            return ["X"] * len(prompts)

    class Flaky:
        """Deterministic for the shared rollout, then drifts — exactly the H100's failure shape:
        the same call with the same batch returns different text on a later invocation."""
        def __init__(self):
            self.calls = 0

        def generate(self, prompts, max_new_tokens=256):
            self.calls += 1
            if self.calls <= 1:
                return ["X"] * len(prompts)
            return ["X" if i % 4 else "W" for i in range(len(prompts))]

    trajs = [Trajectory(id=f"t{i}", source="glaive_r1", prefix=f"p{i}", step="s", index=0)
             for i in range(16)]
    obs = Obs()

    # a deterministic box: the identity is exact, to the last bit
    shared = build_shared(trajs, Steady(), obs, "obs")
    ok, info = identity_check(shared, Steady(), obs)
    assert ok, info
    assert info["score"] == 1.0, info

    # a box whose generation drifts: caught, with the shortfall quantified
    flaky = Flaky()
    shared2 = build_shared(trajs, flaky, obs, "obs")
    ok2, info2 = identity_check(shared2, flaky, obs)
    assert not ok2, info2
    assert info2["shortfall"] > 0.01, info2
    assert "disagrees with itself" in info2["verdict"], info2

    # and the round ABORTS rather than crowning on a box that fails it — annotating the record
    # would leave the crown standing, which is the wrong side to fail on
    import json as _json
    import struct
    import tempfile
    from pathlib import Path
    from eval.economics import RegistrationLedger
    from eval.gates import TierBudget
    from eval.identity import commit_value, content_hash
    from eval.koth import Tier, Tournament
    from eval.validator_observer_loop import CommittedSubmission, run_observer_round

    with tempfile.TemporaryDirectory() as dd:
        hb = _json.dumps({"w": {"dtype": "F32", "shape": [4],
                                "data_offsets": [0, 16]}}).encode()
        (Path(dd) / "model.safetensors").write_bytes(struct.pack("<Q", len(hb)) + hb + b"\0" * 16)
        (Path(dd) / "config.json").write_text('{"hidden_size":8}')
        h, salt = content_hash(dd), "s0"
        tiers = [Tier("t", 10 ** 12, 1.0)]
        out = run_observer_round(
            1, "root", "nonce",
            [CommittedSubmission("hot0", "cold0", "t", dd, 1.0, revealed_hash=h, salt=salt,
                                 committed_value=commit_value(h, salt),
                                 make_runner=lambda: Steady())],
            trajs, Flaky(), {"kimi": obs}, tiers,
            {"t": TierBudget(name="t", max_params=10 ** 12, max_effective_bits=32.0)},
            Tournament(tiers, margin=0.03), RegistrationLedger(), {}, n_items=16)
        assert any(e.get("action") == "abort" and "identity" in e.get("reason", "")
                   for e in out.events), out.events
        assert not out.weights, "crowned on a box that cannot reproduce the identity"


def test_pool_is_language_balanced_or_the_round_refuses():
    """The anti-clone axis has to actually be in the pool.

    Pinning Qwen3-8B pins the same parent PrismML compress, so the cheapest attack is concrete:
    download their Bonsai-8B, submit it, and it passes every gate — same architecture, genuinely
    ~1.71 bpw measured from the bytes, shipped as GGUF which intake accepts. No gate can catch that,
    because it IS a real low-bit compression of the real parent.

    What separates it from honest work is where it is weak. A cloned artifact arrives already broken
    on non-Latin text (measured elsewhere: Persian 79.8% -> 45.2% at 1-bit while English held
    97-100%), and worst-slice aggregation over (observer x language x depth) is what turns that into
    a losing score. But every harness built its pool from glaive_r1 alone — English — so the language
    slice existed with exactly one value in it and the aggregation was a no-op."""
    from eval.pool import DEFAULT_MIX, build_pool, check_balance, language_balance
    from eval.observer_round import _lang_of
    from eval.steps import Trajectory

    # the source->language table and the scorer's own derivation must not disagree, or the pool
    # thinks it is balanced while the crown is scored on different slices
    from eval.pool import LANG_OF_SOURCE
    for src, lang in LANG_OF_SOURCE.items():
        assert _lang_of(src) == lang, f"{src}: pool says {lang}, scorer says {_lang_of(src)}"

    def fake_loader(name, want, max_steps):
        return [Trajectory(id=f"{name}-{i}", source=name, prefix=f"{name} prefix {i}",
                           step="s", index=i % 4) for i in range(want)]

    pool, spec = build_pool(240, loader=fake_loader)
    bal = language_balance(pool)
    assert set(bal) == {"en", "hi", "zh"}, bal
    # non-Latin is a third of the pool, not a garnish — worst-slice means the smallest LIVE slice
    # decides the crown, and a token presence would be dropped by the per-slice floor
    non_latin = bal["hi"] + bal["zh"]
    assert non_latin / len(pool) > 0.30, f"non-Latin share {non_latin/len(pool):.2f} too thin"
    ok, reasons = check_balance(pool)
    assert ok, reasons

    # the corpus spec records the mix, the revision and the realised balance — an index into a pool
    # is meaningless without the ordering that produced it
    cs = spec.as_corpus_spec()
    for token in ("samvaad_hi", "zh_reasoning", "rev=main", "langs=", "n=240"):
        assert token in cs, cs

    # ---- the two failures this must refuse ----
    english_only = [t for t in pool if _lang_of(t.source) == "en"]
    ok2, why2 = check_balance(english_only)
    assert not ok2 and any("single-language" in r for r in why2), why2

    # a language present but too thin to survive the scorer's per-slice floor is worse than absent,
    # because the pool looks balanced and the slice is silently dropped
    thin = english_only + [t for t in pool if _lang_of(t.source) == "hi"][:3]
    ok3, why3 = check_balance(thin)
    assert not ok3 and any("below the" in r for r in why3), why3

    # ---- and it must actually produce multiple slices in a real round ----
    from eval.observer_round import build_shared
    d3 = lambda x, y, z: {"a": x, "b": y, "c": z}

    class Obs:
        def generate(self, prompts, max_new_tokens=128):
            return ["cont cont"] * len(prompts)

        def distributions(self, prefix, continuation):
            return [d3(0.8, 0.1, 0.1)] * 6 if prefix.rstrip().endswith("X") \
                else [d3(0.34, 0.33, 0.33)] * 6

    class Step:
        def generate(self, prompts, max_new_tokens=256):
            return ["X"] * len(prompts)

    # THE DRAW, not a prefix. build_pool concatenates by source, so any prefix of the pool is
    # single-language — the round never takes a prefix, it takes a nonce-derived sample.
    from eval.observer_round import select_trajectories

    # ARITHMETIC FIRST. Slices are (language x depth), so a 3-language pool has 6 of them, and a
    # draw smaller than 6 x the per-slice floor cannot fill them however it is stratified. That is
    # not a degradation — every slice is dropped, the score is 0.0, the identity check fails and
    # the round ABORTS. It aborted the first real publish to HF, and the identity check's own
    # diagnosis blamed nondeterministic generation, which would have sent an operator hunting a
    # GPU bug that did not exist. So the draw refuses, with the numbers.
    try:
        select_trajectories(pool, "root", "nonce-A", 24)
        raise AssertionError("a draw too small to fill its slices was accepted")
    except ValueError as e:
        assert "cannot fill 6 scoring slices" in str(e) and "at least 48" in str(e), e

    drawn, idx = select_trajectories(pool, "root", "nonce-A", 48)
    shared = build_shared(drawn, Step(), Obs(), "obs")
    slices = {s.slice_key for s in shared if s.usable}
    langs_in_play = {k.split("lang=")[1].split("|")[0] for k in slices}
    assert langs_in_play == {"en", "hi", "zh"}, langs_in_play

    # AND every language must clear the scorer's per-slice floor, or it is dropped and the axis
    # stops binding. A uniform draw of 24 from a 67%-English pool gives ~4 Hindi and ~5 Chinese —
    # both under the floor, both silently discarded, crown decided on English alone. That is
    # exactly where a cloned artifact is strong.
    import inspect
    from eval.observer_kl import score_miner
    floor = inspect.signature(score_miner).parameters["min_per_slice"].default
    # count by the FULL slice key the scorer uses — balancing languages alone left 6 thin slices
    from collections import Counter
    from eval.observer_round import _slice_key
    per_slice = Counter(_slice_key(t, "obs") for t in drawn)
    assert all(v >= floor for v in per_slice.values()), (dict(per_slice), floor)
    assert len(per_slice) == 6, dict(per_slice)

    # stratification must not cost the properties the nonce draw exists for
    a, _ = select_trajectories(pool, "root", "nonce-A", 48)
    b, _ = select_trajectories(pool, "root", "nonce-A", 48)
    c, _ = select_trajectories(pool, "root", "nonce-B", 48)
    assert [t.id for t in a] == [t.id for t in b], "draw is not reproducible"
    assert [t.id for t in a] != [t.id for t in c], "draw does not move with the nonce"
    assert idx == sorted(idx) and len(idx) == len(set(idx))


def test_density_and_model_card_are_derived_not_asserted():
    """Intelligence density is the unit this market competes on, so it must be RECOMPUTABLE.

    PrismML publish a density number from a private method. The whole difference here is that both
    halves are measured on every submission — retention from observer-KL, true bits per weight from
    the tensor data — so the record publishes the INPUTS and anyone divides them. A card that
    asserted the ratio could disagree with the round that produced it; one generated from the record
    cannot."""
    from eval.density import Density, compression_ratio, from_record, rank, size_gb
    from eval.model_card import render
    from eval.round_record import SubmissionRecord

    P = 8_190_735_360   # the pinned Qwen3-8B

    # ternary, the Bonsai operating point
    d = Density(params=P, code_bits=1.71, container_bits=2.125, retention=0.94)
    assert abs(size_gb(P, 16) - 16.38) < 0.02, size_gb(P, 16)
    assert abs(d.download_gb - 2.176) < 0.01, d.download_gb
    assert abs(d.shrink - 16 / 2.125) < 0.01
    # the unit: retention POINTS per GB downloaded, using the shipped size not the achieved one —
    # density is a claim about what you get per byte you actually store
    assert abs(d.retention_per_gb - (94.0 / d.download_gb)) < 1e-6

    # a 1-bit model in a 16-bit container is a real compression and an undeployable file: the
    # achievement and the download must not be the same number
    unpacked = Density(params=P, code_bits=1.125, container_bits=16.0, retention=0.94)
    assert unpacked.code_bits < unpacked.container_bits
    assert unpacked.retention_per_gb < d.retention_per_gb, "shipping fat must cost density"

    # derived from a published record, never stored as a ratio
    sub = SubmissionRecord(model_id="abc", miner="hot1", tier="ternary", retention=0.94,
                           retention_lb=0.94, per_point=[], gates_ok=True,
                           params=P, code_bits=1.71, container_bits=2.125)
    d2 = from_record(sub)
    assert abs(d2.retention_per_gb - d.retention_per_gb) < 1e-9
    assert "retention_per_gb" in d2.as_dict() and "params" in d2.as_dict()

    # the leaderboard PrismML cannot publish, because they have one entry
    worse = SubmissionRecord(model_id="def", miner="hot2", tier="ternary", retention=0.94,
                             retention_lb=0.94, per_point=[], gates_ok=True,
                             params=P, code_bits=4.0, container_bits=4.25)
    order = rank([worse, sub])
    assert order[0][0].model_id == "abc", "denser artifact must rank first"

    # ---- the card ----
    card = render(model_id="RalphLabsAI/Qwen3-8B-ternary", parent="Qwen/Qwen3-8B",
                  parent_params=P, tier="ternary", density=d, miner="hot1", round_no=7,
                  observer="qwen1.5b", languages={"en": 30, "hi": 12, "zh": 12},
                  record_url="https://example/record.json")

    # their structure, adopted deliberately: functional headline, three ratios, quickstart BEFORE
    # benchmarks, density section, stated limitations
    for section in ("## Quickstart", "## Model overview", "## Intelligence density",
                    "## Benchmarks", "## How this was scored", "## Limitations"):
        assert section in card, section
    assert card.index("## Quickstart") < card.index("## Benchmarks"), \
        "quickstart must precede benchmarks — you can run it before you argue about it"
    assert "architecture unchanged" in card
    assert "not a smaller model trained to imitate" in card

    # the honesty rules that must survive contact with a marketing surface
    assert "have not been run" in card, "an unmeasured benchmark table must say so out loud"
    assert "saturates on badly damaged models" in card, "the known limitation must ship with it"
    # assert the word, not the formatting — the card renders it as *not* interchangeable and a
    # literal match on the phrase breaks the moment someone adjusts the emphasis
    assert "interchangeable" in card, "retention must not be passed off as absolute capability"
    assert "Retention says nothing about how good the parent was" in card
    assert "worst-slice" in card and "hi 12" in card
    assert "python -m eval.rerun" in card, "the card must tell you how to check it"


def test_fetch_refuses_hostile_artifacts_before_downloading():
    """fetch_dir_for is the ONLY place untrusted input enters the validator.

    A miner controls the URI, the repo behind it, the file names, the count and the total size.
    None of it has been checked upstream: the on-chain commitment binds the CONTENT HASH, and a
    hash cannot be verified until after the bytes are already on the disk of a box that also holds
    a signing key. So the limits have to bind BEFORE the download, which is what plan() is for."""
    import json, os, struct, tempfile
    from pathlib import Path
    from eval.fetch import (ALLOWED_SUFFIXES, FetchRefused, MAX_FILES, MAX_TOTAL_BYTES,
                            fetch, parse_uri, plan, resolver)

    def refuses(uri, files=None, why=""):
        lister = (lambda r, v: files) if files is not None else None
        try:
            plan(uri, lister=lister)
        except FetchRefused as e:
            assert why in str(e), f"refused for the wrong reason: {e}"
            return
        raise AssertionError(f"should have refused: {uri} {files}")

    ok = [("model.safetensors", 1000), ("config.json", 40)]

    # scheme allowlist — file:// would read the validator's own disk
    refuses("file:///etc/passwd", ok, "scheme not allowed")
    refuses("ftp://x/y", ok, "scheme not allowed")
    refuses("https://evil.example/repo", ok, "not allowlisted")
    refuses("", ok, "no artifact URI")
    assert parse_uri("hf://org/model@abc123") == ("hf", "org/model", "abc123")
    assert parse_uri("hf://org/model")[2] == "main", "missing revision must default, not crash"

    # path escape is REFUSED, not sanitised — sanitising invites a bypass
    refuses("hf://a/b", [("../../etc/cron.d/x.json", 10)], "path escape")
    refuses("hf://a/b", [("/abs/path.json", 10)], "path escape")
    refuses("hf://a/b", [("we!rd;name.json", 10)], "unsafe characters")
    refuses("hf://a/b", [(("x" * 200) + ".json", 10)], "longer than")

    # size ceiling from advertised metadata
    refuses("hf://a/b", [("model.safetensors", MAX_TOTAL_BYTES + 1)], "exceeds the")
    refuses("hf://a/b", [(f"m{i}.safetensors", 1) for i in range(MAX_FILES + 1)], "exceeds the")
    refuses("hf://a/b", [("README.md", 10)], "no weight files")

    # non-weight files are skipped quietly — repos legitimately carry READMEs and .gitattributes
    kept = plan("hf://a/b", lister=lambda r, v: ok + [("README.md", 5), (".gitattributes", 1)])
    assert [n for n, _ in kept] == ["model.safetensors", "config.json"], kept

    # ---- the advertised size is a HINT, not a promise ----
    with tempfile.TemporaryDirectory() as dd:
        def liar(repo, rev):
            return [("model.safetensors", 10)]          # claims 10 bytes

        def floods(repo, rev, name, out):               # actually writes far more
            with open(out, "wb") as f:
                f.write(b"\0" * (4 * 1024 * 1024))

        try:
            # small ceiling, real behaviour: the guard is injectable precisely so it can be
            # exercised without writing 60 GB to prove a 60 GB limit
            fetch("hf://a/b", dd, lister=liar, downloader=floods, max_bytes=1024 * 1024)
            raise AssertionError("a repo that lied about its size was accepted")
        except FetchRefused as e:
            assert "stream exceeded" in str(e) or "ceiling" in str(e), e
        assert not os.listdir(dd) or all(not os.listdir(os.path.join(dd, d))
                                         for d in os.listdir(dd)), "partial download left on disk"

    # ---- hash mismatch is caught AT THE DOOR, not only at intake ----
    with tempfile.TemporaryDirectory() as dd:
        hb = json.dumps({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}).encode()

        def good(repo, rev):
            return [("model.safetensors", 100), ("config.json", 20)]

        def writes(repo, rev, name, out):
            if name.endswith(".safetensors"):
                Path(out).write_bytes(struct.pack("<Q", len(hb)) + hb + b"\0" * 16)
            else:
                Path(out).write_text('{"hidden_size":8}')

        r = fetch("hf://a/b", dd, lister=good, downloader=writes)
        assert r.files == 2 and len(r.content_hash) == 64
        try:
            fetch("hf://a/b", dd, expect_hash="f" * 64, lister=good, downloader=writes)
            raise AssertionError("bytes that do not match the commitment were accepted")
        except FetchRefused as e:
            assert "does not match the revealed" in str(e), e

        # a refusal must skip ONE miner, not take the round down
        log = []
        res = resolver(dd, reveals={"hot1": {"content_hash": "f" * 64}}, log=log)
        assert res("hot1", "file:///etc/passwd") == "", "refusal must return empty, not raise"
        assert log and log[-1][1] == "refused"


def _auditor_fixture(dd):
    """Two published rounds on a real publisher + a single-slot anchoring chain.

    Shared by the auditor tests so each one can concentrate on the POLICY rather than on rebuilding
    a trail. Returns (sink_root, chain, signer, rounds_index)."""
    import json as _json
    import struct
    from dataclasses import asdict
    from pathlib import Path
    from eval.chain import Commitment, run_v2_observer_epoch
    from eval.economics import RegistrationLedger
    from eval.gates import TierBudget
    from eval.identity import commit_value, content_hash
    from eval.koth import Tier, Tournament
    from eval.publish import LocalSink, RecordPublisher, anchor_of
    from eval.shadow_axis_epoch import FakeChain
    from eval.signing import Ed25519Signer
    from eval.steps import Trajectory

    pool = [Trajectory(id=f"t{i}", source="glaive_r1", prefix=f"p{i}", step="s", index=0)
            for i in range(200)]
    d3 = lambda x, y, z: {"a": x, "b": y, "c": z}

    class Obs:
        def generate(self, prompts, max_new_tokens=128):
            return ["cont cont"] * len(prompts)

        def distributions(self, prefix, continuation):
            t = prefix.rstrip()[-1:]
            return [d3(0.8, 0.1, 0.1)] * 6 if t == "X" else \
                [d3(0.55, 0.25, 0.20)] * 6 if t == "W" else [d3(0.34, 0.33, 0.33)] * 6

    class Step:
        def __init__(self, tok):
            self.tok = tok

        def generate(self, prompts, max_new_tokens=256):
            return [self.tok] * len(prompts)

    class AnchoringChain(FakeChain):
        def __init__(self, commits):
            super().__init__(commits)
            self.slot = ""

        def publish_record(self, record):
            self.slot = anchor_of(getattr(record, "prev_anchor", ""), record.sha256())

        def head_anchor(self):
            return self.slot

    def ck(name, n):
        q = Path(dd) / name
        q.mkdir()
        hb = _json.dumps({"w": {"dtype": "F32", "shape": [n],
                                "data_offsets": [0, 4 * n]}}).encode()
        (q / "model.safetensors").write_bytes(struct.pack("<Q", len(hb)) + hb + b"\0" * 4 * n)
        (q / "config.json").write_text('{"hidden_size":8}')
        return str(q)

    weak, strong = ck("weak", 4), ck("strong", 6)
    runners = {weak: Step("W"), strong: Step("X")}
    tiers = [Tier("t", 10 ** 12, 1.0)]
    bud = {"t": TierBudget(name="t", max_params=10 ** 12, max_effective_bits=32.0)}
    sig = Ed25519Signer(seed=b"z" * 32)
    root = str(Path(dd) / "sink")
    pubr = RecordPublisher(LocalSink(root), window=8, state_path=str(Path(dd) / "hwm.json"))
    chain = AnchoringChain([])
    tour, led, reg = Tournament(tiers, margin=0.03), RegistrationLedger(), {}

    for rnd, cd in ((1, weak), (2, strong)):
        hh, ss = content_hash(cd), f"s{rnd}"
        chain._commits = [Commitment(f"hot{rnd}", f"cold{rnd}", "t", cd, 1.0, revealed_hash=hh,
                                     salt=ss, committed_value=commit_value(hh, ss),
                                     artifact_uri=f"file://{cd}")]
        run_v2_observer_epoch(chain, rnd, pool, Step("X"), {"kimi": Obs(), "qwen": Obs()},
                              tiers, bud, tour, led, reg,
                              make_safe_runner=lambda c: runners[c], signer=sig, n_items=20,
                              corpus_spec="glaive_r1@rev=abc|order=stream", publisher=pubr)
    idx = _json.loads(LocalSink(root).get("index.json").decode())
    return root, chain, sig, idx


def test_audit_binds_emission_to_the_crowns():
    """THE WEIGHT VECTOR IS THE ONLY THING THAT MOVES TAO, and it used to be the least-checked
    field in the record.

    The old check keyed the king map off crown/dethrone events and then guarded the whole block
    with `if rec.weights and kings:`. A HOLD round — the steady state of a working subnet — has no
    crown event, so `kings` was empty, the block degraded to a non-required SKIP, and the operator
    could pay an arbitrary hotkey with every level of the audit passing. L1, L2 and L3 never look
    at `weights` or `events`, so running the expensive levels bought nothing either.

    That mattered most for AUDITORS: a validator following the trail and copying `weights` after an
    ACCEPT would have been signing a verdict attesting to a number nobody checked. So each rig
    below has to be caught at L0 — free, no corpus, no models — and the honest record has to keep
    passing."""
    import json as _json
    import tempfile
    from copy import deepcopy
    from dataclasses import asdict
    from pathlib import Path
    from eval.publish import LocalSink
    from eval.rerun import audit, record_from_blob

    ATTACKER = "5AttackerHotkeyNotInThisRoundAtAll"

    with tempfile.TemporaryDirectory() as dd:
        root, chain, sig, idx = _auditor_fixture(dd)
        sink = LocalSink(root)
        e2 = [r for r in idx["rounds"] if r["round"] == 2][0]
        base = _json.loads(sink.get(e2["name"]).decode())
        incumbent = [s for s in base["submissions"] if s["role"] == "incumbent"]
        challenger = [s for s in base["submissions"] if s["role"] == "challenger"][0]
        assert incumbent, "fixture must re-score an incumbent, or the hold case cannot be built"

        def rigged(mutate):
            """Re-signed by the operator's own key, as always: the adversary here holds it."""
            raw = deepcopy(base)
            mutate(raw)
            rec = record_from_blob(_json.dumps(raw).encode())
            rec.signature = rec.signer = rec.sig_scheme = ""
            rec.sign(sig)
            p = Path(dd) / "rigged-weights.json"
            p.write_text(_json.dumps(asdict(rec)))
            return audit(str(p))          # L0 ONLY — the cheap level has to be the one that binds

        def names(a):
            return [c.name for c in a.failed]

        # ---- the honest record still passes, or every assertion below is meaningless
        hp = Path(dd) / "honest.json"
        hp.write_text(_json.dumps(base))
        honest = audit(str(hp))
        assert not honest.failed, [(c.name, c.detail) for c in honest.failed]

        # ---- 1. HOLD ROUND PAYING A STRANGER. No crown event at all, so the old code checked
        #         nothing whatsoever. This is the steady state, which is what made it serious.
        a = rigged(lambda r: (r.update(
            events=[{"tier": "t", "round": 2, "action": "hold",
                     "king": incumbent[0]["model_id"]}],
            weights={ATTACKER: 1.0})))
        assert any("weights name exactly the kings" in n for n in names(a)), names(a)

        # ---- 2. NO EVENTS AT ALL, arbitrary weights. The `if rec.weights and kings` guard made
        #         this the quietest possible way to redirect emission.
        a = rigged(lambda r: r.update(events=[], weights={ATTACKER: 1.0}))
        assert any("weights name exactly the kings" in n for n in names(a)), names(a)

        # ---- 3. CROWNING BYTES NOBODY SCORED. The crown names a model_id that appears in no
        #         submission, so there is nothing binding the paid hotkey to anything measured.
        a = rigged(lambda r: r.update(
            events=[{"tier": "t", "round": 2, "action": "crown", "king": "de" * 32,
                     "miner": ATTACKER}],
            weights={ATTACKER: 1.0}))
        assert any("crowned model was scored" in n for n in names(a)), names(a)

        # ---- 4. RIGHT MODEL, WRONG PAYEE. The event's `miner` field is written by the operator
        #         beside the weights, so validating the weights against it would be circular — the
        #         payee is bound through the SUBMISSION, which commit-reveal ties to a hotkey.
        a = rigged(lambda r: (
            [e.update(miner=ATTACKER) for e in r["events"] if e.get("action") == "dethrone"],
            r.update(weights={ATTACKER: 1.0})))
        assert any("crown event agrees with the submission" in n for n in names(a)), names(a)

        # ---- 5. DROPPING A LEGITIMATE KING is caught too — the old subset test only looked for
        #         extra payees, so zeroing a rival's emission passed cleanly.
        a = rigged(lambda r: r.update(weights={}))
        assert any("weights" in n for n in names(a)), names(a)

        # ---- 6. AND THE ARITHMETIC: a vector that does not sum to 1 is not a normalised vector,
        #         which used to be checked only when a crown happened to occur.
        a = rigged(lambda r: r.update(weights={challenger["miner"]: 0.4}))
        assert any("weights normalised" in n for n in names(a)), names(a)


def test_auditor_verifies_then_diverges():
    """THE POINT OF AN AUDITOR. "Other validators set weights aligned with the owner validator" is
    one word away from weight-copying with extra steps: a copier sets identical weights whether the
    round reproduces or not, so it adds no safety while manufacturing the APPEARANCE of independent
    agreement. This is the other version — verify first, and when verification fails, weight the
    last round that actually checked out.

    The adversary here is the operator, holding every key: the rigged round is re-signed, its index
    entry is rewritten to match, and the on-chain anchor is committed over the rigged bytes. Every
    signature and every hash is self-consistent. Only recomputing the score catches it."""
    import io
    import json as _json
    import tempfile
    from dataclasses import asdict
    from pathlib import Path
    from eval.auditor import ACCEPT, INCOMPLETE, REJECT, Auditor, AuditorConfig
    from eval.publish import INDEX, LocalSink, anchor_of, record_name
    from eval.rerun import record_from_blob
    from eval.signing import Ed25519Signer

    with tempfile.TemporaryDirectory() as dd:
        root, chain, sig, idx = _auditor_fixture(dd)
        sink = LocalSink(root)
        pinned = sig.public_id()

        def make(work, **kw):
            cfg = AuditorConfig(expected_signer=pinned, require=("L0", "L1"),
                                work_dir=str(Path(dd) / work), **kw)
            return Auditor(cfg, sink=sink, signer=Ed25519Signer(seed=b"a" * 32),
                           head_anchor_fn=lambda: chain.head_anchor(), out=io.StringIO())

        # ---- 1. HONEST TRAIL: both rounds accepted, and the auditor follows each one's weights.
        a = make("audit-honest")
        vs = a.once()
        assert [v.round for v in vs] == [1, 2], [v.round for v in vs]
        assert all(v.verdict == ACCEPT for v in vs), [(v.round, v.verdict, v.failed) for v in vs]
        # L1 ran WITHOUT the operator handing over a corpus file: the pool the record pins was
        # fetched from the trail and re-digested against the signed manifest.
        assert vs[0].levels_run == ["L0", "L1"], vs[0].levels_run
        assert vs[1].followed_round == 2 and vs[1].weights, vs[1].weights
        round1_weights = dict(vs[0].weights)
        round2_weights = dict(vs[1].weights)
        assert round1_weights != round2_weights, "fixture must dethrone, or the test proves nothing"

        # verdicts are signed, and chained so one cannot be quietly dropped later
        assert vs[0].verify_signature() and vs[1].verify_signature()
        assert vs[1].prev == vs[0].sha256(), "verdicts must chain"
        vs[1].round = 99
        assert not vs[1].verify_signature(), "a tampered verdict must stop verifying"

        # ---- 2. RIG ROUND 2 THE WAY AN OPERATOR ACTUALLY COULD: inflate the headline score over
        #         honest per-sample measurements, re-sign, rewrite the index entry, and commit the
        #         new anchor. Nothing is inconsistent; the arithmetic is simply wrong.
        e2 = [r for r in idx["rounds"] if r["round"] == 2][0]
        raw = _json.loads(sink.get(e2["name"]).decode())
        raw["submissions"][0]["retention"] = 0.99
        raw["submissions"][0]["retention_lb"] = 0.99
        rec2 = record_from_blob(_json.dumps(raw).encode())
        rec2.signature = rec2.signer = rec2.sig_scheme = ""
        rec2.sign(sig)
        digest = rec2.sha256()
        name = record_name(2, digest)
        sink.put(name, _json.dumps(asdict(rec2), sort_keys=True, separators=(",", ":")).encode())
        e2.update(sha256=digest, name=name, signature=rec2.signature,
                  anchor=anchor_of(e2["prev_anchor"], digest))
        sink.put(INDEX, _json.dumps(idx, sort_keys=True, indent=1).encode())
        chain.slot = e2["anchor"]

        b = make("audit-rigged")
        vs = b.once()
        assert vs[0].verdict == ACCEPT and vs[1].verdict == REJECT, \
            [(v.round, v.verdict) for v in vs]
        assert any("recompute score" in name for _, name, _ in vs[1].failed), vs[1].failed
        # the signature still verifies — which is exactly why signature-checking is not auditing
        assert not any("signature" in name for _, name, _ in vs[1].failed), vs[1].failed

        # HOLD, NOT HALT, and NOT COPY. The rejected crown is not paid; the last round this
        # auditor actually verified keeps earning.
        assert vs[1].followed_round == 1, vs[1].followed_round
        assert vs[1].weights == round1_weights, (vs[1].weights, round1_weights)
        assert vs[1].weights != round2_weights, "an auditor that pays the disputed crown is a copier"

        # ---- 3. A MISSING RECORD IS NOT AN ACCUSATION UNTIL IT PERSISTS. Deleting a published
        #         record is the cheapest tamper there is and per-record checks cannot see it —
        #         there is no record left to check. But a 429 from HuggingFace is byte-for-byte the
        #         same observation, and an auditor that signs a public accusation over a timeout is
        #         one nobody listens to. The distinguishing signal is that it keeps happening.
        Path(root, [r for r in idx["rounds"] if r["round"] == 1][0]["name"]).unlink()
        c = make("audit-gap")
        vs = c.once()
        assert vs[0].verdict == INCOMPLETE and vs[0].retry, (vs[0].verdict, vs[0].retry)
        assert vs[0].trail == "INCOMPLETE", vs[0].trail
        # and it must NOT be written off: the watermark stops at the provisional round, so the
        # next pass re-audits it rather than stepping over it forever
        assert c.state["last_round"] < 1, c.state["last_round"]

        for _ in range(c.cfg.unfetchable_passes):
            vs = c.once()
        assert vs[0].trail == "TRAIL BROKEN", vs[0].trail
        assert any("stays gone" in b.get("why", "") for b in vs[0].trail_detail.get("broken", [])), \
            vs[0].trail_detail
        # once the trail IS broken, even a round whose own record reproduces is rejected
        assert all(v.verdict == REJECT for v in vs), [(v.round, v.verdict) for v in vs]


def test_auditor_verdict_states_what_it_did_not_check():
    """A verdict is worth something only if it cannot claim more than it did.

    Three ways an auditor could quietly overstate itself, each of which has to be visible in the
    signed body: running only the free level and calling it verified; requiring a level whose
    inputs it does not have and passing anyway; and — the one `rerun` structurally cannot catch —
    verifying a validly signed record without knowing WHOSE signature it should be."""
    import io
    import tempfile
    from pathlib import Path
    from eval.auditor import ACCEPT, INCOMPLETE, REJECT, Auditor, AuditorConfig
    from eval.publish import LocalSink
    from eval.signing import Ed25519Signer

    with tempfile.TemporaryDirectory() as dd:
        root, chain, sig, _ = _auditor_fixture(dd)
        sink = LocalSink(root)

        def run(work, **kw):
            cfg = AuditorConfig(work_dir=str(Path(dd) / work), **kw)
            a = Auditor(cfg, sink=sink, signer=Ed25519Signer(seed=b"a" * 32),
                        head_anchor_fn=lambda: chain.head_anchor(), out=io.StringIO())
            return a.once()

        # ---- the CHEAP auditor is legitimate, and says so. L0 needs no corpus and no GPU; it
        #      catches a fabricated aggregate over honest measurements, which is the fraud a lone
        #      operator is most able to commit. What it must not do is imply it ran more.
        vs = run("cheap", expected_signer=sig.public_id(), require=("L0",), pool_from_trail=False)
        assert all(v.verdict == ACCEPT for v in vs), [(v.round, v.failed) for v in vs]
        assert vs[0].levels_run == ["L0"], vs[0].levels_run
        assert vs[0].level_status["L1"] == "SKIP" and vs[0].level_status["L2"] == "SKIP"

        # ---- claiming a level you cannot run is INCOMPLETE, never ACCEPT — and INCOMPLETE does
        #      not advance what the auditor is willing to pay. "Follow what you did not check" is
        #      the copier again, so it is opt-in (`on_incomplete="follow"`) rather than the default.
        vs = run("strict", expected_signer=sig.public_id(), require=("L0", "L1", "L2"),
                 pool_from_trail=False)
        assert all(v.verdict == INCOMPLETE for v in vs), [(v.round, v.verdict) for v in vs]
        assert all(v.followed_round == -1 and not v.weights for v in vs)
        assert any("L2" in (v.note or "") for v in vs), [v.note for v in vs]

        # ---- AN UNPINNED SIGNER IS NOT A PASS. rerun can prove a signature is valid; it has no
        #      way to know whose it should be, so an auditor following a repo without pinning the
        #      validator is verifying whatever the repo holder signs.
        vs = run("unpinned", expected_signer="", require=("L0",), pool_from_trail=False)
        assert all(v.verdict == INCOMPLETE for v in vs), [(v.round, v.verdict) for v in vs]
        assert any("expected validator" in n for _, n, _ in vs[0].skipped), vs[0].skipped

        # ---- and a record signed by SOMEONE ELSE is a rejection, not a shrug. This is the key
        #      rotation / wrong-trail case, and it is the difference between auditing the subnet's
        #      validator and auditing a HuggingFace repo.
        vs = run("wrongkey", expected_signer=Ed25519Signer(seed=b"q" * 32).public_id(),
                 require=("L0",), pool_from_trail=False)
        assert all(v.verdict == REJECT for v in vs), [(v.round, v.verdict) for v in vs]


def test_auditor_touches_the_chain_once_per_pass_and_never_goes_silent():
    """Two ways an auditor validator quietly stops being one.

    ONE EXTRINSIC PER ROUND. Weight-setting used to sit inside the per-round commit, so a cold
    start — where `todo` is the entire published history — fired one set_weights per historical
    round in a tight loop. `weights_rate_limit` rejects everything after the first, and the daemon
    discarded the False return because state had already been written. An auditor starting at round
    80 would land round 1's weights and keep them, and two auditors starting on different days
    would hold different vectors from identical honest input.

    AND NEVER SETTING ANYTHING AT ALL. "Hold" is sound advice for the incumbent operator — its
    previous weights persist on chain — but a third-party auditor that has never set any has
    nothing to persist. Silence there is not conservatism, it is an absent validator: no
    contribution to consensus, no dividends, and eventually `activity_cutoff`."""
    import io
    import tempfile
    from pathlib import Path
    from eval.auditor import ACCEPT, Auditor, AuditorConfig
    from eval.publish import LocalSink
    from eval.signing import Ed25519Signer

    with tempfile.TemporaryDirectory() as dd:
        root, chain, sig, idx = _auditor_fixture(dd)
        wrote, burned = [], []

        def build(work, **kw):
            cfg = AuditorConfig(expected_signer=kw.pop("signer", sig.public_id()),
                                require=kw.pop("require", ("L0",)), pool_from_trail=False,
                                set_weights=True, work_dir=str(Path(dd) / work), **kw)
            return Auditor(cfg, sink=LocalSink(root), signer=Ed25519Signer(seed=b"a" * 32),
                           head_anchor_fn=lambda: chain.head_anchor(),
                           set_weights_fn=lambda w: (wrote.append(dict(w)), True)[1],
                           set_burn_fn=lambda: (burned.append(1), True)[1],
                           out=io.StringIO())

        # ---- a two-round backlog is ONE write, from the newest verified round, not two
        a = build("once-per-pass")
        vs = a.once()
        assert len(vs) == 2 and all(v.verdict == ACCEPT for v in vs)
        assert len(wrote) == 1, f"{len(wrote)} extrinsics for a 2-round backlog"
        assert wrote[0] == vs[1].weights, (wrote[0], vs[1].weights)
        assert not burned

        # ...and a pass with nothing new still writes, or the validator drifts into inactive
        wrote.clear()
        a.once()
        assert len(wrote) == 1 and not burned, (wrote, burned)

        # ---- nothing verified yet -> BURN, not silence. Pinning a signer nobody signed with
        #      makes every round REJECT, which is the cold-start-into-a-bad-round case.
        b = build("cold-reject", signer=Ed25519Signer(seed=b"q" * 32).public_id())
        wrote.clear()
        b.once()
        assert not wrote, "must not pay a crown it rejected"
        assert len(burned) == 1, f"expected exactly one burn, got {len(burned)}"


def test_an_unrevealed_submission_is_refused_not_waved_through():
    """COMMIT-REVEAL WAS FAIL-OPEN, AND UNREACHABLE FROM THE MINER'S SIDE.

    intake ran the seal check only `if revealed_hash and salt and committed_value`, so a submission
    with no reveal skipped it entirely and scored as though it had passed. And no miner could have
    revealed anyway: `submit reveal` printed the two values and told them to publish to "the
    validator's reveal endpoint", which never existed. So the gate that binds the scored bytes to
    what was sealed before the nonce existed was absent on every real submission — while the
    announcement told miners their checkpoint was sealed before the exam was drawn."""
    import json as _json, struct, tempfile
    from pathlib import Path
    from eval.gates import TierBudget
    from eval.identity import commit_value, content_hash
    from eval.intake import intake
    from eval.chain_bittensor import BittensorChainIO, build_commitment_envelope

    with tempfile.TemporaryDirectory() as dd:
        d = Path(dd) / "ck"
        d.mkdir()
        hb = _json.dumps({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}).encode()
        (d / "model.safetensors").write_bytes(struct.pack("<Q", len(hb)) + hb + b"\0" * 16)
        (d / "config.json").write_text('{"hidden_size":8}')
        h, salt = content_hash(str(d)), "s0"
        cv = commit_value(h, salt)
        bud = TierBudget(name="t", max_params=10 ** 12, max_effective_bits=32.0)

        # committed but NOT revealed -> REFUSED with a reason, never silently accepted
        dec = intake(str(d), bud, committed_value=cv)
        assert not dec.accepted and any("not revealed" in r for r in dec.reasons), dec.reasons

        # revealed -> accepted
        assert intake(str(d), bud, revealed_hash=h, salt=salt, committed_value=cv).accepted

        # and a WRONG reveal is still caught (the seal is checked, not merely present)
        bad = intake(str(d), bud, revealed_hash="f" * 64, salt=salt, committed_value=cv)
        assert not bad.accepted, bad.reasons

        # THE REVEAL TRAVELS IN THE ENVELOPE, because the commitment slot is the only channel a
        # miner has. The validator reads it back out with no injected state.
        env = _json.loads(build_commitment_envelope("t", cv, f"file://{d}"))
        env["ch"], env["salt"] = h, salt
        raw = _json.dumps(env, separators=(",", ":"), sort_keys=True)

        class _MG:
            hotkeys = ["hkA"]; coldkeys = ["ckA"]

        class _Sub:
            def metagraph(self, netuid): return _MG()
            def get_commitment(self, netuid, uid): return raw

        io = BittensorChainIO(subtensor=_Sub(), netuid=40,
                              fetch_dir_for=lambda hk, uri: str(d))
        cs = io.read_commitments(0, 100)
        assert len(cs) == 1 and cs[0].revealed_hash == h and cs[0].salt == salt, cs
        assert intake(str(d), bud, revealed_hash=cs[0].revealed_hash, salt=cs[0].salt,
                      committed_value=cs[0].committed_value).accepted


def test_the_throne_is_inherited_from_the_published_trail():
    """A KING NOBODY HAS TO BEAT IS NOT A KING.

    `Tournament` was built fresh every round and nothing rebuilt its kings, so koth took the
    open-throne branch every time and crowned max(retention) outright. The dethrone margin, the
    paired bootstrap, axis_regression and the anti-copy guarantee were unreachable code — the
    subnet documented a king of the hill and ran a leaderboard that reset hourly.

    The lineage is DERIVED from the published trail rather than stored: the records are signed,
    hash-chained and anchored, so an outsider can rebuild the same throne, and there is no local
    file the operator could edit. Both the scorer and `audit_emission` call the same walk, because
    two derivations of who is king means every honest round fails its own emission audit."""
    import json as _json
    import tempfile
    from pathlib import Path
    from eval.lineage import apply_events, replay, replay_from_trail
    from eval.publish import LocalSink, PublishError, RecordPublisher
    from eval.rerun import record_from_blob

    with tempfile.TemporaryDirectory() as dd:
        root, chain, sig, idx = _auditor_fixture(dd)
        sink = LocalSink(root)
        recs = [record_from_blob(sink.get(e["name"]))
                for e in sorted(idx["rounds"], key=lambda r: r["round"])]

        # the fixture crowns in round 1 and dethrones in round 2 — replaying must land on the
        # round-2 winner, bound to the miner through the SUBMISSION rather than the event's own
        # `miner` field (which the operator writes beside the weights, so trusting it is circular)
        kings = replay(recs)
        assert list(kings) == ["t"], kings
        r2 = [s_ for s_ in recs[1].submissions if s_.role == "challenger"][0]
        assert kings["t"].model_id == r2.model_id, (kings["t"].model_id, r2.model_id)
        assert kings["t"].miner == r2.miner
        # ...and it carries a locator, or the incumbent cannot be refetched and re-scored
        assert kings["t"].artifact_uri, kings["t"]

        # ORDER IS NOT OPTIONAL: replaying a dethrone before the crown it beats gives a different
        # throne, so the walk sorts rather than trusting the operator-written index order
        assert replay(list(reversed(recs)))["t"].model_id == kings["t"].model_id

        # VACATE empties the throne
        vac = apply_events(kings, [{"tier": "t", "action": "vacate", "king": kings["t"].model_id}])
        assert vac == {}, vac

        # a HOLD keeps the same king and counts the reign
        held = apply_events(kings, [{"tier": "t", "action": "hold", "king": kings["t"].model_id}])
        assert held["t"].model_id == kings["t"].model_id and held["t"].reign == 1

        # AND THE SAME WALK BACKS THE AUDIT. audit_emission reconstructs through apply_events, so
        # the scorer and the auditor cannot disagree about who holds the throne.
        from eval.rerun import audit_emission, Audit
        a = Audit()
        audit_emission(recs[1], a)
        assert not a.failed, [(c.name, c.detail) for c in a.failed]

        # A PARTIAL HISTORY MUST ABORT, NOT GUESS. Skipping an unfetchable round would rebuild the
        # throne from an incomplete lineage — most likely handing the crown back to whoever held it
        # before the gap, which is a live emission change caused by a network error.
        pub = RecordPublisher(sink, state_path=str(Path(dd) / "lin-hwm.json"))
        Path(root, idx["rounds"][0]["name"]).unlink()
        try:
            replay_from_trail(pub)
            raise AssertionError("a missing round must abort the lineage, not be skipped")
        except RuntimeError as e:
            assert "cannot be rebuilt" in str(e), e


def test_gguf_is_loadable_because_nothing_else_can_pass_the_tiers():
    """GGUF is not one option among several — it is the ONLY artifact that can clear the bit tiers.

    `measure_checkpoint` credits integer dtypes their full container width, so an int8 safetensors
    measures 8.0/8.0 against sub4's 4.0/5.0 and clears nothing; a genuinely 4-bit-packed
    safetensors would carry half the element count and fail the pinned-parent gate instead. So
    until the loader could open a GGUF, the subnet accepted a format it could not score and had no
    submittable format at all."""
    import json as _json, struct, tempfile
    from pathlib import Path
    from eval.runners import GGUFStudentRunner, student_runner

    with tempfile.TemporaryDirectory() as dd:
        d = Path(dd) / "sub"
        d.mkdir()
        (d / "model.gguf").write_bytes(b"GGUF" + b"\0" * 32)
        (d / "config.json").write_text('{"model_type":"qwen3"}')

        seen = {}

        def fake_llama(path):
            seen["path"] = path

            def call(prompt, **kw):
                seen["kw"] = kw
                return {"choices": [{"text": f"step<{prompt[-2:]}>"}]}

            return call

        # dispatch picks the GGUF loader for a gguf-only artifact...
        r = student_runner(str(d), backend=fake_llama)
        assert isinstance(r, GGUFStudentRunner), type(r)
        got = r.generate(["aX", "bY"], max_new_tokens=16)
        assert got == ["step<aX>", "step<bY>"], got

        # ...GREEDY ONLY. A sampled student is not reproducible, and a round that cannot be
        # re-run bit-exactly cannot be audited, which is the entire product.
        assert seen["kw"]["temperature"] == 0.0 and seen["kw"]["top_k"] == 1, seen["kw"]
        assert seen["kw"]["max_tokens"] == 16

        # two gguf files is a REFUSAL, never a guess: picking one would silently score something
        # other than what was committed
        (d / "second.gguf").write_bytes(b"GGUF" + b"\0" * 32)
        try:
            student_runner(str(d), backend=fake_llama)
            raise AssertionError("two gguf files must be refused, not guessed between")
        except ValueError as e:
            assert "expected exactly one" in str(e), e


def test_one_bad_artifact_cannot_take_the_round_down():
    """A SUBMISSION THAT PASSES THE GATES AND THEN FAILS TO LOAD IS A REJECTION, NOT AN OUTAGE.

    The runner constructor is the first code that touches miner-controlled bytes as a MODEL rather
    than as a file, and it was called bare inside the intake loop. So one artifact that cleared all
    six gates and then raised unwound the entire round: the GPU rented, the parent downloaded,
    nothing scored for anybody. Any registered miner could mount that for the cost of one
    commitment, indefinitely — and it was not hypothetical, because a GGUF passes every gate while
    SafeStudentRunner has no GGUF path at all."""
    from eval.koth import Tier, Tournament
    from eval.economics import RegistrationLedger
    from eval.gates import TierBudget
    from eval.identity import commit_value, content_hash
    from eval.validator_observer_loop import CommittedSubmission, run_observer_round
    from eval.steps import Trajectory
    import json as _json, struct, tempfile
    from pathlib import Path

    pool = [Trajectory(id=f"t{i}", source="glaive_r1", prefix=f"p{i}", step="s", index=0)
            for i in range(200)]
    d3 = lambda x, y, z: {"a": x, "b": y, "c": z}

    class Obs:
        def generate(self, prompts, max_new_tokens=128):
            return ["cont cont"] * len(prompts)

        def distributions(self, prefix, continuation):
            t = prefix.rstrip()[-1:]
            return [d3(0.8, 0.1, 0.1)] * 6 if t == "X" else [d3(0.34, 0.33, 0.33)] * 6

    class Step:
        def generate(self, prompts, max_new_tokens=256):
            return ["X"] * len(prompts)

    with tempfile.TemporaryDirectory() as dd:
        def ck(name, n):
            q = Path(dd) / name
            q.mkdir()
            hb = _json.dumps({"w": {"dtype": "F32", "shape": [n],
                                    "data_offsets": [0, 4 * n]}}).encode()
            (q / "model.safetensors").write_bytes(struct.pack("<Q", len(hb)) + hb + b"\0" * 4 * n)
            (q / "config.json").write_text('{"hidden_size":8}')
            return str(q)

        good, poison = ck("good", 4), ck("poison", 6)
        tiers = [Tier("t", 10 ** 12, 1.0)]
        bud = {"t": TierBudget(name="t", max_params=10 ** 12, max_effective_bits=32.0)}

        def cm(hot, cd, runner):
            hh, ss = content_hash(cd), "s" + hot
            return CommittedSubmission(hotkey=hot, coldkey="c" + hot, tier="t", ckpt_dir=cd,
                                       declared_compute_h100h=1.0, bond_posted=1.0,
                                       make_runner=runner, revealed_hash=hh, salt=ss,
                                       committed_value=commit_value(hh, ss),
                                       artifact_uri=f"file://{cd}")

        def boom():
            raise ValueError("GGUF is not loadable by this runner")

        # A TIER THIS ROUND DOES NOT RUN IS A REJECTION, NOT A CRASH. `binary` is a real tier; a
        # miner naming one outside this round's config hit a bare dict lookup and KeyError'd the
        # whole round. Verified against a live round: it killed a round with five submissions.
        wrong_tier = cm("wrongtier", good, lambda: Step())
        wrong_tier.tier = "binary"

        out = run_observer_round(
            1, "root", "nonce",
            [cm("attacker", poison, boom), wrong_tier, cm("honest", good, lambda: Step())],
            pool, Step(), {"kimi": Obs(), "qwen": Obs()}, tiers, bud,
            Tournament(tiers, margin=0.03), RegistrationLedger(), {}, n_items=20,
            corpus_spec="glaive_r1@rev=abc|order=stream")

        # the poison artifact is REJECTED WITH A REASON, and the honest miner is still scored
        assert "attacker" not in out.accepted, out.accepted
        assert "honest" in out.accepted, (out.accepted, out.rejected)
        why = dict((h, r) for h, r in out.rejected).get("attacker", [])
        assert any("could not be loaded" in x for x in why), why
        wt = dict((h, r) for h, r in out.rejected).get("wrongtier", [])
        assert any("not being scored this round" in x for x in wt), wt
        assert out.record is not None, "the round must still produce a record"


def test_orchestrator_audits_its_own_scorer_before_signing():
    """The validator is split so the SIGNING KEY never reaches the rented GPU. That only helps if
    the orchestrator refuses to rubber-stamp what comes back.

    A signature applied to whatever the box returned would launder it: a compromised or merely
    buggy rental could hand back a record for a different round, a pruned exam, an invented crown,
    or numbers from the wrong hardware — and our key would make any of it authoritative. So the
    orchestrator runs the SAME L0+L1 an outsider runs, against its own scorer, and checks that the
    round's identity is the one it issued rather than one the box chose."""
    import io
    import json as _json
    import tempfile
    from pathlib import Path
    from eval.orchestrator import (GpuSpec, Instance, RoundPlan, RemoteRoundError,
                                   run_remote_round, verify_returned_record)
    from eval.publish import LocalSink
    from eval.rerun import load_pool, record_from_blob

    with tempfile.TemporaryDirectory() as dd:
        root, chain, sig, idx = _auditor_fixture(dd)
        sink = LocalSink(root)
        e2 = [r for r in idx["rounds"] if r["round"] == 2][0]
        honest = _json.loads(sink.get(e2["name"]).decode())
        pool_sha = honest["manifest"]["pool_sha256"]
        pool_blob = sink.get(f"pool/{pool_sha}.jsonl")
        pool_path = Path(dd) / "pool.jsonl"
        pool_path.write_bytes(pool_blob)
        pool = load_pool(str(pool_path))

        GPU = "NVIDIA H100 PCIe"
        honest["manifest"].setdefault("versions", {})["gpu"] = GPU
        # what the rented box actually returns: UNSIGNED. The key never went there.
        honest["signature"] = honest["signer"] = honest["sig_scheme"] = ""
        plan = RoundPlan(round=honest["round"], commit_root=honest["commit_root"],
                         round_nonce=honest["round_nonce"],
                         prev_anchor=honest.get("prev_anchor", ""))
        spec = GpuSpec(require_gpu=GPU)

        def check(mutate=None):
            raw = _json.loads(_json.dumps(honest))
            if mutate:
                mutate(raw)
            rec = record_from_blob(_json.dumps(raw).encode())
            return verify_returned_record(rec, plan, pool, spec, {"gpu": GPU})

        # the honest return signs
        assert check() == [], check()
        # ...and a SIGNED return is itself a finding: the key is supposed to live only here
        bad = check(lambda r: r.update({"signature": "de" * 32, "signer": "ab" * 32,
                                        "sig_scheme": "ed25519"}))
        assert any("key leaked" in b for b in bad), bad

        # ---- 1. A RECORD FOR A DIFFERENT ROUND. The box does not get to choose the nonce; a
        #         box that could would be able to grind it.
        for f, v in (("round", 999), ("commit_root", "ff" * 32), ("round_nonce", "ab" * 32),
                     ("prev_anchor", "cd" * 32)):
            bad = check(lambda r, f=f, v=v: r.update({f: v}))
            assert any(f in b for b in bad), (f, bad)

        # ---- 2. THE WRONG HARDWARE. Cross-box spread is ~0.03 retention against a 0.05 dethrone
        #         margin, so "whatever GPU was cheapest" silently makes the crown a function of
        #         the spot market.
        bad = check(lambda r: r["manifest"]["versions"].update({"gpu": "NVIDIA L40S"}))
        assert any("not comparable" in b for b in bad), bad

        # ---- 3. A NONDETERMINISTIC BOX. The parent scored against itself is 1.000 by
        #         construction; anything else means every number from that box is suspect.
        bad = check(lambda r: r["manifest"]["identity"].update({"score": 0.8754}))
        assert any("deterministic" in b for b in bad), bad

        # ---- 4. AND A RIGGED SCORE still has to clear L0 — the audit runs against our OWN scorer,
        #         not only against strangers.
        bad = check(lambda r: r["submissions"][0].update({"retention": 0.99,
                                                          "retention_lb": 0.99}))
        assert any("recompute score" in b for b in bad), bad

        # ---- 5. THE INSTANCE IS DESTROYED EVEN WHEN THE ROUND FAILS. A leaked GPU bills until
        #         somebody notices, and the run that leaked it is the one that already went wrong.
        events = []

        class _Provider:
            def rent(self, spec, name, exclude=()):
                # `exclude` carries the (cloud, region) pairs already tried this round: a region
                # can fail to deliver a box at all, and without it the retry picks the same one.
                events.append("rent")
                return Instance(id="i-1", ip="10.0.0.1", price_per_hour=2.0)

            def wait_ready(self, inst, timeout_s=900, out=None):
                # `out` is part of the Provider protocol: the real wait_ready blocks for up to
                # fifteen minutes with the meter running and has to say so as it goes.
                events.append("ready")
                return inst

            def destroy(self, inst):
                events.append("destroy")

        def _boom(*a, **k):
            raise RemoteRoundError("scoring blew up")

        try:
            run_remote_round(plan, _Provider(), spec, str(Path(dd) / "w"),
                             runner=_boom, out=io.StringIO())
            raise AssertionError("a failed round must not return quietly")
        except RemoteRoundError:
            pass
        assert events == ["rent", "ready", "destroy"], events

        # ---- 5b. AND TEARDOWN IS VERIFIED, NOT ASSUMED. The first version of destroy() used the
        #          HTTP verb the endpoint name suggests; the API wants POST and answers DELETE with
        #          a 405, so teardown would have failed on every real round and left an H100
        #          billing. A write whose return value you trust is a write you have not verified —
        #          the same lesson publish.py learned about `put`.
        from eval.orchestrator import ShadeformProvider

        class _Stubborn(ShadeformProvider):
            DESTROY_VERIFY_S = 0.05  # the real budget is 120s; don't sleep it in a test

            def __init__(self):
                self.calls = []

            def _key(self):
                return "k"

            def _api(self, method, path, body=None):
                self.calls.append((method, path))
                if path == "/instances":
                    return {"instances": [{"id": "i-9", "name": "ralph-x", "status": "active"}]}
                return {}

        sp = _Stubborn()
        try:
            sp.destroy(Instance(id="i-9"))
            raise AssertionError("a delete that did not take must raise, not return quietly")
        except RuntimeError as e:
            assert "STILL ACTIVE" in str(e), e
        assert all(m == "POST" for m, p_ in sp.calls if p_.endswith("/delete")), sp.calls

        # ---- 5c. BUT A SLOW LIST IS NOT A LEAK. Deletion is accepted long before it is visible:
        #          killing the idle keepalive box POSTed fine, the box really went, and destroy()
        #          still raised STILL ACTIVE because it gave the listing ~9s to catch up. A false
        #          leak alarm fires inside teardown, makes a round that succeeded look failed, and
        #          sends whoever reads it into the console to hand-delete "the leak" — next to
        #          instances that are not ours. Patience and retries are separate budgets now.
        class _SlowList(ShadeformProvider):
            """Reports the instance as present for the first few listings, then drops it."""

            DESTROY_VERIFY_S = 30.0

            def __init__(self):
                self.lists, self.deletes = 0, 0

            def _key(self):
                return "k"

            def _api(self, method, path, body=None):
                if path == "/instances":
                    self.lists += 1
                    if self.lists <= 4:
                        return {"instances": [{"id": "i-9", "name": "ralph-x",
                                               "status": "active"}]}
                    return {"instances": []}
                if path.endswith("/delete"):
                    self.deletes += 1
                return {}

        sp2 = _SlowList()
        sp2.destroy(Instance(id="i-9"))  # must NOT raise
        assert sp2.lists > 4, "destroy gave up before the listing caught up"
        assert sp2.deletes <= _SlowList.DESTROY_POSTS, (
            f"destroy kept POSTing while merely waiting ({sp2.deletes} deletes)")

        # ---- 6. AND NO SECRET EVER REACHES THE JOB SPEC, asserted rather than assumed: the spec
        #         is written to disk and copied to somebody else's machine.
        # the guard checks for the VALUES this process holds, not for field names. The first
        # version matched the word "coldkey" — which every real spec carries as a miner's PUBLIC
        # ss58 — so it blocked round 1 outright while protecting nothing.
        import os as _o
        _o.environ["RALPH_RECORD_SEED"] = "de" * 32
        leaky = RoundPlan(round=1, commit_root="a", round_nonce="b", prev_anchor="",
                          committed=[{"hotkey": "h", "note": "de" * 32}])
        try:
            run_remote_round(leaky, _Provider(), spec, str(Path(dd) / "w2"),
                             runner=_boom, out=io.StringIO())
            raise AssertionError("a job spec carrying a seed must be refused")
        except RemoteRoundError as e:
            assert "refusing to ship" in str(e), e
        finally:
            _o.environ.pop("RALPH_RECORD_SEED", None)

        # ...and a spec carrying an ordinary public coldkey is NOT refused
        fine = RoundPlan(round=1, commit_root="a", round_nonce="b", prev_anchor="",
                         committed=[{"hotkey": "h", "coldkey": "5Fabc"}])
        try:
            run_remote_round(fine, _Provider(), spec, str(Path(dd) / "w3"),
                             runner=_boom, out=io.StringIO())
        except RemoteRoundError as e:
            assert "refusing to ship" not in str(e), f"public coldkey must not trip the guard: {e}"


def test_auditor_publishes_its_verdicts_or_stops_voting():
    """"The verdict is the product" is a claim about PUBLISHING, and until this existed it was
    false: rulings went to a local directory, so nobody could read them, the `prev` chain protected
    nothing, and an auditor setting weights was on-chain indistinguishable from a copier.

    Three properties, each the mirror of one the operator's publisher has — but note where they
    DIVERGE, because the adversary is different. A record may never be replaced; a verdict may
    legitimately be revised, since an auditor that could not fetch a record rules INCOMPLETE and
    must be able to rule again when the sink recovers. Refusing that would force it to lie the
    first time or stay silent. So a revision APPENDS and both rulings stay readable."""
    import io
    import json as _json
    import tempfile
    from pathlib import Path
    from eval.auditor import ACCEPT, Auditor, AuditorConfig
    from eval.publish import LocalSink, PublishError
    from eval.signing import Ed25519Signer
    from eval.verdicts import VerdictPublisher, verify_verdict_trail

    with tempfile.TemporaryDirectory() as dd:
        root, chain, sig, idx = _auditor_fixture(dd)
        committed = []
        vroot = str(Path(dd) / "verdict-trail")

        def build(work, vsink, **kw):
            cfg = AuditorConfig(expected_signer=sig.public_id(), require=("L0",),
                                pool_from_trail=False, work_dir=str(Path(dd) / work), **kw)
            return Auditor(cfg, sink=LocalSink(root), signer=Ed25519Signer(seed=b"a" * 32),
                           head_anchor_fn=lambda: chain.head_anchor(), verdict_sink=vsink,
                           commit_head_fn=committed.append, out=io.StringIO())

        # ---- 1. verdicts reach a sink a third party can read, and the chain is anchored
        a = build("pub", LocalSink(vroot))
        vs = a.once()
        assert all(v.verdict == ACCEPT for v in vs)
        pub = VerdictPublisher(LocalSink(vroot))
        published = pub.load_index()["verdicts"]
        assert [e["round"] for e in published] == [1, 2], published
        assert published[0]["sha256"] == vs[0].sha256()
        # the chain head was written to the auditor's own commitment slot -- without that, the
        # chain is the auditor's computation over the auditor's files against the auditor's index
        assert committed and committed[-1] == published[-1]["chain"], committed

        # a third party can walk it, and says so honestly when it has no on-chain head to check
        rep = verify_verdict_trail(pub)
        assert not rep["broken"] and not rep["chain_breaks"], rep
        assert rep["ok"] is False and rep["head_checked"] is False, "unanchored must not read ok"
        rep = verify_verdict_trail(pub, head_chain=committed[-1])
        assert rep["ok"] and rep["n"] == 2, rep

        # ---- 2. DELETING A VERDICT BREAKS THE CHAIN. That is the whole point of `prev`: an
        #         auditor must not be able to quietly drop the ruling it later regrets.
        Path(vroot, published[0]["name"]).unlink()
        rep = verify_verdict_trail(pub, head_chain=committed[-1])
        assert rep["broken"] and not rep["ok"], rep

        # ---- 3. A FAILED PUBLISH STOPS THE VOTE. A validator writing weights with no published
        #         reasoning is exactly the copier this role exists to be distinguishable from.
        class Broken(LocalSink):
            def put(self, name, blob):
                raise PublishError("sink is down")

        wrote = []
        b = build("hold", Broken(str(Path(dd) / "broken")), set_weights=True)
        b.set_weights_fn = lambda w: (wrote.append(w), True)[1]
        b.set_burn_fn = lambda: (wrote.append("burn"), True)[1]
        b.once()
        assert b.publish_error, "a failed publish must be recorded, not swallowed"
        assert not wrote, "must not vote on a ruling it could not publish"

        # ---- 4. AND A REVISION APPENDS. Same round, different ruling, both readable — unlike a
        #         round record, which may never be replaced.
        pub2 = VerdictPublisher(LocalSink(vroot))
        revised = vs[0]
        revised.note = "re-ruled after the sink recovered"
        revised.signature = ""
        revised.sign(Ed25519Signer(seed=b"a" * 32))
        pub2.publish(revised)
        rounds = [e["round"] for e in pub2.load_index()["verdicts"]]
        assert rounds == [1, 2, 1], f"a revision must append, not overwrite: {rounds}"

        # ...but a DIFFERENT auditor cannot write into this trail, or nobody could attribute it
        alien = vs[1]
        alien.signature = ""
        alien.sign(Ed25519Signer(seed=b"z" * 32))
        try:
            pub2.publish(alien)
            raise AssertionError("a second identity must not share one verdict trail")
        except PublishError as e:
            assert "separate repo per auditor" in str(e), e


def test_auditor_treats_silence_as_a_finding():
    """v1's failure was not a wrong number. The validator stopped publishing on 7 July and kept
    setting weights; king.json went stale, events.json was never published at all, and nothing
    alarmed for four days. An auditor that only ever reacts to NEW records is blind to exactly
    that, because the symptom is the absence of one."""
    import io
    import tempfile
    from pathlib import Path
    from eval.auditor import ACCEPT, STALE, Auditor, AuditorConfig
    from eval.publish import LocalSink
    from eval.signing import Ed25519Signer

    with tempfile.TemporaryDirectory() as dd:
        root, chain, sig, _ = _auditor_fixture(dd)
        clock = [1000.0]
        cfg = AuditorConfig(expected_signer=sig.public_id(), require=("L0",),
                            pool_from_trail=False, stale_after_s=3600.0,
                            work_dir=str(Path(dd) / "audit"))
        a = Auditor(cfg, sink=LocalSink(root), signer=Ed25519Signer(seed=b"a" * 32),
                    head_anchor_fn=lambda: chain.head_anchor(), out=io.StringIO(),
                    now_fn=lambda: clock[0])

        assert all(v.verdict == ACCEPT for v in a.once())
        clock[0] += 1800.0                      # half an hour later: quiet, but not yet late
        assert a.once() == []
        clock[0] += 7200.0                      # now the trail has been silent past the threshold
        vs = a.once()
        assert len(vs) == 1 and vs[0].verdict == STALE, [(v.round, v.verdict) for v in vs]
        assert "emission continues" in vs[0].note
        # one verdict per stale EPISODE, not one per pass — an alarm that repeats every interval
        # is an alarm operators filter, and the condition is already on the record
        clock[0] += 7200.0
        assert a.once() == [], "staleness must not re-alarm every pass"
        # a stale verdict still carries the weights it last verified — the finding is that the
        # trail went quiet, not that the last good crown became invalid
        assert vs[0].followed_round == 2 and vs[0].weights
        assert vs[0].verify_signature()


def _isolate_env() -> None:
    """Clear the production RALPH_* switches before running anything.

    Found by deploying: on the orchestrator box, where `RALPH_REQUIRE_PUBLISH=1` is correctly set,
    this suite reported 53/57 and looked broken. It was not — the gate was doing its job, refusing
    four epochs that construct no publisher. But a suite whose result depends on the operator's
    shell is a suite you cannot trust on the machine you actually deploy to, which is the one
    machine where you most want to run it. The two tests that exercise the env var set and unset it
    themselves, so isolating here does not weaken them."""
    import os as _o
    for k in [k for k in _o.environ if k.startswith("RALPH_")]:
        _o.environ.pop(k, None)


_isolate_env()          # also covers `pytest tests/test_crown_path.py`, which never calls main()


def test_a_sparse_field_cannot_move_an_already_published_digest():
    """THE TRAP IN ADDING `rejected`. `canonical()` covers every dataclass field, so a new field
    defaulting to `[]` puts `"rejected":[]` into the canonical form of records published BEFORE it
    existed. Those records are re-loaded and re-digested by `verify_window` on EVERY publish, so
    round 1 — already anchored — would have gone stale and the gate would have withheld round 2.

    An absent field and an empty one are the same statement; they must produce the same bytes."""
    from eval.round_record import RoundRecord

    def _rec(**kw):
        d = dict(round=1, commit_root="c", round_nonce="n", teacher="t", judge="j", base="b",
                 pile_id="p", points=[], submissions=[], events=[], weights={})
        d.update(kw)
        return RoundRecord(**d)

    empty = _rec()
    assert '"rejected"' not in empty.canonical(), \
        "an empty sparse field entered the signed payload — every published record just moved"

    # the digest a pre-`rejected` record would have had: canonical() minus the sparse key
    import json as _j
    from dataclasses import asdict
    legacy = {k: v for k, v in asdict(empty).items()
              if k not in RoundRecord._SIG_FIELDS and k != "rejected"}
    assert empty.canonical() == _j.dumps(legacy, sort_keys=True, separators=(",", ":")), \
        "the canonical form drifted from what a record published before this field would digest to"

    # ...and a real rejection IS signed
    full = _rec(rejected=[["5XYZ", ["committed but not revealed"]]])
    assert '"rejected"' in full.canonical()
    assert full.sha256() != empty.sha256(), "a rejection that does not change the digest is unsigned"

    # a record round-trips through the reader with the field intact
    from eval.rerun import record_from_blob
    back = record_from_blob(_j.dumps({**asdict(full)}).encode())
    assert back.rejected == [["5XYZ", ["committed but not revealed"]]], back.rejected
    assert back.sha256() == full.sha256()


def main() -> int:
    _isolate_env()
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
             test_script_aware_numbers, test_gguf_intake, test_pinned_parent,
             test_observer_kl_step_scoring, test_step_extraction, test_observer_round,
             test_determinism_gate, test_observer_epoch_end_to_end, test_artifact_manifest,
             test_miner_submit_and_chain_adapter,
             test_nonce_selects_items_and_record_is_rerunnable,
             test_rerun_audits_and_catches_a_rigged_record,
             test_publisher_is_fail_closed,
             test_identity_canary_catches_a_nondeterministic_box,
             test_pool_is_language_balanced_or_the_round_refuses,
             test_density_and_model_card_are_derived_not_asserted,
             test_fetch_refuses_hostile_artifacts_before_downloading,
             test_saturation_guard_retires_flat_axes,
             test_audit_binds_emission_to_the_crowns,
             test_auditor_verifies_then_diverges,
             test_auditor_verdict_states_what_it_did_not_check,
             test_auditor_touches_the_chain_once_per_pass_and_never_goes_silent,
             test_an_unrevealed_submission_is_refused_not_waved_through,
             test_the_throne_is_inherited_from_the_published_trail,
             test_gguf_is_loadable_because_nothing_else_can_pass_the_tiers,
             test_one_bad_artifact_cannot_take_the_round_down,
             test_orchestrator_audits_its_own_scorer_before_signing,
             test_auditor_publishes_its_verdicts_or_stops_voting,
             test_auditor_treats_silence_as_a_finding]
    # COLLECTED, NOT LISTED. The hand-maintained list above is kept only so the intended ORDER
    # survives; anything defined and not named in it is appended rather than silently skipped.
    # A test file where writing a test does nothing is worse than no test file: a security
    # regression for the crown path was added here, the suite reported 61/61, and the new test had
    # never run. Discovering that by counting is not a control.
    named = {t.__name__ for t in tests}
    tests += [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v) and k not in named]
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
