"""Regression tests for what the round pays for, and what it refuses to crown.

    python -m tests.test_incentives
    pytest tests/test_incentives.py

Two defects, both visible in the first two scored rounds and both in the path that ranks miners.

THE FIELD WAS PRICED CORRECTLY. Rounds 1 and 2 drew 9 sub4 and 6 ternary submissions and nothing at
all in binary or sub2. That is not miners ignoring the hard tiers, it is miners reading them: every
tier paid the same, and `Tournament.weights` divided emission over the tiers that HAD a king — so
the two empty tiers handed their share to the two occupied ones, and sub4, which a miner enters with
one `llama-quantize` invocation, collected half of everything. The surest way to raise your income
was for nobody to attempt the hard tiers.

THE TERNARY TIER NEARLY CROWNED TOKEN SOUP. Four entries, three broken. `degeneracy_flags` existed
and was imported by the live scoring loop, but only `round_engine` and `axis_round` ever called it,
so nothing looked at the text. The floor that did apply was `retention_lb > 0.02`, and the soup
scored 0.1230-0.1740. One miner uploading a working model is the only reason netuid 40 did not crown
a broken quantiser and publish it as the state of the compression art.

The degeneracy cases run against `fixtures/round2_steps.json` — the actual steps those eleven
submissions emitted, taken from the signed record. A hand-written "looks broken" string would prove
nothing about the field the gate has to survive.
"""
from __future__ import annotations

import json
import os

from eval.bitrate import TIER_EMISSION_WEIGHT, TIERS, emission_weight
from eval.gates import degeneracy_flags
from eval.koth import King, Tier, Tournament

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
NAMES = ("binary", "ternary", "sub2", "sub4")


def _round2():
    with open(os.path.join(FIXTURES, "round2_steps.json")) as fh:
        return json.load(fh)


def _tournament(kings: dict):
    t = Tournament(tiers=[Tier(n, max_params=10 ** 10, weight=emission_weight(n))
                          for n in NAMES])
    for tier, miner in kings.items():
        t.kings[tier] = King(miner, f"{miner}-model", 0.3, 1)
    return t


# --- what a tier pays ----------------------------------------------------------------------------

def test_the_smallest_tier_pays_most():
    """Three orderings coincide, so the ladder is steepest at the bottom: fewer bits is a smaller
    file, a smaller file fits more phones and decodes faster, and fewer bits is the harder problem.

    This was briefly weighted the other way, on the belief that only Q2_0 reached a phone and
    sub-2-bit needed a private fork. Mainline llama.cpp carries Metal kernels for Q1_0, IQ1_S,
    IQ1_M, IQ2_XXS and Q2_0, so there is no reason to pay less for the smaller artifact."""
    w = TIER_EMISSION_WEIGHT
    by_bits = [t.name for t in sorted(TIERS, key=lambda t: t.max_code_bits)]
    pay = [w[n] for n in by_bits]
    assert pay == sorted(pay, reverse=True), \
        f"emission must fall as the bit budget rises: {list(zip(by_bits, pay))}"
    assert abs(sum(w.values()) - 1.0) < 1e-9, sum(w.values())
    assert set(w) == {t.name for t in TIERS}, "the weight table and the tier table disagree"


def test_a_format_that_cannot_run_is_refused_by_every_tier():
    """A format that fits the bit budget but cannot run on the target device is not a crown.

    TQ1_0 fits binary and ternary at 1.6875 bpw and has no Metal kernel — a champion that crashes
    on every iPhone.

    THIS TEST USED TO ASSERT "its sibling TQ2_0 is fine". IT IS NOT. Checked against upstream
    ggml-org/llama.cpp @ 1c3c967 (2026-08-04): `GGML_TYPE_TQ2_0` appears only in ggml-quants.c,
    ggml.c and the two ggml-cpu files — zero occurrences under ggml-metal/ AND zero under
    ggml-cuda/. TQ2_0 is CPU-only, so it would not even run GPU-accelerated on our own scorer; it
    would silently fall back to CPU, which is the exact trap `eval/gpu_check` was hardened against.
    Worse, the TQ1_0 refusal used to RECOMMEND TQ2_0, so a miner following our own instructions
    would have cleared intake with an artifact that cannot ship.

    Q2_0 is the control: same family, same ~2-bit neighbourhood, and it genuinely has Metal matmul
    kernels — so this cannot be waved through as "the low-bit formats are all broken"."""
    from eval.bitrate import BitReport, bit_tier_gate
    from eval.gguf import GGML_TYPES, code_bits, type_bits

    def gate_all(name):
        tid = next(k for k, v in GGML_TYPES.items() if v[0] == name)
        rep = BitReport(params=8_190_000_000, code_bits=code_bits(name),
                        container_bits=type_bits(tid), formats={name: 8_190_000_000})
        return [t.name for t in TIERS if bit_tier_gate(rep, t)[0]]

    for bad in ("TQ1_0", "TQ2_0"):
        assert gate_all(bad) == [], f"{bad} has no Metal kernel and was accepted into a tier"
    for ok in ("Q2_0", "Q1_0", "IQ1_S", "IQ1_M"):
        assert gate_all(ok), f"{ok} runs on Apple GPU and must not be refused"


def test_the_phone_tier_can_accept_the_phone_format():
    """sub2's cap is 2.3 SO THAT Q2_0 FITS. At 2.0 the one phone-native format on the board could
    enter no tier but sub4, where models carrying twice the bits would beat it — a tier built for
    a format that could not enter it."""
    from eval.bitrate import BitReport, bit_tier_gate
    from eval.gguf import GGML_TYPES, code_bits, type_bits
    by_name = {t.name: t for t in TIERS}
    q2 = next(tid for tid, v in GGML_TYPES.items() if v[0] == "Q2_0")
    rep = BitReport(params=8_190_000_000, code_bits=code_bits("Q2_0"), container_bits=type_bits(q2))
    ok, why = bit_tier_gate(rep, by_name["sub2"])
    assert ok, f"Q2_0 cannot enter the tier built for it: {why}"
    assert abs(type_bits(q2) - 2.25) < 1e-9, type_bits(q2)     # 18 bytes / 64 elems
    # and PrismML's Q1_0 lands in binary, at its published 1.125 bpw
    q1 = next(tid for tid, v in GGML_TYPES.items() if v[0] == "Q1_0")
    assert abs(type_bits(q1) - 1.125) < 1e-9, type_bits(q1)
    assert bit_tier_gate(BitReport(params=8_190_000_000, code_bits=code_bits("Q1_0"),
                                   container_bits=type_bits(q1)), by_name["binary"])[0]


def test_unknown_tiers_fall_back_to_an_equal_split():
    """`rehearsal`, `open` and the simulation tiers are not in the table and must still work."""
    assert emission_weight("rehearsal", 1) == 1.0
    assert emission_weight("open", 4) == 0.25


# --- the subsidy ---------------------------------------------------------------------------------

def test_nothing_is_ever_burned():
    """NO BURN, DELIBERATELY. A subnet writing part of its weight vector to a burn uid reads in
    this ecosystem as taxing its own miners. Whatever the field looks like, the payout is whole."""
    for live in (["ternary", "sub4"], ["sub2"], NAMES, ["binary", "sub4"]):
        t = _tournament({n: f"m-{n}" for n in live})
        assert abs(sum(t.weights().values()) - 1.0) < 1e-9, (live, t.weights())


def test_moving_up_a_tier_pays_more_with_the_field_held_fixed():
    """What the unequal weights buy: under an equal split the same field paid sub4 50% for one
    `llama-quantize` invocation.

    THE COMPARISON MUST HOLD THE LIVE TIERS FIXED. An earlier version of this test compared a
    two-tier field against a three-tier one and read the extra dilution as a lost incentive — the
    renormalisation changes every share when a tier is added, so a miner moving up looked worse
    off. Swap which tier one miner holds and leave the field alone."""
    W = TIER_EMISSION_WEIGHT
    live = ["ternary", "sub2", "sub4"]
    tot = sum(W[t] for t in live)
    got = _tournament({t: f"k-{t}" for t in live}).weights()
    for t in live:
        assert abs(got[f"k-{t}"] - W[t] / tot) < 1e-9, (t, got)
    # the same miner, same field, one rung up each time
    assert got["k-sub2"] > got["k-sub4"], got
    assert got["k-ternary"] > got["k-sub2"], got

    # and the two-tier field we actually have today
    today = _tournament({"ternary": "alice", "sub4": "bob"}).weights()
    two = W["ternary"] + W["sub4"]
    assert abs(today["alice"] - W["ternary"] / two) < 1e-9, today
    assert abs(today["bob"] - W["sub4"] / two) < 1e-9, today
    assert today["bob"] < 0.5, "sub4 must no longer take half of everything"


def test_an_empty_tier_is_redistributed_not_withheld():
    """An unclaimed share goes to the occupied tiers, and `unclaimed()` REPORTS how much did.

    It is reporting only — the point is that a record can say "this round paid a four-tier schedule
    to two tiers" rather than leaving a concentrated payout looking accidental."""
    t = _tournament({"ternary": "alice", "sub4": "bob"})
    assert abs(sum(t.weights().values()) - 1.0) < 1e-9
    assert abs(t.unclaimed() - (TIER_EMISSION_WEIGHT["binary"]
                                + TIER_EMISSION_WEIGHT["sub2"])) < 1e-9
    assert _tournament({n: f"m-{n}" for n in NAMES}).unclaimed() == 0.0


def test_all_tiers_claimed_pays_the_nominal_schedule():
    t = _tournament({n: f"m-{n}" for n in NAMES})
    w = t.weights()
    for n in NAMES:
        assert abs(w[f"m-{n}"] - TIER_EMISSION_WEIGHT[n]) < 1e-9, (n, w)
    assert t.unclaimed() == 0.0


def test_no_kings_pays_nobody():
    t = _tournament({})
    assert t.weights() == {}
    assert abs(t.unclaimed() - 1.0) < 1e-9


def test_one_miner_holding_several_tiers_accumulates():
    t = _tournament({"binary": "solo", "sub4": "solo"})
    assert abs(t.weights()["solo"] - 1.0) < 1e-9, "sole holder of every live tier takes it all"
    W = TIER_EMISSION_WEIGHT
    tot = W["binary"] + W["sub4"] + W["sub2"]
    both = _tournament({"binary": "solo", "sub4": "solo", "sub2": "rival"}).weights()
    assert abs(both["solo"] - (W["binary"] + W["sub4"]) / tot) < 1e-9, both
    assert abs(both["rival"] - W["sub2"] / tot) < 1e-9, both


# --- what cannot be crowned ----------------------------------------------------------------------

def test_every_broken_round2_submission_is_rejected():
    """The three that were not language. `andreas11112`'s ternary wrote lucid prose early and
    `0 0 0 0 …` by step 30; both `Jordun01` entries emitted subword salad throughout."""
    broken = ["andreas11112/qwen3-8b-sn40-ternary", "tern-mix-tB", "tern-mix-v3"]
    d = _round2()
    for key in broken:
        hits = [v for k, v in d.items() if key in k]
        assert hits, f"fixture missing {key}"
        ok, reasons = degeneracy_flags(hits[0]["steps"])
        assert not ok, f"{key} (retention {hits[0]['retention']}) passed the gate"
        assert reasons


def test_every_working_round2_submission_survives():
    """The gate is worthless if it also rejects the field. All eight coherent submissions pass,
    including the two crowns and the challenger that lost on worst-slice."""
    d = _round2()
    broken = ("andreas11112/qwen3-8b-sn40-ternary", "tern-mix-tB", "tern-mix-v3")
    for uri, v in d.items():
        if any(b in uri for b in broken):
            continue
        ok, reasons = degeneracy_flags(v["steps"])
        assert ok, f"{uri} (retention {v['retention']}) was wrongly rejected: {reasons}"


def test_the_soup_would_have_been_crowned_without_this_gate():
    """Pins the counterfactual, so nobody weakens the gate without seeing what it prevents.

    The best broken ternary scored 0.1740 against a floor of 0.02. Absent one working submission it
    would have taken the crown on retention alone."""
    from eval.koth import MIN_CROWN_LB
    d = _round2()
    soup = max((v for k, v in d.items()
                if any(b in k for b in ("qwen3-8b-sn40-ternary", "tern-mix"))),
               key=lambda v: v["retention"])
    assert soup["retention"] > MIN_CROWN_LB, "retention alone would not have crowned it"
    assert not degeneracy_flags(soup["steps"])[0], "the gate is what refuses it"


def test_ordinary_content_is_not_mistaken_for_salad():
    """Markdown rules, table borders and ellipses are runs of a repeated character, and the corpus
    is a third Hindi and Chinese. Script-mixing was the tempting signal and is the wrong one — a
    genuine Chinese step in this corpus switches script MORE often than the salad does."""
    for name, text in (
            ("markdown rule", "Plan.\n\n-----------------------\n\n1. read it\n2. patch it"),
            ("table border", "| col | col |\n|-----|-----|\n| a | b |\nThat is the mapping."),
            ("code fence", "```bash\nls -la /path/to/bleach\n```\nThe output shows the tree."),
            ("ellipsis", "The result was inconclusive............ so I re-ran it and got 42."),
            ("chinese", "嗯，用户现在需要回答关于大西洋奴隶贸易对非洲社会长期经济影响的问题，需要仔细分析每个选项。"),
            ("hindi", "मुझे यह समझने की आवश्यकता है कि यह प्रश्न किस बारे में है और फिर उत्तर देना होगा।"),
            ("arithmetic", "We compute 0 + 0 = 0, then 1 + 1 = 2, then 2 + 2 = 4, and sum them."),
    ):
        ok, reasons = degeneracy_flags([text] * 10)
        assert ok, f"{name} was flagged as degenerate: {reasons}"


# COLLECTED AFTER EVERY TEST IS DEFINED. Keep this immediately above main().
TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    bad = 0
    for t in TESTS:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except Exception as e:
            bad += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - bad}/{len(TESTS)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
