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

def test_hard_tiers_pay_more_than_easy_ones():
    w = TIER_EMISSION_WEIGHT
    assert w["binary"] > w["ternary"] > w["sub2"] > w["sub4"], w
    assert abs(sum(w.values()) - 1.0) < 1e-9, sum(w.values())
    assert set(w) == {t.name for t in TIERS}, "the weight table and the tier table disagree"


def test_unknown_tiers_fall_back_to_an_equal_split():
    """`rehearsal`, `open` and the simulation tiers are not in the table and must still work."""
    assert emission_weight("rehearsal", 1) == 1.0
    assert emission_weight("open", 4) == 0.25


# --- the subsidy ---------------------------------------------------------------------------------

def test_an_empty_tier_does_not_pay_the_occupied_ones():
    """THE BUG, stated as the case that produced it: binary and sub2 empty, ternary and sub4 held.

    Under the old rule the two kings split 1.0 between them and sub4 took half of all emission for
    being the easiest tier on the board."""
    t = _tournament({"ternary": "alice", "sub4": "bob"})
    w = t.weights()
    assert abs(w["bob"] - TIER_EMISSION_WEIGHT["sub4"]) < 1e-9, w
    assert abs(w["alice"] - TIER_EMISSION_WEIGHT["ternary"]) < 1e-9, w
    assert sum(w.values()) < 1.0, "an unclaimed tier's share was redistributed"
    assert abs(t.unclaimed() - (TIER_EMISSION_WEIGHT["binary"]
                                + TIER_EMISSION_WEIGHT["sub2"])) < 1e-9


def test_entering_an_empty_hard_tier_cannot_be_diluted_by_the_easy_ones():
    """The incentive the whole change exists to create: taking the binary crown pays its full share
    no matter how crowded sub4 is."""
    alone = _tournament({"binary": "carol"})
    crowded = _tournament({"binary": "carol", "ternary": "a", "sub2": "b", "sub4": "c"})
    assert abs(alone.weights()["carol"] - crowded.weights()["carol"]) < 1e-9
    assert abs(alone.weights()["carol"] - TIER_EMISSION_WEIGHT["binary"]) < 1e-9


def test_all_tiers_claimed_pays_out_in_full():
    t = _tournament({n: f"m-{n}" for n in NAMES})
    assert abs(sum(t.weights().values()) - 1.0) < 1e-9
    assert t.unclaimed() == 0.0


def test_no_kings_pays_nobody():
    t = _tournament({})
    assert t.weights() == {}
    assert abs(t.unclaimed() - 1.0) < 1e-9


def test_one_miner_holding_several_tiers_accumulates():
    t = _tournament({"binary": "solo", "sub4": "solo"})
    assert abs(t.weights()["solo"]
               - (TIER_EMISSION_WEIGHT["binary"] + TIER_EMISSION_WEIGHT["sub4"])) < 1e-9


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
