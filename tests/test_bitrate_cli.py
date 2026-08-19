"""A miner must be able to find out what intake will decide BEFORE committing to it.

`TQ1_0` is rejected at intake because mainline llama.cpp has no Metal kernel for it, so the artifact
could never run on the device the crown exists to reach. That rule lived in `eval/bitrate.py` and
appeared in no document a miner reads — and it has cost two miners their entry, including the only
`binary` submission this subnet has ever received. The README now names the formats, and this CLI
lets them run the same code the validator runs.
"""
from eval.bitrate import TIERS, UNRUNNABLE_FORMATS, BitReport, bit_tier_gate, main


def _rep(fmt, code_bits, container=2.0, params=8_000_000_000):
    return BitReport(params=params, code_bits=code_bits, container_bits=container,
                     formats={fmt: params})


def test_both_tq_types_are_refused_in_every_tier_however_small():
    """Even at a bit budget that would otherwise win `binary` outright.

    Verified against mainline llama.cpp 1c3c967 (2026-08-04): `ggml/src/ggml-metal/` contains zero
    occurrences of either type, while q1_0/q2_0/iq1_s/iq1_m/iq2_xxs all have matmul kernels."""
    for fmt in ("TQ1_0", "TQ2_0"):
        rep = _rep(fmt, 1.05)
        for tier in TIERS:
            ok, why = bit_tier_gate(rep, tier)
            assert not ok, f"{fmt} passed {tier.name}"
            assert any(fmt in w for w in why)


def test_the_refusal_never_recommends_another_unrunnable_format():
    """THE BUG THIS PINS. The TQ1_0 message used to say "Use TQ2_0, ...", and TQ2_0 was missing
    from UNRUNNABLE_FORMATS — so a miner who followed our own advice exactly would have swapped one
    unrunnable format for another, cleared intake, and could have been crowned with an artifact
    that cannot run on a phone. Advice worse than the rule it explained."""
    for fmt in ("TQ1_0", "TQ2_0"):
        _ok, why = bit_tier_gate(_rep(fmt, 1.05), TIERS[0])
        msg = " ".join(why)
        assert "Q1_0" in msg or "IQ1_S" in msg, "must name a format that works"
        for bad in UNRUNNABLE_FORMATS:
            assert f"Use {bad}" not in msg and f", {bad}" not in msg, \
                f"the {fmt} refusal recommends {bad}, which is itself unrunnable"


def test_a_runnable_ternary_format_at_the_same_budget_passes():
    """Controls the test above: it is the FORMAT being refused, not the bit budget."""
    ok, why = bit_tier_gate(_rep("Q1_0", 1.05, container=2.0), TIERS[0])
    assert ok, why


def test_cli_usage_and_missing_path():
    assert main(["eval.bitrate"]) == 2
    assert main(["eval.bitrate", "/definitely/not/here.gguf"]) == 2
