"""The bond is off by default, and the per-coldkey cap is what actually bounds a round.

`bond_posted` is a number the miner writes into their own commitment envelope
(`miner/submit.py --bond` -> `env["bond"]`). No stake moves, nothing is escrowed, and the bond
extrinsic the economics docstring refers to was never built — `bonds_held`/`settle()` account for
it in a dict discarded at the end of the round.

So the gate rejected miners who trusted the error message and stopped nobody who read the source.
In live round 2 it refused three real artifacts from one operator, any of which would have passed
by typing `--bond 1.0`.
"""
from eval.economics import RegistrationLedger


def test_a_second_submission_is_not_blocked_by_default():
    """The exact round-2 case: same coldkey, different hotkey, no bond declared."""
    led = RegistrationLedger()
    assert led.can_submit("hot1", "cold1").ok
    led.record("hot1", "cold1")
    d = led.can_submit("hot2", "cold1")
    assert d.ok, f"a real artifact was refused for an uncollectible bond: {d.reason}"
    assert d.bond_required == 0.0


def test_the_coldkey_cap_still_binds():
    """The control that IS enforceable — the coldkey comes from chain and cannot be self-declared.
    Without this, dropping the bond would leave nothing bounding one operator's share of a round."""
    led = RegistrationLedger(per_coldkey_round_cap=2)
    for hk in ("hot1", "hot2"):
        assert led.can_submit(hk, "cold1").ok
        led.record(hk, "cold1")
    d = led.can_submit("hot3", "cold1")
    assert not d.ok and "cap" in d.reason


def test_a_different_operator_is_unaffected():
    led = RegistrationLedger(per_coldkey_round_cap=2)
    for hk in ("hot1", "hot2"):
        led.record(hk, "cold1")
    assert led.can_submit("other", "cold2").ok


def test_the_machinery_still_works_when_a_real_bond_exists():
    """Disabled, not deleted. The day stake actually moves, one constructor argument turns it back
    on — so this must keep passing or the re-enable path is gone."""
    led = RegistrationLedger(base_bond=1.0)
    led.record("hot1", "cold1")
    d = led.can_submit("hot2", "cold1")
    assert not d.ok and d.bond_required == 1.0
    assert led.can_submit("hot2", "cold1", bond_posted=1.0).ok


def test_an_operator_can_enter_every_tier():
    """THE ROUND-2 BUG. There are four tiers and a commitment declares ONE, so competing in all
    four needs four hotkeys under one coldkey. The cap was per COLDKEY, so it scored two of them
    and refused the rest by iteration order — punishing breadth, which is the behaviour the subnet
    wants most. algodhf hit this with one artifact per tier."""
    led = RegistrationLedger()
    for hk, tier in (("hk_b", "binary"), ("hk_t", "ternary"),
                     ("hk_2", "sub2"), ("hk_4", "sub4")):
        d = led.can_submit(hk, "cold1", tier=tier)
        assert d.ok, f"{tier} refused for an operator entering every tier: {d.reason}"
        led.record(hk, "cold1", tier=tier)


def test_best_of_n_inside_one_tier_is_still_bounded():
    """What the cap is actually for: two artifacts in `binary`, keep whichever scores better."""
    led = RegistrationLedger(per_coldkey_round_cap=2)
    for n in range(2):
        assert led.can_submit(f"hk{n}", "cold1", tier="binary").ok
        led.record(f"hk{n}", "cold1", tier="binary")
    d = led.can_submit("hk_extra", "cold1", tier="binary")
    assert not d.ok, "unbounded best-of-N within a tier"
    assert "binary" in d.reason, "the refusal must name the tier that is full"
    # ...and a DIFFERENT tier is untouched by that
    assert led.can_submit("hk_other", "cold1", tier="ternary").ok
