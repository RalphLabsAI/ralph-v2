"""Events must say which round they happened in.

`Tournament` initialises `self.round = 0` and every event copies it. `round_engine`, `env_round`
and `axis_round` all set it before scoring; `score_job` — the SPLIT VALIDATOR, the only path that
actually publishes — did not. So live rounds 1 and 2 both shipped signed, anchored records whose
events are stamped `round: 0`, which is what an auditor reads to say when a tier changed hands and
what a `crown` event means by "since".
"""
import inspect

from eval import score_job
from eval.koth import Tier, Tournament

TIERS = (Tier(name="sub4", max_params=10**10, weight=0.15),)


def test_events_carry_the_tournament_round():
    t = Tournament(TIERS, margin=0.05)
    t.round = 7
    ev = t.consider("sub4", [], None)
    assert ev["round"] == 7, "an event that cannot say when it happened is not a lineage"


def test_default_is_zero_which_is_why_it_must_be_set():
    """Pins the trap: the default is a VALID-looking round number, not None, so a path that forgets
    to set it publishes records that look fine and are wrong."""
    assert Tournament(TIERS, margin=0.05).round == 0


def test_score_job_stamps_the_round_from_the_job():
    """The regression itself. Asserted on the source because reaching that line otherwise needs a
    GPU, a parent checkpoint and a whole round; what it must never go back to is the constant 0."""
    src = inspect.getsource(score_job)
    assert 'tournament.round = int(job.get("round", 0) or 0)' in src, (
        "score_job must stamp the round on the Tournament, or every event it emits says round 0")
    # and before the kings are installed, which is the first thing that reads the tournament
    assert src.index("tournament.round =") < src.index("tournament.kings[tier]"), \
        "the round must be stamped before the tournament is used"
