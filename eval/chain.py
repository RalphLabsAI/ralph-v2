"""Chain-I/O boundary for a v2 epoch.

`run_validator_round` (validator_loop.py) is pure: models + pile in, decisions out. The
only thing between it and a live subnet is chain I/O — reading what miners committed,
drawing the round nonce, and writing back weights / crown / the round-record hash.

This module defines that boundary as a small PROTOCOL (`ChainIO`) and wires it to the
round in `run_v2_epoch`. It does NOT import or modify Ralph's live v1 validator — the v1
service (karpa/validator/service.py) already implements every method here
(get_king/set_king/set_weights/current_block/get_commitment/blacklist), so bringing v2 up
is: construct the pile + pinned (glm, base, judge), then call `run_v2_epoch(chain, ...)`
from the existing epoch loop. Keeping it a protocol lets the whole thing be tested against
`FakeChain` with zero chain dependency, and keeps this session off the live signer.

Commit-then-generate ordering enforced here:
  1. read commitments (sealed H(content_hash‖salt)) that locked BEFORE this block
  2. draw round_nonce = hash of a block AFTER the commit window closed
  3. only now are points minted (seed binds commit_root‖nonce) and scoring runs
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .koth import Tier, Tournament
from .economics import RegistrationLedger
from .gates import TierBudget
from .round_record import RoundRecord
from .trajectory import Rollout, StepJudge
from .validator_loop import CommittedSubmission, RoundOutcome, run_validator_round


@dataclass
class Commitment:
    """One miner's sealed on-chain commitment, revealed and resolved to a fetched dir."""
    hotkey: str
    coldkey: str
    tier: str
    ckpt_dir: str              # local path; intake fetch-verifies it against the reveal
    declared_compute_h100h: float
    bond_posted: float = 0.0
    # commit-reveal: the sealed on-chain value + what the miner revealed. intake
    # recomputes the artifact hash and refuses a mismatch (bait-and-switch).
    revealed_hash: str = ""
    salt: str = ""
    committed_value: str = ""
    # where the bytes live (hf://repo@rev, ipfs://cid, https://...). Without this a crown is an
    # unresolvable digest and "every crown ships a downloadable model" is not deliverable.
    artifact_uri: str = ""


@runtime_checkable
class ChainIO(Protocol):
    """Exactly what a v2 epoch needs from the chain. v1's service satisfies all of it."""

    def current_block(self) -> int: ...
    def block_hash(self, block: int) -> str: ...
    def read_commitments(self, min_block: int, max_block: int) -> list[Commitment]: ...
    def commit_root(self, min_block: int, max_block: int) -> str: ...
    def set_weights(self, weights: dict[str, float]) -> bool: ...
    def set_king(self, tier: str, hotkey: str, model_id: str) -> None: ...
    def get_king(self, tier: str) -> object | None: ...
    def blacklist(self, hotkey: str, reason: str) -> None: ...
    def publish_record(self, record: RoundRecord) -> None: ...
    # The anchor currently committed on chain. REQUIRED, and in the protocol rather than probed
    # with getattr, because the first version of the publish gate read it with
    # getattr(chain, "record_anchors", None): no adapter implemented it, the anchor comparison was
    # skipped in silence, and the heartbeat degraded to comparing the operator's bytes against the
    # operator's index. A missing method must be a type error, not a quietly weaker guarantee.
    def head_anchor(self) -> str: ...
    # settle refundable anti-grind bonds. Without this the ledger computes refunds
    # and drops them, so the bond is never actually returned (or forfeited) on
    # chain and the anti-grind economics are theatre.
    def settle_bonds(self, refunds: dict) -> None: ...


@dataclass
class EpochResult:
    round_no: int
    outcome: RoundOutcome
    record_sha256: str
    # what publishing did. `publish.ok is False` means NO weights were set this round — the
    # previous crown keeps earning until the operator fixes publishing.
    publish: object = None
    weights_set: bool = False
    # set when the gate withheld payment. Kept OFF the signed record on purpose (see _write_back).
    withheld: dict | None = None


def run_v2_epoch(
    chain: ChainIO, round_no: int,
    experience: list[Rollout], glm, base, judge: StepJudge,
    tiers: list[Tier], tier_budgets: dict[str, TierBudget],
    tournament: Tournament, ledger: RegistrationLedger, registry: dict,
    commit_window: int = 100, make_safe_runner=None,
    pile_id: str = "pile", n_points: int = 120, self_frac: float = 0.0,
    publisher=None, require_publish: bool | None = None,
) -> EpochResult:
    """One v2 epoch: read the sealed commit window, draw the nonce, score, write back.

    `make_safe_runner(ckpt_dir) -> ModelRunner` builds the locked-down loader for an
    accepted checkpoint (SafeStudentRunner in prod; a sim factory in tests). It is NOT
    called until intake gates pass — no untrusted weights load on a rejected submission.
    """
    if make_safe_runner is None:
        # fail-safe default: real untrusted checkpoints load ONLY through SafeStudentRunner
        # (safetensors-only, trust_remote_code=False). A caller must consciously inject a
        # different factory (tests) — production never runs without the locked-down loader.
        from .runners import SafeStudentRunner
        make_safe_runner = lambda cd: SafeStudentRunner(cd)
    now = chain.current_block()
    lo, hi = now - commit_window, now
    commits = chain.read_commitments(lo, hi)
    commit_root = chain.commit_root(lo, hi)
    round_nonce = chain.block_hash(now)   # drawn AFTER the window closed -> unpredictable

    committed = [
        CommittedSubmission(
            hotkey=c.hotkey, coldkey=c.coldkey, tier=c.tier, ckpt_dir=c.ckpt_dir,
            declared_compute_h100h=c.declared_compute_h100h, bond_posted=c.bond_posted,
            make_runner=(lambda cd=c.ckpt_dir: make_safe_runner(cd)),
            revealed_hash=c.revealed_hash, salt=c.salt, committed_value=c.committed_value,
        )
        for c in commits
    ]

    outcome = run_validator_round(
        round_no, commit_root, round_nonce, committed, experience, glm, base, judge,
        tiers, tier_budgets, tournament, ledger, registry,
        pile_id=pile_id, n_points=n_points, self_frac=self_frac,
    )

    return _write_back(chain, tiers, tournament, outcome, round_no,
                       publisher=publisher, require_publish=require_publish)


def run_v2_env_epoch(
    chain: ChainIO, round_no: int,
    pile, base_agent, teacher,
    tiers: list[Tier], tier_budgets: dict[str, TierBudget],
    tournament: Tournament, ledger: RegistrationLedger, registry: dict,
    commit_window: int = 100, make_agent=None,
    teacher_id: str = "glm", base_id: str = "base", pile_id: str = "env-pile",
    n_points: int = 300, self_frac: float = 0.4,
    publisher=None, require_publish: bool | None = None,
) -> EpochResult:
    """One v2 epoch on the ENV substrate (the validated, production scoring path). Same
    chain I/O as run_v2_epoch; scores via run_env_round. `make_agent(ckpt_dir) -> Agent`
    builds the locked-down ModelAgent for an accepted checkpoint (only after gates pass).
    The reference is the pinned `teacher` (GLM) agent — reproduce-GLM, checked by the
    deterministic env oracle (no LLM judge)."""
    from .validator_env_loop import CommittedSubmission as EnvCommitted, run_env_round

    if make_agent is None:
        # fail-safe default: untrusted checkpoints run only as a ModelAgent wrapping the
        # locked-down SafeStudentRunner. Tests inject their own agent factory.
        from .runners import SafeStudentRunner
        from .multiturn import ModelAgent
        make_agent = lambda cd: ModelAgent(SafeStudentRunner(cd))

    now = chain.current_block()
    lo, hi = now - commit_window, now
    commits = chain.read_commitments(lo, hi)
    commit_root = chain.commit_root(lo, hi)
    round_nonce = chain.block_hash(now)

    committed = [
        EnvCommitted(hotkey=c.hotkey, coldkey=c.coldkey, tier=c.tier, ckpt_dir=c.ckpt_dir,
                     declared_compute_h100h=c.declared_compute_h100h, bond_posted=c.bond_posted,
                     make_agent=(lambda cd=c.ckpt_dir: make_agent(cd)),
                     revealed_hash=c.revealed_hash, salt=c.salt, committed_value=c.committed_value)
        for c in commits
    ]
    outcome = run_env_round(
        round_no, commit_root, round_nonce, committed, pile, base_agent, teacher,
        tiers, tier_budgets, tournament, ledger, registry,
        teacher_id=teacher_id, base_id=base_id, pile_id=pile_id,
        n_points=n_points, self_frac=self_frac,
    )
    return _write_back(chain, tiers, tournament, outcome, round_no,
                       publisher=publisher, require_publish=require_publish)


def run_v2_axis_epoch(
    chain: ChainIO, round_no: int,
    specs, glm, base,
    tiers: list[Tier], tier_budgets: dict[str, TierBudget],
    tournament: Tournament, ledger: RegistrationLedger, registry: dict,
    commit_window: int = 100, make_safe_runner=None,
    teacher_id: str = "glm", base_id: str = "base",
    items_per_axis: int = 150, max_new_tokens: int = 512,
    overfit_check=None, signer=None, surprise_k: int | None = None,
    publisher=None, require_publish: bool | None = None,
) -> EpochResult:
    """One v2 epoch on the AXIS substrate — the GLM-COVER production path (verifiable-outcome
    retention across deterministic-checker axes). Same chain I/O as run_v2_epoch; scores via
    run_axis_round, which carries the full front door (intake + commit-reveal), the
    genre-overfit crown precondition, and a signed record. `make_safe_runner(ckpt_dir)`
    builds the locked-down loader, invoked only after intake gates pass."""
    from .validator_axis_loop import CommittedSubmission as AxisCommitted, run_axis_round

    if make_safe_runner is None:
        from .runners import SafeStudentRunner
        make_safe_runner = lambda cd: SafeStudentRunner(cd)

    now = chain.current_block()
    lo, hi = now - commit_window, now
    commits = chain.read_commitments(lo, hi)
    commit_root = chain.commit_root(lo, hi)
    round_nonce = chain.block_hash(now)   # drawn AFTER the window closed -> unpredictable

    committed = [
        AxisCommitted(hotkey=c.hotkey, coldkey=c.coldkey, tier=c.tier, ckpt_dir=c.ckpt_dir,
                      declared_compute_h100h=c.declared_compute_h100h, bond_posted=c.bond_posted,
                      make_runner=(lambda cd=c.ckpt_dir: make_safe_runner(cd)),
                      revealed_hash=c.revealed_hash, salt=c.salt, committed_value=c.committed_value)
        for c in commits
    ]
    outcome = run_axis_round(
        round_no, commit_root, round_nonce, committed, specs, glm, base,
        tiers, tier_budgets, tournament, ledger, registry,
        teacher_id=teacher_id, base_id=base_id,
        items_per_axis=items_per_axis, max_new_tokens=max_new_tokens,
        overfit_check=overfit_check, signer=signer, surprise_k=surprise_k,
    )
    return _write_back(chain, tiers, tournament, outcome, round_no,
                       publisher=publisher, require_publish=require_publish)


def run_v2_observer_epoch(
    chain: ChainIO, round_no: int,
    trajectory_pool, parent, observers: dict,
    tiers: list[Tier], tier_budgets: dict[str, TierBudget],
    tournament: Tournament, ledger: RegistrationLedger, registry: dict,
    commit_window: int = 100, make_safe_runner=None, parent_id: str = "parent",
    max_step_tokens: int = 256, max_cont_tokens: int = 128,
    signer=None, canary=None, noise_safety: float = 3.0,
    n_items: int = 64, corpus_spec: str = "",
    publisher=None, require_publish: bool | None = None,
) -> EpochResult:
    """One v2 epoch on the OBSERVER-KL substrate — the crown path.

    Same chain I/O as the axis epoch: read the commitment window, take the nonce from a block
    drawn AFTER it closed, run the round, write back crown/weights/record. The round itself
    draws its observer from that nonce, so which observer a miner faces is unknowable at seal
    time (const's rotation idea, made un-pre-fittable rather than merely varied)."""
    from .validator_observer_loop import CommittedSubmission as ObsCommitted, run_observer_round

    if make_safe_runner is None:
        from .runners import SafeStudentRunner
        make_safe_runner = lambda cd: SafeStudentRunner(cd)

    now = chain.current_block()
    lo, hi = now - commit_window, now
    commits = chain.read_commitments(lo, hi)
    commit_root = chain.commit_root(lo, hi)
    round_nonce = chain.block_hash(now)

    committed = [
        ObsCommitted(hotkey=c.hotkey, coldkey=c.coldkey, tier=c.tier, ckpt_dir=c.ckpt_dir,
                     declared_compute_h100h=c.declared_compute_h100h, bond_posted=c.bond_posted,
                     make_runner=(lambda cd=c.ckpt_dir: make_safe_runner(cd)),
                     revealed_hash=c.revealed_hash, salt=c.salt, committed_value=c.committed_value,
                     # the locator travels with the submission so the RECORD can name where the
                     # scored bytes live; without it L3 has nothing to fetch and the frozen miner
                     # steps are unfalsifiable
                     artifact_uri=c.artifact_uri)
        for c in commits
    ]
    # The anchor link goes INSIDE the signed body, so the previous head has to be known before the
    # record is built and signed — it cannot be back-filled at publish time.
    prev_anchor = publisher.head_anchor() if publisher is not None else ""
    outcome = run_observer_round(
        round_no, commit_root, round_nonce, committed, trajectory_pool, parent, observers,
        tiers, tier_budgets, tournament, ledger, registry, parent_id=parent_id,
        prev_anchor=prev_anchor,
        signer=signer, max_step_tokens=max_step_tokens, max_cont_tokens=max_cont_tokens,
        noise_safety=noise_safety, canary=canary, n_items=n_items, corpus_spec=corpus_spec,
    )
    # The pool goes up BEFORE the record. The record's manifest pins its digest, so publishing the
    # record first would leave a window where the trail names a corpus nobody can fetch — and L1 is
    # exactly the level that stops being runnable when that happens.
    if publisher is not None and outcome.record is not None:
        from .pool import dump_pool
        want = (outcome.record.manifest or {}).get("pool_sha256") or ""
        if want:
            publisher.publish_pool(dump_pool(trajectory_pool), digest=want)
    return _write_back(chain, tiers, tournament, outcome, round_no,
                       publisher=publisher, require_publish=require_publish)


def _write_back(chain: ChainIO, tiers, tournament, outcome, round_no,
                publisher=None, require_publish: bool | None = None) -> EpochResult:
    """PUBLISH FIRST, then crown and pay. The old order was set_king -> set_weights ->
    publish_record, which paid out a round before anyone could check it and kept paying when
    publishing broke. That is not hypothetical: v1's publisher was fail-open, its on-chain anchor
    went 22 days stale, and the lineage regressed from 50 entries to 6 while scoring ran fine.

    `require_publish` defaults to True whenever a publisher is configured, and can be forced on
    with RALPH_REQUIRE_PUBLISH=1 so a production box cannot be started without one by accident."""
    import os as _os
    from .publish import PublishError, publish_and_gate

    forced = _os.environ.get("RALPH_REQUIRE_PUBLISH", "").strip() not in ("", "0", "false")
    if require_publish is None:
        require_publish = publisher is not None
    # The env var is a HARD floor, not a default. It used to only fill in when require_publish was
    # None, so a single require_publish=False argument anywhere in the call path silently defeated
    # the one switch an operator sets to guarantee the gate is on.
    require_publish = require_publish or forced
    if require_publish and publisher is None:
        raise PublishError("RALPH_REQUIRE_PUBLISH is set but no publisher was configured — "
                           "refusing to run a round whose record nobody can fetch")

    res = EpochResult(round_no, outcome, outcome.record.sha256() if outcome.record else "")
    if publisher is not None:
        # pass the RESOLVER, not a snapshot: the current round is anchored inside
        # publish_and_gate, so a snapshot taken here would never contain it
        rep = publish_and_gate(publisher, outcome.record,
                               anchors=getattr(chain, "record_anchors", None),
                               anchor_fn=getattr(chain, "publish_record", None),
                               head_anchor_fn=getattr(chain, "head_anchor", None),
                               allow_unanchored=not require_publish)
        res.publish = rep
        if not rep.ok and require_publish:
            # FAIL CLOSED. No crown written on chain, no weights set. Emission does not stop —
            # the previously set weights persist, so the last verifiably published crown keeps
            # earning. A hold, not a halt; see publish.py for why that residual is stated openly.
            # ROLL BACK THE CROWN. tournament.consider() already mutated tournament.kings during
            # scoring, so returning here left the withheld round's king in memory — and the NEXT
            # round's set_king wrote it on chain and paid it, which defeats the whole gate.
            if getattr(outcome, "kings_before", None) is not None:
                tournament.kings = dict(outcome.kings_before)
            # Do NOT append to outcome.events: the record holds that list, is already signed and
            # already published, so mutating it invalidates the signature of published bytes and
            # makes the round permanently unrepublishable. The reason lives on the result instead.
            res.withheld = {
                "round": round_no, "action": "withhold_weights",
                "reason": "; ".join(rep.reasons) or "record not verifiably published",
                "stale_rounds": rep.stale_rounds}
            return res
    elif outcome.record is not None:
        chain.publish_record(outcome.record)

    for tier in tiers:
        king = tournament.kings.get(tier.name)
        if king is not None:
            chain.set_king(tier.name, king.miner, king.model_id)
    chain.set_weights(outcome.weights)
    res.weights_set = True
    if getattr(outcome, "refunds", None):
        settle = getattr(chain, "settle_bonds", None)
        if settle is not None:
            settle(outcome.refunds)
    return res
