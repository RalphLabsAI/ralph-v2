"""Auditable round record — the signed, reproducible artifact of one scoring round.

The paired dethrone test rests on per-point verdicts. If those aren't logged, a verdict
is not reproducible and a compromised validator could rig a crown silently (a red-team
ship-blocker). So every round emits a canonical record:

  * the exact points scored (rollout id, k, mode)
  * per-submission, per-point verdicts (the judge's YES/NO, 0/1)
  * the derived retention + the crown decision
  * the seeds and model/judge/pile identities that make it re-runnable

It hashes canonically (sorted keys) so the hash can be committed on-chain and anyone can
re-run the round and reproduce it within a documented tolerance (the judge's float logits
can flip near threshold across hardware — record the tolerance, don't pretend to
bit-exactness).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass
class SubmissionRecord:
    model_id: str
    miner: str
    tier: str
    retention: float
    retention_lb: float
    per_point: list          # aligned to points[]
    gates_ok: bool
    reasons: list = field(default_factory=list)
    # FROZEN miner steps, aligned to points[]. With these plus the frozen parent step and
    # continuation in points[], an auditor recomputes the observer's distributions itself — so
    # the JUDGMENT layer is re-derived, not trusted. That is the difference from a subnet that
    # publishes LLM verdicts: aggregating asserted judgments only catches arithmetic fraud,
    # whereas recomputing a KL from text catches a rigged score.
    steps: list = field(default_factory=list)
    # per-sample [slice_key, s, d_parent, d_miner] as scored, aligned to points[]. Carrying the
    # slice key makes the score ARITHMETICALLY recomputable with no models at all: regroup, mean
    # per slice, take the worst. A validator that publishes honest KLs but a fabricated aggregate
    # is caught for free.
    effects: list = field(default_factory=list)
    # slice_key -> per-sample scores, i.e. the exact paired vectors the dethrone bootstrap ran on.
    slices: dict = field(default_factory=dict)
    # Where the scored bytes live. Without a locator, "re-generate the steps yourself" is not an
    # instruction anyone can follow, so the frozen steps are unfalsifiable by construction.
    artifact_uri: str = ""
    manifest_root: str = ""
    # INPUTS to intelligence density, published so the number is recomputable rather than asserted.
    # `params` and the two bit figures come from intake's measurement of the actual tensor data;
    # retention is already above. eval/density.py divides them. Publishing the inputs rather than
    # the ratio is the same rule the rest of this record follows.
    params: int = 0
    code_bits: float = 0.0
    container_bits: float = 0.0
    # "challenger" | "incumbent". The incumbent's re-score is the OTHER HALF of the paired
    # dethrone test; publishing only challengers left the comparison one-sided and the margin
    # unverifiable.
    role: str = "challenger"


@dataclass
class RoundRecord:
    round: int
    commit_root: str
    round_nonce: str
    teacher: str
    judge: str
    base: str
    pile_id: str
    points: list             # [{rollout_id, k, mode}]
    submissions: list        # [SubmissionRecord...]
    events: list             # tournament events (crown/hold/dethrone)
    weights: dict
    # RE-RUN MANIFEST. Rule: if a value moved the crown, it is in canonical(). Anything omitted
    # is something an auditor has to guess, and a guess is where v1's auditor died — its replay
    # diverged on 37 of 40 real reports because the report omitted state the scorer carried.
    manifest: dict = field(default_factory=dict)
    # The measured noise floor the crown was gated on. determinism.py promises "publish both
    # numbers either way"; keeping it OUT of the signed body made the gating number unverifiable.
    noise: dict = field(default_factory=dict)
    # HASH-CHAIN LINK. The anchor for round n is H(prev_anchor ‖ sha256(record)), so the single
    # on-chain commitment slot commits to the WHOLE history rather than only the latest round.
    # Without this, "the anchor is the half the operator cannot rewrite" was not true: one slot
    # holds one digest, so every past record was checkable only against an index the operator also
    # owns, and deleting or swapping an old round broke nothing.
    prev_anchor: str = ""
    # WHY A MINER IS NOT IN `submissions`. Round 1 accepted eleven artifacts and published nine,
    # and the two reasons — one miner never revealed, one shipped a GGUF llama.cpp cannot open —
    # existed only in a summary file on the orchestrator, on a box the miner cannot see. From
    # their side they vanished from a round they had been accepted into. A rejection is a
    # judgement about someone's work, so it belongs in the signed, anchored record beside the
    # scores, exactly as tamper-evident as a retention. Shape: [[hotkey, [reason, ...]], ...]
    rejected: list = field(default_factory=list)
    # Share of emission belonging to tiers with NO KING this round, routed to the burn uid rather
    # than to the kings that do exist. Recorded because it is a payout decision: a reader
    # reconciling `weights` against the emission actually written needs to see where the remainder
    # went, and "the vector does not sum to 1" is otherwise indistinguishable from a bug.
    unclaimed: float = 0.0
    # WHEN THE ROUND RAN. Epoch seconds, operator-asserted — nothing in a signed record can prove
    # its own clock. Published so a reader can see the subnet's CADENCE: how often rounds happen,
    # how long one takes, whether a gap was a pause or a failure. Rounds 1 and 2 predate these and
    # carry 0; they are sparse for the same reason `rejected` is, so those two still digest to
    # exactly what they signed. A timestamp nobody has to take on trust is in the trail repo's own
    # commit history.
    started_at: float = 0.0
    published_at: float = 0.0
    # Tolerance is derived from the MEASURED floor, not assumed. The old 0.02 default was ~200x
    # looser than the measured forward-pass floor, i.e. it would certify a materially wrong
    # re-run. build_round_record() sets this from `noise` when one is supplied.
    reproduction_tolerance: float = 0.02   # allowed |retention| drift on re-run
    # signature over canonical() — attributes the record to a validator identity. A bare
    # hash proves self-consistency only; anyone can mint an alternative record + hash.
    signature: str = ""
    signer: str = ""          # ed25519 pubkey hex, or the validator hotkey ss58
    sig_scheme: str = ""

    # excluded from the signed payload: a payload cannot contain its own signature
    _SIG_FIELDS = ("signature", "signer", "sig_scheme")

    # OMITTED FROM THE PAYLOAD WHEN EMPTY, and this is what makes the field addable at all.
    # `canonical()` covers every dataclass field, so a new field defaulting to `[]` would put
    # `"rejected":[]` into the canonical form of records published BEFORE it existed — and those
    # records are re-loaded and re-digested by `verify_window` on every publish. Round 1 was
    # already anchored when this was added; without this rule its digest would have moved, the
    # window check would have called it stale, and round 2 would have been WITHHELD by the gate
    # that exists to catch exactly that. An absent field and an empty one say the same thing, so
    # they get the same bytes.
    #
    # `unclaimed` joins it for the same reason and one more: it is a FLOAT, so the "empty" value
    # that must vanish is 0.0 rather than an empty list. Rounds 1 and 2 are anchored and predate
    # it; both had every live tier claimed, so 0.0 is also the honest value for them, and omitting
    # it leaves their bytes exactly as signed.
    _SPARSE = ("rejected", "unclaimed", "started_at", "published_at")

    def canonical(self) -> str:
        d = {k: v for k, v in asdict(self).items()
             if k not in self._SIG_FIELDS and not (k in self._SPARSE and not v)}
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    def sign(self, signer) -> "RoundRecord":
        """Sign the canonical payload with a Signer (eval/signing.py). Returns self."""
        self.signature = signer.sign(self.canonical().encode())
        self.signer = signer.public_id()
        self.sig_scheme = getattr(signer, "scheme", "")
        return self

    def verify_signature(self) -> bool:
        """True iff the record carries a signature valid over its canonical payload — so
        any tampering with the points/verdicts/crown invalidates it."""
        from .signing import verify
        if not (self.signature and self.signer and self.sig_scheme):
            return False
        return verify(self.canonical().encode(), self.signature, self.signer, self.sig_scheme)

    def verify_reproduction(self, other: "RoundRecord") -> tuple[bool, list]:
        """Re-run auditor: same crown decisions, and per-submission retention within
        tolerance. Judge float-flips near threshold mean we check the DECISION and the
        aggregate, not bit-exact per-point equality."""
        problems = []
        a = {s.model_id: s for s in self.submissions}
        b = {s.model_id: s for s in other.submissions}
        if set(a) != set(b):
            problems.append("submission set differs")
        for mid in set(a) & set(b):
            if abs(a[mid].retention - b[mid].retention) > self.reproduction_tolerance:
                problems.append(f"{mid}: retention {a[mid].retention} vs {b[mid].retention} "
                                f"exceeds tolerance {self.reproduction_tolerance}")
        ev_a = [(e.get("tier"), e.get("action"), e.get("king")) for e in self.events]
        ev_b = [(e.get("tier"), e.get("action"), e.get("king")) for e in other.events]
        if ev_a != ev_b:
            problems.append(f"crown decisions differ: {ev_a} vs {ev_b}")
        return (not problems), problems


def build_round_record(round_no: int, commit_root: str, round_nonce: str, teacher: str,
                       judge: str, base: str, pile_id: str, points, scored: dict,
                       events: list, weights: dict, manifest: dict | None = None,
                       noise: dict | None = None, safety: float = 3.0,
                       prev_anchor: str = "", rejected: list | None = None,
                       unclaimed: float = 0.0, started_at: float = 0.0,
                       published_at: float = 0.0) -> RoundRecord:
    def _pt(p):
        if not isinstance(p, dict):
            return {"rollout_id": getattr(p, "rollout_idx", None), "k": getattr(p, "k", None),
                    "mode": getattr(p, "mode", None)}
        # PASS THROUGH every key the caller supplied. This used to project onto
        # {rollout_id, k, mode}, which silently discarded the frozen parent_step and
        # continuation — i.e. exactly the fields that make a re-run a pure forward pass. A
        # record that drops what the scorer used is how v1's auditor came to diverge on 92% of
        # reports, so the default here is KEEP, not filter.
        return dict(p)

    pts = [_pt(p) for p in points]
    # `scored` may key the re-scored incumbent as "<hash>#incumbent" so it cannot collide with the
    # same model submitted as a challenger. The RECORD identifies it by content hash and `role`, so
    # the suffix is stripped here — an auditor matching an event's `beaten` field against model_id
    # must not have to know about an internal dict-key convention.
    subs = [SubmissionRecord(
        model_id=mid.split("#", 1)[0], miner=s.sub.miner, tier=s.sub.tier,
        retention=round(s.retention, 6), retention_lb=round(s.retention_lb, 6),
        per_point=[round(x, 4) for x in s.per_point], gates_ok=s.gates_ok, reasons=s.reasons,
        steps=list(getattr(s, "steps", []) or []), effects=list(getattr(s, "effects", []) or []),
        slices={k: [round(x, 6) for x in v] for k, v in (s.per_axis or {}).items()},
        role=getattr(s, "role", "challenger"),
        artifact_uri=getattr(s, "artifact_uri", ""),
        manifest_root=getattr(s, "manifest_root", ""),
        params=getattr(s.sub, "params", 0) or 0,
        code_bits=round(getattr(s, "code_bits", 0.0) or 0.0, 4),
        container_bits=round(getattr(s, "container_bits", 0.0) or 0.0, 4))
        for mid, s in scored.items()]
    rec = RoundRecord(round_no, commit_root, round_nonce, teacher, judge, base, pile_id,
                      # COPY the events list. It used to be stored by reference, so a caller
                      # appending one more event after signing silently invalidated the signature
                      # of an already-published record — demonstrated, and it would have made that
                      # round permanently unrepublishable.
                      pts, subs, list(events), weights,
                      manifest=dict(manifest or {}), noise=dict(noise or {}),
                      prev_anchor=prev_anchor,
                      # COPIED, for the same reason `events` is
                      rejected=[[str(hk), list(rs)] for hk, rs in (rejected or [])],
                      unclaimed=round(float(unclaimed or 0.0), 6),
                      started_at=float(started_at or 0.0),
                      published_at=float(published_at or 0.0))
    if noise:
        # derive the acceptance band from the MEASURED floor instead of a fixed 0.02, which was
        # ~200x looser than measured and would certify a materially wrong re-run.
        mk = float(noise.get("max_kl", 0.0) or 0.0)
        rec.reproduction_tolerance = max(1e-4, safety * mk)
    return rec
