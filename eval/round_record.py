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

    def canonical(self) -> str:
        d = {k: v for k, v in asdict(self).items() if k not in self._SIG_FIELDS}
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
                       noise: dict | None = None, safety: float = 3.0) -> RoundRecord:
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
    subs = [SubmissionRecord(
        model_id=mid, miner=s.sub.miner, tier=s.sub.tier,
        retention=round(s.retention, 6), retention_lb=round(s.retention_lb, 6),
        per_point=[round(x, 4) for x in s.per_point], gates_ok=s.gates_ok, reasons=s.reasons)
        for mid, s in scored.items()]
    rec = RoundRecord(round_no, commit_root, round_nonce, teacher, judge, base, pile_id,
                      pts, subs, events, weights,
                      manifest=dict(manifest or {}), noise=dict(noise or {}))
    if noise:
        # derive the acceptance band from the MEASURED floor instead of a fixed 0.02, which was
        # ~200x looser than measured and would certify a materially wrong re-run.
        mk = float(noise.get("max_kl", 0.0) or 0.0)
        rec.reproduction_tolerance = max(1e-4, safety * mk)
    return rec
