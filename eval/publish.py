"""Publishing round records, fail-CLOSED — and refusing to pay out a round nobody can check.

WHY THIS IS ITS OWN MODULE AND ITS OWN GATE. eval/rerun.py can audit a record; nothing was putting
records anywhere an auditor could fetch them. That is not a small omission, it is the exact way v1
failed: `publish_king.py` was fail-OPEN, so when publishing broke the epoch carried on setting
weights, the on-chain anchor went 22 days stale, `king.json`'s lineage regressed from 50 entries to
6 because a partial write overwrote a fuller one, and `events.json` was never published at all. The
scoring was fine the whole time. The audit trail was the thing that rotted, silently, because
nothing depended on it.

So the ordering is inverted here. The old `_write_back` did:

    set_king -> set_weights -> publish_record          # weights already paid when publish throws

and this does:

    publish -> READ BACK AND VERIFY -> heartbeat -> set_king -> set_weights

FOUR PROPERTIES, each one a v1 post-mortem:

  1. VERIFY AFTER WRITE. A `put` that returns success having stored nothing, or having stored a
     truncated body, is the common failure with object stores and the HF API alike. We read the
     blob back and re-hash it. Trusting the return value of the write is how you get an index that
     claims 50 rounds and serves 6.

  2. NEVER SHRINK. The index is append-only and merges by round number. A publisher holding a
     shorter history than what is already live REFUSES rather than overwrites — v1 lost 44 lineage
     entries to exactly this, from a process that had simply started with cold state.

  3. HEARTBEAT OVER A WINDOW, not just the current round. Publish-then-delete is the obvious
     tamper: score honestly, publish, then remove the round you rigged. So the gate re-fetches the
     last N records every round and re-checks their digests against the on-chain anchors. A record
     that vanishes or changes stops the payout for the CURRENT round, which is the only leverage
     the mechanism has.

  4. FAIL-CLOSED MEANS HOLD, NOT HALT — and it is worth being precise, because overstating this
     would be its own dishonesty. Refusing `set_weights` does not stop emission; the previously
     set weights persist on chain, so the last VERIFIABLY PUBLISHED crown keeps earning until the
     operator fixes publishing. That is the right direction (an unauditable new crown cannot take
     over) but it is a hold, not a stop, and an operator who breaks publishing while their own
     model is king benefits from the freeze. That residual is why the gate re-verifies the WINDOW
     rather than only the current round, and why `stale_rounds` is reported for a watcher to alarm
     on rather than being silently tolerated.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Protocol


class Sink(Protocol):
    """Somewhere bytes can be put and later fetched by anyone. Deliberately tiny: HF, S3, IPFS and
    a plain directory all satisfy it, and the gate's guarantees do not depend on which."""

    def put(self, name: str, blob: bytes) -> str: ...   # returns a resolvable URI
    def get(self, name: str) -> bytes | None: ...


class LocalSink:
    """Directory-backed. Real for a cold-standby mirror, and the only sink the tests need."""

    def __init__(self, root: str, uri_prefix: str = ""):
        self.root = root
        self.uri_prefix = uri_prefix or f"file://{os.path.abspath(root)}"
        os.makedirs(root, exist_ok=True)

    def _p(self, name: str) -> str:
        p = os.path.join(self.root, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def put(self, name: str, blob: bytes) -> str:
        p = self._p(name)
        # atomic-ish: a reader must never see a half-written record, because a truncated record
        # fails its own signature check and looks like tampering rather than a bad write
        tmp = p + ".part"
        with open(tmp, "wb") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
        return f"{self.uri_prefix}/{name}"

    def get(self, name: str) -> bytes | None:
        try:
            with open(os.path.join(self.root, name), "rb") as fh:
                return fh.read()
        except OSError:
            return None


class HFSink:
    """HuggingFace dataset repo. v1's audit-reports repo was the one part of the trail that never
    broke — 57 crowns, 9 measurement failures and 426 submissions all still fetchable — so it is
    the default in production. Import is local so nothing here needs `huggingface_hub` installed."""

    def __init__(self, repo_id: str, token: str = "", revision: str = "main"):
        self.repo_id, self.revision = repo_id, revision
        self._token = token or os.environ.get("RALPH_HF_TOKEN", "")

    def put(self, name: str, blob: bytes) -> str:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=blob, path_in_repo=name, repo_id=self.repo_id,
                            repo_type="dataset", token=self._token, revision=self.revision)
        return f"hf://datasets/{self.repo_id}@{self.revision}/{name}"

    def get(self, name: str) -> bytes | None:
        from huggingface_hub import hf_hub_download
        try:
            # force_download: a cache hit would make "verify after write" verify our own memory
            # instead of what the repo actually serves, which is the whole point of the check
            p = hf_hub_download(self.repo_id, name, repo_type="dataset", token=self._token,
                                revision=self.revision, force_download=True)
            with open(p, "rb") as fh:
                return fh.read()
        except Exception:
            return None


INDEX = "index.json"


def record_name(round_no: int, digest: str) -> str:
    """Content-addressed in the filename, so a record cannot be swapped for a different one at the
    same path without the name changing."""
    return f"rounds/round-{round_no:08d}-{digest[:16]}.json"


@dataclass
class Published:
    round: int
    sha256: str
    uri: str
    name: str
    signer: str = ""


@dataclass
class PublishReport:
    published: Published | None = None
    verified: bool = False
    stale_rounds: list = field(default_factory=list)   # in-window rounds that no longer verify
    checked_rounds: list = field(default_factory=list)
    anchors_checked: int = 0
    reasons: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verified and not self.stale_rounds and not self.reasons


class PublishError(RuntimeError):
    """Raised INSTEAD of setting weights. The name is deliberately not `PublishWarning`."""


class RecordPublisher:
    """Publishes records, maintains the append-only index, and re-verifies a trailing window."""

    def __init__(self, sink: Sink, window: int = 8, state_path: str = ""):
        self.sink = sink
        self.window = window
        # LOCAL HIGH-WATER MARK, and it is load-bearing rather than an optimisation. `sink.get`
        # returning None is ambiguous: it means "no index yet" on a first run and "the read failed"
        # on a network blip — and treating the second as the first is PRECISELY how v1 overwrote a
        # 50-entry lineage with 6. So the count we last saw is remembered locally and a sink that
        # suddenly claims to hold less is refused rather than believed. Persisted so a restart is
        # not a fresh start; kept out of the sink deliberately, because a compromised sink must not
        # be able to lower our own idea of how much history exists.
        self.state_path = state_path
        self._hwm = self._load_hwm()

    def _load_hwm(self) -> int:
        if not self.state_path:
            return 0
        try:
            with open(self.state_path) as fh:
                return int(json.load(fh).get("count", 0))
        except Exception:
            return 0

    def _save_hwm(self, count: int) -> None:
        self._hwm = max(self._hwm, count)
        if not self.state_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.state_path)), exist_ok=True)
        tmp = self.state_path + ".part"
        with open(tmp, "w") as fh:
            json.dump({"count": self._hwm}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.state_path)

    # ---- index -------------------------------------------------------------

    def load_index(self) -> dict:
        raw = self.sink.get(INDEX)
        if not raw:
            if self._hwm > 0:
                raise PublishError(
                    f"index is unreadable but {self._hwm} rounds were published before; refusing "
                    f"to treat a failed read as an empty history (this is the v1 lineage bug)")
            return {"rounds": [], "head": -1}
        try:
            idx = json.loads(raw.decode())
        except Exception:
            # A corrupt index must NOT be silently replaced with an empty one — that is the
            # never-shrink violation with extra steps. Refuse and let a human look.
            raise PublishError(f"{INDEX} is present but unparseable; refusing to overwrite it")
        idx.setdefault("rounds", [])
        idx.setdefault("head", max((r["round"] for r in idx["rounds"]), default=-1))
        if len(idx["rounds"]) < self._hwm:
            raise PublishError(f"index serves {len(idx['rounds'])} rounds but {self._hwm} were "
                               f"published before — history has shrunk; refusing to write")
        return idx

    def _merge(self, idx: dict, entry: dict) -> dict:
        by_round = {r["round"]: r for r in idx["rounds"]}
        prev = by_round.get(entry["round"])
        if prev and prev.get("sha256") != entry["sha256"]:
            # Re-publishing a round with DIFFERENT content is a rewrite of history. Allowing it
            # would let an operator replace the round an auditor is objecting to.
            raise PublishError(
                f"round {entry['round']} already published as {prev.get('sha256')[:12]}…; "
                f"refusing to replace it with {entry['sha256'][:12]}…")
        by_round[entry["round"]] = entry
        merged = sorted(by_round.values(), key=lambda r: r["round"])
        if len(merged) < len(idx["rounds"]):
            raise PublishError("merged index is shorter than the live one — refusing to shrink")
        return {"rounds": merged, "head": merged[-1]["round"]}

    # ---- publish -----------------------------------------------------------

    def publish(self, record) -> Published:
        digest = record.sha256()
        blob = json.dumps(asdict(record), sort_keys=True, separators=(",", ":")).encode()
        name = record_name(record.round, digest)
        uri = self.sink.put(name, blob)

        # VERIFY AFTER WRITE. Not paranoia: this is the failure that made v1's index claim more
        # rounds than it served.
        back = self.sink.get(name)
        if back is None:
            raise PublishError(f"wrote {name} but it cannot be fetched back")
        if hashlib.sha256(back).hexdigest() != hashlib.sha256(blob).hexdigest():
            raise PublishError(f"{name} reads back different from what was written")
        # and the round-trip must still be a valid, signed record — a body that survives byte
        # comparison but fails its own signature means the record was already broken upstream
        if not self._roundtrip_ok(back, digest):
            raise PublishError(f"{name} does not round-trip to a signature-valid record")

        pub = Published(round=record.round, sha256=digest, uri=uri, name=name,
                        signer=getattr(record, "signer", ""))
        idx = self._merge(self.load_index(), asdict(pub))
        self.sink.put(INDEX, json.dumps(idx, sort_keys=True, indent=1).encode())
        back_idx = self.sink.get(INDEX)
        if back_idx is None:
            raise PublishError("index write could not be read back")
        if len(json.loads(back_idx.decode()).get("rounds", [])) != len(idx["rounds"]):
            raise PublishError("index reads back with a different number of rounds")
        self._save_hwm(len(idx["rounds"]))
        return pub

    @staticmethod
    def _roundtrip_ok(blob: bytes, digest: str) -> bool:
        from .round_record import RoundRecord, SubmissionRecord
        try:
            raw = json.loads(blob.decode())
            subs = [SubmissionRecord(**s) for s in raw.pop("submissions", [])]
            rec = RoundRecord(**{**raw, "submissions": subs})
        except Exception:
            return False
        if rec.sha256() != digest:
            return False
        return rec.verify_signature() if rec.signature else True

    # ---- heartbeat ---------------------------------------------------------

    def verify_window(self, upto_round: int, anchors=None) -> tuple[list, list, int]:
        """Re-fetch the trailing window and re-check it. Returns (stale, checked, n_anchors).

        Guards publish-then-delete: score honestly, publish, then remove the round you rigged.
        Without this, the gate only ever proves that the CURRENT round exists at the moment it is
        written, which an operator can undo a second later."""
        idx = self.load_index()
        recent = [r for r in idx["rounds"] if r["round"] <= upto_round][-self.window:]
        stale, checked, n_anchor = [], [], 0
        for r in recent:
            checked.append(r["round"])
            blob = self.sink.get(r["name"])
            if blob is None:
                stale.append({"round": r["round"], "why": "no longer fetchable"})
                continue
            if not self._roundtrip_ok(blob, r["sha256"]):
                stale.append({"round": r["round"], "why": "digest or signature no longer matches"})
                continue
            if anchors is not None:
                a = anchors.get(r["round"]) if hasattr(anchors, "get") else None
                if a is None:
                    stale.append({"round": r["round"], "why": "no on-chain anchor"})
                    continue
                n_anchor += 1
                if a != r["sha256"]:
                    # The chain is the authority. A published record that disagrees with what was
                    # anchored is a swap, and the anchor is the half the operator cannot rewrite.
                    stale.append({"round": r["round"],
                                  "why": f"anchor {a[:12]}… != published {r['sha256'][:12]}…"})
        return stale, checked, n_anchor


def publish_and_gate(publisher: RecordPublisher, record, anchors=None, anchor_fn=None,
                     require_signature: bool = True) -> PublishReport:
    """Publish, verify, re-check the window. `report.ok` is False -> DO NOT set weights.

    `anchors` may be a mapping OR a zero-arg callable returning one. Prefer the callable: the
    current round is anchored partway through this function, so a snapshot taken by the caller
    would always be missing it and every round would look unanchored.

    Returns a report rather than raising for the recoverable cases, so a caller can log precisely
    what is wrong and keep serving the previous crown. Raises only when continuing would corrupt
    the published history (never-shrink, history rewrite, unparseable index)."""
    rep = PublishReport()
    if record is None:
        rep.reasons.append("round produced no record — nothing to publish, so nothing to pay out")
        return rep
    if require_signature and not (getattr(record, "signature", "") and record.verify_signature()):
        # An unsigned record is attributable to nobody, so publishing it buys no accountability.
        # Refusing here (rather than at audit time) keeps unsigned rounds out of the history.
        rep.reasons.append("record is unsigned or its signature does not verify")
        return rep
    rep.published = publisher.publish(record)
    rep.verified = True
    # Anchor AFTER the bytes are fetchable and BEFORE the window check, so the current round is
    # covered by the same anchor comparison as every other round in the window. Anchoring first
    # would leave a window where the chain points at bytes nobody can get.
    if anchor_fn is not None:
        try:
            anchor_fn(record)
        except Exception as e:
            rep.reasons.append(f"on-chain anchor failed: {type(e).__name__}: {e}")
            return rep
    resolved = anchors() if callable(anchors) else anchors
    rep.stale_rounds, rep.checked_rounds, rep.anchors_checked = publisher.verify_window(
        record.round, resolved)
    return rep


@dataclass
class HistoryReport:
    """Is the TRAIL complete? Distinct from "is this round correct", which is eval/rerun.py."""
    head: int = -1
    n_rounds: int = 0
    gaps: list = field(default_factory=list)        # round numbers claimed by no entry
    broken: list = field(default_factory=list)      # entries that no longer fetch/verify
    unanchored: list = field(default_factory=list)  # published but not anchored on chain
    mismatched: list = field(default_factory=list)  # published != anchored

    @property
    def ok(self) -> bool:
        return not (self.gaps or self.broken or self.mismatched)


def verify_history(publisher: RecordPublisher, anchors=None, first_round: int | None = None
                   ) -> HistoryReport:
    """Walk the WHOLE index, not just the trailing window.

    The window heartbeat is what the gate can afford every round; this is what an outsider runs
    once to ask whether the history has holes. GAPS matter on their own: a round that was scored,
    paid out and never published is invisible to any per-record check, because there is no record
    to check. v1 had exactly that shape — the HF report repo held 57 crowns while king.json's
    lineage held 6, and nothing compared the two."""
    idx = publisher.load_index()
    rounds = sorted(idx.get("rounds", []), key=lambda r: r["round"])
    rep = HistoryReport(head=idx.get("head", -1), n_rounds=len(rounds))
    if not rounds:
        return rep
    have = {r["round"] for r in rounds}
    lo = rounds[0]["round"] if first_round is None else first_round
    rep.gaps = [n for n in range(lo, rounds[-1]["round"] + 1) if n not in have]
    resolved = anchors() if callable(anchors) else anchors
    for r in rounds:
        blob = publisher.sink.get(r["name"])
        if blob is None:
            rep.broken.append({"round": r["round"], "why": "not fetchable"})
            continue
        if not publisher._roundtrip_ok(blob, r["sha256"]):
            rep.broken.append({"round": r["round"], "why": "digest or signature mismatch"})
            continue
        if resolved is not None:
            a = resolved.get(r["round"])
            if a is None:
                rep.unanchored.append(r["round"])
            elif a != r["sha256"]:
                rep.mismatched.append({"round": r["round"], "anchor": a[:16],
                                       "published": r["sha256"][:16]})
    return rep
