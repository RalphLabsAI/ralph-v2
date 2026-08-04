"""Publishing an auditor's verdicts — the half that makes "the verdict is the product" true.

WHY THIS IS A SEPARATE MODULE FROM publish.py. That one publishes the OPERATOR's round records and
its adversary is the operator. This publishes an AUDITOR's rulings and its adversary is the auditor
— a different party with a different incentive, so the guarantees are not the same shape:

  * A RECORD MAY NEVER BE REPLACED. Re-publishing round N with different content is a rewrite of
    history, and `RecordPublisher._merge` refuses it outright.
  * A VERDICT MAY LEGITIMATELY BE REVISED. An auditor that could not fetch a record rules
    INCOMPLETE, then rules again once the sink recovers. Refusing that would force it to either lie
    the first time or stay silent. So verdicts are content-addressed and the index holds a LIST per
    round: a revision is an APPEND, never an overwrite, and both rulings stay readable. Changing
    your mind is allowed; pretending you never held the first view is not.

THE CHAIN NEEDS AN OUTSIDE ANCHOR OR IT PROVES NOTHING. Verdicts chain (`prev` = the previous
verdict's hash, inside the signed body) so an auditor cannot quietly drop the one it later regrets.
But a chain the auditor computes, stored in a repo the auditor owns, compared against an index the
auditor writes, is exactly the self-referential comparison that made the first version of the
operator's anchor check worthless — `record_anchors` was implemented by no adapter, so it silently
compared operator bytes to operator memory. The same trap, one party over. So `commit_head_fn`
writes the verdict-chain head to the AUDITOR's own on-chain commitment slot, and deleting any past
verdict then changes a head that block history already recorded.

Publishing is fail-closed in the same direction as everything else: an auditor that cannot publish
its ruling must not go on setting weights as if it had, because a validator writing weights with no
published reasoning is indistinguishable from a weight-copier — which is the one thing this whole
role exists to be distinguishable from.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict

from .publish import PublishError, anchor_of

INDEX = "index.json"


def verdict_name(round_no: int, digest: str) -> str:
    """Content-addressed, so a revision cannot land on top of the ruling it revises."""
    return f"verdicts/round-{round_no:08d}-{digest[:16]}.json"


def put_verified(sink, name: str, blob: bytes) -> str:
    """Write, then READ BACK AND RE-HASH. One implementation, used for verdicts and the index.

    A `put` that returns success having stored nothing — or a truncation — is the ordinary failure
    with object stores and the HF API alike, and trusting the return value is how v1 ended up with
    an index claiming 50 rounds and serving 6."""
    uri = sink.put(name, blob)
    back = sink.get(name)
    if back is None:
        raise PublishError(f"wrote {name} but it cannot be fetched back")
    if hashlib.sha256(back).hexdigest() != hashlib.sha256(blob).hexdigest():
        raise PublishError(f"{name} reads back different from what was written")
    return uri


class VerdictPublisher:
    """Append-only, verified-after-write, hash-chained. The auditor's mirror of RecordPublisher."""

    def __init__(self, sink, state_path: str = "", auditor: str = ""):
        self.sink = sink
        self.auditor = auditor
        self.state_path = state_path
        self._hwm = self._load_hwm()

    # ---- never-shrink mark, kept OUTSIDE the sink ---------------------------

    def _load_hwm(self) -> int:
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
                # Same ambiguity the operator's publisher guards: `get` returning None means "no
                # index yet" on a first run and "the read failed" on a blip, and treating the
                # second as the first is how a full history gets overwritten with an empty one.
                raise PublishError(
                    f"verdict index is unreadable but {self._hwm} verdicts were published before; "
                    f"refusing to treat a failed read as an empty history")
            return {"verdicts": [], "head": -1, "auditor": self.auditor}
        try:
            idx = json.loads(raw.decode())
        except Exception:
            raise PublishError(f"{INDEX} is present but unparseable; refusing to overwrite it")
        idx.setdefault("verdicts", [])
        idx.setdefault("head", max((v["round"] for v in idx["verdicts"]), default=-1))
        idx.setdefault("auditor", self.auditor)
        if len(idx["verdicts"]) < self._hwm:
            raise PublishError(f"verdict index serves {len(idx['verdicts'])} but {self._hwm} were "
                               f"published before — history has shrunk; refusing to write")
        return idx

    def head_chain(self) -> str:
        """The value an auditor should commit on chain. Empty before the first verdict."""
        vs = self.load_index().get("verdicts", [])
        return vs[-1].get("chain", "") if vs else ""

    # ---- publish -----------------------------------------------------------

    def publish(self, v, commit_head_fn=None) -> dict:
        """Put one verdict, extend the chain, update the index, optionally anchor the head.

        Returns the index entry. Raises rather than returning a status: a caller that ignores a
        publish failure is the fail-open pattern this whole codebase exists to avoid."""
        digest = v.sha256()
        idx = self.load_index()
        existing = idx["verdicts"]
        prev_chain = existing[-1].get("chain", "") if existing else ""

        # ONE AUDITOR PER TRAIL. A repo holding verdicts from two identities is not an auditor's
        # record, it is a mailbox — and nobody reading it could tell which rulings to attribute to
        # whom without checking every signature individually.
        if v.auditor and idx.get("auditor") and v.auditor != idx["auditor"]:
            raise PublishError(
                f"this trail belongs to auditor {str(idx['auditor'])[:16]}… but the verdict is "
                f"signed by {v.auditor[:16]}… — use a separate repo per auditor identity")

        blob = json.dumps(asdict(v), sort_keys=True, separators=(",", ":")).encode()
        name = verdict_name(v.round, digest)
        already = self.sink.get(name)
        if already is not None:
            if hashlib.sha256(already).hexdigest() != hashlib.sha256(blob).hexdigest():
                raise PublishError(f"{name} already exists with different content")
            uri = f"(already published) {name}"
        else:
            uri = put_verified(self.sink, name, blob)

        entry = {"round": int(v.round), "verdict": v.verdict, "sha256": digest, "name": name,
                 "uri": uri, "levels_run": list(v.levels_run), "ts": float(v.ts),
                 "record_sha256": v.record_sha256, "prev": v.prev,
                 "chain": anchor_of(prev_chain, digest), "signature": v.signature}
        if any(e.get("sha256") == digest for e in existing):
            return entry                                   # idempotent re-publish, nothing to add

        merged = existing + [entry]                        # APPEND, never replace — see the header
        out = {"verdicts": merged, "head": max(e["round"] for e in merged),
               "auditor": v.auditor or idx.get("auditor", "")}
        put_verified(self.sink, INDEX, json.dumps(out, sort_keys=True, indent=1).encode())
        self._save_hwm(len(merged))

        if commit_head_fn is not None:
            # AFTER the bytes are fetchable, so the chain never points at something nobody can get.
            commit_head_fn(entry["chain"])
        return entry


def verify_verdict_trail(publisher: VerdictPublisher, head_chain: str = "") -> dict:
    """Walk an auditor's published verdicts. What a THIRD PARTY runs on an auditor.

    Answers a question the auditor cannot answer about itself: are these all of the rulings it made,
    in the order it made them? Without `head_chain` — read from the auditor's on-chain commitment —
    this compares the auditor's files against the auditor's index and proves nothing, which is
    reported rather than glossed."""
    idx = publisher.load_index()
    vs = list(idx.get("verdicts", []))
    rep = {"auditor": idx.get("auditor", ""), "n": len(vs), "broken": [], "chain_breaks": [],
           "unsigned": [], "head_chain": "", "head_checked": False}
    run = ""
    for e in vs:
        blob = publisher.sink.get(e["name"])
        if blob is None:
            rep["broken"].append({"round": e["round"], "why": "not fetchable"})
        elif hashlib.sha256(blob).hexdigest() != hashlib.sha256(
                json.dumps(json.loads(blob.decode()), sort_keys=True,
                           separators=(",", ":")).encode()).hexdigest():
            rep["broken"].append({"round": e["round"], "why": "not canonically encoded"})
        else:
            body = json.loads(blob.decode())
            if not body.get("signature"):
                rep["unsigned"].append(e["round"])
        want = anchor_of(run, e["sha256"])
        if e.get("chain") != want:
            rep["chain_breaks"].append({"round": e["round"], "have": str(e.get("chain"))[:16],
                                        "want": want[:16]})
        # carry the RECOMPUTED value, never the recorded one, or a deleted middle verdict
        # re-synchronises the walk and the final head matches a history the chain never saw
        run = want
    rep["head_chain"] = run
    if head_chain:
        rep["head_checked"] = True
        if head_chain != run:
            rep["chain_breaks"].append({"round": "HEAD", "have": head_chain[:16],
                                        "want": run[:16]})
    rep["ok"] = (not rep["broken"] and not rep["chain_breaks"] and rep["head_checked"])
    return rep
