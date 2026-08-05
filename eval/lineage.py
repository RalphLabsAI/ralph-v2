"""Who is king, derived from the published trail — the ONE derivation, used by scorer and auditor.

WHY THIS EXISTS. `Tournament` was constructed fresh every round and nothing rebuilt its `kings`, so
`koth.consider` took the OPEN-THRONE branch every single time and crowned `max(retention)` outright.
The dethrone margin, the paired bootstrap, `axis_regression`, the noise gate and the whole anti-copy
guarantee were unreachable code. The subnet documented a king of the hill and ran a leaderboard that
reset hourly.

WHY IT IS DERIVED AND NOT STORED. A local `kings.json` would be faster and it would be a second
source of truth that the operator controls — precisely what `BittensorChainIO.get_king` returns None
to avoid. The published records are already signed, already hash-chained, and already anchored on
chain, so the crown lineage is recoverable from them by anyone. Deriving it means an outsider can
check who the king should be, and it means the operator cannot quietly install one.

WHY BOTH SIDES CALL THIS FUNCTION. `rerun.audit_emission` reconstructs the same map to decide
whether a round's weights were legitimate. If the scorer derived kings one way and the audit another,
they would disagree about who holds the throne and every honest round would fail its own audit —
the same "two copies of a rule diverge" failure that disabled the per-slice floor earlier, but on
the field that moves emission. There is one walk, here, and both call it.

THE EVENT GRAMMAR, in the order `koth.Tournament.consider` emits it:
    crown     an empty throne is taken outright            -> king := event.king
    dethrone  the challenger beat the incumbent by margin  -> king := event.king
    hold      the incumbent survived                       -> king unchanged (event.king names it)
    vacate    the incumbent re-scored below the floor      -> throne emptied
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Reign:
    """A tier's current king, with enough to RE-SCORE it: the bytes have to be refetchable, or the
    incumbent cannot be re-measured on this round's items and the dethrone test has nothing to
    compare against."""
    tier: str = ""
    model_id: str = ""
    miner: str = ""
    artifact_uri: str = ""
    manifest_root: str = ""
    retention: float = 0.0
    crowned_round: int = -1
    reign: int = 0

    def as_king(self):
        from .koth import King
        return King(miner=self.miner, model_id=self.model_id, retention=self.retention,
                    crowned_round=self.crowned_round, reign=self.reign,
                    artifact_uri=self.artifact_uri, manifest_root=self.manifest_root)


def apply_events(kings: dict, events, round_no: int = -1, submissions=None) -> dict:
    """Apply ONE record's events to a tier->Reign map. Returns a new map; does not mutate.

    `submissions` is that record's submission list, used to bind a crowned model_id to the miner
    and the artifact locator. The event's own `miner` field is deliberately NOT trusted for that:
    it is written by the operator beside the weights, so using it to decide who gets paid would be
    circular — the same reasoning `audit_emission` uses."""
    out = {t: Reign(**vars(r)) for t, r in kings.items()}
    by_model = {}
    for s in (submissions or []):
        by_model.setdefault(getattr(s, "model_id", None) or s.get("model_id"), s)

    def field(s, name, default=""):
        return (getattr(s, name, None) if not isinstance(s, dict) else s.get(name)) or default

    for e in events or []:
        tier = e.get("tier")
        action = e.get("action")
        if action == "vacate":
            out.pop(tier, None)
            continue
        mid = e.get("king")
        if not mid:
            continue
        if action in ("crown", "dethrone"):
            s = by_model.get(mid)
            out[tier] = Reign(
                tier=tier, model_id=mid,
                miner=field(s, "miner") if s is not None else (e.get("miner") or ""),
                artifact_uri=field(s, "artifact_uri"),
                manifest_root=field(s, "manifest_root"),
                retention=float(field(s, "retention", 0.0) or e.get("retention", 0.0) or 0.0),
                crowned_round=int(e.get("round", round_no)), reign=0)
        elif tier in out and out[tier].model_id == mid:
            out[tier].reign += 1
    return out


def replay(records) -> dict:
    """Walk published records IN ROUND ORDER and return the current tier->Reign map.

    Order is not optional: a dethrone applied before the crown it beats produces a different throne.
    The records are sorted here rather than trusted to arrive sorted, because the index is operator
    -written and the anchor chain — not the list order — is what makes the sequence authoritative."""
    kings: dict = {}
    for rec in sorted(records, key=lambda r: int(getattr(r, "round", 0) or 0)):
        kings = apply_events(kings, getattr(rec, "events", None) or [],
                             round_no=int(getattr(rec, "round", -1) or -1),
                             submissions=getattr(rec, "submissions", None) or [])
    return kings


def replay_from_trail(publisher, max_rounds: int = 0, out=None) -> dict:
    """Fetch the published trail and replay it. What the orchestrator calls before a round.

    A record that will not fetch or will not verify ABORTS rather than being skipped. Skipping one
    would silently rebuild a throne from an incomplete history — most likely handing the crown back
    to whoever held it before the missing round, which is a live emission change produced by a
    network error. Refusing means the round does not run, which is the honest failure."""
    from .publish import roundtrip_reason
    from .rerun import record_from_blob

    idx = publisher.load_index()
    entries = sorted(idx.get("rounds", []), key=lambda r: int(r["round"]))
    if max_rounds:
        entries = entries[-max_rounds:]
    recs = []
    for e in entries:
        blob = publisher.sink.get(e["name"])
        if blob is None:
            raise RuntimeError(
                f"round {e['round']} is indexed but not fetchable, so the crown lineage cannot be "
                f"rebuilt. Refusing to score against a partial history — the throne it produces "
                f"would be wrong, and wrong thrones move emission.")
        ok, why = roundtrip_reason(blob, e.get("sha256", ""), expect_round=int(e["round"]),
                                   expect_signature=e.get("signature", ""))
        if not ok:
            raise RuntimeError(f"round {e['round']} does not verify ({why}); refusing to rebuild "
                               f"the crown lineage from it")
        recs.append(record_from_blob(blob))
    kings = replay(recs)
    if out is not None:
        out.write(f"  lineage   : {len(recs)} round(s) replayed -> "
                  f"{ {t: (r.model_id[:10] + '…') for t, r in kings.items()} or 'no king yet'}\n")
    return kings
