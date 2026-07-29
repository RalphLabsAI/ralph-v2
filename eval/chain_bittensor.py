"""Real Bittensor ChainIO — what makes miners able to actually submit.

Until this existed the whole v2 stack ran against `FakeChain`, so no miner could submit
anything even if they knew the subnet existed. This is the adapter that closes that: it
satisfies `chain.ChainIO` on top of the v1 validator's `BittensorChain`, which already
implements weights, king, blacklist, block reads and commitment writes against mainnet.

HOW A MINER SUBMITS. Bittensor gives each hotkey ONE commitment slot, written with
`set_commitment` and read with `get_commitment`. A miner writes a compact JSON envelope:

    {"v": 2, "tier": "ternary-4b", "cv": "<commit_value>", "uri": "hf://acme/x@rev"}

`cv` is `H(content_hash‖salt)` — the sealed value, published BEFORE the round nonce exists, so
the artifact is bound before the miner can know what it will be scored on. The reveal
(`content_hash` + `salt`) is published after the round opens, and `intake` recomputes the hash
from the fetched bytes and refuses a mismatch. `uri` is where the bytes live, so a crown resolves
to something downloadable.

Overwriting the slot is the update mechanism — the same pattern v1 uses. The consequence a miner
must understand: the slot holds only your LATEST commitment, so committing again before a round
closes replaces what you had.

TWO SAFETY PROPERTIES, deliberate:

  * WRITES ARE OPT-IN. `read_only=True` (the default) makes every mutating call a no-op that
    logs what it WOULD have done. Weight-setting on mainnet signs with the validator hotkey, and
    nothing here should touch a live signer by accident — bringing this up on mainnet must be a
    conscious act by the operator, not a side effect of importing a module.
  * FAIL-CLOSED READS. A commitment that is missing, unparseable, or not v2 is SKIPPED with a
    reason rather than guessed at. A malformed envelope must not become a submission.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .chain import Commitment


@dataclass
class BittensorChainIO:
    """`chain.ChainIO` over the live network. `inner` is a v1 `BittensorChain`."""

    inner: object                       # karpa.chain_layer.bittensor_chain.BittensorChain
    netuid: int = 40
    read_only: bool = True              # writes disabled unless the operator opts in
    fetch_dir_for: object = None        # callable(hotkey, uri) -> local path (resolver)
    reveals: dict = field(default_factory=dict)   # hotkey -> {"content_hash", "salt"}
    log: list = field(default_factory=list)
    skipped: list = field(default_factory=list)

    # ---- reads -------------------------------------------------------------------------

    def current_block(self) -> int:
        return int(self.inner.get_current_block())

    def block_hash(self, block: int) -> str:
        return str(self.inner.get_block_hash(block))

    def read_commitments(self, min_block: int, max_block: int) -> list:
        """Every registered hotkey's current v2 commitment, resolved to a local dir.

        Bittensor exposes one slot per hotkey with no block filtering, so the window is
        advisory here: what a miner has committed IS its latest commitment. The round nonce is
        still drawn from a block after the window closes, so the commit-then-generate ordering
        that makes the score un-pre-fittable is unaffected."""
        out: list[Commitment] = []
        for hotkey in self._hotkeys():
            raw = self._commitment_of(hotkey)
            if not raw:
                continue
            env = self._parse(hotkey, raw)
            if env is None:
                continue
            rev = self.reveals.get(hotkey) or {}
            ckpt = ""
            if self.fetch_dir_for is not None:
                try:
                    ckpt = self.fetch_dir_for(hotkey, env.get("uri", "")) or ""
                except Exception as e:
                    self.skipped.append((hotkey, f"fetch failed: {type(e).__name__}: {e}"))
                    continue
            if not ckpt:
                self.skipped.append((hotkey, "no local artifact (reveal/fetch pending)"))
                continue
            out.append(Commitment(
                hotkey=hotkey, coldkey=self._coldkey_of(hotkey), tier=str(env.get("tier", "")),
                ckpt_dir=ckpt, declared_compute_h100h=float(env.get("h100h", 0.0) or 0.0),
                bond_posted=float(env.get("bond", 0.0) or 0.0),
                revealed_hash=str(rev.get("content_hash", "")), salt=str(rev.get("salt", "")),
                committed_value=str(env.get("cv", "")),
                artifact_uri=str(env.get("uri", "")),
            ))
        return out

    def commit_root(self, min_block: int, max_block: int) -> str:
        """Deterministic digest over the window's sealed values. Binds the item/observer seeds to
        the exact set of submissions, so an auditor can prove which cohort a round scored."""
        import hashlib
        h = hashlib.sha256()
        for c in sorted(self.read_commitments(min_block, max_block), key=lambda x: x.hotkey):
            h.update(f"{c.hotkey}|{c.committed_value}".encode())
            h.update(b"\0")
        return h.hexdigest()

    def get_king(self, tier: str):
        try:
            return self.inner.get_king()
        except Exception as e:
            self.skipped.append((tier, f"get_king failed: {e}"))
            return None

    # ---- writes (no-ops unless read_only=False) ------------------------------------------

    def set_weights(self, weights: dict) -> bool:
        if self.read_only:
            self.log.append(("set_weights", dict(weights)))
            return False
        return bool(self.inner.set_weights(weights))

    def set_king(self, tier: str, hotkey: str, model_id: str) -> None:
        if self.read_only:
            self.log.append(("set_king", tier, hotkey, model_id))
            return
        from karpa.chain_layer.chain_interface import KingRecord  # type: ignore
        self.inner.set_king(KingRecord(hotkey=hotkey, model_id=model_id, tier=tier))

    def blacklist(self, hotkey: str, reason: str) -> None:
        if self.read_only:
            self.log.append(("blacklist", hotkey, reason))
            return
        self.inner.blacklist(hotkey, reason)

    def publish_record(self, record) -> None:
        """Anchor the round record's digest on-chain; the record itself is published off-chain
        (HF/IPFS) because a commitment slot cannot hold it."""
        import hashlib
        digest = hashlib.sha256(record.canonical().encode()).hexdigest()
        if self.read_only:
            self.log.append(("publish_record", digest))
            return
        self.inner.commit_audit_root(digest)

    def settle_bonds(self, refunds: dict) -> None:
        # Bonds are an off-chain ledger rebuilt from commitments each epoch; there is no
        # transfer extrinsic wired yet, so record the intent rather than pretend it settled.
        self.log.append(("settle_bonds", dict(refunds)))

    # ---- helpers -----------------------------------------------------------------------

    def _hotkeys(self) -> list:
        mg = getattr(self.inner, "metagraph", None)
        if mg is not None and getattr(mg, "hotkeys", None):
            return list(mg.hotkeys)
        return []

    def _coldkey_of(self, hotkey: str) -> str:
        mg = getattr(self.inner, "metagraph", None)
        try:
            i = list(mg.hotkeys).index(hotkey)
            return str(list(mg.coldkeys)[i])
        except Exception:
            return ""

    def _commitment_of(self, hotkey: str):
        uid = self.inner.get_uid(hotkey)
        if uid is None:
            return None
        try:
            return self.inner.subtensor.get_commitment(netuid=self.netuid, uid=uid)
        except Exception as e:
            self.skipped.append((hotkey, f"get_commitment failed: {type(e).__name__}"))
            return None

    def _parse(self, hotkey: str, raw) -> dict | None:
        """Fail-closed: a commitment that is not a well-formed v2 envelope is skipped with a
        reason, never guessed at."""
        try:
            env = json.loads(str(raw))
        except Exception:
            self.skipped.append((hotkey, "commitment is not JSON (likely a v1 handshake)"))
            return None
        if not isinstance(env, dict) or env.get("v") != 2:
            self.skipped.append((hotkey, f"not a v2 commitment (v={env.get('v') if isinstance(env, dict) else '?'})"))
            return None
        if not env.get("cv") or not env.get("tier"):
            self.skipped.append((hotkey, "v2 commitment missing cv/tier"))
            return None
        return env


def build_commitment_envelope(tier: str, commit_value: str, artifact_uri: str,
                              declared_compute_h100h: float = 0.0,
                              bond_posted: float = 0.0) -> str:
    """The exact string a miner writes to its on-chain slot. Kept compact — a commitment slot is
    small — and versioned so the validator can refuse anything it does not understand."""
    env = {"v": 2, "tier": tier, "cv": commit_value, "uri": artifact_uri}
    if declared_compute_h100h:
        env["h100h"] = round(float(declared_compute_h100h), 3)
    if bond_posted:
        env["bond"] = round(float(bond_posted), 6)
    return json.dumps(env, separators=(",", ":"), sort_keys=True)
