"""Real Bittensor ChainIO — native, and the only chain layer v2 has.

THIS USED TO WRAP THE V1 VALIDATOR. `inner` was a `karpa.chain_layer.bittensor_chain.BittensorChain`
and every call delegated to it. That made v2 depend on a separate repository which is not on PyPI,
is not declared as a dependency, and pulls in torch just to set a weight — and it made the two
codebases share a fate they should not share. v1 is retired; this talks to bittensor directly.

Two things the delegation was hiding, both of which broke the round in production:

  * `run_round` passed a raw `bt.subtensor` where a v1 `BittensorChain` was expected, so
    `commit_audit_root` did not exist and THE ANCHOR WAS NEVER WRITTEN. `head_anchor()` then fell
    back to `_head` — the value we had just computed in-process — so the publish gate compared the
    operator's bytes against the operator's memory and reported `anchor_verified=True` with nothing
    on chain at all.
  * `_hotkeys()` read `inner.metagraph.hotkeys`, but on a Subtensor `metagraph` is a METHOD, so it
    returned `[]`, no commitments were ever read, and every round scored nobody.

HOW A MINER SUBMITS. Bittensor gives each hotkey ONE commitment slot, written with
`set_commitment` and read with `get_commitment`. A miner writes a compact JSON envelope:

    {"v": 2, "tier": "ternary", "cv": "<commit_value>", "uri": "hf://acme/x@rev"}

`cv` is `H(content_hash‖salt)` — the sealed value, published BEFORE the round nonce exists, so the
artifact is bound before the miner can know what it will be scored on. The reveal (`content_hash` +
`salt`) is published after the round opens, and `intake` recomputes the hash from the fetched bytes
and refuses a mismatch. `uri` is where the bytes live, so a crown resolves to something
downloadable.

Overwriting the slot is the update mechanism. The consequence a miner must understand: the slot
holds only your LATEST commitment, so committing again before a round closes replaces what you had.

TWO SAFETY PROPERTIES, deliberate:

  * WRITES ARE OPT-IN. `read_only=True` (the default) makes every mutating call a no-op that logs
    what it WOULD have done. Weight-setting on mainnet signs with the validator hotkey, and nothing
    here should touch a live signer by accident — bringing this up on mainnet must be a conscious
    act by the operator, not a side effect of importing a module.
  * FAIL-CLOSED READS. A commitment that is missing, unparseable, or not v2 is SKIPPED with a
    reason rather than guessed at. A malformed envelope must not become a submission.

API NOTE. Written against `bittensor>=10.5,<11`: `bt.Subtensor` (capital S — the lowercase
`bt.subtensor` alias is gone), `Subtensor.metagraph(netuid)`, `get_commitment(netuid, uid)`,
`set_commitment(wallet, netuid, data)`. Bittensor 11.x is a ground-up rewrite — no
`Subtensor.metagraph`, no `get_commitment`, a dataclass `Metagraph` carrying `commitments` — so it
needs a new adapter rather than a version bump. requirements.txt pins accordingly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .chain import Commitment


@dataclass
class BittensorChainIO:
    """`chain.ChainIO` over the live network, with no v1 in the path."""

    netuid: int = 40
    network: str = "finney"
    wallet_name: str = "ralph"
    hotkey_name: str = "owner"
    read_only: bool = True              # writes disabled unless the operator opts in
    fetch_dir_for: object = None        # callable(hotkey, uri) -> local path (resolver)
    reveals: dict = field(default_factory=dict)   # hotkey -> {"content_hash", "salt"}
    log: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    burn_uid: int = 0
    # injectable so every path here is testable without a network; production leaves them None
    subtensor: object = None
    wallet: object = None
    _mg: object = None

    # ---- lazy connections --------------------------------------------------------------

    def st(self):
        if self.subtensor is None:
            import bittensor as bt
            self.subtensor = bt.Subtensor(network=self.network)
        return self.subtensor

    def wal(self):
        if self.wallet is None:
            import bittensor as bt
            self.wallet = bt.Wallet(name=self.wallet_name, hotkey=self.hotkey_name)
        return self.wallet

    def mg(self, refresh: bool = False):
        """Synced ONCE per round unless asked again. Every uid lookup used to trigger its own
        sync, which on a full subnet is a lot of RPC for a mapping that does not move mid-round."""
        if self._mg is None or refresh:
            self._mg = self.st().metagraph(netuid=self.netuid)
        return self._mg

    @property
    def me(self) -> str:
        return str(self.wal().hotkey.ss58_address)

    # ---- reads -------------------------------------------------------------------------

    def current_block(self) -> int:
        return int(self.st().get_current_block())

    def block_hash(self, block: int) -> str:
        return str(self.st().get_block_hash(block))

    def read_commitments(self, min_block: int, max_block: int, require_local: bool = True) -> list:
        """Every registered hotkey's current v2 commitment, resolved to a local dir.

        Bittensor exposes one slot per hotkey with no block filtering, so the window is advisory
        here: what a miner has committed IS its latest commitment. The round nonce is still drawn
        from a block after the window closes, so the commit-then-generate ordering that makes the
        score un-pre-fittable is unaffected."""
        out: list[Commitment] = []
        for uid, hotkey in enumerate(self._hotkeys()):
            raw = self._commitment_of(uid, hotkey)
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
            # `require_local=False` is the CPU-ORCHESTRATOR path: the artifacts belong on the
            # rented GPU's disk, not on the box that holds the signing key, so the orchestrator
            # reads the envelopes and ships them onward rather than downloading 60 GB it will
            # never open.
            if not ckpt and require_local:
                self.skipped.append((hotkey, "no local artifact (reveal/fetch pending)"))
                continue
            out.append(Commitment(
                hotkey=hotkey, coldkey=self._coldkey_of(uid), tier=str(env.get("tier", "")),
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
        """v2 HAS NO ON-CHAIN KING STORE, and that is the design rather than a gap.

        v1 kept a king record the validator wrote and read back, which meant the crown's identity
        lived in operator-controlled storage. In v2 the king is whatever the SIGNED, PUBLISHED
        record says it is — reconstructible by anyone from the trail, and covered by the anchor
        chain. Returning None keeps the protocol satisfied without inventing a second source of
        truth that could disagree with the first."""
        return None

    # ---- writes (no-ops unless read_only=False) ------------------------------------------

    def set_weights(self, weights: dict) -> bool:
        """`{hotkey: score}` -> uids, normalised. Unresolvable hotkeys are DROPPED WITH A REASON,
        never silently, because a dropped king is a crown that stops being paid."""
        if self.read_only:
            self.log.append(("set_weights", dict(weights)))
            return False
        uids, vals = [], []
        for hk, w in weights.items():
            uid = self.uid_of(hk)
            if uid is None:
                self.skipped.append((hk, "not registered on this subnet — weight dropped"))
                continue
            uids.append(int(uid))
            vals.append(max(0.0, float(w)))
        if not uids or sum(vals) <= 0:
            self.log.append(("set_weights", "no resolvable uids", dict(weights)))
            return False
        total = sum(vals)
        vals = [v / total for v in vals]
        res = self.st().set_weights(wallet=self.wal(), netuid=self.netuid, uids=uids, weights=vals)
        ok = bool(getattr(res, "success", res))
        self.log.append(("set_weights", dict(zip(uids, vals)), ok))
        return ok

    def set_burn_weights(self) -> bool:
        """Everything to the burn uid. What a validator writes when it has nothing it is willing to
        pay: a validator that sets NO weights contributes nothing to consensus and eventually
        crosses `activity_cutoff` into inactive, which is a worse outcome than an honest burn."""
        if self.read_only:
            self.log.append(("set_burn_weights", self.burn_uid))
            return False
        res = self.st().set_weights(wallet=self.wal(), netuid=self.netuid,
                                    uids=[int(self.burn_uid)], weights=[1.0])
        return bool(getattr(res, "success", res))

    def set_king(self, tier: str, hotkey: str, model_id: str) -> None:
        """Journalled, not written. See get_king: the crown lives in the signed record."""
        self.log.append(("set_king", tier, hotkey, model_id))

    def blacklist(self, hotkey: str, reason: str) -> None:
        # No slashing extrinsic exists for this; a blacklisted miner is simply not weighted, which
        # the round already does. Recording the intent beats pretending it was enforced on chain.
        self.log.append(("blacklist", hotkey, reason))

    def publish_record(self, record) -> None:
        """Anchor the round's HASH-CHAIN LINK on chain; the record itself is published off-chain
        (HF/IPFS) because a commitment slot cannot hold it.

        What is committed is A_n = H(A_{n-1} ‖ sha256(record)), not the bare digest. A commitment
        slot holds exactly one value, so committing the digest alone anchored only the newest round
        and left every earlier one checkable against nothing but the operator's own index —
        deleting or swapping an old record broke no check. Chaining makes the one slot commit to
        the whole history, and the operator cannot retroactively repair it because the superseded
        commitments are in block history."""
        from .publish import anchor_of
        anchor = anchor_of(getattr(record, "prev_anchor", ""), record.sha256())
        if self.read_only:
            self.log.append(("publish_record", anchor))
            return
        self.commit_audit_root(anchor)

    def commit_audit_root(self, value: str) -> None:
        """Write one value to our commitment slot. RAISES on failure — a silent anchor failure is
        how v1's on-chain root went 22 days stale while scoring carried on."""
        if self.read_only:
            self.log.append(("commit_audit_root", value))
            return
        res = self.st().set_commitment(wallet=self.wal(), netuid=self.netuid, data=str(value),
                                       raise_error=True)
        self.log.append(("commit_audit_root", value, bool(getattr(res, "success", res))))

    def head_anchor(self) -> str:
        """The anchor currently committed on chain, read FROM THE CHAIN.

        There is deliberately NO fallback to what we last committed in-process. That fallback used
        to exist, and it made `publish_and_gate` compare a value we computed against the same value
        we computed — the exact self-referential check the anchor exists to escape. An empty string
        means "not anchored", which the gate treats as unverified. That is the honest answer."""
        return self.commitment_of_hotkey(self.me)

    def commitment_of_hotkey(self, hotkey: str) -> str:
        uid = self.uid_of(hotkey)
        if uid is None:
            return ""
        try:
            return str(self.st().get_commitment(netuid=self.netuid, uid=int(uid)) or "")
        except Exception:
            return ""

    def settle_bonds(self, refunds: dict) -> None:
        # Bonds are an off-chain ledger rebuilt from commitments each epoch; there is no transfer
        # extrinsic wired yet, so record the intent rather than pretend it settled.
        self.log.append(("settle_bonds", dict(refunds)))

    # ---- helpers -----------------------------------------------------------------------

    def uid_of(self, hotkey: str):
        try:
            return list(self.mg().hotkeys).index(hotkey)
        except Exception:
            return None

    def _hotkeys(self) -> list:
        try:
            return list(self.mg().hotkeys)
        except Exception as e:
            self.skipped.append(("metagraph", f"could not sync: {type(e).__name__}: {e}"))
            return []

    def _coldkey_of(self, uid: int) -> str:
        try:
            return str(list(self.mg().coldkeys)[uid])
        except Exception:
            return ""

    def _commitment_of(self, uid: int, hotkey: str):
        try:
            return self.st().get_commitment(netuid=self.netuid, uid=int(uid))
        except Exception as e:
            self.skipped.append((hotkey, f"get_commitment failed: {type(e).__name__}"))
            return None

    def _parse(self, hotkey: str, raw) -> dict | None:
        """Fail-closed: a commitment that is not a well-formed v2 envelope is skipped with a
        reason, never guessed at."""
        try:
            env = json.loads(str(raw))
        except Exception:
            self.skipped.append((hotkey, "commitment is not JSON"))
            return None
        if not isinstance(env, dict) or env.get("v") != 2:
            self.skipped.append((hotkey, f"not a v2 commitment "
                                         f"(v={env.get('v') if isinstance(env, dict) else '?'})"))
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
