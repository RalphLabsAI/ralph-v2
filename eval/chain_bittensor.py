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

API NOTE. Written against `bittensor>=11.1,<12`, which is a ground-up rewrite of the 10.x SDK this
file used to target. Nothing carried over: `Subtensor` has no `get_current_block`, no `metagraph`,
no `get_commitment`, no `set_commitment`, no `set_weights` and no `.substrate`. What replaces them:

    block           `st.block` (property)          was get_current_block()
    block hash      `st.block_info(n).hash`        was get_block_hash(n)
    storage read    `st.query_map(item, params)`   was st.substrate.query_map(module=, storage_function=)
                    with `item` from `bt.storage.<Container>.<Item>`
    uid <-> hotkey  storage `SubtensorModule.Keys` / `.Uids`   was metagraph.hotkeys
    coldkey         storage `SubtensorModule.Owner(hotkey)`    was metagraph.coldkeys
    extrinsics      build a call (`bt.SetWeights(...)`, `bt.calls.Commitments.set_commitment(...)`)
                    then `st.submit_call(call, wallet)`

`_decode_commitment` is unchanged and still correct: 11.x returns the same
`{"info": {"fields": [{"RawN": "0x…"}]}}` shape, and it must keep WALKING that structure rather than
indexing it — a payload longer than one Raw field is split across several, and a decoder that reads
only the first one reported ZERO v2 commitments on a subnet that had sixteen.

There is a real sync/async split in 11.x: `bt.Subtensor` is synchronous, but `bt.metagraph.fetch` is
a coroutine. This file stays entirely on the synchronous storage reads rather than dragging an event
loop into the validator, which is why `mg()` builds its own small view instead of calling `fetch`.
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
        """`{uid: hotkey}` for this subnet, read ONCE per round unless asked again.

        Not `bt.metagraph.fetch` — that is a coroutine in 11.x, and the whole validator is
        synchronous. It also returns far more than this needs (stake, emission, ranks, prices) for
        a mapping that never moves mid-round. One storage read answers every uid lookup in a round.
        """
        if self._mg is None or refresh:
            try:
                self._mg = {int(uid): str(getattr(v, "value", v)) for uid, v in
                            self.st().query_map(_item("SubtensorModule", "Keys"), [self.netuid])}
            except Exception as e:
                # `log`, not `skipped`: a failed metagraph read is our problem, not a miner's, and
                # `skipped` is the column an operator reads to find out why a miner was not scored.
                self.log.append(("metagraph", f"could not read Keys: {type(e).__name__}: {e}"))
                self._mg = {}
        return self._mg

    @property
    def me(self) -> str:
        return str(self.wal().hotkey.ss58_address)

    # ---- reads -------------------------------------------------------------------------

    def current_block(self) -> int:
        return int(self.st().block)

    def block_hash(self, block: int) -> str:
        return str(self.st().block_info(int(block)).hash)

    def commitments_map(self) -> dict:
        """`{hotkey: raw commitment}` in ONE query instead of one per uid.

        The per-uid path is `get_commitment(netuid, uid)`, which on a 256-slot subnet means 256
        sequential RPC round-trips AND an SDK error log for each of the ~110 slots that hold
        nothing — minutes of wall clock and a screenful of empty ERROR lines before a round can even
        start. The storage map answers the same question in one call, and an empty slot is simply
        an absent key rather than an exception.

        In 11.x the map is reached as `st.query_map(bt.storage.Commitments.CommitmentOf, [netuid])`
        — the typed storage item replaces the module/storage_function string pair, and there is no
        `.substrate` attribute to reach through any more. The KEY is already the hotkey ss58, so no
        uid round-trip is needed to attribute a commitment."""
        out: dict = {}
        try:
            res = self.st().query_map(_item("Commitments", "CommitmentOf"), [self.netuid])
        except Exception as e:
            # `log`, not `skipped`: skipped is the list of SUBMISSIONS we refused and it is read
            # by operators looking for why a miner was not scored. Our own RPC path degrading is
            # not a miner's problem and must not appear in their column.
            self.log.append(("commitments_map", f"query_map unavailable "
                                                f"({type(e).__name__}) — falling back to per-uid"))
            return out
        for k, v in res:
            val = getattr(v, "value", v)
            raw = _decode_commitment(val)
            if raw:
                # THE WRITE HEIGHT COMES WITH IT. Without it nothing can check WHEN a commitment
                # was made, which is the whole basis of commit-then-generate — see read_commitments.
                blk = 0
                try:
                    blk = int((val or {}).get("block") or 0)
                except Exception:
                    blk = 0
                out[str(k)] = (raw, blk)
        return out

    def read_commitments(self, min_block: int, max_block: int, require_local: bool = True) -> list:
        """Every registered hotkey's current v2 commitment, resolved to a local dir.

        Bittensor exposes one slot per hotkey with no block filtering, so the window is advisory
        here: what a miner has committed IS its latest commitment. The round nonce is still drawn
        from a block after the window closes, so the commit-then-generate ordering that makes the
        score un-pre-fittable is unaffected."""
        out: list[Commitment] = []
        cmap = self.commitments_map()
        for uid, hotkey in enumerate(self._hotkeys()):
            entry = cmap.get(hotkey) if cmap else (self._commitment_of(uid, hotkey), 0)
            raw, wrote_at = entry if isinstance(entry, tuple) else (entry, 0)
            if not raw:
                continue
            # SEALED BEFORE THE NONCE, OR NOT SCORED. `min_block`/`max_block` were accepted and
            # then ignored, and the commitment's write height was never read — so nothing checked
            # WHEN a miner committed. Since one slot holds cv, ch and salt in a single write, a
            # miner could watch for the nonce block, see which items and which observer it draws,
            # fit an artifact to that exam, and write all three values together: verify_reveal
            # passes trivially because they were computed together. That defeats the one property
            # the whole commit-then-generate ordering exists to provide.
            #
            # `max_block` IS the nonce block (run_orchestrated draws the nonce from `now` and
            # passes it as `hi`), so a commitment written at or after it was not sealed first.
            # Reported by a miner, 2026-08-05.
            if max_block and wrote_at and wrote_at >= max_block:
                self.skipped.append((hotkey, f"committed at block {wrote_at}, at or after the "
                                             f"round nonce block {max_block} — not sealed before "
                                             f"the exam was drawn"))
                continue
            env = self._parse(hotkey, raw)
            if env is None:
                continue
            # THE REVEAL RIDES IN THE ENVELOPE. `reveals` was an injectable dict that nothing in
            # production ever populated, so every submission arrived unrevealed and intake skipped
            # its seal check. A miner's commitment slot is the only channel they have, so the
            # reveal is written back into it alongside the `cv` it has to match.
            rev = self.reveals.get(hotkey) or {}
            # BOTH SPELLINGS. `cmd_reveal` writes `ch` but PRINTS `{"content_hash", "salt"}` just
            # above it, and tells a miner without the bittensor SDK to publish the string by hand —
            # so the tool showed one field name and wrote another. A miner did exactly what the
            # output suggested, their reveal was valid and on chain, and round 1 rejected them as
            # "committed but not revealed". The seal check is unchanged and still binding: whichever
            # key carries it, H(content_hash‖salt) must equal the `cv` sealed before the nonce. Be
            # liberal about the spelling, never about the proof.
            if not rev.get("content_hash"):
                got = env.get("ch") or env.get("content_hash")
                if got:
                    rev = {"content_hash": str(got), "salt": str(env.get("salt", ""))}
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
        the exact set of submissions, so an auditor can prove which cohort a round scored.

        `require_local=False` IS THE POINT, and getting this wrong was a blocker. A commit root is a
        digest over what miners wrote ON CHAIN; whether this box could download their bytes is a
        separate and later question. The default (`require_local=True`) drops every submission whose
        artifact is not on local disk — so on the CPU orchestrator, which deliberately never fetches
        artifacts, EVERY submission was dropped and commit_root returned
        e3b0c44298fc1c14… (the digest of nothing) no matter who had submitted. The round's exam
        would then attest to no cohort at all, which is precisely the property the value exists to
        provide. It also stops the single-box path downloading every artifact twice."""
        import hashlib
        h = hashlib.sha256()
        for c in sorted(self.read_commitments(min_block, max_block, require_local=False),
                        key=lambda x: x.hotkey):
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
        ok = self._submit(self._set_weights_call(uids, vals), "set_weights")
        self.log.append(("set_weights", dict(zip(uids, vals)), ok))
        return ok

    def set_burn_weights(self) -> bool:
        """Everything to the burn uid. What a validator writes when it has nothing it is willing to
        pay: a validator that sets NO weights contributes nothing to consensus and eventually
        crosses `activity_cutoff` into inactive, which is a worse outcome than an honest burn.

        REFUSES TO BURN TO A HOTKEY WE CONTROL, because on netuid 40 the default `burn_uid=0` IS
        the validator's own hotkey — verified on chain: uid 0 is
        5HijSRHd9wUmk51UE8Kia7vmx6kD2jwqJLn9bY1frQ4aiTUs. Bittensor has no designated burn address;
        uid 0 is simply whoever registered first, and here that is us. This method is called by the
        AUDITOR, which third parties are meant to run — so with the shipped default an independent
        auditor would have written 100% of its weight to us and called it a burn. That is not a
        burn, it is a self-vote wearing one's name, and it would be found."""
        if self.read_only:
            self.log.append(("set_burn_weights", self.burn_uid))
            return False
        target = self.mg().get(int(self.burn_uid))
        if target and target == self.me:
            self.log.append(("set_burn_weights", "REFUSED",
                             f"uid {self.burn_uid} is this validator's own hotkey — set "
                             f"RALPH_BURN_UID to a uid you do not control"))
            return False
        return self._submit(self._set_weights_call([int(self.burn_uid)], [1.0]),
                            "set_burn_weights")

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
        import bittensor as bt
        payload = str(value).encode()
        # RawN variants are fixed-width, so a value that is not exactly a supported length has to be
        # split or padded rather than silently truncated. The anchor is 64 hex chars, which is Raw64
        # exactly; anything else is a programming error here and should say so.
        if len(payload) != 64:
            raise ValueError(f"anchor must be 64 bytes to fit one Raw64 field, got {len(payload)}")
        call = bt.calls.Commitments.set_commitment(
            self.netuid, {"fields": [{"Raw64": payload}]})
        ok = self._submit(call, "commit_audit_root", raise_on_fail=True)
        self.log.append(("commit_audit_root", value, ok))

    def head_anchor(self) -> str:
        """The anchor currently committed on chain, read FROM THE CHAIN.

        There is deliberately NO fallback to what we last committed in-process. That fallback used
        to exist, and it made `publish_and_gate` compare a value we computed against the same value
        we computed — the exact self-referential check the anchor exists to escape. An empty string
        means "not anchored", which the gate treats as unverified. That is the honest answer."""
        return self.commitment_of_hotkey(self.me)

    def commitment_of_hotkey(self, hotkey: str) -> str:
        try:
            return str(self._commitment_of(0, hotkey) or "")
        except Exception:
            return ""

    def settle_bonds(self, refunds: dict) -> None:
        # Bonds are an off-chain ledger rebuilt from commitments each epoch; there is no transfer
        # extrinsic wired yet, so record the intent rather than pretend it settled.
        self.log.append(("settle_bonds", dict(refunds)))

    # ---- writes: build a call, submit it with the wallet --------------------------------

    def _set_weights_call(self, uids: list, vals: list):
        """11.x has no `Subtensor.set_weights`. An extrinsic is a CALL OBJECT submitted with a
        wallet, which is a better shape for us anyway: the call can be built and inspected without
        a signer, so `read_only` can log exactly what would have gone on chain."""
        import bittensor as bt
        return bt.SetWeights(netuid=self.netuid, uids=[int(u) for u in uids],
                             weights=[float(v) for v in vals])

    def _submit(self, call, what: str, raise_on_fail: bool = False) -> bool:
        """One place where anything is signed. Every mutating path goes through here so there is a
        single line to audit for "what can this process write"."""
        try:
            res = self.st().submit_call(call, self.wal())
        except Exception as e:
            self.log.append((what, f"submit failed: {type(e).__name__}: {e}"))
            if raise_on_fail:
                # A SILENT ANCHOR FAILURE is how v1's on-chain root went 22 days stale while
                # scoring carried on. The anchor is the one write that must never fail quietly.
                raise
            return False
        ok = bool(getattr(res, "success", getattr(res, "is_success", res)))
        if not ok and raise_on_fail:
            raise RuntimeError(f"{what} was submitted but not accepted: {res!r}")
        return ok

    # ---- helpers -----------------------------------------------------------------------

    def uid_of(self, hotkey: str):
        """Reverse of the Keys map. Returns None for an unregistered hotkey, which every caller
        treats as "not on this subnet" rather than as uid 0."""
        for uid, hk in self.mg().items():
            if hk == hotkey:
                return uid
        return None

    def _hotkeys(self) -> list:
        m = self.mg()
        if not m:
            self.skipped.append(("metagraph", "could not read the uid -> hotkey map"))
        return [m[u] for u in sorted(m)]

    def _coldkey_of(self, uid: int) -> str:
        """The OPERATOR behind a hotkey — the identity anti-grind economics key on, and the one
        worth deduplicating by, since hotkeys are cheap and a funded coldkey is not."""
        hk = self.mg().get(int(uid))
        if not hk:
            return ""
        try:
            v = self.st().query(_item("SubtensorModule", "Owner"), [hk])
            # `getattr(v, "value", "")` is WRONG here and returned empty for every hotkey: 11.x
            # hands back a plain str for this item, and a str has no `.value`, so the default won.
            # Fall back to the object ITSELF, never to a blank.
            return str(getattr(v, "value", v) or "")
        except Exception:
            return ""

    def _commitment_of(self, uid: int, hotkey: str):
        """Single-slot read, used only where the whole map would be wasteful. 11.x has no
        `get_commitment`, so this is the same storage item the map walks, keyed by hotkey."""
        try:
            val = self.st().query(_item("Commitments", "CommitmentOf"), [self.netuid, hotkey])
            return _decode_commitment(getattr(val, "value", val) or {})
        except Exception as e:
            self.skipped.append((hotkey, f"commitment read failed: {type(e).__name__}"))
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


def _item(container: str, name: str):
    """Resolve a typed storage item, e.g. `_item("Commitments", "CommitmentOf")`.

    ISOLATED SO THE CROWN-PATH SUITE STAYS INSTALLABLE WITHOUT THE SDK. 10.x addressed storage by
    strings, so this module could be exercised with a stub and no `bittensor` on the box — a
    property requirements.txt states out loud. 11.x addresses it by typed objects that only the
    package can build, which would have made `import bittensor` a hard dependency of running the
    tests. One indirection keeps the seam stubbable."""
    import bittensor as bt
    return getattr(getattr(bt.storage, container), name)


def _decode_commitment(val) -> str:
    """`{"info": {"fields": [{"Raw64": "0x…"}]}}` -> the string a miner wrote.

    Field variants are RawN for assorted N and can arrive nested, so this walks rather than
    indexes: an unexpected shape returns "" and the commitment is skipped with a reason, which is
    the fail-closed direction."""
    parts: list = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            if x.startswith("0x"):
                try:
                    parts.append(bytes.fromhex(x[2:]).decode("utf-8", "ignore"))
                except Exception:
                    pass
            else:
                parts.append(x)

    walk(((val or {}).get("info") or {}).get("fields") or [])
    return "".join(parts)


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
