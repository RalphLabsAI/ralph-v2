"""Round-record signing — makes a published verdict ATTRIBUTABLE, not merely self-consistent.

A bare content hash proves a record is internally consistent; it does NOT prove who
produced it. An unsigned record is repudiable and anyone can mint an alternative one with
a matching hash, so "signed audit trail" was false while the record carried only a sha256.

The signer is pluggable:
  * PRODUCTION — the validator's bittensor HOTKEY (sr25519). Signing with the same key that
    sets weights on-chain is what binds a record to the on-chain validator identity, so an
    auditor can check a published record against the metagraph.
  * LOCAL / SHADOW — ed25519 (pynacl), no wallet required.

Verification is by scheme + public id recorded IN the record, so an auditor needs nothing
but the record and the claimed identity.
"""
from __future__ import annotations

from typing import Protocol


class Signer(Protocol):
    scheme: str
    def public_id(self) -> str: ...
    def sign(self, payload: bytes) -> str: ...


class Ed25519Signer:
    """Local/shadow signer (pynacl). `seed` makes it deterministic for tests."""
    scheme = "ed25519"

    def __init__(self, seed: bytes | None = None):
        from nacl.signing import SigningKey
        self._sk = SigningKey(seed) if seed else SigningKey.generate()

    def public_id(self) -> str:
        return bytes(self._sk.verify_key).hex()

    def sign(self, payload: bytes) -> str:
        return self._sk.sign(payload).signature.hex()


class HotkeySigner:
    """Production signer: the validator's bittensor hotkey (sr25519).

        from bittensor import wallet
        signer = HotkeySigner(wallet(name=..., hotkey=...).hotkey)

    public_id is the ss58 address, so an auditor resolves it against the metagraph.
    """
    scheme = "sr25519"

    def __init__(self, keypair):
        self._kp = keypair

    def public_id(self) -> str:
        return self._kp.ss58_address

    def sign(self, payload: bytes) -> str:
        return self._kp.sign(payload).hex()


def verify(payload: bytes, signature_hex: str, public_id: str, scheme: str) -> bool:
    """Verify a signature against the recorded public id + scheme. Fail-closed on any
    error (bad hex, wrong length, unknown scheme) — never raise into a crown decision."""
    try:
        if scheme == "ed25519":
            from nacl.exceptions import BadSignatureError
            from nacl.signing import VerifyKey
            try:
                VerifyKey(bytes.fromhex(public_id)).verify(payload, bytes.fromhex(signature_hex))
                return True
            except BadSignatureError:
                return False
        if scheme == "sr25519":
            from bittensor import Keypair
            return Keypair(ss58_address=public_id).verify(payload, bytes.fromhex(signature_hex))
    except Exception:
        return False
    return False
