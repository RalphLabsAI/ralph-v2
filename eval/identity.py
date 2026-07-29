"""Content-addressed checkpoint identity + commit-reveal verification.

`model_id` must be the checkpoint's CONTENT HASH, not the submitting hotkey. Keying on the
hotkey leaves three holes:
  * BAIT-AND-SWITCH — nothing binds the artifact we score to the artifact that was
    committed; a miner can commit one checkpoint and serve another.
  * COPY-GUARD is identity-based — koth's `best.model_id != king.model_id` guard compares
    hotkeys, so it says nothing about whether the bytes are the same model.
  * REGISTRY COLLISION — two submissions from one hotkey overwrite each other in the round
    registry, so a later round can re-score the wrong artifact as the reigning king.

The hash covers every file that affects BEHAVIOR — weights AND config / tokenizer /
generation config — because a weights-only hash leaves those swappable after commit while
the hash still matches (miner/package.py hashed only *.safetensors).

Commit-reveal: the miner commits H(content_hash || salt) before the round nonce is drawn;
at reveal the validator recomputes the hash of the artifact it actually fetched and checks
both that it equals the revealed hash and that it opens the on-chain commitment.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# every file type that can change model behavior
HASHED_SUFFIXES = {".safetensors", ".gguf", ".json", ".txt", ".model", ".tiktoken",
                   ".vocab", ".merges"}
_CHUNK = 1 << 20


def content_hash(ckpt_dir: str | Path) -> str:
    """Deterministic SHA-256 over all behavior-affecting files, sorted by relative path.
    Path + size are folded in so a rename or truncation changes the hash."""
    d = Path(ckpt_dir)
    if not d.is_dir():
        raise ValueError(f"not a directory: {d}")
    files = sorted((p for p in d.rglob("*")
                    if p.is_file() and p.suffix.lower() in HASHED_SUFFIXES),
                   key=lambda p: str(p.relative_to(d)))
    if not files:
        raise ValueError("no hashable files in checkpoint")
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(d)).encode())
        h.update(b"\0")
        h.update(str(p.stat().st_size).encode())
        h.update(b"\0")
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(_CHUNK), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def commit_value(content_hash_hex: str, salt: str) -> str:
    """The sealed value a miner publishes on-chain BEFORE the round nonce is drawn."""
    return hashlib.sha256(f"{content_hash_hex}|{salt}".encode()).hexdigest()


def verify_reveal(ckpt_dir: str | Path, revealed_hash: str, salt: str,
                  committed_value: str) -> tuple[bool, str]:
    """Fetch-verify. Recompute the hash of the artifact we ACTUALLY have and require that
    (a) it equals the revealed hash — otherwise the served artifact is not the revealed one
    (bait-and-switch) — and (b) H(hash||salt) opens the on-chain commitment. Returns
    (ok, content_hash_or_reason)."""
    try:
        actual = content_hash(ckpt_dir)
    except Exception as e:
        return False, f"hash failed: {e}"
    if actual != revealed_hash:
        return False, (f"artifact hash {actual[:12]} != revealed {revealed_hash[:12]} "
                       f"(bait-and-switch)")
    if commit_value(actual, salt) != committed_value:
        return False, "commit mismatch (revealed hash+salt do not open the commitment)"
    return True, actual
