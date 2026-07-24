"""Miner-side: package a student checkpoint into a submission.

A miner trains/quantizes a student however they like, then packages it here. The
package is what the on-chain commit binds to and what the validator fetches and gates.

Submission = a checkpoint dir (safetensors only) + a manifest. The commit is
sealed commit-then-reveal: the miner commits H(content_hash ‖ salt) on-chain BEFORE
the round's points are minted (so the reveal is provenance-ordered, earliest wins), and
reveals content_hash + salt after. The winning artifact is published only after a delay
window, so the live king isn't copyable mid-round.

The content hash is over the SORTED safetensors files' bytes, so it is deterministic and
independent of directory layout or upload path — the same weights always hash the same.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Manifest:
    tier: str                     # which size/bit tier this is submitted to
    teacher_pair: str             # the pinned (teacher, base) pair id this student targets
    params: int                   # recomputed from the state dict (validator re-checks)
    effective_bits: float
    declared_compute_h100h: float # generation + training + scoring + rollout (reconciled)
    student_base: str             # which permitted base this was distilled from
    format_version: int = 1


# Identity MUST be byte-identical to the validator's or commit-reveal can NEVER verify.
# This module previously hashed only *.safetensors (leaving config/tokenizer swappable
# after commit) and sealed with a different separator than the validator — either alone
# breaks every reveal. Both now delegate to the single shared implementation.
from eval.identity import commit_value, content_hash  # noqa: F401  (re-exported API)


def build_submission(ckpt_dir: str | Path, tier: str, teacher_pair: str,
                     student_base: str, declared_compute_h100h: float,
                     salt: str) -> dict:
    """Produce everything a miner needs: the manifest, the content hash, and the
    on-chain commit value. Uses the validator's own inspector so the miner sees the
    exact params/bits the validator will re-derive (no surprises at intake)."""
    from eval.gates import inspect_checkpoint

    insp = inspect_checkpoint(ckpt_dir)
    if not insp.ok:
        raise ValueError(f"checkpoint fails intake gates: {insp.reasons}")

    manifest = Manifest(
        tier=tier, teacher_pair=teacher_pair, params=insp.params,
        effective_bits=insp.effective_bits_per_param,
        declared_compute_h100h=declared_compute_h100h, student_base=student_base)
    # Write the manifest BEFORE hashing: the content hash covers .json files, so it must be
    # computed over the exact directory the validator will fetch (which includes the
    # manifest). Hashing first would commit a hash that never matches at reveal -> every
    # honest submission rejected as bait-and-switch. Writing it under the hash also binds
    # the manifest, so a miner cannot swap the declared tier/compute after committing.
    (Path(ckpt_dir) / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2))
    chash = content_hash(ckpt_dir)
    return {
        "manifest": asdict(manifest),
        "content_hash": chash,
        "commit_value": commit_value(chash, salt),   # commit this on-chain now
        "reveal": {"content_hash": chash, "salt": salt},  # reveal after the round opens
    }
