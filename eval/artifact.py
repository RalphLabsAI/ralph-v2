"""Artifact manifest and locator — the path from "king" to bytes you can download.

THE GAP THIS CLOSES. The product promise is "every crown ships as a downloadable, runnable
model", but a crown published only a rolling `content_hash`: an opaque digest with no locator
and no file list. Nobody could fetch the king, and nobody could prove the published file set was
the scored file set. For a compression subnet whose deliverable IS the artifact, that is the
difference between a leaderboard and a product.

A manifest is a per-file {path, size, sha256} list plus a root over it. That buys three things
one rolling hash cannot:

  * RESOLVABLE. `artifact_uri` says where the bytes live, and the manifest says exactly which
    bytes, so a third party can fetch the king and verify it independently.
  * ENUMERABLE. `content_hash` alone cannot answer "which files were scored?" — you can only ask
    "does this whole directory still hash the same?". A per-file list makes the scored set a
    published fact.
  * NO STOWAWAYS. Verification requires an EXACT match, so an extra file is a rejection rather
    than something ignored. That closes a real hole: `gates` only rejects a file whose suffix is
    non-empty and not allowlisted, and `identity.HASHED_SUFFIXES` only hashes known suffixes —
    so a SUFFIX-LESS file passed inspection AND escaped the hash entirely. Uninspected,
    unhashed bytes could ride along inside a committed artifact.

The root is defined to equal `identity.content_hash` so nothing downstream has to change: the
same value keeps binding commit-reveal, and the manifest is the enumeration of what went into
it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .identity import HASHED_SUFFIXES, content_hash

_CHUNK = 1 << 20


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class FileEntry:
    path: str
    size: int
    sha256: str

    def as_dict(self) -> dict:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass
class ArtifactManifest:
    root: str = ""                    # == identity.content_hash(dir)
    artifact_uri: str = ""            # where the bytes can be fetched (hf://, ipfs://, https://)
    total_bytes: int = 0
    files: list = field(default_factory=list)
    extra_files: list = field(default_factory=list)   # present but NOT behavior-affecting

    def as_dict(self) -> dict:
        return {"root": self.root, "artifact_uri": self.artifact_uri,
                "total_bytes": self.total_bytes,
                "files": [f.as_dict() for f in self.files],
                "extra_files": self.extra_files}

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


def build_manifest(ckpt_dir, artifact_uri: str = "") -> ArtifactManifest:
    """Enumerate a checkpoint directory into a verifiable manifest.

    `files` are the behaviour-affecting ones (the set `content_hash` covers). Anything else is
    recorded in `extra_files` rather than silently ignored — including SUFFIX-LESS files, which
    slip past both the gates allowlist and the identity hash and are exactly the stowaway
    channel this manifest exists to surface."""
    d = Path(ckpt_dir)
    if not d.is_dir():
        raise ValueError(f"not a directory: {d}")
    m = ArtifactManifest(artifact_uri=artifact_uri)
    for p in sorted(d.rglob("*"), key=lambda x: str(x.relative_to(d))):
        if not p.is_file():
            continue
        rel = str(p.relative_to(d))
        if p.suffix.lower() in HASHED_SUFFIXES:
            m.files.append(FileEntry(rel, p.stat().st_size, _sha256_file(p)))
            m.total_bytes += p.stat().st_size
        else:
            m.extra_files.append(rel)
    if not m.files:
        raise ValueError("no behaviour-affecting files in checkpoint")
    m.root = content_hash(d)
    return m


def verify_manifest(ckpt_dir, manifest: ArtifactManifest,
                    allow_extra: bool = False) -> tuple[bool, list]:
    """Does the directory on disk match the published manifest EXACTLY?

    Fail-closed and exact by default: a missing file, a changed byte, a size mismatch, or an
    UNDECLARED EXTRA FILE is a rejection. Tolerating extras is what lets uninspected bytes ride
    along inside an artifact that still hashes correctly at the top level."""
    d = Path(ckpt_dir)
    reasons: list[str] = []
    if not d.is_dir():
        return False, [f"not a directory: {d}"]

    on_disk = {}
    extras_on_disk = []
    for p in d.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(d))
        if p.suffix.lower() in HASHED_SUFFIXES:
            on_disk[rel] = p
        else:
            extras_on_disk.append(rel)

    declared = {f.path: f for f in manifest.files}
    for rel, entry in declared.items():
        p = on_disk.get(rel)
        if p is None:
            reasons.append(f"missing file declared in manifest: {rel}")
            continue
        if p.stat().st_size != entry.size:
            reasons.append(f"size mismatch for {rel}: {p.stat().st_size} != {entry.size}")
            continue
        if _sha256_file(p) != entry.sha256:
            reasons.append(f"sha256 mismatch for {rel}")
    for rel in sorted(set(on_disk) - set(declared)):
        reasons.append(f"undeclared file present: {rel}")
    if not allow_extra:
        for rel in sorted(set(extras_on_disk) - set(manifest.extra_files)):
            reasons.append(f"undeclared non-hashed file present: {rel} — uninspected bytes")

    if not reasons:
        root = content_hash(d)
        if root != manifest.root:
            reasons.append(f"root mismatch: {root[:12]} != {manifest.root[:12]}")
    return (not reasons), reasons
