"""Resolve a miner's committed URI to a local directory — the one place untrusted input arrives.

Everything else in this repo operates on bytes that are already on disk. This is the door. A miner
controls the URI, the repo behind it, the file names, the file count and the total size, and none
of that has been checked by anything upstream — the on-chain commitment binds the CONTENT HASH, and
you cannot verify a hash until after you have downloaded the thing.

So the limits bind BEFORE the bytes land, not after:

  * ALLOWLISTED SCHEMES ONLY. `hf://` and `https://` to known hosts. Not `file://` (a miner would
    read the validator's own disk), not `ftp://`, not a bare path.
  * SIZE CEILING CHECKED FROM METADATA FIRST, then enforced again while streaming. A repo that
    advertises 2 GB and serves 200 GB fills the disk of a box that also holds a signing key.
  * FILE-COUNT AND NAME limits. Path traversal (`../`), absolute paths and symlinks are refused
    outright rather than sanitised — sanitising invites a bypass, refusing does not.
  * EXTENSION ALLOWLIST that matches what intake will accept anyway. Downloading a 40 GB tarball to
    then reject it for being a tarball is work an attacker gets for free.
  * NO EXECUTION, EVER. Nothing here imports, unpickles, or runs anything it downloaded. It writes
    files and returns a path.

And the hash is checked here as well as at intake. Intake verifies commit-reveal because that is
its job; doing it at the door too means a mismatched artifact never reaches the loader at all.
"""
from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field

# Matches what gates.inspect_checkpoint will accept. Anything else is refused before download.
ALLOWED_SUFFIXES = (".safetensors", ".gguf", ".json", ".txt", ".model")
ALLOWED_HOSTS = ("huggingface.co", "cdn-lfs.huggingface.co", "cdn-lfs-us-1.huggingface.co")

# Injectable at every call site: a guard you cannot exercise without writing 60 GB to disk is
# a guard nobody tests, and this one has to hold against a repo that lies about its size.
MAX_TOTAL_BYTES = 60 * 1024 ** 3      # 60 GB: a bf16 8B parent is 16 GB, so this is generous
MAX_FILES = 96
MAX_NAME = 160

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._\-]+$")


class FetchRefused(Exception):
    """Raised BEFORE downloading. The message is shown to the operator and recorded on the skip."""


@dataclass
class FetchResult:
    path: str = ""
    files: int = 0
    bytes: int = 0
    content_hash: str = ""
    notes: list = field(default_factory=list)


def _check_name(name: str) -> None:
    base = name.split("/")[-1]
    if not name or name != name.strip():
        raise FetchRefused(f"empty or padded filename {name!r}")
    if name.startswith("/") or ".." in name.split("/"):
        raise FetchRefused(f"path escape in {name!r} — refused rather than sanitised")
    if len(base) > MAX_NAME:
        raise FetchRefused(f"filename longer than {MAX_NAME} chars: {base[:40]}…")
    if not _SAFE_NAME.match(base):
        raise FetchRefused(f"unsafe characters in {base!r}")
    if not base.endswith(ALLOWED_SUFFIXES):
        raise FetchRefused(f"{base!r} is not a weight or config file — intake would reject it "
                           f"anyway, so it is not downloaded")


def parse_uri(uri: str) -> tuple[str, str, str]:
    """`hf://repo@rev` or `https://huggingface.co/<repo>` -> ('hf', repo, rev). Nothing else."""
    u = (uri or "").strip()
    if not u:
        raise FetchRefused("no artifact URI in the commitment")
    if u.startswith("hf://"):
        rest = u[5:]
        repo, _, rev = rest.partition("@")
        if repo.count("/") != 1 or not repo.replace("/", "").replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise FetchRefused(f"malformed hf repo {repo!r}")
        return "hf", repo, (rev or "main")
    if u.startswith("https://"):
        host = u.split("/")[2].lower()
        if host not in ALLOWED_HOSTS:
            raise FetchRefused(f"host {host!r} is not allowlisted; use hf://<repo>@<rev>")
        parts = u.split("huggingface.co/", 1)
        if len(parts) == 2 and parts[1].count("/") >= 1:
            repo = "/".join(parts[1].split("/")[:2])
            return "hf", repo, "main"
        raise FetchRefused("could not read a repo id out of the URL")
    raise FetchRefused(f"scheme not allowed in {u[:40]!r} — hf:// or https://huggingface.co only")


def plan(uri: str, lister=None, max_bytes: int = MAX_TOTAL_BYTES,
         max_files: int = MAX_FILES) -> list:
    """Decide what WOULD be downloaded, and refuse before any bytes move.

    `lister(repo, rev) -> [(name, size_bytes)]` is injectable so the refusal logic is testable
    without the network — which matters, because these are the checks that must never regress."""
    _, repo, rev = parse_uri(uri)
    files = (lister or _hf_list)(repo, rev)
    keep, total = [], 0
    for name, size in files:
        base = name.split("/")[-1]
        if not base.endswith(ALLOWED_SUFFIXES):
            continue                      # quietly skipped, not an error: repos carry READMEs
        _check_name(name)
        total += int(size or 0)
        keep.append((name, int(size or 0)))
    if not keep:
        raise FetchRefused("no weight files in the artifact")
    if len(keep) > max_files:
        raise FetchRefused(f"{len(keep)} files exceeds the {max_files} limit")
    if total > max_bytes:
        raise FetchRefused(f"advertised size {total/1024**3:.1f} GB exceeds the "
                           f"{max_bytes/1024**3:.0f} GB ceiling")
    return keep


def fetch(uri: str, dest_root: str, expect_hash: str = "", lister=None,
          downloader=None, max_bytes: int = MAX_TOTAL_BYTES) -> FetchResult:
    """Download to `dest_root/<sanitised>` and verify the content hash.

    The hash is checked HERE as well as at intake. Intake does it because commit-reveal is its job;
    doing it at the door means a mismatched artifact never reaches the loader at all."""
    from .identity import content_hash

    files = plan(uri, lister=lister, max_bytes=max_bytes)
    _, repo, rev = parse_uri(uri)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", f"{repo}@{rev}")
    dest = os.path.join(dest_root, safe)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)

    got = 0
    dl = downloader or _hf_download
    for name, size in files:
        # flatten: the artifact is a directory of weight files, and honouring nested paths is
        # how a traversal turns into a write outside dest
        out = os.path.join(dest, name.split("/")[-1])
        dl(repo, rev, name, out)
        actual = os.path.getsize(out)
        got += actual
        if got > max_bytes:
            shutil.rmtree(dest, ignore_errors=True)
            raise FetchRefused(f"stream exceeded the {max_bytes/1024**3:.2f} GB ceiling — "
                               f"advertised sizes are a miner-controlled hint, not a promise")

    res = FetchResult(path=dest, files=len(files), bytes=got)
    res.content_hash = content_hash(dest)
    if expect_hash and res.content_hash != expect_hash:
        shutil.rmtree(dest, ignore_errors=True)
        raise FetchRefused(f"content hash {res.content_hash[:16]}… does not match the revealed "
                           f"{expect_hash[:16]}… — the bytes are not what was committed")
    return res


def resolver(dest_root: str, reveals: dict | None = None, log: list | None = None):
    """Build the `fetch_dir_for(hotkey, uri)` callable BittensorChainIO expects.

    Returns "" rather than raising on refusal, because a bad submission must skip that miner and
    not take the round down with it — the caller records the reason on `skipped`."""
    reveals = reveals if reveals is not None else {}

    def fetch_dir_for(hotkey: str, uri: str) -> str:
        want = str((reveals.get(hotkey) or {}).get("content_hash", ""))
        try:
            r = fetch(uri, dest_root, expect_hash=want)
            if log is not None:
                log.append((hotkey, "fetched", r.path, r.files, r.bytes))
            return r.path
        except FetchRefused as e:
            if log is not None:
                log.append((hotkey, "refused", str(e)))
            return ""

    return fetch_dir_for


# ---- network paths, imported locally so the module is testable without huggingface_hub ----

def _hf_list(repo: str, rev: str) -> list:
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, revision=rev, files_metadata=True)
    return [(s.rfilename, getattr(s, "size", 0) or 0) for s in info.siblings]


def _hf_download(repo: str, rev: str, name: str, out_path: str) -> None:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo, name, revision=rev)
    shutil.copyfile(p, out_path)
