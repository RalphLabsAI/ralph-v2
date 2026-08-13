"""Mirror the reigning crowns into one RalphLabsAI repo — verified, not copied.

WHY THIS EXISTS. A crowned artifact lives in the miner's own HuggingFace repo, because the miner
made it and owns it. That is right, and it leaves three problems:

  * NOBODY CAN FIND IT. The crowns have 116 and 119 downloads. They are named after a miner, and
    nothing about them says "this is the reigning champion of netuid 40". PrismML's comparable
    artifact does ~78k downloads a month.
  * THE BYTES CAN MOVE. `main` currently points at exactly the commits that were scored. The miner
    can push over it whenever they like, and then everyone downloading "the crown" gets something
    no round ever measured.
  * THERE IS NO STABLE NAME. A tier's king changes every round, so any link to a specific miner's
    repo goes stale the moment it is dethroned.

So this publishes one repo whose contents are, by construction, the current kings — the same shape
as any multi-quant repo on the Hub, so PocketPal and Enclave list both files and a user picks by
size.

WHAT MAKES IT A MIRROR AND NOT A COPY. Every artifact is re-hashed after download and compared to
the `model_id` in the signed record, which IS its content hash. A mismatch aborts that tier rather
than publishing it. Without that step this script would launder whatever the miner's repo happens
to hold today into a repo carrying our name — which is worse than not mirroring at all, because it
would carry our name.

    python -m eval.publish_champions                 # verify everything, publish nothing
    python -m eval.publish_champions --push          # verify, then upload
    env: HF_TOKEN (write scope), RALPH_CHAMPIONS_REPO
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

TRAIL = "https://huggingface.co/datasets/RalphLabsAI/ralph-v2-shakedown/resolve/main"
CHAMPIONS_REPO = os.environ.get("RALPH_CHAMPIONS_REPO", "RalphLabsAI/ralph-qwen3-8b")
MANIFEST = "champions.json"


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "ralph-publish-champions"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def parse_artifact_uri(uri: str) -> tuple[str, str]:
    """`hf://owner/name@revision` -> ("owner/name", "revision").

    The revision is not optional. Downloading `main` would fetch whatever the miner's repo holds
    now, which is the exact substitution this script exists to make impossible."""
    s = (uri or "").strip()
    if s.startswith("hf://"):
        s = s[5:]
    if "@" not in s:
        raise ValueError(f"artifact_uri has no pinned revision: {uri!r}")
    repo, rev = s.rsplit("@", 1)
    if not repo or not rev:
        raise ValueError(f"unusable artifact_uri: {uri!r}")
    return repo, rev


def latest_record() -> tuple[dict, str]:
    """The newest signed record on the trail, and its URL."""
    index = _get(f"{TRAIL}/index.json")
    rounds = sorted(index.get("rounds", []), key=lambda r: r.get("round", 0))
    if not rounds:
        raise SystemExit("the trail has no rounds")
    head = rounds[-1]
    return _get(f"{TRAIL}/{head['name']}"), f"{TRAIL}/{head['name']}"


def current_kings(record: dict) -> dict:
    """{tier: submission dict} for every occupied throne."""
    from .koth import kings_from_events
    kings = kings_from_events(record.get("events") or [])
    out = {}
    for tier, model_id in kings.items():
        rows = [s for s in record.get("submissions", []) if s.get("model_id") == model_id]
        if not rows:
            print(f"  !! {tier}: record names king {model_id[:12]}… but carries no submission for it")
            continue
        # PREFER THE INCUMBENT ROW. A held crown appears TWICE — once as the challenger it was
        # submitted as, once as the incumbent re-scored on this round's exam — and in round 2 the
        # same bytes scored 0.3030 and 0.2875, a 0.0155 spread that is the measured run-to-run
        # floor. The re-score is the number the crown decision actually used, so publishing the
        # challenger row would put the luckier of two measurements on the card.
        out[tier] = next((s for s in rows if s.get("role") == "incumbent"), rows[0])
    return out


def fetch_and_verify(sub: dict, tier: str) -> str | None:
    """Download the pinned revision and prove it is the scored artifact. Returns the .gguf path."""
    from huggingface_hub import snapshot_download

    from .identity import content_hash
    repo, rev = parse_artifact_uri(sub.get("artifact_uri", ""))
    print(f"  {tier}: fetching {repo}@{rev[:12]}…")
    d = snapshot_download(repo_id=repo, revision=rev, allow_patterns=["*.gguf"])
    got = content_hash(d)
    want = sub.get("model_id", "")
    if got != want:
        # NOT A WARNING. The artifact does not hash to what the round scored, so either the repo
        # changed under a pinned revision or the record is wrong. Either way it must not be
        # republished under our name.
        print(f"     REFUSED — content hash {got[:16]}… != record model_id {want[:16]}…")
        return None
    print(f"     verified {got[:16]}… matches the signed record")
    return next((os.path.join(d, f) for f in os.listdir(d) if f.endswith(".gguf")), None)


def readme(manifest: dict, record_url: str, repo: str) -> str:
    """One card for the whole repo, because the repo holds one file per tier.

    `model_card.render` writes a page about a single artifact, which is the wrong shape here — a
    reader arriving at this repo is choosing between the champions by size, exactly as they would
    between quantisations of any other model. Numbers come from the manifest, which came from the
    signed record, so nothing on this page is typed by hand."""
    rows = []
    for tier, m in sorted(manifest.items(), key=lambda kv: kv[1].get("code_bits") or 0):
        gb = (m.get("container_bits") or 0) * 8.19e9 / 8 / 1e9
        rows.append(f"| `ralph-qwen3-8b-{tier}.gguf` | {tier} | "
                    f"{m.get('code_bits', '?')} | {gb:.2f} GB | {m.get('retention', '?')} | "
                    f"{m.get('round', '?')} | [`{(m.get('source_repo') or '')}`]"
                    f"(https://huggingface.co/{m.get('source_repo')}) "
                    f"@`{(m.get('source_revision') or '')[:12]}` |")
    return f"""---
license: apache-2.0
library_name: gguf
base_model: Qwen/Qwen3-8B
tags: [gguf, quantized, compression, bittensor]
---

# Ralph champions — Qwen3-8B

The **reigning crowned compressions** from [Bittensor netuid 40](https://github.com/RalphLabsAI/ralph-v2),
one file per bit tier. Every round re-scores the incumbents against a fresh exam; when a crown
changes hands, the file here changes with it.

| file | tier | bits/weight | size | retention | round | scored artifact |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

## What "retention" is, and what it is not

Retention measures how much of the **pinned parent's** effect on a third-party observer model each
compression reproduces, aggregated over its **worst** slice of (observer x language x depth) rather
than its average. It is a compression-fidelity measure. **It is not a capability benchmark**, and a
high retention does not by itself mean a model is good at anything in particular.

Round record, with the exam, every per-sample measurement and the crown decision:
{record_url}

## Provenance

Each file is byte-identical to the artifact the round actually scored: it is downloaded from the
miner's own repo at the **pinned commit** named in the signed record, re-hashed, and published only
if the hash matches the `model_id` in that record. `champions.json` carries the source repo and
revision for every file, so you can fetch the original and check it yourself.

Credit for the weights belongs to the miners named in `champions.json`. This repo is a verified
mirror with a stable name, not the origin.

## Running them

Any llama.cpp-based runner. On iPhone, [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai)
and Enclave AI both load GGUF straight from the Hub — search this repo and pick a file by size.
Note that an 8B at ~4.6 GB is close to the per-app memory ceiling on iOS and needs a Pro device;
the smaller tiers are the ones that fit comfortably.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--push", action="store_true",
                    help="actually upload; without it everything is verified and nothing is written")
    ap.add_argument("--repo", default=CHAMPIONS_REPO)
    a = ap.parse_args(argv)

    record, record_url = latest_record()
    print(f"round {record.get('round')} — {record_url}")
    kings = current_kings(record)
    if not kings:
        print("no occupied thrones; nothing to publish")
        return 0
    print(f"thrones: {', '.join(f'{t}={s['model_id'][:12]}…' for t, s in sorted(kings.items()))}\n")

    # ALREADY-PUBLISHED CHECK. Re-uploading multi-gigabyte files that have not changed is slow and
    # pointless, and it churns the repo history a reader uses to see when a crown actually moved.
    try:
        published = _get(f"https://huggingface.co/{a.repo}/resolve/main/{MANIFEST}")
    except Exception:
        published = {}

    manifest, staged = dict(published), []
    for tier, sub in sorted(kings.items()):
        if (published.get(tier) or {}).get("model_id") == sub.get("model_id"):
            print(f"  {tier}: already published at this model_id — skipping")
            continue
        path = fetch_and_verify(sub, tier)
        if not path:
            continue
        repo, rev = parse_artifact_uri(sub["artifact_uri"])
        manifest[tier] = {
            "model_id": sub.get("model_id"), "tier": tier, "round": record.get("round"),
            "retention": sub.get("retention"), "miner": sub.get("miner"),
            "source_repo": repo, "source_revision": rev,
            "code_bits": sub.get("code_bits"), "container_bits": sub.get("container_bits"),
            "record": record_url,
        }
        staged.append((tier, path, f"ralph-qwen3-8b-{tier}.gguf"))

    if not staged:
        print("\nnothing new to publish")
        return 0

    print("\nto publish:")
    for tier, path, name in staged:
        print(f"  {name:<32} {os.path.getsize(path) / 1e9:5.2f} GB   (tier {tier})")

    card = readme(manifest, record_url, a.repo)
    if not a.push:
        print("\nDRY RUN — verified only. Re-run with --push to upload.")
        print(f"\n--- README.md that would be written ({len(card)} chars) ---")
        print(card[:1400] + ("\n…" if len(card) > 1400 else ""))
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("\nHF_TOKEN is not set in the environment; refusing to guess at credentials")
        return 2

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(a.repo, repo_type="model", exist_ok=True)
    for tier, path, name in staged:
        print(f"  uploading {name}…")
        api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=a.repo,
                        repo_type="model",
                        commit_message=f"{tier} champion, round {record.get('round')}")
    api.upload_file(path_or_fileobj=json.dumps(manifest, indent=1, sort_keys=True).encode(),
                    path_in_repo=MANIFEST, repo_id=a.repo, repo_type="model",
                    commit_message=f"champions as of round {record.get('round')}")
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=a.repo, repo_type="model",
                    commit_message=f"champions as of round {record.get('round')}")
    print(f"\npublished -> https://huggingface.co/{a.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
