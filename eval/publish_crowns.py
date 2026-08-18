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

    python -m eval.publish_crowns                 # verify everything, publish nothing
    python -m eval.publish_crowns --push          # verify, then upload
    env: HF_TOKEN (write scope), RALPH_CROWNS_REPO
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

# FOLLOWS THE LIVE TRAIL, from the same env var the round publishes to. This was pinned to the
# shakedown repo, which was correct while that was the only trail — and would have kept mirroring
# crowns from an ARCHIVED chain forever once live rounds moved to their own history, quietly
# publishing a champion no current round had crowned.
TRAIL_REPO = os.environ.get("RALPH_HF_REPO", "RalphLabsAI/ralph-v2-rounds")
TRAIL = f"https://huggingface.co/datasets/{TRAIL_REPO}/resolve/main"
CROWNS_REPO = os.environ.get("RALPH_CROWNS_REPO", "RalphLabsAI/ralph-crowns")
MANIFEST = "crowns.json"


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
        # MERGED, because the two facts live in DIFFERENT ROWS. A held crown appears twice — once
        # as the artifact its miner submitted, once as the incumbent re-scored on this round's
        # exam — and neither row alone is publishable:
        #
        #   retention   comes from the INCUMBENT row. In round 2 the same bytes scored 0.3030 as a
        #               challenger and 0.2875 re-scored, a 0.0155 spread that is exactly the
        #               measured run-to-run floor. The re-score is what the crown decision used;
        #               publishing the challenger's number puts the luckier measurement on the card.
        #   code_bits   comes from the CHALLENGER row. The incumbent is re-scored, not re-ingested,
        #   container   so it carries no bit measurement at all — 0.0/0.0/0 params. Taking those
        #   params      renders the crown as "0.0 bits, 0.00 GB", which is how the first draft of
        #               this card described a 4.61 GB model.
        base = next((s for s in rows if s.get("role") == "incumbent"), rows[0])
        measured = next((s for s in rows if (s.get("params") or 0) > 0), base)
        out[tier] = {**base, **{k: measured.get(k) for k in
                                ("code_bits", "container_bits", "params")},
                     # the locator must come from the row that was actually ingested
                     "artifact_uri": measured.get("artifact_uri") or base.get("artifact_uri", "")}
    return out


def fetch_and_verify(sub: dict, tier: str) -> str | None:
    """Download the pinned revision and prove it is the scored artifact. Returns the .gguf path."""
    from huggingface_hub import snapshot_download

    from .identity import HASHED_SUFFIXES, content_hash
    repo, rev = parse_artifact_uri(sub.get("artifact_uri", ""))
    print(f"  {tier}: fetching {repo}@{rev[:12]}…")
    # THE SAME FILE SET THE SCORER HASHED, derived from HASHED_SUFFIXES rather than guessed.
    # `allow_patterns=["*.gguf"]` fetched only the weights, but `content_hash` covers .json, .txt,
    # .model and more — so an artifact carrying a config.json hashed differently here than it did
    # at intake, and the mirror REFUSED a perfectly good crown as if its bytes had changed. It
    # looked exactly like a miner swapping the file underneath us. The sub4 crown matched only
    # because its repo happens to contain nothing but the .gguf.
    d = snapshot_download(repo_id=repo, revision=rev,
                          allow_patterns=[f"*{ext}" for ext in sorted(HASHED_SUFFIXES)])
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

# Ralph crowns — Qwen3-8B

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
if the hash matches the `model_id` in that record. `crowns.json` carries the source repo and
revision for every file, so you can fetch the original and check it yourself.

Credit for the weights belongs to the miners named in `crowns.json`. This repo is a verified
mirror with a stable name, not the origin.

## Running them

Any llama.cpp-based runner. On iPhone, [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai)
and Enclave AI both load GGUF straight from the Hub — search this repo and pick a file by size.
Note that an 8B at ~4.6 GB is close to the per-app memory ceiling on iOS and needs a Pro device;
the smaller tiers are the ones that fit comfortably.
"""


def publish(repo: str = "", push: bool = True, out=None) -> dict:
    """Mirror the current crowns. Returns a report; NEVER raises.

    CALLED AT THE END OF A ROUND, which is why it cannot raise. The round is already scored,
    signed, published and anchored by the time this runs — that work is done and correct whatever
    happens here. A mirroring failure must show up as a line in the log and a follow-up, not as a
    round that reports failure after having succeeded.

    Idempotent: a crown already published at the same `model_id` is skipped, so an unchanged tier
    costs one small HTTP read rather than re-uploading gigabytes every round."""
    import io as _io
    buf = _io.StringIO()
    real = sys.stdout
    try:
        sys.stdout = buf
        rc = main(["--repo", repo or CROWNS_REPO] + (["--push"] if push else []))
    except SystemExit as e:                       # argparse/`--push` without a token
        rc = int(getattr(e, "code", 1) or 0)
    except Exception as e:
        rc = 1
        buf.write(f"\ncrown publishing raised {type(e).__name__}: {e}\n")
    finally:
        sys.stdout = real
    text = buf.getvalue()
    if out is not None:
        for line in text.splitlines():
            out(f"    {line}\n")
    return {"ok": rc == 0, "rc": rc, "log": text}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--push", action="store_true",
                    help="actually upload; without it everything is verified and nothing is written")
    ap.add_argument("--repo", default=CROWNS_REPO)
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

    manifest, staged, failed = dict(published), [], []
    for tier, sub in sorted(kings.items()):
        if (published.get(tier) or {}).get("model_id") == sub.get("model_id"):
            print(f"  {tier}: already published at this model_id — skipping")
            continue
        # PER TIER, so one bad crown cannot take the others down with it. A miner deleting their
        # repo, a revision going missing, a dependency absent on this box — each of those is a
        # reason to skip ONE tier and say so, not a reason to leave the other crowns unmirrored.
        try:
            path = fetch_and_verify(sub, tier)
        except Exception as e:
            print(f"  {tier}: FAILED — {type(e).__name__}: {e}")
            failed.append(tier)
            continue
        if not path:
            failed.append(tier)
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
        # A NON-ZERO EXIT WHEN SOMETHING FAILED, even though nothing was staged — otherwise a round
        # whose only crown could not be fetched reports "nothing new to publish" and looks healthy.
        if failed:
            print(f"\nnothing published; {len(failed)} tier(s) failed: {', '.join(failed)}")
            return 1
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
                    commit_message=f"crowns as of round {record.get('round')}")
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=a.repo, repo_type="model",
                    commit_message=f"crowns as of round {record.get('round')}")
    print(f"\npublished -> https://huggingface.co/{a.repo}")
    if failed:
        print(f"  !! {len(failed)} tier(s) did NOT mirror: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
