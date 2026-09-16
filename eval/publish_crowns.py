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

THE CARD IS THE MAINTAINER'S; THE PUBLISHER OWNS ONE BLOCK OF IT. The repo's README carries
hand-written launch copy, device receipts and a reviewed license section that no round produced,
and the first version of this script replaced the whole file with its template on every crown
change — one dethrone away from deleting all of it. Now the publisher writes exactly one delimited
block (the crowns table, the checksums and each file's license status) plus the front-matter keys
it can vouch for, and leaves every other byte of the card as it found it. A card without the
markers is adopted once — its "## Current crowns" section becomes the block — and a repo with no
card at all gets the full template.

A LICENSE IS DECLARED FOR BYTES, NOT FOR A TIER. `crowns.json` carries `license` beside
`license_sha256`; the declaration counts only while the mirrored file still hashes to what was
reviewed, so a crown change voids it without anyone remembering to. The repo badge is one id when
every crown declares the same one and `other` + "pending" otherwise.

    python -m eval.publish_crowns                 # verify everything, publish nothing
    python -m eval.publish_crowns --push          # verify, then upload
    python -m eval.publish_crowns --render-card   # print the card the next publish would write
    python -m eval.publish_crowns --declare-license sub2=Apache-2.0 --push    # after review
    env: HF_TOKEN (write scope), RALPH_CROWNS_REPO
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

# FOLLOWS THE LIVE TRAIL, from the same env var the round publishes to. This was pinned to the
# shakedown repo, which was correct while that was the only trail — and would have kept mirroring
# crowns from an ARCHIVED chain forever once live rounds moved to their own history, quietly
# publishing a champion no current round had crowned.
TRAIL_REPO = os.environ.get("RALPH_HF_REPO", "RalphLabsAI/ralph-v2-rounds")
TRAIL = f"https://huggingface.co/datasets/{TRAIL_REPO}/resolve/main"
CROWNS_REPO = os.environ.get("RALPH_CROWNS_REPO", "RalphLabsAI/ralph-crowns")
MANIFEST = "crowns.json"
PARENT = "Qwen/Qwen3-8B"
ARTIFACT_METADATA_FIELDS = ("filename", "file_bytes", "file_sha256",
                            "chat_template_present")
LICENSE_FIELDS = ("license", "license_sha256")

# The one region of the card this script owns. Everything outside the pair is the maintainer's.
CROWNS_BEGIN = "<!-- ralph:managed:crowns:begin -->"
CROWNS_END = "<!-- ralph:managed:crowns:end -->"
# The section the first hand-written card used for the table, adopted into the block once.
_LEGACY_CROWNS_HEADING = re.compile(r"(?im)^##[ \t]+Current crowns\b[^\n]*$")
_NEXT_H2 = re.compile(r"(?m)^##[ \t]+(?!#)")
_H1 = re.compile(r"(?m)^#[ \t]+[^\n]*\n")
_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_FM_KEY = re.compile(r"^([A-Za-z_][\w-]*)[ \t]*:[ \t]*(.*)$")
# Front-matter keys the publisher can vouch for; `license` is derived, `tags` is a union.
OWNED_FRONT_MATTER = {"library_name": "gguf", "pipeline_tag": "text-generation",
                      "base_model": PARENT}
REQUIRED_TAGS = ("gguf", "conversational", "qwen3", "quantized", "compression", "bittensor")


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "ralph-publish-champions"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def _get_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "ralph-publish-champions"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8")


def published_state(repo: str) -> tuple[dict, str]:
    """Return the manifest and the exact Hub commit it came from, when available.

    Reading `main` and later committing against an unrelated head can overwrite another
    publisher's crown transition. Resolve the head first, read the manifest at that immutable
    revision, and pass the revision back to `create_commit(parent_commit=...)` as a compare-and-swap
    guard. A repository without a manifest is valid and returns an empty mapping with its head.
    """
    head = ""
    try:
        info = _get(f"https://huggingface.co/api/models/{quote(repo, safe='/')}")
        head = str(info.get("sha") or "")
    except Exception:
        # Keep dry-run verification usable if the model-info endpoint is temporarily unavailable.
        # The push path will still use one atomic commit; it just cannot add the race guard.
        pass
    try:
        rev = head or "main"
        manifest = _get(f"https://huggingface.co/{repo}/resolve/{rev}/{MANIFEST}")
        return (manifest if isinstance(manifest, dict) else {}), head
    except Exception:
        return {}, head


def published_card(repo: str, revision: str = "") -> str:
    """Read the existing card at the same immutable head used by the commit guard.

    A missing card is a valid empty-repository state. Every other read failure is allowed to
    propagate: rendering without the live card would replace the maintainer's copy with the
    template on the next crown transition, which is the exact loss the managed block exists to
    prevent."""
    try:
        return _get_text(
            f"https://huggingface.co/{repo}/resolve/{revision or 'main'}/README.md")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return ""
        raise


# ---------------------------------------------------------------- front matter

def _split_front_matter(card: str) -> tuple[list, str]:
    """(front-matter lines, body); the list is empty when the card has no front matter."""
    text = (card or "").replace("\r\n", "\n")
    m = _FRONT_MATTER.match(text)
    if not m:
        return [], text
    return m.group(1).split("\n"), text[m.end():]


def front_matter_license(card: str) -> str:
    """The one license id the live card's front matter declares, or "" when it declares none,
    several, or a placeholder. Never read from prose: `license: apache-2.0` in a sentence is
    not Hub metadata."""
    vals = [v for line in _split_front_matter(card)[0]
            for k, v in [_FM_KEY.match(line).groups() if _FM_KEY.match(line) else ("", "")]
            if k == "license"]
    if len(vals) != 1:
        return ""
    v = vals[0].strip().strip("'\"")
    return "" if v.lower() in ("", "other", "unknown") else v


def merge_front_matter(card: str, manifest: dict) -> str:
    """Set the keys the publisher owns and leave every other key as the maintainer wrote it.

    `license` is derived from the manifest (see `repo_license`); `license_name` exists only while
    the badge is `other`. `tags` is a union with the existing order first, so a tag the maintainer
    added for discovery is never dropped."""
    lines, body = _split_front_matter(card)
    lic, name = repo_license(manifest)
    want = dict(OWNED_FRONT_MATTER, license=lic)
    out, seen, tags_done = [], set(), False
    for line in lines:
        m = _FM_KEY.match(line)
        key = m.group(1) if m else None
        if key == "license_name":
            continue                                    # rewritten beside `license`, or dropped
        if key in want:
            out.append(f"{key}: {want[key]}")
            seen.add(key)
            if key == "license" and name:
                out.append(f"license_name: {name}")
            continue
        if key == "tags":
            tags_done = True
            val = m.group(2).strip()
            if val.startswith("["):
                have = [t.strip().strip("'\"") for t in val.strip("[]").split(",") if t.strip()]
                out.append("tags: [" + ", ".join(have + [t for t in REQUIRED_TAGS
                                                          if t not in have]) + "]")
                continue
            # a block-style list is the maintainer's; leave it exactly as written
        out.append(line)
    for key in ("license", "library_name", "pipeline_tag", "base_model"):
        if key not in seen:
            out.append(f"{key}: {want[key]}")
            if key == "license" and name:
                out.append(f"license_name: {name}")
    if not tags_done:
        out.append("tags: [" + ", ".join(REQUIRED_TAGS) + "]")
    return "---\n" + "\n".join(out) + "\n---\n" + body


# ---------------------------------------------------------------- licenses

def licensed(entry: dict) -> str:
    """The id a maintainer declared FOR THESE BYTES, or "". The declaration is bound to the file's
    sha256, so a crown change — new bytes — voids it without anyone having to remember to."""
    lic = str((entry or {}).get("license") or "").strip()
    if lic and entry.get("license_sha256") and entry.get("license_sha256") == entry.get("file_sha256"):
        return lic
    return ""


def declare_license(entry: dict, spdx: str) -> dict:
    out = dict(entry)
    out["license"] = spdx.strip()
    out["license_sha256"] = entry.get("file_sha256")
    return out


def repo_license(manifest: dict) -> tuple[str, str]:
    """(front-matter `license`, `license_name` or "") for the aggregate repo: one id when every
    crown declares the same one for its current bytes, else `other` and pending."""
    rows = [e for e in (manifest or {}).values() if isinstance(e, dict)]
    ids = {licensed(e) for e in rows}
    if rows and len(ids) == 1 and next(iter(ids)):
        return next(iter(ids)).lower(), ""
    return "other", "Artifact license pending"


# ---------------------------------------------------------------- artifact metadata

def file_sha256(path: str | os.PathLike) -> str:
    """Hash the GGUF file itself without loading a multi-gigabyte artifact into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_metadata(path: str, filename: str) -> dict:
    """Direct, byte-level metadata for the exact GGUF that will be uploaded unchanged."""
    from .gguf import read_gguf
    info = read_gguf(path)
    if not info.ok:
        raise ValueError("cannot publish GGUF metadata: " + "; ".join(info.reasons))
    return {
        "filename": filename,
        "file_bytes": info.file_bytes,
        "file_sha256": file_sha256(path),
        "chat_template_present": info.chat_template_present,
    }


def needs_metadata_refresh(entry: dict, filename: str) -> bool:
    """Whether an unchanged crown needs direct artifact metadata filled in."""
    if not isinstance(entry, dict) or entry.get("filename") != filename:
        return True
    if type(entry.get("file_bytes")) is not int or entry["file_bytes"] <= 0:
        return True
    sha = entry.get("file_sha256")
    if (not isinstance(sha, str) or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha.lower())):
        return True
    return type(entry.get("chat_template_present")) is not bool


def manifest_entry(*, tier: str, sub: dict, round_no: int, record_url: str,
                   metadata: dict, previous: dict | None = None,
                   adopt_license: str = "") -> dict:
    """Build one manifest row without rewriting an unchanged crown's historical facts.

    A reigning artifact is re-scored on later rounds. That new score is useful in the round record,
    but it is not the score/round at which the artifact took the crown. Metadata backfills must
    therefore preserve the existing `round`, `retention`, and source fields byte-for-byte and add
    only facts measured directly from the fetched GGUF. `adopt_license` carries a declaration the
    live card already makes into the manifest, for an UNCHANGED crown only; new bytes start with
    no license at all."""
    unchanged = bool(previous and previous.get("model_id") == sub.get("model_id"))
    if unchanged:
        out = dict(previous)
    else:
        repo, rev = parse_artifact_uri(sub.get("artifact_uri", ""))
        out = {
            "model_id": sub.get("model_id"), "tier": tier, "round": round_no,
            "retention": sub.get("retention"), "miner": sub.get("miner"),
            "source_repo": repo, "source_revision": rev,
            "code_bits": sub.get("code_bits"), "container_bits": sub.get("container_bits"),
            "record": record_url,
        }
    out.update({k: metadata[k] for k in ARTIFACT_METADATA_FIELDS})
    if unchanged and adopt_license and not licensed(out):
        out = declare_license(out, adopt_license)
    return out


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
    ggufs = sorted(str(p) for p in Path(d).rglob("*.gguf") if p.is_file())
    if len(ggufs) != 1:
        print(f"     REFUSED — expected exactly one GGUF in the scored package, found {len(ggufs)}")
        return None
    return ggufs[0]


# ---------------------------------------------------------------- the card

def _sorted_rows(manifest: dict):
    return sorted(((t, m) for t, m in (manifest or {}).items() if isinstance(m, dict)),
                  key=lambda kv: kv[1].get("code_bits") or 0)


def crowns_block(manifest: dict, record_url: str, repo: str, record: dict | None = None) -> str:
    """The managed block: table, checksums and license status. Numbers come from the manifest,
    which came from the signed record, so nothing in the block is typed by hand."""
    rounds = [int(m.get("round") or 0) for _, m in _sorted_rows(manifest)]
    latest = int((record or {}).get("round") or (max(rounds) if rounds else 0))
    rows, sums, lic = [], [], []
    for tier, m in _sorted_rows(manifest):
        fn = m.get("filename") or f"ralph-qwen3-8b-{tier}.gguf"
        fb = m.get("file_bytes")
        size = f"{fb:,} B" if isinstance(fb, int) else "not recorded"
        r = m.get("round")
        status = (f"New in Round {r}" if r == latest
                  else f"Crowned in Round {r}; held through Round {latest}")
        owner = (m.get("source_repo") or "").split("/")[0]
        builder = f"[{owner}](https://huggingface.co/{owner})" if owner else "not recorded"
        chat = m.get("chat_template_present")
        chat_s = "Yes" if chat is True else ("No" if chat is False else "not recorded")
        rows.append(f"| `{fn}` | {tier} | {m.get('code_bits', '?')} | {size} | {status} | "
                    f"{m.get('retention', '?')} | {builder} | {chat_s} |")
        sha = m.get("file_sha256")
        sums.append(f"{sha}  {fn}" if sha else f"(not recorded)  {fn}")
        L = licensed(m)
        lic.append(f"- `{fn}` — **{L}**, declared by the source owner for exactly these bytes."
                   if L else
                   f"- `{fn}` — **pending review**: crowned in Round {r}; no artifact-level license "
                   f"has been checked for these bytes yet.")
    nl = "\n"
    return f"""{CROWNS_BEGIN}
## Current crowns — through Round {latest}

| File | Tier | Code bits / weight | Exact size | Crown status | Retention at crown | Builder | Chat-template key |
|---|---|---:|---:|---|---:|---|---|
{nl.join(rows)}

### Exact checksums

```text
{nl.join(sums)}
```

### License status

{nl.join(lic)}

The same filenames, byte counts, hashes, pinned sources, chat-template flags and license
declarations are machine-readable in [`crowns.json`](./crowns.json). Latest round record:
{record_url}
{CROWNS_END}"""


def replace_managed_block(card: str, block: str) -> str:
    """Put the block where the card keeps it, and nowhere else.

    Markers present: replace between them. No markers but the first card's "## Current crowns"
    section: replace that section (through the next H2) — the one-time adoption. Neither: after
    the title. Ambiguous or half-written markers raise rather than guess which copy is live."""
    n_b, n_e = card.count(CROWNS_BEGIN), card.count(CROWNS_END)
    if n_b or n_e:
        if n_b != 1 or n_e != 1:
            raise ValueError("crowns card markers must appear exactly once as a pair")
        s, e = card.index(CROWNS_BEGIN), card.index(CROWNS_END)
        if e < s:
            raise ValueError("crowns card markers are reversed")
        return card[:s] + block + card[e + len(CROWNS_END):]
    m = _LEGACY_CROWNS_HEADING.search(card)
    if m:
        nxt = _NEXT_H2.search(card, m.end())
        end = nxt.start() if nxt else len(card)
        return card[:m.start()] + block + "\n\n" + card[end:]
    h1 = _H1.search(card)
    if h1:
        return card[:h1.end()] + "\n" + block + "\n" + card[h1.end():]
    return card.rstrip("\n") + "\n\n" + block + "\n"


def _judge_copy(record: dict | None) -> str:
    judges = list(((record or {}).get("manifest") or {}).get("observers_scored") or [])
    if len(judges) > 1:
        return (f"All {len(judges)} recorded judges scored the referenced round. For each "
                "judge, Ralph takes the lowest mean fidelity across its live "
                "(language, depth) slices; the displayed retention is the arithmetic mean "
                f"of those {len(judges)} judge-level worst-slice values.")
    if judges:
        return ("The referenced round used the recorded judge and reports its lowest mean "
                "fidelity across the live (language, depth) slices.")
    return ("For a multi-judge record, each judge is reduced to its lowest live "
            "(language, depth) slice and those judge-level values are averaged. See the "
            "record manifest for the judges that actually scored the round.")


def _run_commands(manifest: dict, repo: str) -> str:
    commands = []
    for tier, m in _sorted_rows(manifest):
        filename, chat = m.get("filename"), m.get("chat_template_present")
        if filename and chat is True:
            commands.append(f"""### `{tier}`

```bash
hf download {repo} {filename} --local-dir ./ralph-crowns
llama-cli -m ./ralph-crowns/{filename} -cnv --jinja -p "Explain one practical use of low-bit quantization."
```""")
        elif filename and chat is False:
            commands.append(f"""### `{tier}`

```bash
hf download {repo} {filename} --local-dir ./ralph-crowns
```

No embedded chat template was found, so this card does not print a canonical chat command. First
establish and publish a tested external template/runtime combination; do not repack or replace the
canonical crown file to add one.""")
        else:
            commands.append(f"""### `{tier}`

No exact mirrored filename is recorded yet. Refresh the artifact metadata before publishing a run
command; this card will not guess a path.""")
    return "\n\n".join(commands)


def fresh_card(manifest: dict, record_url: str, repo: str, record: dict | None,
               block: str) -> str:
    """The whole card, for a repo that has none. Everything a later publish will not touch is
    template copy the maintainer is expected to replace."""
    body = f"""
# Ralph crowns — Qwen3-8B

The **reigning crowned compressions** from [Bittensor netuid 40](https://github.com/RalphLabsAI/ralph),
one file per bit tier. Every round re-scores the incumbents against a fresh exam; when a crown
changes hands, the file here changes with it.

{block}

## What "retention" is, and what it is not

Retention measures how closely an artifact reproduces the **pinned comparison parent's** effect on
configured judge models. {_judge_copy(record)}

It is a within-protocol fidelity measure. **It is not a capability benchmark**, a quality score, or
evidence of a particular device speed. A high retention does not by itself establish usefulness on
any downstream task.

## How a crown changes hands

The crown transition uses the recorded **point-estimate lead** on the same fresh exam: a challenger
must clear the protocol's single-round margin or its consecutive-round persistence path, and it
must satisfy the per-judge floor rule. Reign decay can lower those point margins for a long-held
crown.

The paired bootstrap lower confidence bound is a separate rule. A positive bound can allocate a
challenger share in the round's **candidate weight vector** without changing the crown; it is not
the dethrone threshold. A candidate vector in a record is not, by itself, proof that weights were
submitted or rewards paid on chain.

## Provenance

The publisher fetches the miner's full behavior-affecting package at the **pinned revision** in the
round record and recomputes the package content hash. It mirrors the GGUF only when that package
hash matches the record's `model_id`; the GGUF bytes are uploaded without conversion or repacking.
`crowns.json` records the pinned source plus the exact mirrored file byte count and SHA-256.

Credit for the weights belongs to the miners named in `crowns.json`. This repo is a verified
mirror with a stable name, not the origin.

The admission checks compare architecture and tensor shapes with `{PARENT}`. They do **not**
cryptographically prove that a miner derived its weights from that parent. `base_model` identifies
the pinned comparison/admission target, not a lineage attestation.

## License

The comparison parent declares Apache-2.0. A mirrored artifact carries a license of its own only
once its source owner has declared one for exactly these bytes; the per-file status above and the
repo badge say where that stands. Check the pinned source repository and obtain any needed
permission before redistribution or commercial use.

## Running them

These are GGUF files. Runtime support depends on the exact quantization type, runtime build,
backend, available memory, and context settings. An embedded chat-template key is reported above;
its presence does not certify that a runtime supports the template or the quantization. Test the
specific file on the target device before making compatibility or performance claims.

{_run_commands(manifest, repo)}

## What each audit level establishes

- **L0** verifies the signature and recomputes arithmetic, crown decisions, and the candidate
  vector from published measurements. **L1** re-derives item selection and checks the pinned pool.
  These data checks do not reproduce judge inference or prove that frozen text came from a model.
- **L2** re-runs a recorded judge over frozen text; complete multi-judge coverage requires a pass
  for every recorded judge. **L3** loads the artifacts and regenerates their steps, binding the
  frozen text to model execution. L2/L3 require the recorded inputs and compatible runtimes.
"""
    return merge_front_matter(body, manifest)


def readme(manifest: dict, record_url: str, repo: str, record: dict | None = None,
           *, existing_card: str = "") -> str:
    """The card the next publish writes: the live card with its managed block and owned
    front-matter keys refreshed, or the full template when the repo has no card yet."""
    block = crowns_block(manifest, record_url, repo, record)
    if not (existing_card or "").strip():
        return fresh_card(manifest, record_url, repo, record, block)
    return merge_front_matter(replace_managed_block(existing_card, block), manifest)


# ---------------------------------------------------------------- publishing

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


def commit_release(api, *, repo: str, staged: list[tuple[str, str, str]], manifest: dict,
                   card: str, round_no: int, parent_commit: str = ""):
    """Upload artifacts, manifest and card as one Hub commit.

    `upload_file` once per path exposes partial public states: new weights with an old manifest,
    then a new manifest with an old README. `create_commit` pre-uploads any LFS blobs and advances
    the repository ref once for the whole operation. `parent_commit` makes a concurrent publisher
    fail instead of silently replacing its head.
    """
    from huggingface_hub import CommitOperationAdd
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
                  for _tier, path, name in staged]
    operations.extend([
        CommitOperationAdd(path_in_repo=MANIFEST,
                           path_or_fileobj=json.dumps(manifest, indent=1,
                                                      sort_keys=True).encode()),
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode()),
    ])
    kwargs = dict(repo_id=repo, repo_type="model", operations=operations,
                  commit_message=f"crowns as of round {round_no}")
    if parent_commit:
        kwargs["parent_commit"] = parent_commit
    return api.create_commit(**kwargs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--push", action="store_true",
                    help="actually upload; without it everything is verified and nothing is written")
    ap.add_argument("--repo", default=CROWNS_REPO)
    ap.add_argument("--render-card", action="store_true",
                    help="print the card the next publish would write, from the live manifest; "
                         "fetches no artifacts and writes nothing")
    ap.add_argument("--declare-license", action="append", default=[], metavar="TIER=SPDX",
                    help="record a reviewed artifact-level license for a tier's CURRENT bytes "
                         "(a metadata-only commit with --push)")
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
    published, published_head = published_state(a.repo)
    try:
        existing_card = published_card(a.repo, published_head)
    except Exception as e:
        print(f"cannot read the live card ({type(e).__name__}: {e}) — refusing to publish over it")
        return 1
    # A badge the maintainer already put on the card is a declaration for the bytes it describes;
    # carry it into the manifest for unchanged crowns so it survives the next crown change of a
    # DIFFERENT tier. New bytes never inherit it.
    adopt = front_matter_license(existing_card)

    if a.render_card:
        shown = {t: (declare_license(e, adopt) if adopt and isinstance(e, dict) and not licensed(e)
                     else e) for t, e in published.items()}
        print(readme(shown, record_url, a.repo, record=record, existing_card=existing_card))
        return 0

    manifest, staged, refreshed, failed = dict(published), [], [], []
    for tier, sub in sorted(kings.items()):
        previous = published.get(tier) or {}
        unchanged = previous.get("model_id") == sub.get("model_id")
        filename = f"ralph-qwen3-8b-{tier}.gguf"
        if unchanged and not needs_metadata_refresh(previous, filename):
            if adopt and not licensed(previous):
                manifest[tier] = declare_license(previous, adopt)
                print(f"  {tier}: already published — adopting the card's {adopt} declaration "
                      f"for these bytes")
            else:
                print(f"  {tier}: already published with complete byte metadata — skipping")
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
        try:
            meta = artifact_metadata(path, filename)
        except Exception as e:
            print(f"  {tier}: FAILED — {type(e).__name__}: {e}")
            failed.append(tier)
            continue
        manifest[tier] = manifest_entry(
            tier=tier, sub=sub, round_no=record.get("round"), record_url=record_url,
            metadata=meta, previous=previous if unchanged else None,
            adopt_license=adopt if unchanged else "")
        if unchanged:
            # The crown facts are deliberately retained from `previous`; this fetch only fills
            # direct metadata about identical bytes. Re-uploading the same multi-GB file would add
            # no evidence and is unnecessary for the atomic metadata/card refresh.
            refreshed.append(tier)
            print("     metadata refreshed; crown round and retention preserved")
        else:
            staged.append((tier, path, filename))

    for spec in a.declare_license:
        tier, _, spdx = spec.partition("=")
        if tier not in manifest or not spdx.strip():
            print(f"--declare-license {spec!r}: unknown tier or missing license id")
            return 2
        manifest[tier] = declare_license(manifest[tier], spdx)
        print(f"  {tier}: license {spdx.strip()} declared for sha "
              f"{str(manifest[tier].get('file_sha256') or '')[:12]}…")

    if failed and not staged and not refreshed:
        # A NON-ZERO EXIT WHEN SOMETHING FAILED, even though nothing was staged — otherwise a round
        # whose only crown could not be fetched reports "nothing new to publish" and looks healthy.
        print(f"\n{len(failed)} tier(s) could not be mirrored: {', '.join(failed)}")
        return 1

    try:
        card = readme(manifest, record_url, a.repo, record=record, existing_card=existing_card)
    except Exception as e:
        print(f"\nREADME preservation REFUSED — {type(e).__name__}: {e}")
        print("nothing published; the live card could not be merged safely")
        return 1

    if not (staged or refreshed or manifest != published or card != existing_card):
        print("nothing new to publish — every throne is already mirrored at its current model_id")
        return 0

    print("\nto publish:")
    for tier, path, name in staged:
        print(f"  {name:<32} {os.path.getsize(path) / 1e9:5.2f} GB   (tier {tier})")
    for tier in refreshed:
        print(f"  {tier:<32} metadata/card only (crown facts unchanged)")
    if not staged and not refreshed:
        print(f"  {'crowns.json + README.md':<32} card/manifest only")
    if not a.push:
        print("\nDRY RUN — verified only. Re-run with --push to upload.")
        print(f"\n--- README.md that would be written ({len(card)} chars) ---")
        print(card)
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set; cannot push")
        return 2
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(a.repo, repo_type="model", exist_ok=True)
    if staged:
        print(f"  uploading {len(staged)} changed GGUF(s) in one atomic commit…")
    commit_release(api, repo=a.repo, staged=staged, manifest=manifest, card=card,
                   round_no=record.get("round"), parent_commit=published_head)
    print(f"\npublished -> https://huggingface.co/{a.repo}")
    if failed:
        print(f"  !! {len(failed)} tier(s) did NOT mirror: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
