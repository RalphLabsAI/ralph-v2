"""Generate the HuggingFace card for a crowned artifact.

STRUCTURE BORROWED DELIBERATELY. PrismML's Bonsai cards are the best in this category and the shape
is worth adopting wholesale: a functional headline that says where it runs rather than what it is,
three ratios immediately underneath, Quickstart BEFORE benchmarks, a comparison table that names
competitors, an intelligence-density section, and stated limitations. Every one of those is a
choice about respecting the reader, and copying the choices is not copying the copy.

WHAT IS OURS. Their card asserts numbers from a private method. Ours links the round record that
produced them, so every figure on the page can be recomputed by the person reading it — and the
comparison table names Bonsai, because pinning their parent is what makes the comparison legitimate.

The card is GENERATED from a record rather than written, so a crown cannot ship with a number that
disagrees with the round that awarded it.
"""
from __future__ import annotations

from .density import BF16_BITS, Density, size_gb


def _fmt(x, nd=2):
    return "—" if x in (None, 0) else f"{x:,.{nd}f}"


def render(*, model_id: str, parent: str, parent_params: int, tier: str,
           density: Density, miner: str, round_no: int,
           record_url: str = "", artifact_uri: str = "",
           observer: str = "", languages: dict | None = None,
           benchmarks: list | None = None, repo: str = "RalphLabsAI/ralph-v2") -> str:
    """`benchmarks` is a list of dicts with model/company/size_gb/avg — including competitors.

    Left EMPTY until absolute benchmarks have actually been run. An empty table says so out loud
    rather than quietly implying the comparison exists, because the one thing a card like this
    cannot survive is a number nobody measured."""
    parent_gb = size_gb(parent_params, BF16_BITS)
    langs = languages or {}
    d = density

    head = f"""---
license: apache-2.0
base_model: {parent}
tags:
  - compression
  - quantization
  - bittensor
  - ralph
---

# {model_id}

**{d.container_bits:.2f}-bit compression of [{parent}](https://huggingface.co/{parent}) — architecture unchanged, runs on llama.cpp and MLX.**

`{_fmt(d.shrink, 1)}× smaller than bf16` | `{_fmt(d.download_gb, 2)} GB on disk` | `{_fmt(d.retention * 100, 1)}% retention vs the parent`

Crowned on [Bittensor subnet 40](https://taostats.io/subnets/40) in round {round_no} by `{miner}`.
This is not a smaller model trained to imitate a bigger one — it is {parent} itself, stored in
fewer bits. Same architecture, same parameter count.
"""

    quick = f"""
## Quickstart

```bash
huggingface-cli download {model_id} --local-dir ./{model_id.split('/')[-1]}
llama-cli -m ./{model_id.split('/')[-1]}/*.gguf -p "Explain quantization in one paragraph."
```

## Model overview

| | |
|---|---|
| Parent | [{parent}](https://huggingface.co/{parent}) |
| Parameters | {parent_params:,} |
| Architecture | unchanged from parent |
| Bit tier | `{tier}` |
| Bits per weight (measured) | **{_fmt(d.code_bits)}** achieved · {_fmt(d.container_bits)} shipped |
| Size | **{_fmt(d.download_gb)} GB** (parent at bf16: {_fmt(parent_gb)} GB) |

Bits per weight are measured from the tensor data, not read off the dtype header — a 4-bit model
shipped inside a 16-bit container is credited for the compression and rejected as unshippable.
"""

    dens = f"""
## Intelligence density

**{_fmt(d.retention_per_gb)} retention-points per GB.**

Capability per byte is the question a person choosing what to download is actually asking, and
neither size nor score answers it alone.

One caveat that travels with the number: the capability term here is **retention against the pinned
parent**, not an average of absolute benchmarks. It ranks artifacts compressed from the same parent —
which is what the crown is decided on — and it is *not* interchangeable with a published
benchmark-average-per-GB figure. Retention says nothing about how good the parent was.
"""

    if benchmarks:
        rows = "\n".join(
            f"| {b.get('model')} | {b.get('company','—')} | {_fmt(b.get('size_gb'))} | "
            f"{_fmt(b.get('avg'), 1)} |" for b in benchmarks)
        bench = f"""
## Benchmarks

| Model | Company | Size (GB) | Avg |
|---|---|---|---|
{rows}
"""
    else:
        bench = """
## Benchmarks

Absolute benchmark numbers have not been run for this artifact yet. The crown was decided on
retention against the pinned parent, which is a different quantity — publishing a benchmark table
we did not measure, or implying one exists, is the failure mode this project is built to avoid.

When they are run they will appear here alongside the comparable published artifacts, including
PrismML's Bonsai builds of the same parent.
"""

    lang_line = ""
    if langs:
        pretty = " · ".join(f"{k} {v}" for k, v in sorted(langs.items()))
        lang_line = (f"\nScored across {len(langs)} languages ({pretty}), aggregated **worst-slice**: "
                     f"a submission cannot buy a weak language with a strong one.\n")

    how = f"""
## How this was scored

Compression is judged by **downstream effect**, not by imitation. This model and the parent each
continue the same reasoning trajectory; an independent observer model{f" (`{observer}`)" if observer else ""}
reads both, and the score is how closely this model moved it to where the parent did. Matching the
parent's wording earns nothing unless it carries the same information.
{lang_line}
Which trajectories were scored, and which observer graded them, were both drawn from a chain value
that did not exist until the checkpoint was already sealed — so there is no fixed test set to fit
and no grader nameable in advance.

**Every number above is recomputable.** {"The round record is published at " + record_url + "." if record_url else "The round record is published with the crown."}

```bash
python -m eval.rerun <record.json> --pool <items.jsonl> --observer <observer> --artifacts <dir>
```

That re-derives the drawn items from the nonce, recomputes the score from the published
measurements, re-runs the observer over the frozen text, and reloads this checkpoint to regenerate
its steps. Exit 0 means reproduced; 2 means you skipped the expensive half.
"""

    lim = f"""
## Limitations

- Retention is measured against **{parent}**, not against absolute capability. A high number here
  means faithful to that parent, not good in general.
- The scoring metric **saturates on badly damaged models** — past a point, more damage stops
  lowering the score. It ranks well among good compressions, which is where crowns are contested,
  and compresses differences among broken ones.
- Low-bit compression is known to degrade non-Latin languages faster than English. Coverage of the
  languages scored here is listed above; languages outside that set are untested.
- No absolute benchmark suite has been run on this artifact yet.

## Provenance

| | |
|---|---|
| Crowned | round {round_no}, subnet 40 |
| Miner | `{miner}` |
| Artifact | {artifact_uri or "see the round record"} |
| Protocol | [{repo}](https://github.com/{repo}) |

Licensed Apache-2.0, inheriting the parent's terms.
"""
    return head + quick + dens + bench + how + lim


def from_record(record, model_id: str, parent: str, parent_params: int,
                record_url: str = "", benchmarks=None, tier: str | None = None) -> str:
    """Render straight from a signed round record, so the card cannot disagree with the round.

    PER TIER, AND `hold` COUNTS. This used to scan for `crown`/`dethrone` only and keep the last one
    it saw, which is wrong twice: a round that HOLDS both thrones emits neither, so it raised "no
    crowned challenger" on a perfectly good round — and with several tiers live it kept whichever
    happened to be last in the list and wrote that model's card for every tier. Pass `tier` when the
    round has more than one throne."""
    from .density import from_record as density_from
    from .koth import kings_from_events
    kings = kings_from_events(record.events)
    if tier is None:
        if len(kings) > 1:
            raise ValueError(f"record has {len(kings)} thrones ({', '.join(sorted(kings))}); "
                             f"pass tier= to say which card to write")
        tier = next(iter(kings), None)
    king = kings.get(tier)
    # The incumbent's re-score carries role="incumbent"; a freshly crowned model carries
    # "challenger". Accept either, or a held crown writes no card at all.
    sub = next((s for s in record.submissions if s.model_id == king), None)
    if sub is None:
        raise ValueError(f"record contains no submission for the {tier!r} king to write a card for")
    man = record.manifest or {}
    return render(
        model_id=model_id, parent=parent, parent_params=parent_params,
        tier=sub.tier, density=density_from(sub, parent_params), miner=sub.miner,
        round_no=record.round, record_url=record_url, artifact_uri=sub.artifact_uri,
        observer=man.get("observer", ""), languages=man.get("languages"),
        benchmarks=benchmarks)
