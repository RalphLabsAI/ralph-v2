"""Generate a conservative Hugging Face card from a crowned-artifact record.

The generator is intentionally narrower than a launch page. It may publish only facts present in
the signed record or derived directly from the artifact. Device compatibility, throughput,
downstream quality, and artifact licensing all need separate evidence and are not inferred from a
crown, a GGUF suffix, or the parent model's card.
"""
from __future__ import annotations

from .density import BF16_BITS, Density, size_gb


def _fmt(x, nd=2):
    return "—" if x in (None, 0) else f"{x:,.{nd}f}"


def render(*, model_id: str, parent: str, parent_params: int, tier: str,
           density: Density, miner: str, round_no: int,
           record_url: str = "", artifact_uri: str = "",
           observer: str = "", observers: list | tuple | None = None,
           languages: dict | None = None, benchmarks: list | None = None,
           repo: str = "RalphLabsAI/ralph", filename: str = "",
           chat_template_present: bool | None = None) -> str:
    """Render only record-backed or explicitly supplied measurements.

    `benchmarks` remains empty until absolute benchmarks have actually been run. `filename` is the
    exact file in `model_id`; callers should pass it when the repository contains several files.
    """
    parent_gb = size_gb(parent_params, BF16_BITS)
    langs = languages or {}
    d = density
    artifact_name = filename
    local_dir = model_id.split("/")[-1]
    judges = list(observers or ())
    if not judges and observer and observer != "all":
        judges = [observer]

    head = f"""---
license: other
license_name: Artifact license pending
library_name: gguf
pipeline_tag: text-generation
base_model: {parent}
tags:
  - gguf
  - compression
  - quantization
  - bittensor
  - ralph
---

# {model_id}

**A `{tier}`-tier GGUF admitted against the pinned comparison parent
[{parent}](https://huggingface.co/{parent}).**

`{_fmt(d.shrink, 1)}× smaller than the parent's bf16 tensor estimate` | `{_fmt(d.download_gb, 2)} GB estimated from measured container bits` | `{_fmt(d.retention * 100, 1)}% recorded fidelity`

Listed as the reigning crown in record round {round_no}, credited to `{miner}`. Retention is a
within-protocol fidelity measurement, not an absolute benchmark, downstream quality claim,
device-compatibility result, or speed result.
"""

    if artifact_name and chat_template_present is True:
        run_copy = f"""```bash
hf download {model_id} {artifact_name} --local-dir ./{local_dir}
llama-cli -m ./{local_dir}/{artifact_name} -cnv --jinja -p "Explain one practical use of low-bit quantization."
```"""
    elif artifact_name:
        why = ("no embedded chat template was found" if chat_template_present is False
               else "chat-template presence was not supplied to the generator")
        run_copy = f"""```bash
hf download {model_id} {artifact_name} --local-dir ./{local_dir}
```

The exact file command is shown, but {why}. This card therefore does not print a canonical chat
command; establish a tested external template/runtime combination first."""
    else:
        run_copy = ("No exact filename was supplied to the generator, so it will not print a "
                    "guessed path. Use the filename and SHA-256 from the mirror manifest.")

    quick = f"""
## Quickstart

These commands name the exact repository file; the `*.gguf` glob is intentionally avoided.

{run_copy}

A GGUF filename does not guarantee support in every llama.cpp build or hardware backend. Confirm
the file's quantization type, embedded template, memory use, and output on the target runtime.

## Model overview

| | |
|---|---|
| Pinned comparison parent | [{parent}](https://huggingface.co/{parent}) |
| Parent parameter count | {parent_params:,} |
| Admission requirement | architecture and tensor-shape compatibility with the parent |
| Bit tier | `{tier}` |
| Bits per weight (measured) | **{_fmt(d.code_bits)}** code bits · {_fmt(d.container_bits)} container bits |
| Estimated tensor size | **{_fmt(d.download_gb)} GB** (parent bf16 estimate: {_fmt(parent_gb)} GB) |

Code bits describe the quantized indices; container bits include block scales and other packing
overhead. The exact file byte count and SHA-256 should be published with the mirrored artifact.
"""

    dens = f"""
## Recorded fidelity per GB

**{_fmt(d.retention_per_gb)} retention-points per estimated GB.**

This is arithmetic over two recorded inputs: protocol retention and measured container size. It is
not "intelligence density" and is not interchangeable with benchmark accuracy per GB. It compares
submissions admitted against the same pinned parent. Retention says nothing about how good the parent was,
nor how useful this artifact is on a task.
"""

    if benchmarks:
        rows = "\n".join(
            f"| {b.get('model')} | {b.get('company','—')} | {_fmt(b.get('size_gb'))} | "
            f"{_fmt(b.get('avg'), 1)} |" for b in benchmarks)
        bench = f"""
## Benchmarks

| Model | Company | Size (GB) | Reported average |
|---|---|---|---|
{rows}

Only explicitly supplied measurements appear above; their protocols still need to be compared
before drawing conclusions.
"""
    else:
        bench = """
## Benchmarks

Absolute downstream benchmarks have not been supplied for this artifact. The crown was decided on
observer-effect fidelity to the pinned parent, which is a different quantity.
"""

    if len(judges) > 1:
        judge_text = (f"All {len(judges)} recorded judges scored the same generated answers. "
                      "Within each judge, Ralph takes the lowest mean fidelity across the live "
                      "(language, depth) slices; the headline retention is the arithmetic mean "
                      f"of those {len(judges)} judge-level worst-slice values. The recorded "
                      f"judges are: {', '.join(f'`{j}`' for j in judges)}.")
    elif judges:
        judge_text = (f"The recorded judge `{judges[0]}` reports the lowest mean fidelity across "
                      "its live (language, depth) slices.")
    else:
        judge_text = ("See the round manifest for the judge set. In a multi-judge round, each "
                      "judge is reduced to its lowest live (language, depth) slice, then the "
                      "judge-level values are averaged.")

    lang_line = ""
    if langs:
        pretty = " · ".join(f"{k} {v}" for k, v in sorted(langs.items()))
        lang_line = f"\nThe record's language counts are {pretty}.\n"

    how = f"""
## How this was scored

The parent and artifact generate separate steps from the same selected prefixes. Configured judge
models measure how each step changes their predictive distribution over a fixed continuation; the
artifact is scored by how closely its effect matches the parent's effect. {judge_text}
{lang_line}
Item selection is derived from post-commit round data. The judges and environment that actually
ran are named in the signed record; the card does not infer them from the model family.

Round record: {record_url or "published with the crown"}

## Crown rule versus challenger share

A crown transition is decided by the challenger **point-estimate lead** on the incumbent's fresh
exam, together with the single-round/persistence margins, per-judge floor rule, and any recorded
reign decay. The paired bootstrap lower confidence bound is separate: a positive bound can assign
a challenger share in the record's candidate weight vector without changing the crown. A recorded
candidate vector does not itself establish that weights were submitted on chain.

## What the audit levels establish

- **L0** verifies the signature and recomputes scores, crown arithmetic, and the candidate vector
  from published measurements. **L1** re-derives item selection and checks the pinned corpus.
  Neither level reruns a judge or proves that frozen steps came from model execution.
- **L2** reruns a recorded judge over frozen text. Full multi-judge coverage requires one pass for
  every recorded judge. **L3** loads the artifacts and regenerates the recorded steps; this is the
  level that binds frozen text to model execution.

Start with `python -m eval.rerun <record.json>` for L0. The command reports an incomplete verdict
until the inputs and runtimes required for the requested higher levels are provided.
"""

    lim = f"""
## Limitations

- Retention is measured against **{parent}**, not against absolute capability.
- Observer-effect fidelity can saturate on badly damaged models, compressing distinctions among
  poor artifacts.
- Languages and tasks outside the recorded exam are untested.
- No universal runtime, phone, accelerator, memory, throughput, or output-quality claim is made.

## Provenance and parent relationship

| | |
|---|---|
| Reigning-crown record | round {round_no}, subnet 40 |
| Miner | `{miner}` |
| Scored package | {artifact_uri or "see the round record"} |
| Protocol | [{repo}](https://github.com/{repo}) |

The scored package is content-addressed and its locator is pinned in the record. Admission checks
architecture and tensor-shape compatibility with `{parent}`; that is **not cryptographic proof of
derivation from the parent's weights**.

## License

The parent model's Apache-2.0 metadata does not automatically license a miner's modified weights.
No separate artifact-level license was supplied to this generator, so this card uses the `other`
license tag and reports the artifact license as pending. Check the source repository and obtain any
needed permission before reuse or redistribution.
"""
    return head + quick + dens + bench + how + lim


def from_record(record, model_id: str, parent: str, parent_params: int,
                record_url: str = "", benchmarks=None, tier: str | None = None,
                filename: str = "", chat_template_present: bool | None = None) -> str:
    """Render straight from a signed round record for one occupied tier."""
    from .koth import kings_from_events
    kings = kings_from_events(record.events)
    if tier is None:
        if len(kings) > 1:
            raise ValueError(f"record has {len(kings)} thrones ({', '.join(sorted(kings))}); "
                             f"pass tier= to say which card to write")
        tier = next(iter(kings), None)
    king = kings.get(tier)
    # A held crown may have both challenger and incumbent rows. The incumbent supplies the re-score
    # used by the crown decision, while the measured challenger row supplies params and bit widths.
    # Neither row alone is enough to produce a truthful card.
    rows = [s for s in record.submissions if s.model_id == king]
    sub = next((s for s in rows if getattr(s, "role", "") == "incumbent"),
               rows[0] if rows else None)
    if sub is None:
        raise ValueError(f"record contains no submission for the {tier!r} king to write a card for")
    measured = next((s for s in rows if getattr(s, "params", 0)), sub)
    density = Density(params=measured.params, code_bits=measured.code_bits,
                      container_bits=measured.container_bits, retention=sub.retention)
    man = record.manifest or {}
    return render(
        model_id=model_id, parent=parent, parent_params=parent_params,
        tier=sub.tier, density=density, miner=sub.miner,
        round_no=record.round, record_url=record_url,
        artifact_uri=measured.artifact_uri or sub.artifact_uri,
        observer=man.get("observer", ""), observers=man.get("observers_scored"),
        languages=man.get("languages"), benchmarks=benchmarks, filename=filename,
        chat_template_present=chat_template_present)
