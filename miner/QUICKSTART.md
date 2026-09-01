# Miner quickstart — Ralph SN40 v2 (compression subnet)

You take a **pinned parent** model, leave its architecture untouched, and re-store its weights at
a **low bit budget**. The best compression per bit tier wears the crown and earns emissions, and
every crown ships as a downloadable, runnable model.

This is Bonsai's task, not distillation-into-a-different-model: same architecture, fewer bits.

## TL;DR

1. Take the tier's **pinned parent** (the validator publishes it; you do not choose it).
2. Compress it — QAT, PTQ, whatever you like. The architecture must stay unchanged.
3. Ship **safetensors or GGUF**. GGUF is preferred: it is what runs on a phone, and its
   per-tensor `ggml_type` gives an exact bit budget with no estimation.
4. Package (`miner/package.py`), commit `H(hash‖salt)` on-chain, reveal after the round opens.
5. You are scored on whether your model's steps carry the **same information** the parent's do.

## How you are scored — read this part

Not on wording. Not on answering generated questions. On **downstream effect**.

For a trajectory prefix `K`:

1. The pinned parent produces a step `K → K+1`. So do you.
2. An **independent observer model** continues from `K + parent_step`; call it `C`.
3. The validator measures the observer's distribution over that same `C`, conditioned on
   `K + parent_step`, on `K + your_step`, and on `K` alone.

You score well when your step moves the observer **in the same direction** and **by about as
much**. Measured:

| what you did | score |
|---|---|
| identical effect | 1.00 |
| same information, different words | **0.80** |
| half the effect | 0.36 |
| contributed nothing | 0.14 |
| moved it the wrong way | 0.02 |

**Design around these consequences:**

- **Copying the parent's style earns nothing.** There is no wording channel. A paraphrase
  carrying the same information scores as well as an exact match, and mimicking phrasing while
  losing content scores near zero. You have to extract what the parent knew at that step.
- **You cannot pre-fit the questions, because there are none** — nor the observer, which is
  drawn from the round nonce after your weights are sealed.
- **Your worst slice is your score.** Aggregation is worst-slice over
  (observer × language × depth). Excellent English and broken Chinese does not average out.
- **Saying nothing is not a hedge.** An inert step is scored as inert, never skipped.
- **Copying the king earns nothing.** An exact copy ties on every sample and cannot dethrone.

## Rules and limits

- **Formats:** safetensors or GGUF. No `*.py`, no pickle, no `auto_map`. Every behaviour-
  affecting file is hashed into your commitment — including config and tokenizer.
- **Bit budget:** measured from your actual tensor data, not a dtype header or a filename. A
  1-bit model stored in a bf16 container is credited as real compression **and rejected as an
  unshippable artifact** — ship it packed.
- **Pinned parent:** your artifact must be shape-compatible with the tier's parent (architecture,
  weight-element count, config essentials). A different model fails at the door.
- **Decoding is validator-owned.** Your `generation_config.json` does not affect scoring.
- **Economics:** one free eval per **coldkey** per round; extra submissions cost a refundable
  bond, returned only if you improve your own best. Rotating hotkeys does not reset this.
- **Publish your bytes.** Include an `artifact_uri` so the crown resolves to something people can
  actually download.

## Package + submit

```python
from miner.package import build_submission
import secrets

sub = build_submission(
    ckpt_dir="path/to/your/compressed/model",   # safetensors or .gguf + config/tokenizer
    tier="ternary",                              # binary | ternary | sub2 | sub4
    teacher_pair="qwen3-8b",                     # the pinned parent (Qwen/Qwen3-8B)
    student_base="Qwen/Qwen3-8B",
    declared_compute_h100h=42.0,
    salt=secrets.token_hex(16),                  # keep secret until reveal
)
# 1) commit sub["commit_value"] on-chain BEFORE the round's nonce is drawn
# 2) publish the checkpoint and note its artifact_uri
# 3) after the round opens, reveal sub["reveal"] = {content_hash, salt}
```

**The order of 1 and 2 is protection, not pedantry.** A commitment binds bytes, not authorship:
anyone can hash a *public* artifact and seal a commitment to it, and duplicate reveals are settled
first-commit-wins — so an artifact that sits public before your commitment lands is, for that
window, up for grabs by whoever commits it faster. Commit first, or upload to a **private** repo,
commit, and only then flip it public: a private upload cannot be hashed by anyone else, so the
race disappears. (Salt discipline is the same idea at reveal time: the salt stays secret until
you reveal, or your sealed value can be replayed.)

`build_submission` runs the validator's own inspector, so an artifact that would fail intake
tells you immediately rather than at scoring time.

## Winning strategy, honestly

The moat is compression quality at a fixed bit budget — nothing else is paid for. Two places the
incumbent is measurably weak, and where a downloaded artifact loses:

- **Non-Latin languages.** Independent testing puts 1-bit Bonsai at 45.2% on Persian against
  79.8% for a conventional 4-bit build, while English holds at 97–100%. Multilingual is its own
  scored slice here, so it cannot be averaged away.
- **Long-horizon, multi-step work**, where compression damage compounds.

## Status

The subnet has not run on mainnet, and the scoring stack has not yet had a run against real
models on a GPU. Read the "Not yet done" section of the top-level README before treating any of
this as live.
