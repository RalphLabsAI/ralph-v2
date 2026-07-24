# Miner quickstart — Ralph SN40 v2 (compression subnet)

You compress a pinned teacher (**GLM**) into a **small student** that keeps GLM's
capability. The best student per size tier wears the crown and earns emissions; every crown
ships as an open checkpoint. This is what you submit and how you're judged.

## TL;DR

1. Distill/quantize/prune GLM into a student that fits a size tier — **any** method you like.
2. Save it as **safetensors** (weights + config + tokenizer). No custom code files.
3. Package it (`miner/package.py`), commit `H(hash‖salt)` on-chain, reveal after the round opens.
4. The validator mints **fresh** items, GLM answers them, your student is scored on the same
   items across the axes, and the **worst axis** sets your score.

## What actually earns the crown

Your student is scored by **verifiable-outcome retention of GLM** across capability axes,
each with a deterministic checker (no LLM judge, no style/logit matching):

| axis | you must be able to… |
|---|---|
| math | solve multi-step word problems (exact numeric answer) |
| code | implement a function that **passes hidden unit tests** |
| instruction | follow verifiable format constraints (exact word count, JSON keys, bullets) |
| long_context | retrieve + aggregate over a long haystack |
| multihop | compose several stated facts (with a shortcut distractor to trap pattern-matching) |

Retention is `(student_pass − base_pass) / (teacher_pass − base_pass)` on the items GLM
itself passes, then aggregated **worst-domain** (a soft-min). **Three consequences that
decide whether you win:**

- **Your weakest axis is your score.** Being brilliant at math and code buys nothing if you
  cratered long_context or multihop — those two (compression breaks them first) are usually
  the ceiling. Distill for *breadth*, not a spike.
- **Style earns zero.** A wrong answer fails the checker no matter how GLM-like it reads.
  Distilling GLM's *tone* is worthless; only correct *outcomes* count.
- **Every declared axis must be live.** If your student is degenerate/looping on an axis it
  can be disqualified; and the crown only lands when all axes have signal.

## Anti-gaming you should know about

- **Genre-overfit gate (diff-in-diff).** You cannot win by distilling GLM only over the
  announced distribution. The validator scores you on **stale** (pre-commit) *and* **fresh**
  (post-commit) documents; if your edge over base **collapses on genuinely-new** same-genre
  content, you're flagged as an overfitter and demoted regardless of your axis scores. The
  only way through is a student that actually **generalizes** GLM's capability.
- **Copy-the-king earns nothing.** The reigning king is re-scored on the same fresh items
  every round; an exact copy ties (zero margin) and cannot dethrone. Dethroning needs a
  **strict improvement on the worst axis** past the noise margin.
- **Fresh items.** Items don't exist until after checkpoints lock (seeded by the commit
  nonce), so there's nothing to memorize offline.
- **Content-addressed identity + commit-reveal.** You're judged on the exact bytes you
  committed to — no bait-and-switch. The hash covers weights **and** config/tokenizer, so
  don't plan to swap those after committing.

## Rules / limits

- **safetensors only.** No `*.py`, no pickle, no `auto_map`/`custom_pipelines` in configs
  (the loader runs `trust_remote_code=False`). Params + effective-bits are recomputed from
  the tensor headers, so declare your tier honestly — you can't smuggle a bigger model in.
- **Tier budget.** Submit to a size/bit tier; over-budget params or effective-bits are
  rejected at the door.
- **Economics.** One **free** eval per **coldkey** per round; extra submissions from the
  same operator cost a refundable bond (refunded only if you improve your own best). Running
  two hotkeys to get two free shots doesn't work — the free eval is per coldkey.
- **Permitted base.** Distill from a base on the allowlist; declare it in the manifest.

## Test locally before you submit

Score your student the same way the validator will (needs a GPU + the pinned GLM):

```bash
RALPH_STUDENTS=you/your-student \
RALPH_TEACHER=THUDM/glm-4-9b-chat-hf \
python -m eval.run_capability_axis        # prints per-axis retention + which axis is your ceiling
```

Read the per-axis line: the axis with the lowest retention is what caps you. Iterate there.

## Package + submit

```python
from miner.package import build_submission
import secrets

sub = build_submission(
    ckpt_dir="path/to/your/student",     # safetensors + config + tokenizer
    tier="open",                          # the tier you're competing in
    teacher_pair="glm-4-9b/qwen-0.5b",   # the pinned (teacher, base) you target
    student_base="Qwen/Qwen2.5-0.5B",    # your permitted base
    declared_compute_h100h=42.0,          # total compute you spent (be honest)
    salt=secrets.token_hex(16),           # keep this secret until reveal
)
# 1) commit sub["commit_value"] on-chain NOW (before the round's nonce is drawn)
# 2) upload the checkpoint dir (build_submission wrote manifest.json into it)
# 3) after the round opens, reveal sub["reveal"] = {content_hash, salt}
```

`build_submission` runs the validator's own inspector, so if your checkpoint would fail
intake you find out immediately (same params/bits the validator will re-derive) — no
surprises at scoring.

## Winning strategy, honestly

The moat is **training efficiency and breadth**, not a trick. The method is open (distill
from GLM's verifiable outputs, filter by the checkers), so everyone can do it; the edge is a
student that retains GLM's capability **on the axes compression breaks first** (long-context,
multi-hop) at your tier's budget — and that keeps generalizing on documents it never saw.
