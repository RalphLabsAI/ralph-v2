# Miner quickstart — Ralph SN40 v2

Take the pinned `Qwen/Qwen3-8B` parent, preserve its architecture, and store its weights at a lower
bit budget. Ralph runs a separate crown in each of four tiers: `binary`, `ternary`, `sub2`, and
`sub4`.

Crown records and artifacts are public; a crown earns its tier's share once validators verify the
round that awarded it.

## TL;DR

1. Compress `Qwen/Qwen3-8B` without changing its architecture or weight-element count.
2. Ship GGUF or safetensors. GGUF exposes exact per-tensor types for the bit-budget check.
3. Run `python -m eval.bitrate your-model.gguf` before committing.
4. `python -m miner.submit commit` seals `H(content_hash ‖ salt)` on chain; `python -m miner.submit
   reveal` publishes the hash and salt only after the round opens (see below).
5. Serve the exact immutable bytes you committed.
6. Read the signed record for intake, score, and crown results.

## What the score means

Ralph measures **retention by downstream effect**, not wording agreement and not benchmark task
accuracy. For a trajectory prefix `K`:

1. The pinned parent and your model each produce a next step.
2. A judge continues from the parent's step to establish a fixed continuation `C`.
3. The judge's distribution over `C` is measured after the parent step, after your step, and after
   the prefix alone.
4. Your score reflects whether your step moved that judge in the same direction and by a similar
   amount as the parent's step.

Every item is scored by all three configured judges:

- `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- `microsoft/Phi-3-mini-4k-instruct`
- `allenai/OLMo-2-1124-7B-Instruct`

Each judge is reduced to its worst language/depth slice, and the displayed metric is the mean of
those three values. The post-commit nonce selects the trajectory items; it does not select a single
judge.

This score is a competition-specific retention proxy. It is **not** evidence of general capability,
accuracy, or device performance.

## Crown rule

The incumbent is re-scored on the same fresh exam as the challengers. A challenger may dethrone by:

- a displayed-metric lead of at least `0.02` in one round, or
- a lead of at least `0.01` in two consecutive rounds with the same artifact.

After a king has held three rounds, both thresholds are halved. In every case, the floor rule also
requires that no challenger slice fall below the incumbent's worst slice under the same judge. An
exact copy has a zero paired lead and cannot dethrone.

## Intake rules

| tier | maximum code bits/weight | maximum container bits/weight |
|---|---:|---:|
| `binary` | 1.15 | 2.5 |
| `ternary` | 1.75 | 2.5 |
| `sub2` | 2.3 | 3.0 |
| `sub4` | 4.0 | 5.0 |

- **Formats:** GGUF or safetensors; no `*.py`, pickle weights, or tokenizer `auto_map`.
- **Measured budgets:** code and container bits come from the served tensor data and GGUF types,
  not a filename or declaration. Both caps bind.
- **Pinned parent:** architecture, weight-element count, and essential config must match
  `Qwen/Qwen3-8B`.
- **Runtime format:** `TQ1_0` and `TQ2_0` are refused because they lack the required mainline Metal
  kernels. This gate is not a benchmark of any particular device.
- **Admission:** one artifact per `(coldkey, tier)` per round. Previously scored bytes are skipped.
  There is no operational resubmission bond; `base_bond` is zero because no escrow extrinsic exists.
- **Commit-reveal:** every behaviour-affecting file is included in the content hash. The fetched
  artifact must match the revealed hash.
- **Decoding:** validator-owned generation settings are used during scoring.

## Commit and reveal (the CLI does the chain writes)

```bash
pip install -r requirements.txt -r requirements-chain.txt

# 1. seal the exact bytes you will serve — BEFORE the artifact is public, before the round's nonce
python -m miner.submit commit --ckpt ./my-model --tier sub2 \
    --uri hf://you/your-repo@<commit-sha> --wallet <wallet> --hotkey <hotkey>   # add --dry-run first
# 2. publish the artifact (flip it public if you uploaded private), then, once the round opens:
python -m miner.submit reveal --ckpt ./my-model --wallet <wallet> --hotkey <hotkey>
```

`commit` writes the salt to `my-model.ralph-submission.json` beside the checkpoint — keep it; no
salt, no reveal, no score. A commitment slot holds 384 bytes (three 128-byte fields), and the
reveal is the bigger write, so `commit` sizes the reveal first and refuses a `--uri` that would not
fit. Both commands accept `--dry-run`, which prints exactly what would go on chain.

## Package and submit

```python
from miner.package import build_submission
import secrets

sub = build_submission(
    ckpt_dir="path/to/your/compressed/model",
    tier="ternary",                         # binary | ternary | sub2 | sub4
    teacher_pair="qwen3-8b",
    student_base="Qwen/Qwen3-8B",
    declared_compute_h100h=42.0,
    salt=secrets.token_hex(16),             # keep secret until reveal
)

# 1) Commit sub["commit_value"] before the round nonce exists   (`miner.submit commit` does this)
# 2) Publish the exact checkpoint and record an immutable artifact_uri.
# 3) Reveal sub["reveal"] after the round opens                   (`miner.submit reveal`)
```

`build_submission` writes `manifest.json`, runs the validator's own inspector, and hashes the full
directory. That means the correct hash is the hash of what the validator will fetch, not an earlier
local file. Download your published immutable revision into a clean directory and verify it before
you commit.

Commit before making the artifact public. Duplicate content is first-commit-wins, so publishing the
bytes first gives someone else a chance to commit the same hash. A private upload can be made public
after your commitment lands; keep the salt secret until reveal.

## Start from a crown

All four tiers had crowns after Round 7. The mirrored files and their provenance are in
[`RalphLabsAI/ralph-crowns`](https://huggingface.co/RalphLabsAI/ralph-crowns). Reusing a crown as a
starting point is allowed, but serving identical bytes cannot win; the same artifact has a zero
paired lead and previously scored content is skipped.

## Status snapshot

As of **2026-09-11**, seven real-model rounds are public, signed, hash-chained, and anchored in
[`RalphLabsAI/ralph-v2-rounds`](https://huggingface.co/datasets/RalphLabsAI/ralph-v2-rounds). Round 7
was the first round scored by all three judges: `binary`, `sub2`, and `sub4` changed hands, while
`ternary` held on the floor rule. A public chain snapshot after the round showed six later
submissions waiting for a future score; the count is point-in-time and does not imply intake
acceptance or a schedule.
