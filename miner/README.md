# Mining SN40 v2 — how to participate

Compress the pinned `Qwen/Qwen3-8B` parent into a lower-bit representation without changing its
architecture. Ralph evaluates the artifact in one of four bit-budget tiers, and each tier's current
crown is published as a downloadable model.

Ralph's observer-KL score measures how closely a compressed model preserves the parent's downstream
effect on the configured judges. It is a retention metric, **not** a capability benchmark.

## What you submit

A content-addressed checkpoint in **GGUF or safetensors**:

- No `*.py`, pickle weights (`.bin`/`.pt`), or tokenizer `auto_map`. Safetensors load with
  `trust_remote_code=False`; GGUF uses the constrained runner.
- Parameter count, code bits, container bits, and architecture identity are recomputed from the
  served artifact. Declarations and filenames do not decide intake.
- The artifact must match the pinned parent's architecture and weight-element count.
- GGUF formats without the required mainline Metal kernels (`TQ1_0` and `TQ2_0`) are refused. This
  is a format-compatibility rule, not a device-performance claim.

The current tier caps are:

| tier | code bits/weight | container bits/weight |
|---|---:|---:|
| `binary` | 1.15 | 2.5 |
| `ternary` | 1.75 | 2.5 |
| `sub2` | 2.3 | 3.0 |
| `sub4` | 4.0 | 5.0 |

Run the same bit inspector as the validator before committing:

```bash
python -m eval.bitrate path/to/model.gguf
```

## How scoring works

The post-commit round nonce selects a fresh set of trajectory items. For each item, the parent and
the submitted model produce a step. Three configured judges then measure how each step changes
their distribution over the same continuation:

- `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- `microsoft/Phi-3-mini-4k-instruct`
- `allenai/OLMo-2-1124-7B-Instruct`

Each judge scores every item. The displayed retention is the mean of the judges' worst-slice scores
over language and trajectory depth. The reigning king is re-scored on the same exam. A challenger
must clear the published point-margin or persistence rule without putting any slice below the
incumbent's floor; exact copies tie and cannot dethrone.

## Package and submit

```python
from miner.package import build_submission
import secrets

sub = build_submission(
    ckpt_dir="my_qwen3_compression/",       # GGUF or safetensors + config/tokenizer
    tier="sub2",                            # binary | ternary | sub2 | sub4
    teacher_pair="qwen3-8b",
    student_base="Qwen/Qwen3-8B",
    declared_compute_h100h=42.0,
    salt=secrets.token_hex(16),              # keep secret until reveal
)

# 1. Commit sub["commit_value"] on chain before the round nonce exists.
# 2. Publish the exact artifact and record its immutable artifact_uri.
# 3. Reveal sub["reveal"] after the round opens.
```

`build_submission` runs the validator's intake inspector and hashes the complete served directory.
After upload, download the exact immutable revision and confirm that its directory hash matches the
one you committed; extra or changed files cause a commit-reveal rejection.

Admission is one artifact per `(coldkey, tier)` per round, and an artifact already scored in the
public trail is skipped. The old resubmission-bond accounting remains disabled (`base_bond=0`)
because no on-chain escrow/refund extrinsic exists.

For the CLI flow and commit-order details, continue with [`QUICKSTART.md`](QUICKSTART.md).

## Public status

As of **2026-09-11**, seven real-model rounds are public, signed, hash-chained, and anchored. Round 7
scored all three judges; `binary`, `sub2`, and `sub4` changed hands while `ternary` held on the floor
rule. The resulting four artifacts are mirrored in
[`RalphLabsAI/ralph-crowns`](https://huggingface.co/RalphLabsAI/ralph-crowns), and the records live in
[`RalphLabsAI/ralph-v2-rounds`](https://huggingface.co/datasets/RalphLabsAI/ralph-v2-rounds).

Each record carries its weight vector; validators set it after verifying the round. A public
snapshot after Round 7 showed six later submissions awaiting a future score; that queue can change
and does not mean an artifact has passed intake.
