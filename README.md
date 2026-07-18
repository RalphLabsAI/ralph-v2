# SN40 v2 — The Model Compression Subnet

**One line:** miners compete to compress frontier open models — starting with GLM — into low-bit checkpoints that run on hardware people actually own; the best quality-per-bit model wears the crown.

**Reference point:** PrismML's Ternary Bonsai showed what one lab gets from 1.58-bit models (9× smaller, ~5× faster, HF-trending). v2 makes that a standing market: a permissionless swarm doing it to GLM, continuously, with every winner shipped as an open checkpoint.

## Objective

Fix a teacher model. Miners submit **student checkpoints** — quantized/distilled versions of the teacher — into per-bit-budget tiers. The validator scores the artifact itself. Nothing about the miner's process needs to be trusted or verified: download the checkpoint, measure it.

**Teacher ladder** (all MIT-licensed):

| Rung | Teacher | bf16 today | 4-bit tier | ~1.58-bit tier |
|---|---|---|---|---|
| 1 | GLM-4-9B (dense) | 19 GB · 24GB GPU | 4.7 GB · any laptop | **2.5 GB · a phone** |
| 2 | GLM-4.5-Air (106B MoE, 12B active) | 212 GB · multi-GPU server | 53 GB · one H100 / Mac Studio | **26 GB · one RTX 5090** |

Rung 2 is the headline: a frontier-class 106B agentic model on a single consumer GPU.

## Scoring

Per bit-tier (1.58 / 2 / 4-bit), king-of-the-hill:

```
score = capability retention vs teacher (see covering set) — hard floor, per-domain
gate  = effective-bits audit                               — codebooks, scales, outliers,
                                                             embeddings all counted; per-tensor
                                                             cardinality checks kill fake-ternary
bonus = measured decode tok/s in a pinned container        — real speed, not theoretical
```

## The covering set — the problem that killed distillation KOTH v1

Distillation contests fail when the eval is narrow: students learn the teacher's *style* on
the eval slice, win on KL/CE, and lose the actual capabilities. v2 therefore never scores
imitation — it scores **verifiable capability retention**:

- **Multi-domain, verifiable basket** — math with checkable answers, code scored by
  execution, knowledge QA, long-context retrieval, instruction-following with programmatic
  checkers. Style cannot pass a unit test.
- **Per-domain retention ratio** (student/teacher), aggregated by **soft-min** — sacrificing
  any one capability tanks the score. No domain left behind.
- **Naive-quantization control** — every candidate must beat a round-to-nearest int4
  baseline on retention; matching the teacher's tone is worth nothing against the control.
- **Secret, rotated sampling + fresh items** — domains are public, samples are secret,
  rotated on schedule, and partly *generated fresh* (seeded item generators +
  post-training-cutoff documents). This is the exact machinery that caught an
  eval-memorization king on sn40 in production.
- **Paraphrase-invariant slice** — a portion of items scored under paraphrase; style
  mimicry collapses under rewording, capability survives.

KL-to-teacher is kept as a *diagnostic* only. It never enters the score.

Crown moves when a challenger beats the tier king past a noise-floor margin (bootstrap-LCB, same statistics SN3 uses). Emissions split score-proportionally across the Pareto frontier — miners iterate and resubmit continuously; no one-shot registrations, no winner-take-all cliff.

## Why it can't be gamed (short version)

All scoring machinery is running today on sn40 and has caught real fraud in production:
- **Secret, rotated held-out sets** + control-model diff-in-diff — caught an eval-memorization king in July; neighbors evaluating on public corpora are exposed to exactly this.
- **Throughput plausibility ceiling** (hardware-calibrated MFU bound) — caught a forged-wall-clock king in July.
- **Effective-bits audit** — a "ternary" model smuggling fp16 outliers fails intake.
- And because students are small, **anyone can re-run the scoring** on one modest GPU — every verdict is independently checkable, and the full history renders live at ralphlabs.ai.

A kernel track (sandboxed miner-submitted CUDA/Triton, correctness-gated against the dequantized reference) makes the speed numbers real — low-bit is only fast with real kernels, so kernels are part of the competition.

## Positioning vs the family

SN3 trains full-precision models on a frozen arch (compression out of scope by design). SN97 fine-tunes a fixed 35B, quantization banned. SN120 runs RL environments on full-size students. **Nobody in the ecosystem owns the efficiency axis.** v2 is the missing quadrant: same proven KOTH shell, competing on quality-per-bit.

## Deliverables & narrative

- Every crowned king ships as an **open runnable checkpoint** (safetensors + GGUF) on HF, Apache-2.0/MIT.
- Public leaderboard: quality-per-bit Pareto frontier, live at ralphlabs.ai.
- Standing demo: **"run the king"** — this week's best compressed GLM, on your laptop.
- Recurring headline, on tap with every crown: *"Bittensor's swarm just compressed GLM another X% with no quality loss."*

## Migration & timeline

sn40's validator, KOTH statistics, held-out rotation, fraud gates, dashboard, and publishing pipeline all carry over — this is a re-aim of a running system, not a rebuild. Rung 1 (GLM-4-9B, 4-bit + ternary tiers) live within weeks of green light; current miners migrate by pointing their agents at a new objective.
