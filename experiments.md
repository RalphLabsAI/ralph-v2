# Experiments — first real-GPU run

Three GPUs, same items, same code. Qwen2.5-0.5B parent (small on purpose — the question was whether
the metric discriminates and reproduces, not whether it scales). Everything here is measured.

## The metric holds up

Parent scored against itself: **exactly 1.000**, all three boxes. And it orders compressions right —
8-bit above 4-bit everywhere (gaps 0.058 / 0.074 / 0.040), fp16 above both.

| | fp16 | 8-bit | 4-bit |
|---|---|---|---|
| H100 PCIe | 1.000 | 0.610 | 0.552 |
| A100 SXM4 | 1.000 | 0.636 | 0.563 |
| L40S | 1.000 | 0.678 | 0.638 |

## The residual

**The score saturates at the bottom.** Perturbing every weight by σ×std and sweeping σ:

| σ | 0 | .01 | .02 | .05 | .10 | .20 | .30 |
|---|---|---|---|---|---|---|---|
| L40S | 1.000 | .674 | .589 | .541 | .555 | .524 | .572 |
| H100 | 1.000 | .612 | .544 | .526 | .451 | .469 | .433 |

Monotone to σ=0.05, then it flattens around 0.43–0.57 and wobbles. It never falls to the 0.135 inert
floor, because a wrecked model still emits *something* and something still moves the observer.

So the metric cannot tell a broken model from a very broken model. Two things stop that mattering:
the tail wobble (0.036–0.048) is below the dethrone margin, so it can't flip a crown; and a real
4-bit beats the luckiest wrecked model by 0.066–0.083, above it. Crowns are contested at the top, and
the top is clean.

It is still the closest thing here to a model scoring well while being useless. That is why the
capability canary stays: cheap deterministic checks run alongside, and a submission that moves the
observer correctly while being unusable fails them and is not crowned. Observer-KL is structurally
blind to exactly that one thing, so it does not decide alone.

## On overfitting

Nothing is fitted to a fixed test set, because there isn't one. The scored items are drawn from
`H(commit_root ‖ round_nonce)` after checkpoints lock — so the operator cannot pick them either. The
same draw picks which observer scores you, from a pool. You cannot pre-fit an exam that does not
exist yet, or a grader you cannot name.

The one thing a miner *can* target is the parent, which is the point — the task is "reproduce this
model's contribution", so fitting it harder is the product, not the exploit.

## What nearly broke it

An H100 gave four different outputs from four identical `generate()` calls on the same batch. An A100
and an L40S were bit-exact on the same code. Length-dependent — fine at 64 new tokens, not at 128+ —
a KV-length kernel heuristic switching to a split-K attention reduction. Torch's determinism flags
did not fix it; pinning the attention implementation did.

Two boxes would have shipped "noise floor is zero" as a property of the mechanism. It is a property
of the box. The identity check catches it instantly: on that H100 the parent scored **0.8754 against
itself** while the noise probe reported clean. It now runs every round and aborts rather than
crowning, and the GPU, batch size and attention implementation are pinned so a re-run on other
hardware reads as hardware rather than fraud.

## The pinned parent, on production hardware

Everything above was measured on a 0.5B. The parent is now `Qwen/Qwen3-8B`, and an 8B is a
different memory and kernel regime, so the identity check had to be re-run against it before any
crown could mean anything.

| | |
|---|---|
| Parent | `Qwen/Qwen3-8B` — 8,190,735,360 weight params, 16.4 GB bf16 |
| Observer | `HuggingFaceTB/SmolLM2-1.7B-Instruct` — deliberately not Qwen |
| Box | H100 PCIe, torch 2.11+cu128 |
| Items | 72, drawn from a language-balanced pool of 300 real trajectories |
| **Identity** | **1.000000** — exact, shortfall 0.0, all 72 samples scored |
| Generation | deterministic across three identical calls (`575ee1fe4b3f5548` ×3) |

This is the same H100 PCIe model that returned **0.8754** before the attention implementation was
pinned, so the fix holds at 8B as well as at 0.5B.

The observer is from a different family on purpose. The metric asks whether a submission moved an
*independent* model to where the parent moved it; an observer sharing the parent's tokenizer and
training mixture shares its blind spots, and would under-detect exactly the failures that family
has — silently, since the scores would still look reasonable.

One thing the real corpus showed that synthetic pools did not: the draw produced **five** scoring
slices, not six. The Chinese source yielded no deep-step trajectories, so `zh|deep` does not exist.
The stratifier allocates across the slices that are actually present (all ≥11, clear of the floor
of 8) rather than the ones arithmetic predicts — slice count is a property of the data, not of the
language list.

## Where it isn't ready

**A round record has now been published to a real sink** — HuggingFace, read back, verified and
anchored, with the trail confirmed against the chain head. It is private until launch.

What has still not happened:

- **No anchor has been committed by a real chain.** The hash chain that makes one commitment slot
  cover the whole history is computed and verified, but only against a fake chain so far.
- **No miner has submitted.** The path is wired end to end now — including the fetcher, which was
  the piece that made a submission impossible to score at all — but no third party has used it.
- **Weights have never been set by this validator.** It runs read-only by default, and will keep
  doing so until there is something worth crowning; a second signer on a live hotkey fights the
  first.
- **Non-Latin coverage is Hindi and Chinese.** That is where the anti-clone axis binds today. The
  published evidence for low-bit collapse is Persian and Cyrillic, so the pool does not yet test
  where the proof is.
- **The saturation limitation above is unchanged.** Nothing measured since has moved it.

## To poke at it

```bash
python -m tests.test_crown_path        # 50/50, CPU
python -m eval.simulate_submission     # miner -> validator -> auditor, seconds
```

The second walks the intake gates, draws the observer from the nonce, runs the identity check,
crowns, and then re-verifies the round itself — including a pass that reloads the checkpoint and
regenerates its steps, since every cheaper check takes the miner's steps on the operator's word.

Reproduce the numbers above:

```bash
python -m eval.ladder_probe            # the σ curve and the quantization ladder (GPU)
python -m eval.shakedown_round_noise   # whole-round noise floor, in retention units (GPU)
python -m eval.batch_invariance        # is the score a function of the artifact alone? (GPU)
```
