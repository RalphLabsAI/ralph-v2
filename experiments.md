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

## Where it isn't ready

Nothing has been published to a real sink and no anchor has been committed by a real chain — both
work end to end against a local directory and a fake chain, nothing more. No miner has submitted.
The parent is Qwen, not GLM; scaling it is a config line, but it has not been run.

## To poke at it

```bash
python -m tests.test_crown_path        # 47/47, CPU
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
