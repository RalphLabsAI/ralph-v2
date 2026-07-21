# Mining SN40 v2 — how to participate

You compress a pinned teacher (GLM) into a small student that keeps as much of the
teacher's behavior as possible, and submit that student. The best student in each size
tier wears the crown; every crown is published as a downloadable open model.

## What you submit

A **student checkpoint** — a smaller/quantized model — in **safetensors only**. Anything
else is rejected before it's loaded:

- **No `*.py`, no pickles (`.bin`/`.pt`), no tokenizer `auto_map`.** The validator loads
  your weights with `trust_remote_code=False`, always. A pickle is arbitrary code
  execution on load; it's an automatic reject, not a warning.
- **Params, bits and dtype are recomputed from your actual tensors** — never from what
  you declare. Shipping the teacher behind a small decoy, or under-declaring your size,
  fails the tier gate.
- **Size/bit tier**: you submit into a tier (e.g. sub-1B / sub-3B, or a 4-bit / 2-bit
  budget). Your checkpoint must actually fit it.

## How you're scored

Not on imitating the teacher's *style* — on doing what the teacher *does*.

The validator samples states from a large pile of agentic trajectories, has the pinned
GLM take its genuine next step from each state, and asks a grounded judge (a different
model) whether *your* student's step did the same thing. That agreement, normalized
against a pinned base model, is your retention. Two properties to know:

- **A slice is scored from the states your student itself reaches** (not just the
  teacher's), so a student that imitates locally but drifts when it runs on its own is
  caught.
- **The crown moves only on a strict improvement past a noise-floor margin**, measured on
  the same fresh points as the reigning king. A copy of the king ties, and a tie earns
  nothing — copying is not a strategy.

Train however you like — quantization-aware training, on-policy distillation, SFT on
teacher traces, whatever wins. The reward only counts what the student can do; we don't
pay for resemblance.

## Submitting

```python
from miner.package import build_submission

sub = build_submission(
    ckpt_dir="my_student/",           # safetensors + config + tokenizer
    tier="sub-1B",
    teacher_pair="glm4-9b/qwen2.5-0.5b",   # the pinned (teacher, base) pair
    student_base="Qwen/Qwen2.5-0.5B",
    declared_compute_h100h=180.0,     # generation + training + scoring + rollout
    salt="<random secret>",
)
# 1. commit sub["commit_value"] on-chain now (seals your content hash, orders discovery)
# 2. upload the checkpoint dir
# 3. reveal sub["reveal"] after the round opens
```

`build_submission` runs the validator's own intake inspector first, so you see the exact
params/bits the validator will re-derive — no surprises at scoring.

## Economics

- **One free eval per registration.** Extra submissions in an epoch cost a **bond**,
  refunded if the submission improves your own best score, forfeit otherwise — so
  best-of-N grinding costs you and honest iteration doesn't.
- **Per-coldkey round cap** bounds how many submissions one operator runs per round.
- **Declared compute is metered and reconciled** against a throughput envelope;
  under-declaring is fraud and forfeits the bond.

## The honest state (pre-launch)

The eval mechanism is validated on real models; the subnet is being brought up in
**shadow mode first** — you can submit and see yourself ranked with no emission at risk —
then emission flips to v2 crowns once the launch checklist clears. Nothing here is a
promise of a date; watch the announcements.
