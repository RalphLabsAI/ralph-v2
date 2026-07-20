# SN40 v2 — The Model Compression Subnet

**One line:** miners compete to compress frontier open models — starting with GLM —
into low-bit checkpoints that run on hardware people actually own; the best
capability-per-bit model in each size tier wears a crown, and every crown ships as a
downloadable open model.

**Reference point:** PrismML's Ternary Bonsai showed what one lab gets from 1.58-bit
models (9× smaller, ~5× faster, HF-trending). v2 makes that a standing market: a
permissionless swarm doing it to GLM, continuously, producing a *family* of runnable
open checkpoints — with an anti-gaming design built from the public lessons of earlier
distillation-KOTH attempts.

## Objective

Fix a teacher model. Miners submit **student checkpoints** — quantized/distilled
versions of the teacher — into size/bit tiers. The validator scores the artifact
itself on fresh, verifiable tasks. Nothing about the miner's private training process
is trusted or inspected: download the checkpoint, measure what it can do, measure how
small and fast it is.

**Teacher ladder** (all MIT-licensed):

| Rung | Teacher | bf16 today | 4-bit tier | ~1.58-bit tier |
|---|---|---|---|---|
| 1 | GLM-4-9B (dense) | 19 GB · 24GB GPU | 4.7 GB · any laptop | **2.5 GB · a phone** |
| 2 | GLM-4.5-Air (106B MoE, 12B active) | 212 GB · multi-GPU server | 53 GB · one H100 / Mac Studio | **26 GB · one RTX 5090** |

Rung 2 is the headline: a frontier-class 106B agentic model on a single consumer GPU.

## Scoring — capability, not imitation

Distillation contests die when they score *imitation* (KL / distribution-match to the
teacher): students learn the teacher's **style** — its "let me reconsider…" filler —
win the metric, and lose the actual capabilities. (This is not hypothetical; earlier
distillation subnets documented it in their own public post-mortems — see
[`docs/prior-art-and-lessons.md`](docs/prior-art-and-lessons.md).) v2 never scores
imitation. It scores **verifiable capability retention**:

```
gate   effective-bits audit        — bits/params/dtype recomputed from the LOADED
                                      state dict, never metadata; serialized-bytes
                                      after canonical recompression (kills zero-pad,
                                      sparse-stuffing, fp16-outlier "fake ternary")
gate   safety + sanity             — safetensors only, no miner *.py (RCE ban),
                                      anti-finetune probe (grad-explosion / absurd
                                      norms), degeneracy probe (loop rate / length
                                      blowup on trivial prompts)
score  capability on FRESH tasks   — see below
bonus  measured decode tok/s       — on pinned hardware, plausibility-ceilinged
```

**The score is downstream capability on freshly-minted, machine-checkable tasks** —
math with checkable answers, code scored by execution, templated reasoning
(GSM-Symbolic style, with NoOp distractors), tool-call/format tasks with programmatic
checkers. Style cannot pass a unit test. Retention is measured *relative to the pinned
GLM teacher run through the identical harness every round* (a live control row), so
"retention" means the student kept what the teacher had — not that it maxed a narrow
benchmark.

**Fresh by construction (commit-then-generate).** The eval items for a round are
generated *after* the round's checkpoint hashes are committed — from seeded procedural
generators and post-cutoff documents. There is no static answer key to memorize and
nothing to harvest; the secret set never exists on a miner-reachable box and is deleted
after the round. A paraphrase-invariant slice and a stale-vs-fresh diff-in-diff (our
GPT-2-control method, which caught a real memorization king on sn40) flag any model
whose gains are benchmark-recovery rather than genuine compression.

## Anti-gaming — economics, not detection

The last team proved you **cannot detect** copying on a shared public base — every
fingerprint detector either failed on legit fine-tunes or got weaponized. v2 makes
copying and gaming *unprofitable* instead:

- **Tiered multi-slot king-of-the-hill.** One crown per size/bit tier (e.g. sub-1B /
  sub-3B / sub-7B, or 1.58 / 2 / 4-bit). Within a tier the ranking is a **total
  order** (capability retention above the effective-bits gate) — no undefined
  "Pareto partial order" stalls. Across tiers you get a downloadable *family*.
- **A copy earns nothing.** Dethroning requires being *strictly better* — higher
  retention at equal-or-smaller budget, past a margin **well above the noise floor**
  (bootstrap-LCB, paired against the incumbent on the same fresh items). A copy of the
  king ties; a tie pays zero; micro-shaving is below the margin. No detector to evade.
- **Margin over a naive-quant control.** Emission share ∝ (retention − round-to-nearest
  baseline at that budget). Shipping the obvious quantization, or a copy of it, has
  ~zero expected value; every emitted TAO is attached to demonstrated algorithmic delta.
- **First-committer-owns-hash + delayed reveal.** Sealed commit-then-reveal (Ralph's
  X25519 bundle encryption); the winning artifact publishes only after a delay window,
  so the live king isn't copyable mid-round; the earliest on-chain commit of a given
  content hash wins, which DQs the *later* committer only (no griefing).
- **Anti-grind economics** (the only defenses that held for the last team):
  one-eval-per-registration, a per-coldkey round cap, and a bond/fee for extra
  submissions *refunded when the submission improves the miner's own best* — spam and
  best-of-N crown-farming become strictly unprofitable; honest iteration stays cheap.
- **Never-shrink ratchet + headroom pay.** The retention floor only rises; payout
  scales with distance above it, so a saturated tier self-retires (emission migrates to
  unsaturated tiers) instead of Goodharting on noise.

## Where confidential compute fits (and where it doesn't)

We asked the hard question directly and researched it: **CC-attested proof bundles do
NOT solve the gaming problem.** They prove *computational integrity* — that the
official eval ran on real hardware over a sealed set — but an attested run of a
gameable objective just faithfully certifies a gamed model. And miner-run CC is a trap:
the miner owns the box, so "sealed from the miner" is outside the hardware vendor's
threat model (enclave memory is extractable by the box owner), and miner-run eval would
delete the one-eval-per-registration defense.

So the core eval is **validator-side** — which is cheap *because the students are
small* (that is exactly what structurally fixes the eval-economics death that bankrupted
the last attempt). CC has one honest, optional role: **phase-2 throughput
attestation** — proving "this student really runs this fast on genuine hardware," a
non-forgeable version of the tok/s bonus, which reuses Ralph's existing TDX+H100-CC
stack. Nice-to-have, not the security foundation.

## Deliverables & narrative

- Every crowned king ships as an **open runnable checkpoint** (safetensors + GGUF) on
  HF, Apache-2.0/MIT — a *family* across tiers, not one model.
- Public leaderboard: capability-per-bit frontier, live at ralphlabs.ai, every verdict
  independently re-runnable on one modest GPU (signed audit trail).
- Standing demo: **"run the king"** — this week's best compressed GLM, on your laptop.
- Recurring headline, on tap with every crown: *"Bittensor's swarm just put a
  frontier-class model on your laptop / phone."*

## Positioning vs the family

SN3 trains full-precision models on a frozen arch (compression out of scope by design).
SN97 fine-tunes a fixed 35B, quantization banned. SN120 runs RL environments on
full-size students. **Nobody owns the efficiency axis.** v2 is the missing quadrant —
and it fuses three things no other subnet can assemble: a fresh commit-then-generate
*verifiable* eval, reward-indifference-to-copies (economic, not detection), and a
tiered family of downloadable artifacts.

## Honest open problems — to prove before locking

The design is not claimed solved. Three problems are genuinely open and will be
prototyped and measured on the live subnet in a low-stakes shadow mode *before* any
public capability claim:

1. **Covering-set breadth / capability laundering.** A miner could distill from a
   smuggled frontier-model corpus and win narrow math/code without truly compressing
   GLM. Only eval *breadth* plus teacher-relative retention defeats it. The honest
   promise is **"verifiable retention of checkable capability,"** not "broad
   capability" — the state of the art forces this tradeoff, and we will not overclaim.
2. **Dethrone margin vs noise.** The strict-improvement margin must sit well above
   measurement noise, or micro-shaving flips crowns. Calibrate empirically.
3. **Eval-cost scaling.** As the covering set widens, run a tiered cheapest-first
   funnel (bits audit → safety/degeneracy → small slice → full suite only for
   survivors); GLM must stay validator-hostable — never a cloud-API dependency in the
   scoring path.

## Migration & timeline

sn40's validator, KOTH statistics, held-out rotation, fraud gates, bundle encryption,
dashboard, and signed publishing pipeline all carry over — this is a re-aim of a
running system, not a rebuild. Rung 1 (GLM-4-9B, one bit tier) ships to the live subnet
in shadow mode first — scoring runs and the frontier is published, but the open problems
above are measured and settled before crowns carry full emission weight.
