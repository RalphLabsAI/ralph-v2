<p align="center">
  <img src="docs/assets/ralph-banner.png" alt="Ralph Labs — open model compression" width="100%">
</p>

# Ralph — open model compression on Bittensor SN40

**Compress one pinned open model into fewer bits, without touching its architecture. Best
compression per bit tier wears the crown, and every crown ships as downloadable weights.**

`Qwen/Qwen3-8B` pinned · four bit-budget tiers · [netuid 40](https://taostats.io/subnets/40)

This is not distillation into a smaller architecture. It is the same architecture stored at a
lower bit budget. A submission whose shape does not match the parent is refused before any weights
load. Ralph's score measures retention by downstream effect; it is **not** a standalone capability
benchmark.

**Public status:** seven signed, hash-chained rounds are published and anchored.
[Round 7](https://huggingface.co/datasets/RalphLabsAI/ralph-v2-rounds/blob/main/rounds/round-00000007-f42ae20c6ed2a796.json)
scored every submission with all three judges, changed the `binary`, `sub2`, and `sub4` crowns, and
held `ternary` on the floor rule. The four crown artifacts are mirrored in
[`RalphLabsAI/ralph-crowns`](https://huggingface.co/RalphLabsAI/ralph-crowns). Emission follows
the signed records: a validator sets a round's weight vector after its own audit accepts the
record (see *Running as an auditor validator*).

---

## For miners

You compress privately, however you like. The subnet never inspects your method, only your artifact.

```bash
git clone https://github.com/RalphLabsAI/ralph && cd ralph
pip install -r requirements.txt -r requirements-chain.txt

# 1. compress Qwen/Qwen3-8B however you want — GPTQ, AWQ, bitsandbytes, your own scheme.
#    safetensors or GGUF, architecture unchanged.

# 2. COMMIT FIRST — this seals a hash of your exact bytes on chain before the round exists
python -m miner.submit commit \
    --ckpt ./my-compressed-qwen3 --tier ternary \
    --uri hf://<you>/<repo>@<rev> \
    --wallet <your-wallet> --hotkey <your-hotkey> --netuid 40
#   add --dry-run first; it prints what would be committed without touching the chain

# 3. reveal after the round opens
python -m miner.submit reveal --ckpt ./my-compressed-qwen3 \
    --wallet <your-wallet> --hotkey <your-hotkey> --netuid 40
```

Full walkthrough: [`miner/QUICKSTART.md`](miner/QUICKSTART.md).

### What you have to clear

Six gates, in order. Nothing loads your weights until all six pass.

| # | gate | fails if |
|---|---|---|
| 1 | economics | not registered, or the coldkey already entered this tier in this round |
| 2 | safety | pickles, remote code, or files that are not weights |
| 3 | tier fit | parameter count or dtype headers inconsistent |
| 4 | **bit budget** | measured bits/weight over the tier cap — read from tensor DATA, not the dtype header |
| 5 | **pinned parent** | architecture or weight-element count does not match `Qwen/Qwen3-8B` |
| 6 | commit-reveal | bytes do not hash to what you committed before the nonce existed |

### What the tournament allocates

The signed record computes the weight vector for each tier: a tier is split between its king and
the best challenger who **provably beat them**. Validators pay that vector once their own audit
accepts the record — a vector in a record is the rule applied, not proof of payment; the chain is.

| | |
|---|---|
| **king** | 80% of the tier |
| **best challenger with a strictly positive paired margin** | 20% of the tier |
| everyone else | nothing |

The crown changes hands on the **displayed metric itself**, on the same exam as the re-scored king: a lead of **0.02** in one round, or **0.01 in two consecutive rounds** — the best challenger of a round stays in as the tier's *contender* and is re-scored on the next exam, so a small real edge accumulates instead of being lost to one draw. One rule on top: no slice of the challenger may sit below the king's worst slice — reshaping is allowed, the worst case getting worse is not. A tied pair false-dethrones ~2% of contests at 288-item exams; a copy leads by exactly 0 and a copy with English polish by ~0.008, under both margins. Every round is scored by **all three judges**, and the crown metric is the average of their worst-slice scores — which judge a round happens to draw no longer decides a crown, and fitting one judge moves a third of your score, not all of it. A king that has held **three rounds running** defends against half the margins (0.01 in one round, or 0.005 twice), so a throne nobody has beaten gets easier to take, not harder. A paired lower bound above zero allocates the runner-up share in the candidate vector.

**Start from the reigning crown if you want to.** Every one is published, and improving a published
artifact is the compounding this trail exists for — not an attack on it. The king keeping 80% and
the 0.02 margin in the candidate allocation is what protects the original author. What protects
everyone is that a *copy* earns nothing in that allocation: an unchanged artifact is not re-scored
at all, and a near-copy scores what the original scores, which puts its paired margin at zero.

**One artifact has one owner.** If two hotkeys commit the same bytes, the earlier commitment wins
and the later is refused, with a row in the signed record naming the block that beat it. Submit
your own work, or someone else's made genuinely better.

### Bit tiers

| tier | max code bits/weight | ≈ code size at 8B |
|---|---|---|
| `binary` | 1.15 | 1.18 GB |
| `ternary` | 1.75 | 1.79 GB |
| `sub2` | 2.3 | 2.35 GB |
| `sub4` | 4.0 | 4.10 GB |

A 4-bit model shipped inside a 16-bit container is credited for the compression it achieved and
rejected as an unshippable artifact — both budgets bind.

### Commit the hash of what will be SERVED, not what you uploaded

The most common self-inflicted rejection is a commit-reveal mismatch: the validator fetches your
repo and its recomputed hash does not equal what you revealed, which reads — and is recorded — as
the served bytes not being the committed ones. Nearly every time, the cause is hashing the wrong
thing on your side: `content_hash` covers the whole directory (paths, sizes, bytes of `.gguf`,
`.json`, and friends), so a hash of your local pre-upload file misses whatever the repo actually
serves. Two miners lost round entries to this.

Do what the validator does: download your own repo at the revision you will commit, hash that
directory with the same rule, and commit that. Credit to the miner who diagnosed this after being
rejected for it.

### Formats that can win

An eligible GGUF format must have a mainline `llama.cpp` Metal path. A format without one is
rejected even if it runs in another backend. This is a format-level compatibility gate, not a
device benchmark or a claim that every crown has been tested on a particular phone.

| | |
|---|---|
| **use** | `Q1_0`, `Q2_0`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, and the `Q*_K` family |
| **rejected** | **`TQ1_0`, `TQ2_0`** — mainline has no Metal kernel for either |

**The `TQ*` types are the trap.** They are the obvious choice by name at ~1.1 and ~2.1 bits and pack
beautifully, but neither has the required Metal kernels; `TQ2_0` also lacks the required CUDA path.
Both are refused. Use `Q1_0` or `IQ1_S` at the binary end, `Q2_0` or `IQ2_XXS` at the sub-2 end.

Check before you commit — the same code the validator runs:

```bash
python -m eval.bitrate path/to/model.gguf
```

### How to reach the binary tier

**Binary is a research tier. Stock `llama-quantize` will not get you there, and we were wrong to
imply otherwise.** An earlier version of this section said a good importance matrix makes `Q1_0`
coherent. A miner ran it end to end on mainline and reported perplexity in the millions; we then
checked the reference and they were right.

Here is what the reference actually is. `prism-ml/Bonsai-8B-unpacked` ships F16 weights, and every
group of 128 in them holds **exactly two distinct values — one magnitude and a sign**:

```
model.layers.0.mlp.up_proj.weight  F16 [12288, 4096]
  group 0: 2 distinct values, 1 magnitude (±0.02966309)
  group 2: 2 distinct values, 1 magnitude (±0.0255127)
```

Those weights were already 1-bit before any GGUF existed. The `Q1_0` file is a packing step, not a
compression step — llama.cpp supplies the kernels that RUN 1-bit weights, not a method that
PRODUCES good ones. Round-to-nearest with group scales, which is what `llama-quantize` does, throws
away the information at one bit no matter how good the imatrix is.

So winning `binary` means producing the 1-bit weights yourself, then exporting to `Q1_0` — the bit
gate is satisfied by the packing, and the quality has to be there before it.

**You do not have to invent the method. Two published 1-bit PTQ frameworks are open source and need
no retraining:**

| | |
|---|---|
| [BiLLM](https://github.com/Aaronhuang-778/BiLLM) (ICML 2024) | splits salient / non-salient weights, binary residual approximation for the salient ones. Reports 8.41 ppl on LLaMA2-70B at **1.08 bit** |
| [ARB-LLM](https://github.com/ZHITENGLI/ARB-LLM) | alternating refined binarization with row/column-wise scaling factors |

**1.08 bit fits under this tier's 1.15 cap**, so the published operating point clears the budget
with room for a 2-bit embedding on top. The remaining work is real but it is engineering rather
than research: run the binarizer on the pinned parent, then get those weights into a `Q1_0` GGUF.

**We have not run either on Qwen3-8B ourselves** — we are pointing at the state of the art, not
handing you a verified recipe, and the export path in particular is unproven. If you get one
working we would rather hear about it than have you assume we already know.

**The bit budget is not the hard part**, and it is more generous than the reference: the cap is
1.15 against Bonsai's 1.125, which buys **15% of parameters at 2-bit** or 5% at 4-bit. On Qwen3-8B
the token embedding is 7.60% of parameters, so `--token-embedding-type Q2_K` measures **1.0760**
and passes. Embedding *and* output both at `Q2_K` comes to 1.1520 and just misses.

**What the bit gate actually measures, because it changes what you can build.** For a **GGUF** the
code bits come from the TYPE NAME — `Q1_0` is 1, `IQ2_XXS` is 2 — mixed per tensor and averaged by
parameter count. Nothing is level-counted, so an auditor recomputing it cannot disagree with us.
(For safetensors we count distinct levels per group instead, which is a different rule and is why
an affine int4 checkpoint measures 4 and not 9.)

The consequence for a 1-bit method: **GGUF types are per-tensor and per-block, so a scheme whose
salience varies per WEIGHT has nowhere to live.** BiLLM-style binary-plus-salient-outliers carries
roughly 1.1 bits of information, and there is no GGML type that expresses it — pack it as `Q1_0`
and you throw the salient handling away; pack it as anything wider and you pay that width across
the whole tensor. The tier is therefore not purely an information bound: it is a bound on
information **that llama.cpp can represent and run**, which is a real and deliberate narrowing, and
it is why the reference is a 2-values-per-group structure rather than an arbitrary 1.1-bit code.

Per-TENSOR mixing is fully supported and is where your headroom is — a 1-bit body with a 2-bit
embedding is measured correctly at 1.0760. Per-weight salience is not, until a GGML type for it
exists.

All four tiers are crowned as of Round 7. The published crown for a tier is a permitted starting
point, and a `Q2_0` body with a higher-precision embedding can measure around 2.23 against the
`sub2` cap of 2.3. Measure the artifact you actually intend to serve; mixed tensors and container
overhead still bind independently.

Check what intake will say before you commit:

```bash
python -m eval.bitrate qwen3-8b-binary.gguf
```

And `python -m eval.precheck` (no arguments, stdlib only) header-measures every artifact currently
awaiting scoring and prints what intake will decide — the whole pending field, yours and your
competitors', without downloading a single model.

---

## For anyone checking a crown

Every round publishes what it scored, which items the post-commit nonce selected and which judges
graded them, so a result can be recomputed rather than believed. Round 7 and later score all three
configured judges; a full L2 re-check repeats the observer pass once per judge.

```bash
python -m eval.rerun <record.json>                                    # arithmetic only, no GPU
python -m eval.rerun <record.json> --pool <items.jsonl> \
    --observer <one-recorded-hf-id> --artifacts <ckpt-dir>            # one judge + model binding
python -m eval.rerun --history <dir> --head <on-chain anchor>         # is the trail complete?
python -m eval.auditor --follow --require L0,L1 --signer <key>        # ... as a standing role
```

Exit `0` reproduced · `1` diverged · `2` incomplete. Skipping the expensive levels exits 2, never
0 — nobody can call a round verified by running the cheap half.

The adversary assumed throughout is **the operator holding the signing key**, so a valid signature
attributes a record and never validates it.

---

## Run it locally

```bash
pip install -r requirements-dev.txt
python -m pytest -q                    # local logic/integration suite
python -m eval.bitrate path/to/model.gguf
```

Most logic tests are CPU-only; sandbox tests need `bubblewrap` to be permitted, and production
scoring plus L2/L3 model re-derivation require the recorded inference environment. The older
`eval.simulate_submission` path is test scaffolding and does not currently represent the
multi-judge production flow.

---

## The crown: semantic equivalence by downstream effect

Every distillation-KOTH before this got gamed the same way. Score **token agreement with the
teacher** and you pay for *style*: SN97's KL-crowned king mimicked "wait, let me reconsider"
filler, never produced an answer, and was worse than the un-finetuned base on 5/5 reasoning
benchmarks. Score **generated questions** and you pay for question-answering: probe formats are
public, so fitting them is a cheap narrow skill that transfers to nothing.

So we score neither. Two steps are equivalent when they move a **configured judge model** into
the same predictive state:

1. From trajectory prefix `K`, the **pinned parent** produces its step, and so does the miner.
2. The observer continues from `K + parent_step`; call that continuation `C`.
3. Measure the observer's per-position distribution over **the same `C`**, conditioned on
   `K + parent_step`, on `K + miner_step`, and on `K` alone.

That gives disagreement `s`, parent effect `d_G`, and miner effect `d_A`. The score rewards
**effect similarity** (low `s`) and **effect magnitude** (low `|d_A − d_G|`), both normalised by
`d_G` so the metric is scale-invariant across trajectories.

**There is no style channel.** A paraphrase carrying the same information scores 0.80; doing
nothing scores 0.14; moving the observer the wrong way scores 0.02. Miners cannot overfit to
the parent's wording alone — they have to preserve the information the parent added at that step.

Three properties that keep it fair, each pinned by a test:

- **The exam is drawn from the post-commit round nonce**, so a miner cannot know which trajectory
  items will be scored when its bytes are sealed. Every configured judge scores every item.
- **Discards are decided by the parent's effect alone.** Samples where the parent moves the
  observer nowhere carry no signal and are dropped — but never based on miner output, or a miner
  could bury its hard samples by emitting bland steps.
- **Worst-slice aggregation** over (observer × language × depth), so a strong slice cannot buy a
  weak one.

Scale is the anti-overfit lever, and here it works for a specific reason: the pre-fit attack is
"memorize the parent's step at every `K`", which requires running the parent across the corpus
and training the student to match — **that is distillation**. The cheat and the job are the same
activity. ~42M verified trajectories across reasoning, agentic-code, dialogue and non-Latin
sources ([`eval/steps.py`](eval/steps.py)); SWE-ZERO is deduped to ≤5 rollouts per task because
its 12.29M rows are only ~122,908 unique pull requests.

Observer-KL is a retention proxy, not a capability score. The production `score_job` does not wire
in a separate task-accuracy suite, so capability claims require a distinct, published benchmark.

## Auditability — scoring is expensive, checking is cheap

Ralph validators run the GPU scoring. Anyone can check the signed arithmetic and exam selection on
CPU; reproducing judge effects or regenerating model steps is a separate, expensive check that
requires the recorded models and a matching inference environment.

Publishing artifacts is not enough. A subnet can publish every prompt, every judge verdict and
every score and still be unfalsifiable, because if the grade came from an unpinned LLM with no seed
you can prove the operator added the numbers up wrong but never that they **graded** wrong. So the
crown here is built to be **recomputable**, not merely transparent:

* **The operator does not choose the exam.** Which trajectory items are scored is derived from
  `commit_root ‖ round_nonce` — a block hash drawn *after* the commitment window closes. The chosen
  indices are in the signed record, and the whole configured judge pool scores them.
* **The record is a re-run manifest.** Corpus + revision + ordering, item indices, observer and
  observer pool, token budgets, stack versions, and the measured noise floor the crown was gated
  on. Every scored point carries the parent step and continuation as literal text, and both the
  challenger's and the **incumbent's** steps are frozen — so a re-run is a pure forward pass and
  never has to reproduce batched generation.
* **Anyone re-runs it**, at four levels of cost:

```bash
python -m eval.rerun record.json                                     # L0 arithmetic, free, no GPU
python -m eval.rerun record.json --pool items.jsonl                  # + L1 re-derive the exam
python -m eval.rerun record.json --pool i.jsonl --observer <hf>      # + L2 re-derive the grades
python -m eval.rerun record.json --pool i.jsonl --observer <hf> \
                                --artifacts ./ckpts                  # + L3 bind to the models
python -m eval.rerun --history ./published --head <on-chain anchor>   # is the trail complete?
```

**L0** recomputes the score, the crown floor, the paired challenger-share bound and the candidate
weights from the published measurements with no models at all — by calling the *same* scorer the
round ran, not a second copy of the rule. **L1** re-derives which items were scored from the nonce,
and checks the exam was neither pruned nor padded and that slice keys follow from the items. **L2**
recomputes a recorded judge's distributions over the frozen text.

**L3 is the one that makes a crown non-forgeable**, and it is worth being exact about why. The
miner's steps are frozen into the record by the same operator who signs it, so an operator can
write ideal steps and L0–L2 will faithfully confirm that those strings produce those numbers. A
forged perfect score reproduces at every level below L3. Only loading the checkpoint and
re-generating binds the record to the models.

**Exit 0 REPRODUCED / 1 DIVERGED / 2 INCOMPLETE.** Skipping the expensive levels exits 2, never 0,
so nobody can call a round verified by running the cheap half — and an artifact the auditor could
not fetch is reported as unchecked, not as a pass.

The adversary assumed throughout is the **operator holding the signing key**, so a valid signature
is treated as attribution and never as evidence. The tests re-sign every rigged record and require
each rig to be caught by recomputation instead.

**Fail-closed means hold, not halt** — stated plainly because overstating it would be its own
dishonesty. Withholding `set_weights` does not stop emission; the previous weights persist, so the
last verifiably published crown keeps earning until publishing is fixed. That residual is why the
gate re-verifies a *window* of past rounds every round rather than only the current one.

### Running as an auditor validator

`eval.auditor` follows the published trail, runs the verification levels its operator can support,
and writes a **signed verdict**. L0/L1 are CPU checks; L2/L3 are model-inference checks.

```bash
python -m eval.auditor --once  --require L0     --signer <validator record key>   # free, no models
python -m eval.auditor --follow --require L0,L1 --signer <key> --interval 600     # + the exam
python -m eval.auditor --follow --require L0,L1,L2 --signer <key> \
    --observer <one-recorded-hf-id>                                                # one judge
python -m eval.auditor --follow --signer <key> --validator-hotkey <ss58> \
    --wallet <wallet> --hotkey <hotkey> --interval 1200 --set-weights             # act on it
```

The last form is what a weight-setting validator runs: every pass it re-verifies any new round,
then sets the vector of the newest round it accepted with its own hotkey — every pass, so the vector
is refreshed inside the subnet's activity cutoff and the validator never reads as absent. Rounds
1–4 were drawn under an earlier remainder rule; records now name their `selection_rule`, and an
untagged legacy record is accepted under whichever of the two rules reproduces it (both are
deterministic in the nonce — see `eval/rerun.py`).

L1 needs no corpus file: the record pins the pool's digest inside its signed body, so the auditor
fetches the pool from the trail and re-digests it.

In a multi-judge record, one L2 pass re-derives one recorded judge's points. Run the check once per
judge for full coverage; the audit reports partial coverage rather than implying that one model
reproduced the aggregate.

**L2 only compares numbers on matching hardware.** Measured cross-box spread is ~0.03 retention on
a genuine compression and ~0.17 on a control, far above `reproduction_tolerance` — so on a
different GPU the effects comparison is reported as *not run*, never as a divergence. Hardware is
not evidence.

**Two identities, and they are not the same string.** `--signer` is the key the operator signs
records with; `--validator-hotkey` is the ss58 whose on-chain commitment holds the anchor. Weight
setting needs both, and preflight refuses to start without them rather than letting a
misconfigured daemon look identical to a quiet subnet.

**The verdict is the primary evidence; setting weights is a separate action.** The signed verdict
says which round and digest were checked, at which levels, against which on-chain head. An auditor
may separately opt into an on-chain weight vote.

But nothing on chain distinguishes an auditor that **verified** from one that **copied**: identical
vectors are never clipped, vtrust and bonds are maximal, and there is no copier detection anywhere in
the pallet. Only the published verdict makes that difference visible — round, record digest, on-chain
head, which levels ran, which did not, and why. It is free, it costs no vtrust, and anyone can check
it. Weight-setting is a second channel with a known price, which is why it is opt-in.

Four properties keep a verdict honest:

* **It says what it did not do.** `levels_run` and `levels_required` are inside the signed body, so
  an auditor that ran only the free arithmetic cannot be mistaken for one that reloaded checkpoints.
* **Any failure is a failure.** A FAIL at a level the auditor did not declare as required still
  rejects the round. `required` only decides whether a *skip* makes the verdict INCOMPLETE.
* **The signer is pinned.** `eval.rerun` can prove a signature is valid; it cannot know whose it
  should be. Without `--signer`, the verdict says the record is internally consistent — **not** that
  the subnet's validator wrote it — and reads INCOMPLETE.
* **Silence is a finding.** A trail with no new round past the threshold emits a `STALE` verdict.
  v1's validator stopped publishing on 7 July 2026 and kept setting weights; nothing alarmed for
  four days.

On a rejected round the auditor **holds** — it weights the last round it actually verified, the same
direction the operator's own publish gate fails. An auditor that pays the disputed crown anyway is a
weight-copier, which manufactures the appearance of independent agreement while adding no safety.
That is the distinction the whole role turns on, and it is what the tests assert.

With **nothing yet verified**, it burns to uid 0 rather than setting nothing. "Hold" is sound for the
incumbent operator — its previous weights persist on chain — but a fresh auditor has none to persist,
and a validator that writes no weights contributes nothing to consensus and eventually crosses
`activity_cutoff` into inactive while believing it is validating.

### Publishing your verdicts

```bash
python -m eval.auditor --follow --require L0,L1 --signer <key> \
    --verdicts-repo <your-hf-org>/ralph-audits --commit-verdicts
```

Verdicts chain (`prev` = the previous verdict's hash, inside the signed body) so an auditor cannot
quietly drop the one it later regrets — and `--commit-verdicts` writes the chain head to **your own**
on-chain commitment slot, because a chain you compute over files you own, checked against an index
you write, proves nothing. That is the same trap the operator's anchor check fell into first.

Anyone can then walk your trail with `eval.verdicts.verify_verdict_trail`, and it reports
`ok=False` when there is no on-chain head to check against rather than glossing it.

Unlike a round record, **a verdict may be revised** — an auditor that could not fetch a record rules
INCOMPLETE and must be able to rule again when the sink recovers. So a revision *appends* and both
rulings stay readable. Changing your mind is allowed; pretending you never held the first view is not.

**A failed publish stops the vote.** If a verdicts repo is configured and publishing fails, the
auditor holds rather than setting weights — a validator writing weights with no published reasoning
is on chain indistinguishable from the copier this whole role exists to be distinguishable from.

**A missing record is not an accusation until it persists.** A deleted record and a 429 from
HuggingFace are byte-for-byte the same observation; only recurrence distinguishes them. So an
unfetchable record reads INCOMPLETE for three passes, is re-audited rather than written off, and
only then escalates to a broken trail.

## Anti-gaming — rules, not detection

The last team proved you **cannot detect** copying on a shared base. v2 makes gaming
*unprofitable* instead — all of the following are built and tested:

- **A copy cannot take a crown or challenger share.** The king is re-scored on the same fresh items
  every round; an exact copy has zero displayed lead and a zero paired lead. Crown transitions use
  the displayed point lead, the one-round/persistence margins, any reign decay, and the same-judge
  floor. The paired lower bound separately gates challenger share in the candidate allocation.
- **Content-addressed identity + commit-reveal.** You're bound to the exact bytes you
  committed (weights *and* config/tokenizer) — no post-commit swap.
- **Anti-grind admission.** One artifact per `(coldkey, tier)` can enter a round, and bytes already
  scored in the public trail are skipped. The bond accounting code is disabled (`base_bond=0`)
  because no on-chain escrow/refund extrinsic exists.
- **Signed round records.** Every round record is signed by the validator and independently
  re-runnable from the recorded seeds.

## Current operating state and limits

As of **2026-09-11**, seven real-model rounds are public, signed, hash-chained, and anchored on
mainnet. Round 7 was the first all-three-judges round: `binary`, `sub2`, and `sub4` changed hands;
`ternary` held because one challenger slice fell below the incumbent's floor. All four resulting
artifacts are public in the crown repository. A public chain snapshot after that round showed six
later submissions awaiting a future score; that count is a point-in-time queue, not a promise that
they will pass intake or be evaluated on a particular date.

The remaining boundaries matter:

- **Weights follow verified records.** A validator sets a round's vector only after its own
  audit accepts the record (`eval.auditor --set-weights`); until then the previous on-chain
  vector persists.
- **Retention is not capability.** The production metric compares downstream observer effects.
  No task-accuracy suite is wired into the crown path, and no benchmark result should be inferred
  from a retention number.
- **Reproduction is environment-sensitive.** GPU model, batch size, attention implementation, and
  stack versions are recorded. L2 reports mismatched hardware as incomplete, not as evidence of
  fraud.
- **Non-Latin coverage is currently Hindi and Chinese.** It is not a claim of broad multilingual
  coverage.
- **`eval/budget.py` is not in the crown path.** Score-at-budget and convergence logic remains an
  offline tool.

## Positioning

Ralph v2 focuses on a specific lane: architecture-preserving compression under explicit bit
budgets, a fresh commit-then-generate evaluation, economic indifference to copies, and a tiered
family of downloadable crowned artifacts.
