<p align="center">
  <img src="docs/assets/ralph-banner.png" alt="Ralph Labs — open model compression" width="100%">
</p>

# Ralph — open model compression on Bittensor SN40

**Compress one pinned open model into fewer bits, without touching its architecture. Best
compression per bit tier wears the crown, and every crown ships as downloadable weights.**

`Qwen/Qwen3-8B` pinned · `16.38 GB` at bf16 → `1.75 GB` at ternary · [netuid 40](https://taostats.io/subnets/40)

This is not a smaller model trained to imitate a bigger one. It is the model you already know,
stored differently — smaller, not dumber. A submission whose shape does not match the parent is
refused before any weights load.

---

## For miners

You compress privately, however you like. The subnet never inspects your method, only your artifact.

```bash
git clone https://github.com/RalphLabsAI/ralph-v2 && cd ralph-v2
pip install -r requirements.txt

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
| 1 | economics | not registered, or no bond outside your free evaluation |
| 2 | safety | pickles, remote code, or files that are not weights |
| 3 | tier fit | parameter count or dtype headers inconsistent |
| 4 | **bit budget** | measured bits/weight over the tier cap — read from tensor DATA, not the dtype header |
| 5 | **pinned parent** | architecture or weight-element count does not match `Qwen/Qwen3-8B` |
| 6 | commit-reveal | bytes do not hash to what you committed before the nonce existed |

### Bit tiers

| tier | max bits/weight achieved | ≈ size at 8B |
|---|---|---|
| `binary` | 1.15 | 1.18 GB |
| `ternary` | 1.75 | 1.79 GB |
| `sub2` | 2.0 | 2.05 GB |
| `sub4` | 4.0 | 4.10 GB |

A 4-bit model shipped inside a 16-bit container is credited for the compression it achieved and
rejected as an unshippable artifact — both budgets bind.

### Formats that can win

**The crown has to run on a phone.** That is the product, so a format mainline `llama.cpp` cannot
execute on Apple GPU is rejected at intake no matter how good its retention would have been — we
score on an H100 where it might run perfectly well, and reject it anyway.

| | |
|---|---|
| **use** | `Q1_0`, `Q2_0`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, and the `Q*_K` family |
| **rejected** | **`TQ1_0`, `TQ2_0`** — mainline has no Metal kernel for either |

**The `TQ*` types are the trap.** They are the obvious choice by name at ~1.1 and ~2.1 bits, they
pack beautifully, and they run fine on the CUDA box you built them on — and they exist only in
llama.cpp's CPU and CUDA paths. There is no Metal kernel for either, so they cannot run on a phone.
`TQ1_0` has already cost two miners their entry, including the only `binary` submission this subnet
has ever received. Use `Q1_0` or `IQ1_S` at the binary end, `Q2_0` or `IQ2_XXS` at the sub-2 end.

Check before you commit — the same code the validator runs:

```bash
python -m eval.bitrate path/to/model.gguf
```

### What wins

Your model and the parent each continue the same reasoning trajectory. An independent observer
model reads both, and your score is how closely you moved it to where the parent did. Scores
aggregate **worst-slice across languages**, so a weak language cannot be bought with a strong one.

Which trajectories get scored, and which observer grades them, are both drawn from a chain value
that does not exist until your checkpoint is sealed. There is no fixed test set to fit and no
grader you can name in advance.

---

## For anyone checking a crown

Every round publishes what it scored, which items it drew and which observer graded them, so a
result can be recomputed rather than believed.

```bash
python -m eval.rerun <record.json>                                    # arithmetic only, no GPU
python -m eval.rerun <record.json> --pool <items.jsonl> \
    --observer <hf-id> --artifacts <ckpt-dir>                         # full re-derivation
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
python -m tests.test_crown_path        # 57/57, CPU, no GPU needed
python -m eval.simulate_submission     # miner -> validator -> auditor in seconds
```

The second walks the six gates, draws the observer from the nonce, runs the identity check, crowns,
publishes fail-closed with an on-chain anchor, then re-verifies its own round at all four levels.

---

## The crown: semantic equivalence by downstream effect

Every distillation-KOTH before this got gamed the same way. Score **token agreement with the
teacher** and you pay for *style*: SN97's KL-crowned king mimicked "wait, let me reconsider"
filler, never produced an answer, and was worse than the un-finetuned base on 5/5 reasoning
benchmarks. Score **generated questions** and you pay for question-answering: probe formats are
public, so fitting them is a cheap narrow skill that transfers to nothing.

So we score neither. Two steps are equivalent when they move an **independent observer** into
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
GLM's wording — they have to extract the information GLM added at that step.

Three properties that keep it fair, each pinned by a test:

- **The observer is drawn from the round nonce**, so a miner cannot pre-fit which observer it
  will face.
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

The old capability axes remain as a cheap **canary**, not the crown, because observer-KL is
structurally blind to exactly one failure: a student that moves the observer correctly while
being unusable.

## Auditability — scoring is expensive, checking is cheap

Ralph validators run the GPU scoring; **auditors run on CPU**, so anyone can check a crown without
a datacentre. The question that matters is not how many validators there are, but whether an
outsider can *check* the ones that exist.

Publishing artifacts is not enough. A subnet can publish every prompt, every judge verdict and
every score and still be unfalsifiable, because if the grade came from an unpinned LLM with no seed
you can prove the operator added the numbers up wrong but never that they **graded** wrong. So the
crown here is built to be **recomputable**, not merely transparent:

* **The operator does not choose the exam.** Which trajectory items are scored, and which observer
  scores them, are derived from `commit_root ‖ round_nonce` — a block hash drawn *after* the
  commitment window closes. The chosen indices are in the signed record.
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

**L0** recomputes the score, the crown floor, the paired dethrone margin and the emission weights
from the published measurements with no models at all — by calling the *same* scorer the round
ran, not a second copy of the rule. **L1** re-derives which items were scored from the nonce, and
checks the exam was neither pruned nor padded and that slice keys follow from the items. **L2**
recomputes the observer's distributions over the frozen text — the level a judge-based subnet
cannot have.

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
last verifiably published crown keeps earning until publishing is fixed. An operator who breaks
publishing while their own model is king benefits from the freeze. That residual is why the gate
re-verifies a *window* of past rounds every round rather than only the current one.

### Running as an auditor validator

One GPU scorer, many CPU checkers. `eval.auditor` is the daemon for the second role: it follows the
published trail, re-runs each new round, and writes a **signed verdict**.

```bash
python -m eval.auditor --once  --require L0     --signer <validator record key>   # free, no models
python -m eval.auditor --follow --require L0,L1 --signer <key> --interval 600     # + the exam
python -m eval.auditor --follow --require L0,L1,L2 --signer <key> \
    --observer HuggingFaceTB/SmolLM2-1.7B-Instruct,google/gemma-2-2b-it           # + the grades
python -m eval.auditor --follow --signer <key> \
    --validator-hotkey <ss58> --set-weights                                       # act on it
```

L1 needs no corpus file: the record pins the pool's digest inside its signed body, so the auditor
fetches the pool from the trail and re-digests it.

**L2 takes the whole observer pool, not one model.** The grader rotates per round out of the nonce,
so a daemon pinned to a single observer is re-deriving the grades with the wrong one about half the
time — and since any failure rejects, it would publish a signed accusation each time. Pass the same
pool the operator declares and it loads whichever one that round drew. A round whose observer is
not in your pool is a skip, not a fault.

**L2 only compares numbers on matching hardware.** Measured cross-box spread is ~0.03 retention on
a genuine compression and ~0.17 on a control, far above `reproduction_tolerance` — so on a
different GPU the effects comparison is reported as *not run*, never as a divergence. Hardware is
not evidence.

**Two identities, and they are not the same string.** `--signer` is the key the operator signs
records with; `--validator-hotkey` is the ss58 whose on-chain commitment holds the anchor. Weight
setting needs both, and preflight refuses to start without them rather than letting a
misconfigured daemon look identical to a quiet subnet.

**The verdict is the primary product; the weight is a paid vote.** Simulating the shipped Yuma steps
on the two-tier layout this repo runs, an auditor with 30% stake that dissents on one disputed crown
settles at about half its dividends (0.30 → 0.15, vtrust 1.0 → 0.5) — and drives the disputed king's
incentive down 18%, because clipping only clips downward, so a dissenting zero survives and removes
that stake share from the rigged king's rank. Dissent is a proportional vote at a proportional price,
not a futile gesture.

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

## Anti-gaming — economics, not detection

The last team proved you **cannot detect** copying on a shared base. v2 makes gaming
*unprofitable* instead — all of the following are built and tested:

- **A copy earns nothing.** The king is re-scored on the same fresh items every round; an
  exact copy ties (zero margin, paired bootstrap-LCB on the **worst axis**) and cannot
  dethrone. Dethroning needs a *strict* worst-axis improvement past the noise floor.
- **Genre-overfit gate (stale-vs-fresh diff-in-diff), armed by default.** A student whose
  edge over base **collapses on genuinely-new same-genre documents** is demoted. This gate is
  only *sighted* when the fresh half is real text (on a synthetic generator, stale and fresh
  are the same closed distribution, so a specialist shows no gap) — which is exactly why the
  crown axis is real-corpus. Freshness alone doesn't stop distribution-overfit; this does.
- **Content-addressed identity + commit-reveal.** You're bound to the exact bytes you
  committed (weights *and* config/tokenizer) — no post-commit swap.
- **Anti-grind economics.** One free eval per **coldkey**, a per-coldkey round cap, and a
  bond for extra submissions refunded only when you improve your own best — spam and
  best-of-N crown-farming are unprofitable; honest iteration stays cheap.
- **Signed round records.** Every verdict is signed by the validator and independently
  re-runnable from the recorded seeds.

## Not yet done — read this before running it anywhere real

The mechanism is complete, has been run against real models on real GPUs
([`experiments.md`](experiments.md)), and has published a real round record. What has **not**
happened:

- **Determinism is a property of the BOX, and two of the three knobs are now pinned in the
  record.** Measured on H100 PCIe / A100 SXM4 / L40S with byte-identical items: within a fixed
  configuration a round is bit-exact (0.0 spread, byte-identical generations), but the H100
  returned four different outputs from four identical `generate()` calls until the attention
  implementation was pinned. Across GPUs the boxes still generate different step text, so a score
  is only meaningful against a recorded (gpu, batch_size, attn_implementation) — all three now
  live in the signed manifest, and the audit reports a mismatch as hardware rather than fraud.
- **The score saturates at the bottom.** Perturbing every weight and sweeping the magnitude, the
  score falls monotonically to σ=0.05 and then flattens around 0.43–0.57 — it never reaches the
  0.135 inert floor, because a wrecked model still emits *something* and something still moves the
  observer. So the metric cannot tell a broken model from a very broken one. Two things stop that
  deciding a crown: the tail wobble (0.036–0.048) is below the dethrone margin, and a real 4-bit
  beats the luckiest wrecked model by 0.066–0.083, above it. Crowns are contested at the top and
  the top is clean, but this is the closest thing here to scoring well while being useless, and it
  is why the capability canary stays.
- **The anchor chain has never been committed by a real chain.** `A_n = H(A_{n-1} ‖ digest)` makes
  one commitment slot cover the whole history, and `BittensorChainIO` computes and reads it — but
  no `commit_audit_root` call has landed on mainnet. Until one does, the strongest guarantee here
  is untested on the network it is for.
- **No miner has ever submitted anything.** The path is wired end to end — including the fetcher,
  which was the piece that made a submission impossible to score at all — but no third party has
  used it, and no auditor other than the operator's own has ever ruled on a round.
- **Weights have never been set by this validator.** It runs read-only by default and will keep
  doing so until there is something worth crowning; a second signer on a live hotkey fights the
  first.
- **Non-Latin coverage is Hindi and Chinese.** That is where the anti-clone axis binds today; the
  published evidence for low-bit collapse is Persian and Cyrillic, so the pool does not yet test
  where the proof is.
- **The validator box needs `bubblewrap`** for the sandboxed code canary.
- **`eval/budget.py`** (score-at-budget / convergence gate) is written and tested but not wired
  into the crown path.

## Positioning vs the family

SN3 trains full-precision models on a frozen arch (compression out of scope). SN97 fine-tunes
a fixed model, quantization banned. SN120 runs RL environments on full-size students. **Nobody
owns the efficiency axis** — v2 is the missing quadrant: a fresh, commit-then-generate
*verifiable* eval; reward-indifference-to-copies (economic, not detection); and a tiered
family of downloadable open checkpoints.
