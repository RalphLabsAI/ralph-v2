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
```

Exit `0` reproduced · `1` diverged · `2` incomplete. Skipping the expensive levels exits 2, never
0 — nobody can call a round verified by running the cheap half.

The adversary assumed throughout is **the operator holding the signing key**, so a valid signature
attributes a record and never validates it.

---

## Run it locally

```bash
python -m tests.test_crown_path        # 49/49, CPU, no GPU needed
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

## Auditability — one validator, many auditors

v2 runs **one owner validator**, like several production subnets. The question that matters is not
how many validators there are, but whether an outsider can *check* the one that exists.

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

The mechanism is complete and tested end to end against a fake chain. What has **not** happened:

- **No run with real models, ever.** Every number in this repo comes from stubs or from real
  *data*; parent + observer + a genuinely quantized student on a GPU has never been executed
  once. Four of the last five real bugs here only appeared against real inputs, so expect that
  first run to surface more.
- **Determinism is a property of the BOX, and two of the three knobs are now pinned in the
  record.** Measured on H100 PCIe / A100 SXM4 / L40S with byte-identical items: within a fixed
  configuration a round is bit-exact (0.0 spread, byte-identical generations), but the H100
  returned four different outputs from four identical `generate()` calls until the attention
  implementation was pinned. Across GPUs the boxes still generate different step text, so a score
  is only meaningful against a recorded (gpu, batch_size, attn_implementation) — all three now
  live in the signed manifest, and the audit reports a mismatch as hardware rather than fraud.
- **The noise floor is still unmeasured in the old KL sense, and it sets the audit tolerance.**
  `eval/determinism.py` gates crowns on it and the record derives `reproduction_tolerance` from
  it, so the same unmeasured number decides both whether a crown is real and whether a re-run
  counts as reproducing. Worse, it is currently probed only on the single-sequence
  `distributions` path — never over a whole round, and never across two GPU models — so the 0.0
  it reports under stubs is an artefact, not a result. SN97 measured 2-5 percentage points of run-to-run variance on
  logit-derived metrics; if ours lands there, the crown may be unable to resolve honest
  differences and the aggregation needs retuning. **This single number decides whether the
  mechanism works.**
- **The anchor chain has never been committed by a real chain.** `A_n = H(A_{n-1} ‖ digest)` makes
  one commitment slot cover the whole history, and `BittensorChainIO` computes and reads it — but
  against FakeChain only. Until a real `commit_audit_root` call lands, the strongest guarantee here
  is untested on the network it is for.
- **No round record has ever been published to a real sink.** The publisher, its fail-closed
  gate and the re-run tool are tested end to end against a local directory and a fake chain;
  `HFSink` has never uploaded anything. Until it has, "anyone can re-run a crown" is a property of
  the code and not yet a fact about the network.
- **`verify_history` is not run automatically.** Every round re-checks a trailing window; nothing
  walks the full trail on a schedule, so a gap or a deletion outside the window waits for a human
  to look.
- **No miner has ever submitted anything.** The end-to-end submission flow is untested by a
  third party, and `fetch_dir_for` is unwired — the validator cannot yet resolve a committed URI
  to bytes at all.
- **Non-Latin coverage is Hindi and Chinese.** That is where the anti-clone axis binds today; the
  published evidence for low-bit collapse is Persian and Cyrillic, so the pool does not yet test
  where the proof is.
- **Chain integration is FakeChain only**, and the validator box needs `bubblewrap` for the
  sandboxed code canary.
- **`eval/budget.py`** (score-at-budget / convergence gate) is written and tested but not wired
  into the crown path.

## Positioning vs the family

SN3 trains full-precision models on a frozen arch (compression out of scope). SN97 fine-tunes
a fixed model, quantization banned. SN120 runs RL environments on full-size students. **Nobody
owns the efficiency axis** — v2 is the missing quadrant: a fresh, commit-then-generate
*verifiable* eval; reward-indifference-to-copies (economic, not detection); and a tiered
family of downloadable open checkpoints.
