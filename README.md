# SN40 v2 — The Model Compression Subnet

**One line:** miners compress a pinned open model to a low bit budget with its architecture
unchanged; the best compression per bit tier wears a crown, and every crown ships as a
downloadable, runnable open model.

**Pinned parent today: `Qwen/Qwen2.5-0.5B-Instruct`.** GLM is the intended target and has never
been run — the mechanism was validated on a 0.5B parent deliberately, because "does it
discriminate and does it reproduce" is answered as well by a small model as a large one, for a
fraction of the rent. Scaling the parent is a config change (`eval/parent.py`); it is not a
claim this repo has earned yet.

**Reference point:** PrismML's Ternary Bonsai showed what one lab gets from aggressive
compression (much smaller, faster, HF-trending). v2 makes that a standing market — a
permissionless swarm doing it continuously — with an anti-gaming design built from
the public lessons of earlier distillation-KOTH attempts
([`docs/prior-art-and-lessons.md`](docs/prior-art-and-lessons.md)).

## How it works

The task is **PrismML/Bonsai's task**: take one **pinned parent** model, leave the architecture
untouched, and re-store its weights at a low bit budget. Miners submit a compressed checkpoint
(safetensors **or GGUF**) into a bit tier. The validator scores the artifact itself and never
inspects or trusts the private training process.

Each round:

1. **Front door** — economics → safety inspect (no pickles, no remote code) → **bit budget
   measured from the tensor data** → **pinned-parent shape check** → **content-hash +
   commit-reveal**. No untrusted weights load until all of it passes.
2. **Score** — observer-KL over trajectory steps (below).
3. **Identity canary** — the pinned parent is scored *as if it were a submission* and must come
   back at exactly **1.0**. Not a modelling assumption: when the submitted step IS the parent's
   step, `s = 0` and `|d_A − d_G| = 0`, so any shortfall is the pipeline disagreeing with itself.
   Measured on real hardware this caught an H100 returning **0.8754 while the noise probe
   reported a clean floor** — batched generation there is nondeterministic above ~64 new tokens.
   A validator that cannot reproduce the identity **aborts the round** instead of crowning.
4. **Crown** — KOTH, incumbent re-scored *and re-gated* every round, dethrone on a paired
   bootstrap lower bound of the worst-slice difference — **and refused if the margin sits inside
   the validator's own measured noise floor.**
5. **Publish, then pay** — the signed round record is written, **read back and verified**, and
   anchored on chain *before* any weight is set. If it cannot be published and verified, no
   crown is written and no weights are set (see [Auditability](#auditability-one-validator-many-auditors)).

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

## The honest claim

The crown certifies that a submitted artifact, at a **measured bit budget**, adds
**approximately the same information to a trajectory as the pinned parent does**, as judged by
independent observer models on trajectories the miner did not choose.

It does *not* certify lineage. The pinned-parent check is a shape/architecture ADMISSION gate,
not a proof of descent — nothing behavioural can prove descent, since a different model of
identical shape would pass the same checks. And observer-KL measures informational equivalence,
not capability: a student could move the observer correctly and still be unusable. That is the
one blind spot, it is why the capability canary is kept, and we claim exactly what the eval
proves and no more.

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
  third party.
- **Chain integration is FakeChain only**, and the validator box needs `bubblewrap` for the
  sandboxed code canary.
- **`eval/budget.py`** (score-at-budget / convergence gate) is written and tested but not wired
  into the crown path.

## Run it

```bash
pip install -r requirements.txt        # one dep for the tests: pynacl
python -m tests.test_crown_path        # 47/47, CPU
python -m eval.simulate_submission     # THE WHOLE CYCLE: miner -> validator -> auditor, seconds
```

`simulate_submission` is the fastest way to see what this actually is. It walks the six intake
gates in order, draws the observer from the post-commit nonce, runs the identity canary, crowns,
publishes fail-closed with an on-chain anchor, and then re-audits its own published record at all
four levels. Stub models, but every gate, the crown logic, publishing, anchoring and the audit are
real. Add `--real` for the pinned parent and a genuine 4-bit quantization on a GPU.

```bash
python -m eval.shakedown_round_noise   # whole-round noise floor, in retention units (GPU)
python -m eval.ladder_probe            # does the metric ORDER compressions? (GPU)
python -m eval.batch_invariance        # is the score a function of the artifact alone? (GPU)
```

Miners: see [`miner/QUICKSTART.md`](miner/QUICKSTART.md).

## Positioning vs the family

SN3 trains full-precision models on a frozen arch (compression out of scope). SN97 fine-tunes
a fixed model, quantization banned. SN120 runs RL environments on full-size students. **Nobody
owns the efficiency axis** — v2 is the missing quadrant: a fresh, commit-then-generate
*verifiable* eval; reward-indifference-to-copies (economic, not detection); and a tiered
family of downloadable open checkpoints.
