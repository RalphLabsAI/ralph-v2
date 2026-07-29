# SN40 v2 — The Model Compression Subnet

**One line:** miners compress a pinned open model — starting with **GLM** — to a low bit budget
with its architecture unchanged; the best compression per bit tier wears a crown, and every
crown ships as a downloadable, runnable open model.

**Reference point:** PrismML's Ternary Bonsai showed what one lab gets from aggressive
compression (much smaller, faster, HF-trending). v2 makes that a standing market — a
permissionless swarm doing it to GLM continuously — with an anti-gaming design built from
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
3. **Crown** — KOTH, incumbent re-scored *and re-gated* every round, dethrone on a paired
   bootstrap lower bound of the worst-slice difference — **and refused if the margin sits inside
   the validator's own measured noise floor.**
4. **Publish** — a signed, reproducible round record.

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
- **The noise floor is unmeasured.** `eval/determinism.py` gates crowns on it, but the number
  itself needs a GPU run. SN97 measured 2-5 percentage points of run-to-run variance on
  logit-derived metrics; if ours lands there, the crown may be unable to resolve honest
  differences and the aggregation needs retuning. **This single number decides whether the
  mechanism works.**
- **No artifact locator.** A crown publishes a content hash with no `artifact_uri` and no
  per-file manifest, so there is no path from "king" to bytes you can download.
- **No miner has ever submitted anything.** The end-to-end submission flow is untested by a
  third party.
- **Chain integration is FakeChain only**, and the validator box needs `bubblewrap` for the
  sandboxed code canary.
- **`eval/budget.py`** (score-at-budget / convergence gate) is written and tested but not wired
  into the crown path.

## Run it

```bash
pip install -r requirements.txt        # one dep for the tests: pynacl
python -m tests.test_crown_path        # 41/41, CPU
python -m eval.run_capability_axis     # the capability CANARY on real models (GPU)
python -m eval.shadow_axis_epoch       # the full operator epoch end to end (fake chain)
```

Miners: see [`miner/QUICKSTART.md`](miner/QUICKSTART.md).

## Positioning vs the family

SN3 trains full-precision models on a frozen arch (compression out of scope). SN97 fine-tunes
a fixed model, quantization banned. SN120 runs RL environments on full-size students. **Nobody
owns the efficiency axis** — v2 is the missing quadrant: a fresh, commit-then-generate
*verifiable* eval; reward-indifference-to-copies (economic, not detection); and a tiered
family of downloadable open checkpoints.
