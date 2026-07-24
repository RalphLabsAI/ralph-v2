# SN40 v2 — The Model Compression Subnet

**One line:** miners compress a frontier open model — starting with **GLM** — into a small
student that keeps the teacher's *capability*; the best student per size tier wears a crown,
and every crown ships as a downloadable open model.

**Reference point:** PrismML's Ternary Bonsai showed what one lab gets from aggressive
compression (much smaller, faster, HF-trending). v2 makes that a standing market — a
permissionless swarm doing it to GLM continuously — with an anti-gaming design built from
the public lessons of earlier distillation-KOTH attempts
([`docs/prior-art-and-lessons.md`](docs/prior-art-and-lessons.md)).

## How it works

Fix a **(teacher, permitted student-base) pair** — the pair, not just the teacher, because
*match beats strength* (OpenThoughts, [arXiv:2506.04178](https://arxiv.org/abs/2506.04178)
Table 8: QwQ-32B is a better teacher than the stronger DeepSeek-R1). Miners submit a
**student checkpoint** into a size tier. The validator scores the artifact itself — download
it, measure what it can do — and never inspects or trusts the private training process.

Each round:

1. **Front door** — economics → safety inspect (safetensors-only, no remote code, params/
   bits from tensor headers) → tier fit → **content-hash + commit-reveal** (you're judged on
   the exact bytes you committed to; no bait-and-switch). No untrusted weights load until all
   of this passes.
2. **Score** — the crown is decided on **real, post-commit content** (see below). GLM authors
   the reference on the same items, then the student and the reigning king answer them.
3. **Aggregate** — per-axis normalized retention `(student − base) / (teacher − base)` on the
   items GLM itself passes, combined **worst-domain** (soft-min). Your weakest axis is your
   score. Cap 1.25 (beating the teacher is the interesting case); negative retention on any
   axis is an automatic reject.
4. **Crown** — KOTH dethrone-on-margin, plus a signed reproducible round record.

## The crown is real∩verifiable, not synthetic generators

Every distillation-KOTH before this (SN97/Distil's own post-mortem is explicit about it) got
gamed the same way: score on a **synthetic/procedural generator**, and a miner learns the
*generator* — a cheap specialist that aces every "fresh" instance without retaining the
teacher's capability. Freshness of *instances* doesn't help: the generator is public, so
there is no held-out information. The score climbs while real capability stays flat (Goodhart;
GSM-Symbolic / GSM1k / RULER all show the same decoupling).

So the crown is set by a **deterministic checker over REAL text that is fresh after the
commit**, the one place where CPU-auditable *and* un-pre-distillable both hold:

| role | axis | checker |
|---|---|---|
| **crown** | **extractive QA over fresh real docs** | the answer is a **verbatim span in the document** — exact-match, no judge, no KL. A doc published *after* the seal carries information the miner could not have distilled. |
| floor | math / code / instruction / long-context / multi-hop | the synthetic generators — deterministic, but the distribution is closed. **Demoted to a liveness/calibration floor:** they gate (a broken or degenerate student is disqualified) but **cannot set the crown.** |

A wrong answer fails the checker no matter how GLM-like it reads, so **style earns nothing**
(no KL in the reward — bring any KL you like to your *trainer*). And because the crown rides
on the fresh-real axis, a generator-specialist that aces every synthetic floor axis but reads
fresh documents only at base level is **not crownable** — necessity, not just sufficiency, is
tested (`tests/test_crown_path.py::test_generator_specialist_denied_crown`).

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

The crown certifies **verifiable capability retention of GLM** across a broad, freshly-minted
distribution. It does *not* certify "is specifically a compressed GLM" — no behavioral eval
can prove lineage (a different, equally-capable model passes the same checkers). We claim
exactly what the eval proves.

## Roadmap (not yet built)

The current subnet crowns capability retention per size tier. These are designed but not
implemented, and the code does not claim them:

- **Bit-tier structure + capability-per-compute** — 1.58/2/4-bit tiers, effective-bits audit
  by recompression, and a compute denominator (teacher generation + training + scoring FLOPs,
  reconciled) in the ranking. Today: size tiers, compute declared-not-ranked.
- **pass@k gate** — a mode-collapse detector (pass@1 can't see it); needs a determinism
  redesign (seeded sampling) before it's turned on. Currently a no-op.
- **Throughput attestation** — a non-forgeable "runs this fast on real hardware" bonus
  (reusing Ralph's TDX+H100-CC stack).
- **Live post-commit corpus feed + provenance** — the extractive crown axis and the
  genre-overfit gate are built and run on a timestamped dataset (CC-News); the un-pre-
  distillability guarantee needs the fresh half to come from a *live* post-commit feed with
  source-provenance attestation and a grind-resistant randomness beacon for the round nonce.
  This is the load-bearing remaining infra.
- **Real-repo code axis** — a second crown axis: fresh commits + their real test suites (run
  the tests = deterministic, un-pre-distillable), pinned dependency snapshot, sandboxed.
- **Delivery** — GGUF / "run the king on your phone", GLM-4.5-Air as a second rung,
  first-committer-wins + delayed reveal, never-shrink ratchet.
- **Chain integration** — the round is wired to a `ChainIO` boundary and tested against a
  fake chain; the live-validator wiring + testnet shakedown is the remaining engineering.

## Run it

```bash
pip install -r requirements.txt        # one dep for the tests: pynacl
python -m tests.test_crown_path        # 22/22, CPU (incl. the generator-specialist denial)
python -m eval.run_capability_axis     # real GLM + student ladder: per-axis retention, crown
python -m eval.shadow_axis_epoch       # the full operator epoch end to end (fake chain)
```

Miners: see [`miner/QUICKSTART.md`](miner/QUICKSTART.md).

## Positioning vs the family

SN3 trains full-precision models on a frozen arch (compression out of scope). SN97 fine-tunes
a fixed model, quantization banned. SN120 runs RL environments on full-size students. **Nobody
owns the efficiency axis** — v2 is the missing quadrant: a fresh, commit-then-generate
*verifiable* eval; reward-indifference-to-copies (economic, not detection); and a tiered
family of downloadable open checkpoints.
