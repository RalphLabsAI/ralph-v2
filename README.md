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
2. **Score** — items are minted **fresh from the commit nonce** (they don't exist until
   checkpoints lock, so there's nothing to memorize offline). GLM authors the reference,
   then the student and the reigning king answer the same items.
3. **Aggregate** — per-axis normalized retention `(student − base) / (teacher − base)` on the
   items GLM itself passes, combined **worst-domain** (soft-min). Your weakest axis is your
   score. Cap 1.25 (beating the teacher is the interesting case); negative retention on any
   axis is an automatic reject.
4. **Crown** — KOTH dethrone-on-margin, plus a signed reproducible round record.

## Scoring is capability, not imitation

Distillation contests die when they score *imitation* (KL / distribution-match): students
learn the teacher's **style**, win the metric, lose the capability. v2 never scores
imitation — every axis has a **deterministic checker**, so a wrong answer fails no matter how
GLM-like it reads:

| axis | checker |
|---|---|
| math | exact numeric answer (multi-step word problems) |
| code | **execute hidden unit tests** |
| instruction-following | verifiable format constraints (parsers) |
| long-context | retrieval + aggregation over a haystack |
| multi-hop | compose several stated facts (with a shortcut distractor) |

Two of these — long-context and multi-hop — are what compression breaks *first*, so the
worst-domain crown is set by what a compressed model **loses**, not what it keeps. **No KL in
the reward; any KL you like in your trainer** — banning imitation as a *method* would outlaw
the state of the art (on-policy distillation), we only refuse to *pay* for style.

## Anti-gaming — economics, not detection

The last team proved you **cannot detect** copying on a shared base. v2 makes gaming
*unprofitable* instead — all of the following are built and tested:

- **A copy earns nothing.** The king is re-scored on the same fresh items every round; an
  exact copy ties (zero margin, paired bootstrap-LCB on the **worst axis**) and cannot
  dethrone. Dethroning needs a *strict* worst-axis improvement past the noise floor.
- **Genre-overfit gate (stale-vs-fresh diff-in-diff).** Distilling GLM only over the
  announced distribution doesn't win: a student whose edge over base **collapses on
  genuinely-new same-genre documents** is demoted regardless of its axis scores. Freshness
  alone doesn't stop distribution-overfit; this does.
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
- **Live fresh-corpus feed** — the genre-overfit gate is validated on a static timestamped
  dataset; production draws the fresh half from a live post-commit feed.
- **Delivery** — GGUF / "run the king on your phone", GLM-4.5-Air as a second rung,
  first-committer-wins + delayed reveal, never-shrink ratchet.
- **Chain integration** — the round is wired to a `ChainIO` boundary and tested against a
  fake chain; the live-validator wiring + testnet shakedown is the remaining engineering.

## Run it

```bash
pip install -r requirements.txt        # one dep for the tests: pynacl
python -m tests.test_crown_path        # 19/19, CPU
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
