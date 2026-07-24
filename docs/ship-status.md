# Ralph SN40 v2 — ship status

A **compression subnet**: miners distill a pinned teacher (GLM) into a small student; the
best student per size tier wears the crown and ships as an open checkpoint. This doc is the
honest handoff map — what is built, what is tested, what is stubbed, and what is
owner/operator-gated. It reflects the code as of the current commit; when it disagrees with
an older comment or docstring, this doc is right.

## What the crown measures (and the honest claim)

**Verifiable-outcome retention of GLM across deterministic-checker capability axes,
worst-domain aggregated.** Each axis is a capability GLM has, with a checker that grades
pass/fail by *outcome* (execute the code, check the number, match the entity) — never an
LLM judge, never logit/KL matching. So style/imitation earns nothing: a wrong answer fails
regardless of how GLM-like it reads, which is the property that structurally avoids the
SN97 "distill the style, lose the capability" trap.

The defensible public claim is **"verifiable capability retention of GLM"** — the eval
proves a student retains GLM's *capability* on a broad fresh distribution. It does **not**
prove "is *specifically* a compressed GLM" (a different, equally-capable model would also
pass) — no behavioral eval can, so don't claim it. The mechanism is const's compression /
capability-retention design; this is the labeling that matches what it actually proves.

## The cover (5 axes, calibrated to the teacher's competence band)

| axis | probes | checker | notes |
|---|---|---|---|
| math | multi-step word problems (work-rate, mixture, ratio, compound, …) | exact numeric | integer-by-construction; 8 families |
| code | implement a function | **execute hidden unit tests** | 10 parameterized families / 15 variants |
| instruction | verifiable format constraints (exact-words, JSON, bullets) | parser | IFEval-style |
| long_context | retrieval + aggregation over a haystack | exact match | *compression-fragile* |
| multihop | compose N stated facts, with a shortcut distractor | exact entity | *compression-fragile* |

An axis only measures compression if the **teacher clears it**. Difficulty and item count
are calibrated per-axis to GLM's band; verified on real models — all 5 live simultaneously,
crown lands, and retention ranks a 3B > 1.5B > 0.5B (`backup_h100/ver1/report.json`). The
crown is set by the *worst* axis, which lands on the two fragile ones — i.e. by what a
compressed model LOSES, not what it keeps. **Broadening any axis requires re-calibrating it
to the teacher (`eval/run_capability_axis.py`).**

## The round (eval/axis_round.py + eval/validator_axis_loop.py)

1. **Front door** (`intake.py` + `identity.py`): economics → safety inspect (safetensors-
   only, no remote code, params/bits from tensor headers) → tier fit → content-hash +
   commit-reveal fetch-verify. No untrusted weights load until all of this passes.
2. **Score** (`axis_round.py`): items minted fresh from the commit nonce (don't exist until
   checkpoints lock), GLM authors the reference, per-axis normalized retention →
   **worst-domain soft-min** → crown gates (negative-axis interval, all-declared-axes-live,
   MIN_CROWN_LB, degeneracy) → **KOTH dethrone on the WORST axis** (not the pooled mean).
3. **Genre-overfit precondition** (`overfit_gate.py`): stale-vs-fresh diff-in-diff — a
   student whose edge over base collapses on genuinely-fresh same-genre documents is demoted
   from crownable regardless of its axis scores. This is const's SN97 gate.
4. **Settle** anti-grind bonds; emit a **signed** reproducible record; write weights.

Chain boundary: `chain.run_v2_axis_epoch` (reads the sealed commit window, draws the
post-window nonce, writes back). Operator shadow: `eval/shadow_axis_epoch.py`.

## Built + tested (16 tests, `tests/test_crown_path.py`, CPU, no GPU/network)

- Worst-axis dethrone (drifter blocked, honest-better dethrones, exact copy ties)
- Crown gates (degeneracy DQ, all-axes-live, negative-axis, MIN_CROWN_LB)
- Deterministic checkers incl. output-style-robust code extraction + first-marker numeric
- Multi-hop composes correctly + the shortcut trap is not accepted
- Content identity + commit-reveal (bait-and-switch rejected; config-swap changes identity)
- Signed record (tampering with a crown decision or score invalidates the signature)
- Economics: free eval per COLDKEY (not hotkey), bonds settled
- Genre-overfit diff-in-diff (flags a memorizer, spares weak/strong honest controls) —
  validated live on 250 real cc_news articles, and wired into the crown path
- Full epoch through a FakeChain end-to-end (commits → nonce → intake → crown → signed
  record → weights)

Verified on real models (GPU, torn down): capability discrimination + 5-axis liveness.

## Not yet real / known gaps

- **pass@k is hardcoded open** (`verdict(..., passk_ok=True)`). It's a mode-collapse
  detector; as specified it conflicts with deterministic CPU-auditability and must be
  **redefined** (seeded sampling with the seed recorded) before it's implemented, not just
  turned on. The subnet's anti-gaming does not depend on it today.
- **Fresh corpus = a live feed in production.** The diff-in-diff is validated on a static
  timestamped dataset; the un-pre-distillable "fresh" half must come from a live post-commit
  feed (news/arXiv filtered by publication date > commit block time).
- **Compute metering** (`declared_compute_h100h`) is carried but not reconciled/ranked.
- **The self_state drift slice** (const's blessed idea) is validated only in a controlled
  env; it did not reproduce cheaply in-regime and is kept as a secondary slice, not the
  headline. Capability-retention is the validated core.

## Owner-gated (not code — decisions/ops only I can't do)

- **Chain integration** — `run_v2_axis_epoch` targets the `ChainIO` protocol that karpa's
  validator/service.py implements, but going live is blocked on the validator migration
  (rotated secrets, signer lock). Nothing reaches mainnet before that.
- **The public claim** — use "verifiable capability retention," per above.
- **The single-signer / off-live-signer discipline** stays as-is.

## Run it

```
# CPU, no deps beyond the venv: the whole test suite
python -m tests.test_crown_path

# real-model end-to-end on a GPU box (FakeChain), all 5 axes:
python -m eval.run_capability_axis                 # capability discrimination check
python -m eval.shadow_axis_epoch                   # full operator epoch on the axis substrate
RALPH_OVERFIT_CORPUS=cc_news python -m eval.shadow_axis_epoch   # + live genre-overfit gate
```
