# SN97 "Distil" v1 — Post-Mortem, and what SN40 v2 inherits

Reconstructed from the public record: `unarbos/distil` (854 commits, 2026-03-29 →
2026-05-20, incl. the team's own diagnostic papers under `paper/`), `unarbos/albedo`
(the successor, whose every rule is a Distil scar), HuggingFace miner artifacts, and
public threads. This is the failure manual for a compression / distillation subnet — we
build v2 to *inherit* it, not rediscover it.

## What Distil was

A KL-divergence king-of-the-hill: teacher GLM-5 (744B) → students ≤10% size, miners
commit an HF repo + pinned SHA on-chain, the validator scores teacher-forced forward
KL on seeded prompts, winner-take-all to the lowest KL. GLM-5 lasted **less than one
day** as teacher — the validator couldn't afford to host it — collapsing to
Qwen3.5-35B-A3B, then Kimi K2.6 (served via cloud API, a fatal trust + rate-limit
dependency). **Eval economics, not miners, set the real constraints from day one.**

## How it died — four independent causes, arriving together

1. **The objective was structurally gameable.** Their own hardening paper
   (`paper/mechanism_hardening.md`), citing the reward-hacking-as-equilibrium
   literature, concludes the answer to "which KL threshold can't miners game" is
   literally *"you can't. Any fixed scalar loss is asymptotically gamed."*

2. **KL rewarded style, not capability — the mechanism-killer.** On 2026-04-17 the
   reigning king (the subnet's *best* model by KL) infinite-looped a 6-word phrase
   102× on the prompt "Hi", ran 10–80× slower than the base, and was **strictly worse
   than the untrained base model on 5/5 reasoning benchmarks** (gsm8k −4.1, ifeval
   −15.3, humaneval −12.2, bbh −16.3, arc −2.0). Their paper
   (`paper/off_policy_cot_collapse.md`): students "learn the cosmetic surface form
   (think longer → lower KL) without the underlying reasoning." **This is const's
   "miners just distill style," confirmed in the team's own forensics — and diagnosed
   by a miner (@allan_ww) before the team's telemetry caught it.**

3. **The covering dataset was never solved.** Eval corpus churned SweInfinite →
   SWE-bench → FineWeb-220 → a procedural block-seeded sampler; no corpus was ever
   simultaneously (a) broad enough to cover capability, (b) too big/procedural to
   memorize, and (c) cheap enough to score per round. Their own
   `goodhart_audit_2026-04-27.md` concedes distribution-level overfit survived even
   fresh items.

4. **Detection-based defense and ops fragility.** Every copy/Sybil detector was
   evadable, weaponizable (attackers uploaded copies of *victims'* models to get them
   DQ'd), or a self-DoS (the anti-collapse probe took production down; the VRAM check
   zeroed a round). The only anti-abuse that held was **economic** (one-eval-per-
   registration). It ended with three validator-repo wipes traced to SSH logins from a
   single IP block, the in-repo tripwire wiped along with the repo, and abandonment.

The pivot to Albedo tells you what they concluded: **zero KL code, no compression at
all, arch-locked same-size fine-tunes, LLM-as-judge.** They gave up on compression, on
distribution-match scoring, and on an unbounded eval corpus — the three things v2
solves with verifiable capability retention. Distil reached verifiable procedural axes
only in its dying weeks and **never once combined them with an actual compression
constraint. That empty combination is v2's entire lane.**

## Exploit catalog → v2 status

| Distil exploit | Their outcome | v2 |
|---|---|---|
| **Teacher-as-student** (upload the teacher behind a 624-byte decoy safetensors; MoE params nested so checker reads "3B" → KL=0) | patched after it crowned + poisoned early-stopping | **Effective-bits audit derives everything from the *loaded state dict*, never config/metadata** — recompute params, per-tensor dtype census, bits = serialized bytes after canonical recompression (kills zero-pad / sparse-stuff / fp16-outlier fake-ternary), peak-VRAM *delta* during eval. Mismatch = fail-closed DQ. Safetensors-only, file allowlist, size caps. |
| **trust_remote_code RCE** (obfuscated `tokenizer.py` monkey-patches `json.dump` to forge the score) | patched: no `.py`, `trust_remote_code=False` | **Inherited verbatim.** The only clean, permanent win they had. |
| **KL style-hacking / CoT collapse** | never solved in-frame; killed the mechanism | **v2 scores verifiable capability on fresh checkable tasks, not distribution match** — the exact fix they only reached post-mortem. + live uncompressed-GLM control row + host-side degeneracy probe (timeout-guarded so it can't self-DoS). |
| **Copy-the-king + recalibrate** (logit-temp scaling, TIES/DARE/Fisher merges, zero-pad to cap, LoRA soups) | detection **never held** | **Make copying pay zero, don't detect it:** strict-improvement dethrone past a margin above noise (a copy ties → earns nothing), margin-over-RTN-control payout, sealed commit-then-reveal, delayed public reveal, earliest-commit-wins (DQs the later committer only). |
| **Eval-distribution harvesting + validator cache leak** (miners harvested teacher continuations; the 30GB logit cache leaked to public HF) | outrun, never remediated | **commit-then-*generate* fresh items** (freshness after weights are sealed); secret set never on a miner-reachable box, deleted post-round; stale-vs-fresh diff-in-diff overfit detector (our GPT-2-control method, proven in our 2026-07-06 bust). |
| **Re-commit spam / eval-budget theft** (13 checkpoints to eat 13× budget) | fixed by **economics** (one-eval-per-registration, per-coldkey cap) | **Inherited** — their single most durable defense. + refunded resubmission fee (taxes noise, not iteration). |
| **Booby-trapped weights** (grad explosion / inflated norms, uncontinuable) | anti-finetune probe **worked** | **Inherited** — one forward+backward, DQ on grad>1000 / absurd norms. Cheap, held. |
| **Throughput forgery** (our own forged-wall-clock king #1217) | — | Measured tok/s in pinned container + max-MFU ceiling (Ralph PR#96) + timing on fresh random inputs with in-run output-correctness so a kernel can't special-case the benchmark. |
| **Tournament state-machine bugs** (cached-score promotion, uid_index collision crown-theft ×3, commit-block spoof) | patched piecemeal, some days before abandonment | Invariants, fail-closed: crown transitions only from a fresh full eval in the current signed round; identity = (hotkey, artifact content hash, harness version), never a mutable uid index; **signed** audit log (our king.json is still unsigned — same bug class); property-test the state machine pre-mainnet. |
| **LLM-judge prompt injection** (Albedo era) | permanent arms race | **Keep every scored slice verifiable (exact-match / execution).** No open-ended judge in the scoring path. |
| **Validator sabotage** (3 repo wipes, in-repo tripwire wiped with it) | **no defense held → abandonment** | Out-of-band tripwires (never in-repo), signed + externally-mirrored publications (our HF audit spine, fail-closed), IaC rebuild, CPU-auditor reproduction. We have already survived one root compromise (2026-06-27) — first-class. |

## Making miners actually work (positive incentives, not just gates)

Distil was winner-take-all on one scalar: one king to copy, maximal incentive for
crown theft, zero incentive for the incumbent to improve. v2 inverts all of it:

- **Margin over the naive-quant control** — the free baseline earns nothing; a copy
  adds no margin over the artifact it copies. This kills the copy meta *economically*
  where detection failed *technically*.
- **Tiered multi-slot crowns** — several simultaneous niches, so miners differentiate
  instead of cloning one leader; a stolen crown captures only one tier.
- **Incumbent decay + beat-yourself reset** — a king's share half-lifes unless its owner
  ships a strict improvement. KOTH becomes a treadmill, not rent.
- **First-discoverer priority share** — originality pays structurally more than
  fast-following.
- **Kernel bounties with adoption royalties** — real measured speedups are monetizable,
  and each adopted kernel lowers the subnet's own eval cost.
- **Delayed-reveal open-recipe bonus** — publish a reproducible method card after a
  protection window → persistent emission multiplier; the recipe lands in the lineage.
- **Exploit bounty lane** — at Distil the fatal bug was found by a miner, and the best
  adversaries spent their talent on theft because theft was the only paid channel. Make
  red-teaming the second-most-profitable strategy after real work.

## The three hard problems v2 must own (Distil solved none)

1. **The covering set** — const's named boss fight. v2's shape (fresh verifiable +
   soft-min domains + teacher-relative retention + control + paraphrase diff-in-diff) is
   stronger than anything Distil shipped, but distribution-overfit / capability
   laundering is a research problem, not settled. Prototype on testnet before any
   public claim; promise "verifiable retention of checkable capability," not "broad."
2. **Eval economics** — cheap *because students are small* + a tiered cheapest-first
   funnel; GLM must stay validator-hostable, never a cloud-API dependency in scoring.
3. **CC is a garnish, not the foundation** — miner-run sealed eval is a trap (attested
   memorization by the box owner; deletes one-eval-per-registration). Core eval stays
   validator-side; CC's honest role is optional throughput attestation.
