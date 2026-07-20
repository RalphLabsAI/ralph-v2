# Prior art: lessons from earlier distillation subnets

Compression-via-distillation as a competitive mechanism has been attempted before on
Bittensor, and those attempts left a detailed public record — repositories, commit
histories, and the teams' own published research notes. v2 is designed to **inherit**
those hard-won lessons rather than re-walk them. Everything below is sourced from public
repositories and public papers; we credit the earlier builders for doing this work in
the open, which is exactly what made it learnable.

## The core lesson: score capability, never imitation

The central finding, which the earlier teams documented in their own public post-mortems,
is that a **distribution-match objective** (KL / teacher-forcing) rewards the teacher's
*surface style*, not its capability. A model can minimize KL to the teacher — win the
leaderboard — while losing the underlying reasoning: in the documented case, the
KL-best model degenerated (looping filler text, far slower) and scored *worse than the
untrained base model on every reasoning benchmark measured*. One team's own hardening
note concluded that any fixed scalar loss is asymptotically gameable.

**v2's response:** never score imitation. Score **downstream capability on
freshly-minted, machine-checkable tasks** — math with checkable answers, code by
execution, templated reasoning — measured relative to the pinned teacher. Style cannot
pass a unit test.

## The unsolved problem: the covering set

None of the prior attempts found an eval corpus that was simultaneously (a) broad
enough to cover real capability, (b) impossible to memorize or distribution-overfit,
and (c) cheap enough to score every round. Corpora were churned repeatedly; the teams'
own audits conceded distribution-level overfit survived even freshly-sampled items.

**v2's response** improves on this (commit-then-generate freshness, teacher-relative
retention, a naive-quant control, a paraphrase-invariant slice, stale-vs-fresh
diff-in-diff) but does **not** claim to have solved it. This is the open research
problem, and v2 promises only "verifiable retention of *checkable* capability," not
"broad capability." See the open-problems section of the [spec](../README.md).

## Exploit → defense map

Every attack below was demonstrated against a live distillation subnet and is visible
in the public record. This is the miner threat model any such subnet faces; v2's
defenses are listed alongside.

| Attack | How it worked | v2 defense |
|---|---|---|
| **Teacher-as-student** | Upload the full teacher behind a tiny decoy safetensors; nest MoE params so the size checker reads a small "active" count → perfect imitation score, crowned | Effective-bits audit derives params/bits/dtype from the **loaded state dict**, never metadata; bits = serialized bytes after canonical recompression (kills zero-pad / sparse-stuff / fp16-outlier fake-ternary); peak-VRAM *delta* during eval. Mismatch = fail-closed DQ. Safetensors-only, file allowlist, size caps. |
| **Remote-code score forgery** | Obfuscated `tokenizer.py` monkey-patched the JSON writer to emit a fake score | No miner `*.py`, `trust_remote_code=False`. The one clean, permanent defense the prior teams had — inherited verbatim. |
| **Style-hacking / degeneration** | Reproduce the teacher's surface form to win KL while the weights are actually damaged | Score verifiable capability on fresh checkable tasks, not distribution match; live uncompressed-teacher control row; timeout-guarded degeneracy probe (loop rate / length blowup). |
| **Copy-the-king + recalibrate** | Logit-temp scaling, model merges, zero-pad to the size cap, LoRA soups — micro-optimize a *copy* of the leader. Detection provably could not separate copies from legit fine-tunes of a shared base. | Make copying pay **zero**: strict-improvement dethrone past a margin above noise (a copy ties → earns nothing), margin-over-naive-control payout, sealed commit-then-reveal, delayed public reveal, earliest-commit-wins (DQs the *later* committer only). No detector to evade or weaponize. |
| **Eval harvesting / cache leak** | Miners harvested teacher continuations; a validator logit cache leaked publicly, letting miners overfit the eval distribution | Commit-then-*generate* fresh items (freshness after weights are sealed); the secret set never lives on a miner-reachable box and is deleted post-round; stale-vs-fresh diff-in-diff overfit detector. |
| **Re-commit spam / budget theft** | Push many checkpoints to consume many times the eval budget | Economic, not heuristic: one-eval-per-registration, per-coldkey round cap, resubmission fee refunded when the submission improves the miner's own best. The single most durable defense the prior teams found. |
| **Booby-trapped weights** | Models scoring well but with exploded grad norms / absurd weights — impossible to continue-train | Anti-finetune probe (one forward+backward, DQ on grad explosion / absurd norms). Cheap, held. |
| **Throughput forgery** | Claim a speed the hardware didn't produce | Measure decode tok/s in a pinned container with a plausibility ceiling; time on fresh random inputs with in-run output-correctness so a kernel can't special-case the benchmark. |
| **Tournament state bugs** | Stale-cache crown promotion, uid-index collision crown theft, commit-block spoofing | Fail-closed invariants: crown transitions only from a fresh full eval in the current signed round; identity = (hotkey, artifact content hash, harness version), never a mutable index; signed audit log; property-test the state machine before mainnet. |
| **LLM-judge prompt injection** | Inject grading instructions into model output when a judge scores open-ended text | Keep every scored slice verifiable (exact-match / execution). No open-ended judge in the scoring path. |

## Making miners do real work (positive incentives, not just gates)

The prior attempts were winner-take-all on one scalar: one leader to copy, maximal
incentive for crown theft, no incentive for the incumbent to improve. v2 inverts that:

- **Margin over a naive-quant control** — the free baseline earns nothing; a copy adds
  no margin over what it copied. Copying dies economically where detection failed.
- **Tiered multi-slot crowns** — several simultaneous niches, so miners differentiate
  instead of cloning one leader; a stolen crown captures only one tier.
- **Incumbent decay + beat-yourself reset** — a crown's share decays unless its owner
  ships a strict improvement. A treadmill, not rent.
- **First-discoverer priority share** — originality pays more than fast-following.
- **Kernel bounties with adoption royalties** — real measured speedups are monetizable,
  and each adopted kernel lowers the subnet's own eval cost.
- **Delayed-reveal open-recipe bonus** — publish a reproducible method card after a
  protection window for a persistent emission multiplier; the recipe lands in the
  public lineage.
- **Exploit bounty lane** — pay for demonstrated mechanism exploits under responsible
  disclosure, so red-teaming is a *paid* channel rather than theft.

## Operational hardening

Prior attempts also taught that validator operational security is first-class, not an
afterthought: out-of-band tripwires (never stored in the repo they protect), signed and
externally-mirrored score publications, infrastructure-as-code rebuilds, and CPU-auditor
reproduction so a single compromised box cannot silently rig results. Ralph already runs
a signed, mirrored audit-report spine and has hardened its own operations after a prior
incident; v2 keeps that posture.
