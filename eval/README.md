# Ralph v2 evaluation and audit

The production crown path evaluates lower-bit, architecture-preserving compressions of
`Qwen/Qwen3-8B`. Its metric is observer-KL retention: how closely a submitted step moves the
configured judges toward the predictive state produced by the parent step.

This is a **retention metric, not a capability benchmark**. The production `score_job` does not
attach a separate task-accuracy canary, so a crown or retention value must not be presented as a
general capability result.

## Public state

As of **2026-09-11**, seven real-model rounds are published in
[`RalphLabsAI/ralph-v2-rounds`](https://huggingface.co/datasets/RalphLabsAI/ralph-v2-rounds). They are
signed, hash-chained, and anchored on mainnet. Round 7 was the first round to score all three judges:

- `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- `microsoft/Phi-3-mini-4k-instruct`
- `allenai/OLMo-2-1124-7B-Instruct`

In that round `binary`, `sub2`, and `sub4` changed hands; `ternary` held on the floor rule. The four
resulting artifacts are mirrored in
[`RalphLabsAI/ralph-crowns`](https://huggingface.co/RalphLabsAI/ralph-crowns).

Each signed record carries the round's weight vector; validators set it after their own audit
accepts the record. A public chain snapshot after Round 7 showed six later submissions waiting for
a future score; that count is a point-in-time queue, not an intake result or schedule.

## Production round

The split production path is:

```text
CPU orchestrator: chain read -> round identity -> rent GPU -> audit returned record
GPU score job:    fetch -> intake -> parent + all judges -> score -> unsigned record
CPU orchestrator: L0/L1 -> sign -> publish -> anchor -> optional weight write -> destroy GPU
```

The scoring validator's entrypoint is `python -m eval.run_orchestrated`. `eval/score_job.py` runs
on the rented GPU and holds no signing or chain-write keys.

For every accepted artifact:

1. The sealed commitments and a later block hash establish `commit_root` and `round_nonce`.
2. The nonce selects trajectory items from the pinned pool under the allocation rule the record
   names (`selection_rule`). It does **not** select one judge.
3. The parent and every accepted submission generate a step on the same selected items.
4. All three judges measure the parent and submission effects over fixed continuations.
5. Each judge is reduced to its worst `(language, depth)` slice; the displayed retention is the mean
   of the three judge-level worst slices.
6. The incumbent is re-scored on the same exam and the tournament applies the point-margin,
   persistence, floor, and reign-decay rules.
7. The round record freezes the exam, model outputs, effects, decisions, and candidate vector before
   it is signed and published.

Round 7 requested 288 trajectory items. One item was dropped because the parent effect was below the
minimum signal threshold; the reason is recorded rather than silently changing the denominator.

## Intake and admission

No miner model loads before the cheap gates complete:

1. registration and one-artifact-per-`(coldkey, tier)` admission;
2. file safety;
3. tier fit;
4. measured code and container bit budgets;
5. pinned-parent architecture identity; and
6. commit-reveal binding to the served bytes.

The active tier code-bit caps are `binary=1.15`, `ternary=1.75`, `sub2=2.3`, and `sub4=4.0`.
Container caps bind independently. GGUF and safetensors are accepted; `TQ1_0` and `TQ2_0` GGUFs are
refused because they lack the required mainline Metal kernels.

The old resubmission-bond bookkeeping is disabled (`RegistrationLedger.base_bond=0`). No chain
extrinsic escrows or refunds that value. The enforceable anti-grind controls are the per-coldkey,
per-tier round cap, registered identities, and skipping bytes already scored in the public trail.

## Crown decision

The current rules operate on the displayed point estimate:

- lead `>= 0.02` on one fresh round, or lead `>= 0.01` on two consecutive rounds by the same
  artifact;
- both thresholds halve after an incumbent has held three rounds;
- no challenger slice may fall below the incumbent's worst slice under the same judge; and
- an exact copy has lead zero and cannot dethrone.

A positive paired lower bound can allocate the 20% challenger share in the weight vector without
moving the crown.

## Audit levels

```bash
python -m eval.rerun record.json                                      # L0
python -m eval.rerun record.json --pool pool.jsonl                    # L0 + L1
python -m eval.rerun record.json --pool pool.jsonl --observer <hf-id> # + one judge's L2
python -m eval.rerun record.json --pool pool.jsonl --observer <hf-id> \
    --artifacts ./ckpts                                                # + L3 model binding
python -m eval.rerun --history ./published --head <on-chain-anchor>
```

- **L0** verifies signatures and recomputes arithmetic, crown decisions, and the candidate vector
  from the published measurements. It needs no models.
- **L1** re-derives item selection under the rule the record names and checks the pool digest and
  recorded prefixes. It is a CPU data-integrity check.
- **L2** re-runs one recorded judge over the frozen text. A multi-judge round needs one pass per
  judge for complete coverage, and numeric comparison requires the recorded GPU, batch size,
  attention implementation, and stack.
- **L3** reloads crowned/submitted artifacts and regenerates their frozen steps. It binds the record
  to the actual model bytes and is not a cheap CPU-only check.

Exit code `0` means the requested checks reproduced, `1` means a demonstrated divergence, and `2`
means incomplete. Running only L0/L1 does not verify the model inference that produced the scores.

`eval.auditor` follows the trail and publishes signed verdicts at configured levels. Its
`--signer` is the record-signing identity; `--validator-hotkey` is the separate ss58 identity whose
on-chain commitment slot carries the anchor. L2 configuration should name recorded judges, and
full multi-judge coverage requires separate passes for all three.

## Useful local commands

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python -m eval.precheck
python -m eval.bitrate path/to/model.gguf
```

The test suite exercises local logic and plumbing; it does not reproduce a public GPU round. Most
tests are CPU-only, while the code-execution sandbox requires `bubblewrap` to be permitted.
Production scoring and L2/L3 model checks require the appropriate model runtime and recorded
environment. `eval.simulate_submission` belongs to the older simulated path and does not currently
represent the multi-judge production flow.

## Main modules

| module | role |
|---|---|
| `run_orchestrated.py` | key-holding CPU orchestration, publish/anchor gate, optional weights |
| `score_job.py` | keyless rented-GPU scoring job |
| `validator_observer_loop.py` | production observer-KL intake and round assembly |
| `observer_kl.py` | downstream-effect measurement |
| `bitrate.py` / `gates.py` | bit budgets, format compatibility, and intake |
| `koth.py` / `lineage.py` | crown decisions and history-derived throne state |
| `round_record.py` / `publish.py` | signed records, hash chain, and public trail |
| `rerun.py` / `auditor.py` | layered re-derivation and independent verdicts |
| `pool.py` / `steps.py` | pinned trajectory pool and balanced exam construction |

Files such as `adversarial.py`, `axes/*`, `gpu_run.py`, and the older simulated round paths are
research history and test scaffolding. They are useful for mechanism experiments but do not
describe the current production Round 7 path. Likewise, `eval/budget.py` is an offline tool and is
not wired into crown decisions.
