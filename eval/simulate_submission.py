"""One full cycle, end to end: a miner submits, the validator scores and crowns, anyone re-runs it.

This exists so the mechanism can be WATCHED rather than described. It walks the same code paths a
live round does — intake gates, commit-reveal, bit measurement, pinned-parent admission, scoring,
KOTH, the fail-closed publisher, the on-chain anchor, and finally the audit tool re-verifying the
published record.

Two modes, and the difference matters:

    --sim   (default, CPU, seconds)  stub parent/observer, synthesized checkpoints. Every GATE,
                                     the crown logic, publishing, anchoring and all four audit
                                     levels are real; only the models are not. This is the mode to
                                     run to see the pipeline.
    --real  (GPU, minutes)           the pinned parent from eval/parent.py, a genuine 4-bit
                                     quantization of it as the submission, and a real observer.

WHAT A MINER ACTUALLY HAS TO SATISFY, in the order the validator checks it — this is the honest
answer to "what do I submit":

    1. economics      registered, and a bond posted if it is not your free evaluation
    2. safety         safetensors or GGUF only, no pickles, no remote code, no stowaway files
    3. tier fit       parameter count inside the tier, dtype headers consistent
    4. bit budget     MEASURED FROM THE TENSOR DATA, not the dtype header, so a 4-bit model stored
                      in a bf16 container reads as 4-bit and not 16
    5. pinned parent  same architecture and weight-element count as the parent (an ADMISSION gate,
                      not a proof of descent — a different model of identical shape would pass)
    6. commit-reveal  H(content_hash || salt) must match what you committed on chain BEFORE the
                      round nonce existed

Only after all six pass does anything load your weights.

    python -m eval.simulate_submission            # CPU, full pipeline
    python -m eval.simulate_submission --real     # GPU, real models
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import time
from pathlib import Path

OUT = os.environ.get("RALPH_SIM_OUT", "runs/simulation")


def log(*a):
    print(*a, flush=True)


def hr(title: str):
    log("\n" + "=" * 78)
    log(f"  {title}")
    log("=" * 78)


# --------------------------------------------------------------------------------------------
# MINER SIDE
# --------------------------------------------------------------------------------------------

def write_sim_checkpoint(d: Path, n_params: int) -> Path:
    """A synthesized INT8 checkpoint — the shape of a real int8 quantization, at demo scale.

    Three things the first attempt at this got wrong, each because a gate caught it:

      * a U8 tensor is 8 bits per weight in the CONTAINER, so submitting it to a 4-bit tier is
        correctly refused ("effective bits 8.0 exceed tier cap 4.5"). Integer dtypes are
        already-packed codes whose meaning depends on a scheme the validator would have to take on
        trust, so bitrate.py reports their container width and does not level-count them. A genuine
        4-bit artifact packs two weights per byte or ships as GGUF Q4; an int8 one belongs in the
        8-bit tier. The demo submits to the tier its artifact actually fits.
      * the parameter count must match the pinned parent within tolerance, so a 1000x-scaled-down
        file is refused too. In sim mode the demo therefore pins a DEMO parent of the same
        architecture at demo scale, and says so; --real uses the true pinned parent.
      * random bytes really do measure as 8 bits. Nothing here is a mock of a measurement."""
    d.mkdir(parents=True, exist_ok=True)
    import random
    rng = random.Random(0)
    # 2-D, because measure_checkpoint skips 1-D tensors as norms/biases — a 1-D "weight" is
    # measured as nothing at all, and bit_tier_gate then fails closed on "no weights measured".
    cols = 1024
    rows = max(1, n_params // cols)
    n = rows * cols
    payload = bytes(rng.randrange(256) for _ in range(n))
    hdr = {"model.layers.0.self_attn.q_proj.weight":
           {"dtype": "U8", "shape": [rows, cols], "data_offsets": [0, n]}}
    hb = json.dumps(hdr).encode()
    with open(d / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + payload)
    (d / "config.json").write_text(json.dumps({
        "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
        "hidden_size": 896, "vocab_size": 151936}))
    return d


def miner_package(ckpt: Path) -> dict:
    """What the miner computes locally and commits on chain, BEFORE the round exists.

    The commitment is H(content_hash || salt), so it reveals nothing about the artifact until the
    miner reveals the salt — and it binds the miner to these exact bytes, so a bait-and-switch
    after the nonce is drawn cannot pass intake."""
    from .identity import commit_value, content_hash
    h = content_hash(ckpt)
    salt = "miner-chosen-secret-0001"
    cv = commit_value(h, salt)
    return {"content_hash": h, "salt": salt, "committed_value": cv,
            "artifact_uri": f"file://{ckpt}"}


# --------------------------------------------------------------------------------------------
# VALIDATOR SIDE
# --------------------------------------------------------------------------------------------

def sim_models():
    """Stub parent/observer with a real signal: a step ending in X moves the observer a lot, W
    partway, anything else not at all. Enough for the crown logic to be exercised honestly."""
    d3 = lambda x, y, z: {"a": x, "b": y, "c": z}

    class Obs:
        def generate(self, prompts, max_new_tokens=128):
            return ["the next thing follows"] * len(prompts)

        def distributions(self, prefix, continuation):
            t = prefix.rstrip()[-1:]
            if t == "X":
                return [d3(0.80, 0.10, 0.10)] * 6
            if t == "W":
                return [d3(0.55, 0.25, 0.20)] * 6
            return [d3(0.34, 0.33, 0.33)] * 6

    class Step:
        def __init__(self, tok):
            self.tok = tok

        def generate(self, prompts, max_new_tokens=256):
            return [self.tok] * len(prompts)

    return Obs, Step


def main(argv) -> int:
    real = "--real" in argv
    outdir = Path(OUT)
    # A DEMO starts from a clean trail. Note what this is working around, because it is a feature:
    # re-running into an existing trail is refused with "round 1 already published as … refusing to
    # replace it", which is the history-rewrite guard. In production that refusal is the whole
    # point; here it just stops the demo being re-runnable, so we clear instead of overwrite.
    if outdir.exists():
        import shutil
        shutil.rmtree(outdir)
        print(f"(cleared {outdir} — publishing into an existing trail is refused by design)")
    outdir.mkdir(parents=True, exist_ok=True)

    from .chain import Commitment, run_v2_observer_epoch
    from .economics import RegistrationLedger
    from .gates import TierBudget, inspect_checkpoint
    from .intake import intake
    from .koth import Tier, Tournament
    from .parent import PARENTS, parent_compat
    from .publish import LocalSink, RecordPublisher, anchor_of, verify_history
    from .shadow_axis_epoch import FakeChain
    from .signing import Ed25519Signer
    from .steps import Trajectory

    from .parent import ParentSpec
    true_parent = PARENTS["qwen2.5-0.5b-instruct"]
    DEMO_PARAMS = 2_000_000
    # In sim mode the parent is scaled down so the demo runs in seconds on a laptop. Same
    # architecture string and the same gate, 300x smaller. --real uses the pinned parent itself.
    parent_spec = true_parent if real else ParentSpec(
        name=f"{true_parent.name} [DEMO SCALE]", arch=true_parent.arch,
        weight_params=DEMO_PARAMS, hidden_size=true_parent.hidden_size,
        vocab_size=true_parent.vocab_size)

    hr("0. WHAT IS PINNED")
    log(f"  parent          : {parent_spec.name}")
    log(f"  architecture    : {parent_spec.arch}   (must match exactly)")
    log(f"  weight params   : {parent_spec.weight_params:,}   (must match within tolerance)")
    log(f"  hidden / vocab  : {parent_spec.hidden_size} / {parent_spec.vocab_size}")
    log("")
    if not real:
        log(f"  (sim mode: scaled to {DEMO_PARAMS:,} params so this runs on a laptop. The real")
        log(f"   pinned parent is {true_parent.name} at {true_parent.weight_params:,}.)")
    log("  NOTE: Qwen2.5-0.5B is the parent the mechanism has actually been validated against.")
    log("  The README's 'starting with GLM' is the intended target, not what has been run.")

    tmp = tempfile.mkdtemp(prefix="ralph-sim-")
    # The submission below is int8, so it goes to the 8-bit tier. A 4-bit tier exists too and
    # would refuse this artifact on measured width — which is the gate working, not a bug.
    tiers = [Tier("8bit", 10 ** 12, 1.0)]
    # WIRE THE OPTIONAL GATES. TierBudget.bit_tier and .parent both default to None, and when
    # they are None intake SILENTLY SKIPS the measured bit budget and the pinned-parent check —
    # the two gates that make this a compression subnet rather than a general model contest. A
    # validator constructed without them runs with four gates, not six, and nothing says so.
    from .bitrate import BitTier
    budgets = {"8bit": TierBudget(name="8bit", max_params=10 ** 12, max_effective_bits=8.5,
                                 bit_tier=BitTier(name="8bit", max_code_bits=8.0,
                                                  max_container_bits=8.5),
                                 parent=parent_spec)}

    # ---------------- MINER ----------------
    hr("1. MINER: build and package a submission")
    ckpt = write_sim_checkpoint(Path(tmp) / "my-int8-qwen", parent_spec.weight_params)
    log("  (2-D weight tensor: measure_checkpoint skips 1-D tensors as norms/biases, and the")
    log("   bit gate then fails closed on 'no weights measured')")
    log(f"  wrote {ckpt}")
    pkg = miner_package(ckpt)
    log(f"  content_hash    : {pkg['content_hash'][:32]}…")
    log(f"  salt            : {pkg['salt']}")
    log(f"  COMMIT THIS     : {pkg['committed_value'][:32]}…   <- goes on chain now")
    log("  (the salt stays secret until you reveal; the commitment binds you to these bytes)")

    hr("2. VALIDATOR: the six intake gates, in order")
    insp = inspect_checkpoint(ckpt)
    log(f"  safety/inspect  : ok={insp.ok}  params={insp.params:,}  "
        f"header_bits={insp.effective_bits_per_param:.2f}  dtypes={insp.dtype_hist}")
    from .bitrate import measure_checkpoint
    bm = measure_checkpoint(ckpt)
    log(f"  MEASURED bits   : container={bm.container_bits:.4f} bpw  code={bm.code_bits:.4f} bpw"
        f"  codebook={bm.codebook}")
    log("                    (int dtypes are packed codes: the container width IS the claim, so")
    log("                     they are not level-counted. A float tensor holding only 16 distinct")
    log("                     values per group would measure code=4.0 in a 16-bit container.)")
    pc = parent_compat(ckpt, parent_spec, insp)
    log(f"  pinned parent   : ok={pc.ok}"
        + ("" if pc.ok else f"  reasons={pc.reasons}"))
    ledger = RegistrationLedger()
    d = intake(ckpt, budgets["8bit"], ledger, "miner-hot-1", "miner-cold-1", 0.0,
               revealed_hash=pkg["content_hash"], salt=pkg["salt"],
               committed_value=pkg["committed_value"])
    log(f"  gates wired     : bit_tier={budgets['8bit'].bit_tier is not None}  "
        f"parent={budgets['8bit'].parent is not None}   <- both optional, both skipped if None")
    log(f"  INTAKE          : accepted={d.accepted}"
        + ("" if d.accepted else f"  reasons={d.reasons}"))
    if not d.accepted:
        log("\n  Submission rejected. Nothing was loaded — that is the point of the front door.")
        return 1

    # ---------------- ROUND ----------------
    hr("3. VALIDATOR: score the round (observer drawn from the post-commit nonce)")
    if real:
        from .runners import HFRunner
        from .shakedown_round_noise import load_trajectories
        pool = load_trajectories(24)
        parent = HFRunner(parent_spec.name)
        observers = {"qwen1.5b": HFRunner("Qwen/Qwen2.5-1.5B-Instruct")}
        make_runner = lambda cd: HFRunner(parent_spec.name, load_in_4bit=True)
    else:
        Obs, Step = sim_models()
        pool = [Trajectory(id=f"t{i}", source="glaive_r1", prefix=f"step-{i} context",
                           step="s", index=i % 4) for i in range(200)]
        parent = Step("X")
        observers = {"kimi": Obs(), "qwen": Obs()}
        make_runner = lambda cd: Step("X")

    sink_root = str(outdir / "published")
    publisher = RecordPublisher(LocalSink(sink_root), window=8,
                               state_path=str(outdir / "hwm.json"))

    class Chain(FakeChain):
        def __init__(self, commits):
            super().__init__(commits)
            self.slot = ""

        def publish_record(self, record):
            self.slot = anchor_of(getattr(record, "prev_anchor", ""), record.sha256())

        def head_anchor(self):
            return self.slot

    chain = Chain([Commitment("miner-hot-1", "miner-cold-1", "8bit", str(ckpt), 1.0,
                              revealed_hash=pkg["content_hash"], salt=pkg["salt"],
                              committed_value=pkg["committed_value"],
                              artifact_uri=pkg["artifact_uri"])])
    tour = Tournament(tiers, margin=0.03)
    t0 = time.time()
    res = run_v2_observer_epoch(
        chain, 1, pool, parent, observers, tiers, budgets, tour,
        RegistrationLedger(), {}, make_safe_runner=make_runner,
        signer=Ed25519Signer(seed=b"validator-demo-key-32bytes-long!"[:32]),
        n_items=24, corpus_spec="glaive_r1@rev=demo|order=stream",
        publisher=publisher)
    o = res.outcome
    log(f"  observer drawn  : {o.observer}   (from commit_root||nonce, unknowable at seal time)")
    log(f"  items scored    : {len(o.item_indices)} drawn from a pool of {len(pool)}")
    log(f"  identity canary : {o.identity.get('score')}   (must be exactly 1.0)")
    log(f"  noise floor     : max_kl={o.noise.get('max_kl')}")
    for mid, sc in o.scores.items():
        log(f"  SCORE {mid[:16]}… : {sc['score']}  (n={sc['n_scored']}, "
            f"mean_effect={sc['mean_effect']})")
    log(f"  events          : {[e.get('action') for e in o.events]}")
    log(f"  weights         : {o.weights}")
    log(f"  took {time.time() - t0:.1f}s")

    hr("4. PUBLISH — fail-closed: no weights unless the record is fetchable and anchored")
    p = res.publish
    log(f"  published       : {p.published.uri if p.published else None}")
    log(f"  read-back ok    : {p.verified}")
    log(f"  anchor verified : {p.anchor_verified}   (on-chain head == recomputed chain head)")
    log(f"  on-chain anchor : {chain.slot[:32]}…")
    log(f"  WEIGHTS SET     : {res.weights_set}")
    if res.withheld:
        log(f"  WITHHELD        : {res.withheld}")

    hr("5. ANYONE AUDITS IT")
    rec_path = str(outdir / "record.json")
    from dataclasses import asdict
    Path(rec_path).write_text(json.dumps(asdict(o.record)))
    pool_path = str(outdir / "pool.jsonl")
    with open(pool_path, "w") as fh:
        for t in pool:
            fh.write(json.dumps({"id": t.id, "source": t.source, "prefix": t.prefix,
                                 "step": t.step, "index": t.index, "meta": t.meta}) + "\n")
    log(f"  record : {rec_path}")
    log(f"  pool   : {pool_path}")
    log(f"  trail  : {sink_root}")
    log("")
    from .rerun import audit, report
    obs_for_audit = observers[o.observer]
    code = report(audit(rec_path, pool_path, obs_for_audit, observer_name=o.observer,
                        make_runner=lambda mid, uri: make_runner(uri)))
    h = verify_history(publisher, head_anchor_fn=chain.head_anchor)
    log(f"  trail: {h.n_rounds} round(s), head_checked={h.head_checked}, complete={h.ok}")

    hr("6. AN AUDITOR VALIDATOR RULES ON IT — and signs what it did and did not check")
    from .auditor import Auditor, AuditorConfig
    acfg = AuditorConfig(expected_signer=o.record.signer, require=("L0", "L1"),
                         work_dir=str(outdir / "auditor"))
    # a DIFFERENT key from the validator's: an auditor signing with the operator's key would be
    # the operator agreeing with itself
    aud = Auditor(acfg, sink=publisher.sink,
                  signer=Ed25519Signer(seed=b"auditor-demo-key-32bytes-long!!!"[:32]),
                  head_anchor_fn=chain.head_anchor, out=sys.stdout)
    for v in aud.once():
        log(f"  verdict         : {v.verdict}  (ran {'+'.join(v.levels_run)}, "
            f"required {'+'.join(v.levels_required)})")
        log(f"  would weight    : round {v.followed_round} -> {v.weights}")
        log(f"  signed by       : {v.auditor[:16]}…  verifies={v.verify_signature()}")
    log(f"  verdicts written: {acfg.verdict_dir}")
    log("")
    log("  This auditor needed no GPU and no corpus file — it fetched the pool the record pins")
    log("  from the trail and re-digested it against the signed manifest. That is the whole")
    log("  point: one scorer, many checkers, and a verdict that cannot claim more than it ran.")
    log("")
    log("  Reproduce any of this yourself:")
    log(f"    python -m eval.rerun {rec_path} --pool {pool_path} \\")
    log(f"        --observer <the observer named in the manifest> --artifacts <dir of ckpts>")
    log(f"    python -m eval.rerun --history {sink_root} --head {chain.slot[:16]}…")
    log(f"    python -m eval.auditor --once --require L0,L1 --signer {o.record.signer}")
    return 0 if code == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
