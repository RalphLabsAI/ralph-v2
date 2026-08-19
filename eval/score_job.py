"""What runs ON THE RENTED GPU. Scores one round and writes an UNSIGNED record.

THIS BOX HOLDS NO KEYS, and that is the whole reason the validator is split in two. The June 2026
compromise of a persistent GPU box gave an attacker root for seven hours and with it the signing
keys — they rigged a crowning and wiped the logs. A rented box is worse, not better, on that axis:
it is somebody else's hardware, briefly. So the wallet, the record seed and the write token all stay
on the CPU orchestrator, and what comes back from here is an unsigned record plus the pool it was
scored against. The orchestrator re-derives it before signing.

WHAT THIS MEANS FOR TRUST. The orchestrator does NOT take these numbers on faith just because it
rented the box. It re-runs L0 and L1 against the returned record before its key ever touches it —
the same audit an outsider runs, pointed at our own scorer. So the failure this design has to
survive is "the rented box returns a plausible lie", and the answer is that a lie has to survive
the operator's own audit before it can be signed, and L3 afterwards.

    python -m eval.score_job job.json out/          # reads a job spec, writes record.json + pool.jsonl
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict


def score(job: dict, out_dir: str) -> dict:
    """Run the round described by `job`. Returns a small summary; writes the artifacts."""
    from .bitrate import TIERS, emission_weight
    from .economics import RegistrationLedger
    from .fetch import resolver
    from .gates import TierBudget
    from .koth import Tier, Tournament
    from .parent import PARENTS
    from .pool import build_pool, check_balance, dump_pool
    from .progress import tick
    from .runners import HFRunner, student_runner
    from .validator_observer_loop import CommittedSubmission, run_observer_round

    os.makedirs(out_dir, exist_ok=True)

    # FAIL IN SECONDS, NOT IN HOURS. Everything below assumes a GPU, and nothing below CHECKS —
    # `device_map="auto"` silently places the parent on CPU when torch cannot see the device, and
    # the round then does the whole job correctly and uselessly: 900 trajectories built, 30 GB of
    # artifacts fetched, an 8B model generating on 12 cores at roughly a tenth the speed, and a
    # result the orchestrator rejects at the end. Observed on a rented H100 whose driver (CUDA 12.8)
    # was older than the torch build pip chose (cu130).
    #
    # This is also the only place that can tell the difference cheaply. Downstream, CPU inference
    # looks BETTER than GPU: it is perfectly deterministic, so the identity canary passes.
    _assert_gpu()
    _assert_student_gpu()

    spec = PARENTS[job["parent_key"]]

    # EVERY STAGE BOUNDARY IS ANNOUNCED. The orchestrator kills this process after twenty minutes
    # of silence, and it is right to: the alternative is the 226-minute hang. But that makes each
    # of these lines part of the contract rather than decoration — a stage added below without a
    # tick is a stage that reads as a hang the first time it runs slowly.
    tick("pool", f"{job['pool_size']} trajectories", force=True)
    pool, pspec = build_pool(int(job["pool_size"]))
    ok, why = check_balance(pool)
    if not ok:
        raise SystemExit(f"pool refused: {why}")

    tick("parent", spec.name, force=True)
    parent = HFRunner(spec.name)
    observers = {o: HFRunner(o) for o in job["observers"]}

    by_name = {t.name: t for t in TIERS}
    budgets = {n: TierBudget(name=n, max_params=int(spec.weight_params * (1 + spec.tol)),
                             max_effective_bits=by_name[n].max_container_bits,
                             bit_tier=by_name[n], parent=spec)
               for n in job["tiers"]}
    # Difficulty-weighted, not an equal split — see bitrate.TIER_EMISSION_WEIGHT for why an equal
    # split predicts exactly the field the first two rounds drew.
    tiers = [Tier(n, max_params=budgets[n].max_params,
                  weight=emission_weight(n, len(job["tiers"])))
             for n in job["tiers"]]

    # Artifacts land HERE, on the rented disk, not on the orchestrator: a miner controls the size
    # and the CPU box also holds the signing key.
    fetch_log: list = []
    reveals = {c["hotkey"]: {"content_hash": c.get("revealed_hash", ""), "salt": c.get("salt", "")}
               for c in job["committed"]}
    # THE INCUMBENT'S BYTES ARE PINNED TOO, and until this line they were not. A challenger's
    # artifact is bound to its revealed content hash, but the king was refetched with
    # `expect_hash=""` — no check at all — from a URI whose repo the king controls. The crown
    # holder could therefore force-push a model fitted to this round's items over the same ref and
    # have it scored AS the incumbent, while the record still carried the ORIGINAL model_id. That
    # is not one bug: it defeats the dethrone margin (the paired comparison is against bytes nobody
    # published), the anti-copy guarantee, and the L2/L3 re-run (model_id no longer identifies the
    # weights). `Reign.model_id` IS the recorded content hash, so pinning to it makes a swap
    # self-refusing — a mutable `@main` ref becomes harmless once the bytes are bound.
    for _tier, _k in (job.get("kings") or {}).items():
        reveals[f"king:{_tier}"] = {"content_hash": str(_k.get("model_id", "")), "salt": ""}
    fetch_dir_for = resolver(os.path.join(out_dir, "artifacts"), reveals=reveals, log=fetch_log)

    committed, skipped = [], []
    for n, c in enumerate(job["committed"], 1):
        # THE LONGEST SILENT STRETCH IN THE ROUND used to be here: a miner may ship up to the 60 GB
        # ceiling and the fetch reports per file, so the hotkey is named before the bytes move.
        tick("fetch", f"[{n}/{len(job['committed'])}] {c['hotkey'][:12]}… "
                      f"{c.get('artifact_uri', '')[:48]}", force=True)
        d = fetch_dir_for(c["hotkey"], c.get("artifact_uri", ""))
        if not d:
            # SAY WHAT ACTUALLY HAPPENED. "could not be fetched" reads as a network blip, and the
            # resolver refuses for reasons that are nothing of the kind — the one that has actually
            # fired is commit-reveal: `content hash … does not match the revealed … — the bytes are
            # not what was committed`. That is a miner serving different bytes than they sealed,
            # the most security-relevant call this system makes, and it was being filed under a
            # transport error. The resolver already recorded the real reason in `fetch_log`; this
            # just stops throwing it away.
            why = next((str(e[-1]) for e in reversed(fetch_log)
                        if e and str(e[0]) == c["hotkey"]), "")
            skipped.append((c["hotkey"], why or "artifact could not be fetched"))
            continue
        committed.append(CommittedSubmission(
            hotkey=c["hotkey"], coldkey=c.get("coldkey", ""), tier=c["tier"], ckpt_dir=d,
            declared_compute_h100h=float(c.get("declared_compute_h100h", 0.0)),
            bond_posted=float(c.get("bond_posted", 0.0)),
            # dispatch by FORMAT: GGUF is the only artifact that can pass the bit tiers,
            # and hardcoding the safetensors loader here made every one of them unscoreable
            make_runner=(lambda cd=d: student_runner(cd)),
            revealed_hash=c.get("revealed_hash", ""), salt=c.get("salt", ""),
            committed_value=c.get("committed_value", ""), artifact_uri=c.get("artifact_uri", "")))

    # THE THRONE IS INHERITED, NOT RE-OPENED. A fresh Tournament takes koth's open-throne branch
    # every round and crowns max(retention) outright, which makes the dethrone margin, the paired
    # bootstrap and the anti-copy guarantee unreachable — a leaderboard that resets hourly wearing
    # the name of a king of the hill. The lineage is derived from the published trail by the
    # orchestrator (eval/lineage.py) and supplied here, and the king's artifact is refetched so the
    # incumbent is RE-SCORED on this round's items rather than defended on last round's number.
    from .lineage import Reign
    tournament = Tournament(tiers, margin=float(job.get("margin", 0.05)))
    # STAMP THE ROUND, or every event this round emits says it happened in round 0. `Tournament`
    # initialises `self.round = 0` and each event copies it, so the number is not decorative: it is
    # what a `crown` event means by "since", and what an auditor reads to say when a tier changed
    # hands. `round_engine`, `env_round` and `axis_round` all set it; THIS path — the split
    # validator, the only one that publishes — did not, so rounds 1 and 2 both shipped events
    # stamped round 0. Same shape as the scorer-parity bug: the money path missed what the older
    # entrypoints did correctly.
    tournament.round = int(job.get("round", 0) or 0)
    registry = {}
    for tier, k in (job.get("kings") or {}).items():
        r = Reign(**k)
        tournament.kings[tier] = r.as_king()
        tick("fetch king", f"{tier} {r.artifact_uri[:48]}", force=True)
        kd = fetch_dir_for(f"king:{tier}", r.artifact_uri)
        if kd:
            try:
                registry[r.model_id] = student_runner(kd)
            except Exception as e:
                # koth emits a "king unavailable" hold for a missing registry entry, which keeps the
                # throne without re-scoring it. That is the right conservative default and it is
                # recorded, not silent.
                skipped.append((f"king:{tier}", f"incumbent not loadable: {type(e).__name__}: {e}"))
        else:
            skipped.append((f"king:{tier}", f"incumbent artifact not fetchable: {r.artifact_uri}"))

    # THE FIXED POINTS. Fetched exactly like a king's artifact, and like a king's artifact a
    # failure here is a NOTE rather than a round failure — a reference is our yardstick, not a
    # miner's livelihood, and losing it must not cost anyone their round.
    references = []
    for r in (job.get("references") or []):
        name, tier, uri = r.get("name", ""), r.get("tier", ""), r.get("artifact_uri", "")
        if not (name and tier and uri):
            continue
        tick("fetch reference", f"{name} {uri[:48]}", force=True)
        rd = fetch_dir_for(f"reference:{name}", uri)
        if not rd:
            skipped.append((f"reference:{name}", f"not fetchable: {uri}"))
            continue
        try:
            references.append((name, tier, student_runner(rd), uri))
        except Exception as e:
            skipped.append((f"reference:{name}", f"not loadable: {type(e).__name__}: {e}"))

    tick("round", f"{len(committed)} accepted, {len(skipped)} skipped", force=True)
    out = run_observer_round(
        int(job["round"]), job["commit_root"], job["round_nonce"], committed, pool, parent,
        observers, tiers, budgets, tournament,
        RegistrationLedger(), registry, parent_id=spec.name,
        prev_anchor=job.get("prev_anchor", ""),
        signer=None,                       # UNSIGNED, deliberately: the key is not on this box
        n_items=int(job["n_items"]), corpus_spec=pspec.as_corpus_spec(),
        references=references)

    with open(os.path.join(out_dir, "record.json"), "w") as fh:
        json.dump(asdict(out.record) if out.record else None, fh)
    with open(os.path.join(out_dir, "pool.jsonl"), "wb") as fh:
        fh.write(dump_pool(pool))
    summary = {"round": int(job["round"]), "accepted": list(out.accepted),
               "kings_before": {t: k.model_id for t, k in (out.kings_before or {}).items()},
               "rejected": [[h, r] for h, r in out.rejected], "skipped": skipped,
               "identity": out.identity, "observer": out.observer,
               "events": out.events, "weights": out.weights,
               "fetch": [list(map(str, e)) for e in fetch_log],
               "gpu": _gpu_name()}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def _assert_gpu() -> None:
    """Refuse to score without a usable GPU, and say exactly why torch cannot see it.

    The diagnosis matters as much as the refusal: `is_available()` returning False has one common
    cause on a rented box — a torch built for a newer CUDA than the host driver supports — and the
    message that torch itself emits is a warning nobody reads on a machine nobody is watching."""
    try:
        import torch
    except Exception as e:
        raise SystemExit(f"refusing to score: torch will not import ({type(e).__name__}: {e})")
    if torch.cuda.is_available():
        from .progress import tick
        tick("gpu", f"{torch.cuda.get_device_name(0)} (torch {torch.__version__})", force=True)
        return
    detail = ""
    try:
        import subprocess
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        detail = f" nvidia-smi reports: {smi or '(nothing)'}."
    except Exception:
        detail = " nvidia-smi could not be run at all."
    raise SystemExit(
        f"refusing to score: torch {getattr(torch, '__version__', '?')} cannot see a GPU "
        f"(torch.cuda.is_available() is False; built for CUDA {torch.version.cuda}).{detail} "
        f"The usual cause is a torch built for a newer CUDA than this host's driver supports — "
        f"scoring would silently fall back to CPU, which is ~10x slower AND perfectly "
        f"deterministic, so the identity canary would pass and the round would look fine.")


def _assert_student_gpu() -> None:
    """On a GPU box, students must run ON the GPU. Refuse early rather than pay for both.

    `_assert_gpu` closed this hole for torch — the parent and observer — and left it open for the
    students, which are the expensive half. llama.cpp's pip wheel is built CPU-only and then
    ACCEPTS AND SILENTLY IGNORES `n_gpu_layers`, so an image without `nvcc` yields a round that
    rents an H100, runs every submission on CPU at roughly 6x the time, and says so in one log line
    nobody is watching. Observed on massedcompute/desmoines, 2026-08-07.

    Within one round it is not a correctness failure — CPU llama.cpp is deterministic and the
    incumbent is re-scored on the same box, so the paired comparison still holds. ACROSS rounds it
    is: llama.cpp on CPU and on CUDA do not emit identical tokens, so a crown measured one way is
    not comparable with one defended the other, which is the whole reason the GPU is pinned. And it
    is always a MONEY failure. The cheapest place to catch all three is before the first token."""
    import os

    try:
        import torch
        on_gpu = torch.cuda.is_available()
    except Exception:
        on_gpu = False
    if not on_gpu or os.environ.get("RALPH_ALLOW_CPU_STUDENTS") == "1":
        return

    from .gpu_check import probe
    from .progress import tick

    ok, why = probe()
    if ok:
        tick("student backend", "cuda", force=True)
        return
    # WHY the reason is quoted rather than summarised: there are now two distinct causes and they
    # need different fixes. A CPU-only wheel means the IMAGE is wrong (no `nvcc`, so the source
    # build fell back). A CUDA build that will not start means the DRIVER is too old for the
    # runtime it was compiled against — a wrong REGION, on an image that looks perfect.
    raise SystemExit(
        f"refusing to score: this box has a GPU but llama.cpp cannot use it — {why}. Every "
        f"submission would run on CPU while the GPU bills, at roughly 6x the time, and "
        f"`n_gpu_layers` would not tell you: the CPU path accepts it and ignores it. Rent an image "
        f"with the CUDA toolkit AND a driver new enough for it, or set RALPH_ALLOW_CPU_STUDENTS=1 "
        f"to accept the time and cost deliberately.")


def _gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except Exception:
        return ""


def main(argv: list) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    job = json.loads(open(argv[0]).read())
    s = score(job, argv[1])
    print(json.dumps({k: s[k] for k in ("round", "accepted", "identity", "observer", "gpu")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
