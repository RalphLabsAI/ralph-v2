"""The public snapshot: what a miner is allowed to know, and nothing that is not true.

WHY A PUSHED FILE AND NOT AN API. This box holds the wallet, the record seed and the HF write
token. Serving traffic from it puts a listener on the one machine the whole split-validator design
exists to keep clean, so the status document is BUILT here and PUSHED to a public repo. Nothing
inbound, ever. `ss -ltnp` on the orchestrator should show sshd and nothing else, and that is a
property of this design rather than an accident of the current firewall.

THE MEASUREMENT ENVELOPE IS THE WHOLE POINT. Zero rounds have completed. Every retention, every
crown, every weight is a number that does not exist yet, and the failure mode of every dashboard
ever written is to render a missing number as `0.00` — which to a miner reads as "I scored zero",
not "nothing has been measured". So a scored quantity is never a bare value:

    {"state": "MEASURED", "value": 0.9123, "round": 7, "record_uri": "hf://…"}
    {"state": "NOT_YET_MEASURED", "because": "no round has completed; the trail index is empty"}

and the `value` key is PHYSICALLY ABSENT unless it was measured. A renderer that forgets to branch
gets a KeyError, which someone fixes, instead of a zero, which nobody notices. `null` is banned for
the same reason — too many templating stacks render it as 0.

WHAT IS WITHHELD WHILE A ROUND IS IN FLIGHT, and this one is not cosmetic. The exam — which 72 of
900 items are scored, and which observer judges them — is a pure function of
`commit_root ‖ round_nonce`. Publishing the nonce while the round is still fetching artifacts would
hand every miner the exam with minutes to spare: attempt 8 had ~7 minutes between writing the nonce
and fetching the first artifact, attempt 9 had eleven. So the exam is a POST-HOC section. It is
released when the attempt terminates, at which point the signed record carries it anyway and an
auditor needs it to re-run the round.

ATTEMPT IS NOT ROUND. Eight attempts have exited; zero rounds have happened. Three of those exits
were `exit 0` on the `no v2 submissions this window` path, which `run_orchestrated` itself calls
"not a round". Counting exits as rounds would put phantom rounds on the board, so `attempts` and
`rounds` are separate fields and the trail — not this process — is what makes a round real.
"""
from __future__ import annotations

import json
import os
import sys
import time

SCHEMA = "ralph-v2-status/1"

# Sections age at very different rates: a chain read is minutes old by construction, an in-flight
# stage is seconds. One document-level freshness number would have to be the loosest of them, so
# every section carries its own and the writer marks the ones it is knowingly publishing stale.
STALE_AFTER_S = 900.0

# PROVISIONAL is the only state that carries a number nobody has verified. It exists because the
# alternative was worse: for the ~2h a round takes, every score read NOT_YET_MEASURED and the page
# looked like nothing was happening while the GPU worked. It is deliberately NOT "MEASURED" — the
# value has not been audited, signed or anchored, and the round can still be WITHHELD after it is
# published. Being in _HAS_VALUE it carries the number; being outside "MEASURED" it is forced by
# `validate` to carry the sentence explaining what the number is not.
STATES = ("MEASURED", "NOT_YET_MEASURED", "UNKNOWN", "NOT_APPLICABLE", "WITHHELD", "STALE",
          "PROVISIONAL")
_HAS_VALUE = ("MEASURED", "WITHHELD", "STALE", "PROVISIONAL")


def m(value, **kw) -> dict:
    """A measurement that was actually taken. `kw` carries its provenance (round, record_uri)."""
    d = {"state": "MEASURED", "value": value}
    d.update({k: v for k, v in kw.items() if v is not None})
    return d


def provisional(value, because: str, **kw) -> dict:
    """A number the scorer produced but nothing has verified. See STATES.

    Never use this for anything read from a published record — that is `m()`. This is only for
    values observed in flight, which a later audit can still overturn."""
    d = {"state": "PROVISIONAL", "value": value, "because": because}
    d.update({k: v for k, v in kw.items() if v is not None})
    return d


def unmeasured(because: str, state: str = "NOT_YET_MEASURED", **kw) -> dict:
    """Everything else. `because` is miner-facing English, not an error code — it is the sentence
    a renderer prints where the number would have been."""
    if state in _HAS_VALUE:
        raise ValueError(f"{state} carries a value; use m() and pass it")
    d = {"state": state, "because": because}
    d.update({k: v for k, v in kw.items() if v is not None})
    return d


def validate(doc) -> None:
    """Walk the document and enforce the envelope. Publishing an unvalidated snapshot is the exact
    failure this file exists to prevent, so this runs before every push and raises rather than
    warns."""
    def walk(node, path):
        if isinstance(node, dict):
            if "state" in node and isinstance(node.get("state"), str):
                st = node["state"]
                if st not in STATES:
                    raise ValueError(f"{path}: unknown state {st!r}")
                has = "value" in node
                if has != (st in _HAS_VALUE):
                    raise ValueError(
                        f"{path}: state={st} but value is {'present' if has else 'absent'} — "
                        f"a renderer would print a placeholder as a measurement")
                if st != "MEASURED" and not node.get("because"):
                    raise ValueError(f"{path}: state={st} needs a `because` a miner can read")
                if has and node["value"] is None:
                    raise ValueError(f"{path}: value is null; null renders as 0 downstream")
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
    walk(doc, "$")


def _parse_retention(detail: str):
    """`… retention=0.9131 gates=ok` -> 0.9131, or None.

    Returns None rather than guessing on anything it does not recognise: this feeds a number onto a
    public page, and the grammar it parses lives in another module and WILL be reworded. A miss
    leaves the score NOT_YET_MEASURED, which is the honest degradation; a lenient parse would put a
    wrong number next to somebody's hotkey."""
    import re
    mo = re.search(r"retention=(-?\d+(?:\.\d+)?)", detail or "")
    if not mo:
        return None
    try:
        v = float(mo.group(1))
    except Exception:
        return None
    # retention is a ratio; anything outside a sane band means the grammar moved under us
    return v if -10.0 <= v <= 10.0 else None


def _trail_unknown(trail_err: str, what: str) -> dict:
    return unmeasured(f"the published trail could not be read ({trail_err}), so {what} cannot be "
                      f"reported — this is a failed read, not an absence of results",
                      state="UNKNOWN")


def _outcome_for(hk, scored_by_hk, rejected_by_hk, rounds_published, rec_round, rec_uri,
                 trail_ok, trail_err):
    """What the SIGNED record says about this submission, or an honest reason there is nothing.

    The four cases are genuinely different and a miner reading their own row can act on which one
    they are in: the trail is unreadable, no round has run, a round ran and did not include them,
    or a round scored them. Collapsing the first into the second is what made the dashboard say
    `0 rounds completed` for an hour after round 1 anchored."""
    if not trail_ok:
        return _trail_unknown(trail_err, "this submission's result")
    if rounds_published == 0:
        return unmeasured("no round has completed, so this submission has not been scored yet")
    sub = scored_by_hk.get(hk)
    if sub is None:
        why_not = rejected_by_hk.get(hk)
        if why_not:
            # NOT_APPLICABLE, not NOT_YET_MEASURED: there is nothing still to come for this miner
            # in this round. The reason is quoted from the signed record verbatim.
            return unmeasured("round %s did not score this submission: %s"
                              % (rec_round, "; ".join(str(r) for r in why_not)),
                              state="NOT_APPLICABLE", round=rec_round,
                              record_uri=rec_uri or None)
        return unmeasured(f"round {rec_round} published without this submission, and without a "
                          f"recorded reason — ask the validator operator")
    return m({"retention": sub.get("retention"),
              "retention_lb": sub.get("retention_lb"),
              "tier": sub.get("tier"),
              "gates_ok": bool(sub.get("gates_ok")),
              "reasons": list(sub.get("reasons") or [])},
             round=rec_round, record_uri=rec_uri or None)


def _crowns_from(rec, rounds_published, rec_round, rec_uri, trail_ok, trail_err):
    if not trail_ok:
        return _trail_unknown(trail_err, "the crowns")
    if rounds_published == 0 or not rec:
        return unmeasured("no round has completed, so no tier has a crown")
    crowns = {}
    for e in (rec.get("events") or []):
        if str(e.get("action")) == "crown" and e.get("tier"):
            crowns[str(e["tier"])] = {"miner": e.get("miner"), "model_id": e.get("king"),
                                      "retention": e.get("retention")}
    if not crowns:
        return unmeasured(f"round {rec_round} published but crowned no tier")
    return m(crowns, round=rec_round, record_uri=rec_uri or None)


def _weights_from(rec, rounds_published, rec_round, rec_uri, trail_ok, trail_err):
    """The weight vector the record DECIDED. Deliberately not "what is on chain": the extrinsic can
    be refused by the rate limit — as it was on round 1 — and reporting the decision as if it were
    the chain state would be the page's one unforgivable lie."""
    if not trail_ok:
        return _trail_unknown(trail_err, "the weights")
    if rounds_published == 0 or not rec:
        return unmeasured("weights are set only after a round publishes and its anchor verifies")
    ws = dict(rec.get("weights") or {})
    if not ws:
        return unmeasured(f"round {rec_round} published without a weight vector")
    return m(ws, round=rec_round, record_uri=rec_uri or None)


def _section(as_of: float, max_age_s: float, now: float, **body) -> dict:
    """Every section dates itself and declares how long it may be believed. The writer stamps
    `expired` because it knows the age at PUSH time, while a renderer only learns it at fetch time
    — and a section that silently goes stale looks exactly like a section that is fine."""
    d = {"as_of": as_of, "max_age_s": max_age_s}
    if as_of and now - as_of > max_age_s:
        d["expired"] = True
    d.update(body)
    return d


# ---------------------------------------------------------------------------------------------

def build(now: float, unit, log, watchdog_state: dict, rentals, commitments, chain_err: str,
          trail_index: dict, live: bool, latest_record: dict | None = None,
          trail_err: str = "") -> dict:
    """Pure over frozen observations, exactly as `watchdog.classify` is, and for the same reason:
    a snapshot builder that reads the world as it goes cannot be tested against the states that
    matter (no round, dead chain, empty trail) without arranging the world first."""
    from .watchdog import budget_for

    rounds_published = len((trail_index or {}).get("rounds") or [])
    trail_ok = not trail_err

    # WHAT THE SIGNED RECORD SAYS, keyed by hotkey. This is the only source on the page allowed to
    # produce a MEASURED score: it was audited, signed, published and anchored, and the verifier
    # re-checked its digest, round number and signature before it got here.
    scored_by_hk: dict = {}
    rejected_by_hk: dict = {}
    rec_round, rec_uri = None, ""
    if latest_record:
        rec_round = latest_record.get("round")
        rec_uri = str(latest_record.get("uri") or "")
        for sub in (latest_record.get("submissions") or []):
            hk = str(sub.get("miner", ""))
            if hk:
                scored_by_hk[hk] = sub
        # WHY A MINER IS NOT IN `submissions`, straight from the signed record. Round 1 dropped two
        # and the reasons lived only on the orchestrator, so from the miner's side they vanished
        # from a round they had been accepted into.
        for entry in (latest_record.get("rejected") or []):
            try:
                hk, reasons = entry[0], list(entry[1])
            except Exception:
                continue
            if hk:
                rejected_by_hk[str(hk)] = reasons

    # --- validator liveness -------------------------------------------------------------------
    last_pass = float(watchdog_state.get("last_pass") or 0)
    val = _section(
        last_pass, 900.0, now,
        watchdog=(m(str(watchdog_state.get("last_state")), )
                  if watchdog_state.get("last_state")
                  else unmeasured("the watchdog has never completed a pass on this box",
                                  state="UNKNOWN")),
        alarms=m(list(watchdog_state.get("last_alarms") or [])),
        # `live` is read from the EFFECTIVE unit config every pass, never hardcoded: a drop-in file
        # is all that separates a dry round from one that sets weights on chain.
        writes_to_chain=m(bool(live)),
    )

    # --- the round in flight ------------------------------------------------------------------
    in_flight = bool(getattr(unit, "running", False))
    if not in_flight:
        rnd = _section(now, 120.0, now,
                       in_flight=m(False),
                       stage=unmeasured("no round is in flight", state="NOT_APPLICABLE"))
    elif not getattr(log, "trusted", False) or not getattr(log, "milestone", ""):
        # The watchdog's own rule: if no line can be attributed to the running process, the stage
        # is unknown. Showing the last matched milestone here would be showing a dead round's.
        rnd = _section(now, 120.0, now,
                       in_flight=m(True),
                       stage=unmeasured("a round is running but no log line can be attributed to "
                                        "it yet", state="UNKNOWN"))
    else:
        budget, _why = budget_for(log.milestone, 2400.0)
        elapsed = now - max(log.mtime, getattr(unit, "start_t", 0) or 0)
        # THE SECTION'S FRESHNESS IS THE STAGE'S BUDGET, not a flat number. A round generating 72
        # items writes nothing for minutes at a time and is entirely healthy — a fixed 120 s here
        # stamped `expired` on a live round, which is the same false-stale mistake as a false kill,
        # and it teaches a reader to ignore the one flag that matters.
        # WHAT THE ROUND IS ACTUALLY DOING. The milestone says "scoring" for the entire expensive
        # leg; these are the scorer's own heartbeats from the rented box — which of eight artifacts
        # is downloading, which miner is in intake, how far through generation. Without it a miner
        # watching a live round sees one unchanging word for forty minutes.
        #
        # `remote_elapsed_s` is the REMOTE's clock and `seen_at` is ours. Both are published because
        # they answer different questions, and neither may be interpolated: a renderer that animates
        # a counter is inventing precision the writer never claimed.
        last = log.progress[-1] if log.progress else None
        sub = (m({"stage": last[1], "detail": last[2][:120],
                  "remote_elapsed_s": last[0], "seen_at": log.mtime})
               if last else
               unmeasured("this attempt has not reached the streamed scoring leg yet"))
        rnd = _section(log.mtime, budget, now,
                       in_flight=m(True),
                       stage=m(log.milestone),
                       substage=sub,
                       # Published as an observation with its own timestamp, never as something a
                       # renderer may animate: it is a duration measured at `as_of`, not a clock.
                       stage_elapsed_s=m(round(elapsed, 1)),
                       stage_budget_s=m(budget),
                       # WITHHELD, not omitted: a miner should be able to see that the exam exists
                       # and know exactly when it is released, rather than wonder what is hidden.
                       exam=unmeasured("the round's nonce selects which 72 of 900 items are scored "
                                       "and which observer judges them, so it is not disclosed "
                                       "until scoring has ended; the signed record carries it",
                                       state="WITHHELD" if False else "NOT_YET_MEASURED"))

    # --- rentals: an empty list is a measurement and needs the envelope too -------------------
    if rentals is None:
        rental = unmeasured("the provider could not be reached on the last pass — this is NOT "
                            "'nothing is running'", state="UNKNOWN")
    else:
        rental = m([{"name": r.get("name"), "status": r.get("status"),
                     "created_at": r.get("created_at")}
                    for r in rentals if str(r.get("name", "")).startswith("ralph-round-")])

    # --- miners ------------------------------------------------------------------------------
    # FAIL CLOSED AS A UNIT. A chain read that half-succeeds would publish a short miner list, and
    # a miner missing from that list reads it as "I was not seen" — a personalised falsehood.
    if chain_err or commitments is None:
        miners = unmeasured(f"the chain could not be read this pass ({chain_err or 'no data'}); "
                            f"an incomplete list would tell some miners they were not seen",
                            state="UNKNOWN")
        cohort = miners
    else:
        # WHAT THE ROUND HAS DONE TO EACH SUBMISSION, so far, from the scorer's own heartbeats.
        # The progress lines carry a TRUNCATED hotkey ("5ERWJp4StMcQ…"), so matching is by prefix.
        handled: dict = {}
        live_scores: dict = {}
        for elapsed, stage, detail in log.progress:
            # `rejected` is the only TERMINAL step here, and it is the one a miner most needs to
            # see. Without it the dashboard's last word on a rejected submission is `intake`, which
            # reads as "still being worked on" for the rest of the round — the round announced 11
            # accepted artifacts and then scored 10, and nothing said which miner fell out or why.
            if stage not in ("fetch", "hash", "intake", "submission", "rejected", "retention"):
                continue
            for c in commitments:
                hk = str(c.get("hotkey", ""))
                if hk and hk[:12] in detail:
                    if stage == "retention":
                        # carried separately: it is a NUMBER, not a step, and it must not displace
                        # `submission` as the answer to "what is the round doing with these bytes"
                        got = _parse_retention(detail)
                        if got is not None:
                            live_scores[hk] = (got, elapsed)
                        continue
                    handled[hk] = (stage, elapsed)

        rows = []
        for c in commitments:
            hk = str(c.get("hotkey", ""))
            step = handled.get(hk)
            rows.append({
                "hotkey": hk,
                "tier": c.get("tier"),
                "artifact_uri": c.get("artifact_uri"),
                "seen": m(True),
                # OBSERVED, NOT CONCLUDED. `intake` is logged when intake BEGINS, so this says what
                # the round was last seen doing with these bytes — never that they passed. The gates
                # decide acceptance, and only a published record can report that.
                "handling": (m({"step": step[0], "remote_elapsed_s": step[1]})
                             if step else
                             unmeasured("this round has not reached this submission yet"
                                        if in_flight else
                                        "no round is currently handling this submission")),
                # Intake runs on the rented GPU, so acceptance is only knowable once a round has
                # scored. Saying "accepted" here would be predicting the gates, not reporting them.
                "outcome": _outcome_for(hk, scored_by_hk, rejected_by_hk, rounds_published,
                                        rec_round, rec_uri, trail_ok, trail_err),
                # THE IN-FLIGHT NUMBER, never merged into `outcome`. `outcome` means "a published,
                # signed, anchored record says this"; this means "the scorer produced this and
                # nothing has checked it". Keeping them as separate fields is what stops a renderer
                # from quietly promoting one into the other.
                "provisional_retention": (
                    provisional(live_scores[hk][0],
                                "the scorer produced this mid-round; it has not been audited, "
                                "signed or anchored, and the round can still be withheld",
                                remote_elapsed_s=live_scores[hk][1])
                    if hk in live_scores else
                    unmeasured("this round has not finished scoring this submission yet"
                               if in_flight else
                               "no round is currently scoring this submission")),
            })
        miners = m(rows)
        by_tier: dict = {}
        for c in commitments:
            by_tier[c.get("tier")] = by_tier.get(c.get("tier"), 0) + 1
        cohort = m({"commitments": len(commitments), "by_tier": by_tier})

    chain_sec = _section(now, 600.0, now, miners=miners, cohort=cohort)

    # --- the trail: where every score would live ----------------------------------------------
    trail = _section(
        now, 900.0, now,
        rounds_published=(m(rounds_published) if trail_ok else
                          unmeasured(f"the published trail could not be read ({trail_err}); a "
                                     f"failed read is not a count of zero", state="UNKNOWN")),
        # ATTEMPTS ARE NOT ROUNDS and the two are never added together.
        note=("no round has completed yet; every scored quantity below is NOT_YET_MEASURED by "
              "fact, not by omission" if trail_ok and rounds_published == 0 else ""),
        crowns=_crowns_from(latest_record, rounds_published, rec_round, rec_uri, trail_ok,
                            trail_err),
        weights=_weights_from(latest_record, rounds_published, rec_round, rec_uri, trail_ok,
                              trail_err),
    )

    doc = {
        "schema": SCHEMA,
        "generated_at": now,
        "stale_after_s": STALE_AFTER_S,
        "netuid": 40,
        "validator": val,
        "round": rnd,
        "rental": rental,
        "chain": chain_sec,
        "trail": trail,
    }
    validate(doc)
    return doc


# ---------------------------------------------------------------------------------------------

def observe(cfg, now: float):
    """The impure half: read the world once, hand `build` frozen facts."""
    from .watchdog import WatchdogConfig, read_log, read_rentals, read_unit

    wc = WatchdogConfig.from_env()
    unit = read_unit(wc.unit)
    log = read_log(wc.log_path, unit)
    try:
        with open(os.path.join(wc.work_dir, "watchdog-state.json")) as fh:
            wstate = json.load(fh)
    except Exception:
        wstate = {}
    try:
        from .orchestrator import ShadeformProvider
        rentals = read_rentals(ShadeformProvider())
    except Exception:
        rentals = None
    return unit, log, wstate, rentals


def _effective_live(unit) -> bool:
    """Read from the unit that RUNS, not from a file in the repo. `--live` currently comes from a
    base unit and is cleared by a drop-in; one deleted file flips it."""
    return "--live" in (getattr(unit, "exec_start", "") or "")


def main(argv: list) -> int:
    from .run_orchestrated import Config

    cfg = Config.from_env()
    now = time.time()
    unit, log, wstate, rentals = observe(cfg, now)

    commitments, chain_err = None, ""
    if "--no-chain" not in argv:
        try:
            from .chain_bittensor import BittensorChainIO
            ch = BittensorChainIO(netuid=cfg.netuid, network=cfg.network, wallet_name=cfg.wallet,
                                  hotkey_name=cfg.hotkey, read_only=True)
            blk = ch.current_block()
            commitments = [{"hotkey": c.hotkey, "tier": c.tier, "artifact_uri": c.artifact_uri}
                           for c in ch.read_commitments(blk - cfg.commit_window, blk,
                                                        require_local=False)]
        except Exception as e:
            chain_err = f"{type(e).__name__}: {e}"

    # THE TRAIL READ USED TO FAIL SILENTLY INTO "NO ROUNDS". `RecordPublisher` refuses to
    # construct without `state_path` — deliberately, it is the never-shrink guard — so this call
    # raised on every single pass and the bare `except` turned it into an empty index. Round 1
    # published and anchored, and the dashboard still said `0 rounds completed` with a MEASURED
    # zero next to it. A failed read is not a measurement of zero; it is now reported as UNKNOWN.
    trail_index: dict = {}
    trail_err = ""
    latest_record = None
    try:
        from .publish import HFSink, RecordPublisher
        hwm = os.path.join(cfg.work_dir,
                           f"publish-hwm-{cfg.records_repo.replace('/', '_')}.json")
        trail_index = RecordPublisher(HFSink(cfg.records_repo), state_path=hwm).load_index()
    except Exception as e:
        trail_err = f"{type(e).__name__}: {e}"

    # and the newest record itself, fetched through the verifier rather than trusted: the scores a
    # miner reads here must be the ones that were signed and anchored, not whatever bytes the repo
    # happens to serve.
    if not trail_err and (trail_index.get("rounds") or []):
        try:
            from .show_round import load as _load_round
            _n, latest_record, _idx = _load_round(None, cfg.records_repo)
        except Exception as e:
            trail_err = f"record unreadable: {type(e).__name__}: {e}"

    doc = build(now, unit, log, wstate, rentals, commitments, chain_err, trail_index,
                _effective_live(unit), latest_record=latest_record, trail_err=trail_err)
    blob = json.dumps(doc, indent=1, sort_keys=True, allow_nan=False)

    out_path = os.path.join(cfg.work_dir, "status.json")
    os.makedirs(cfg.work_dir, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(blob)
    sys.stdout.write(f"  wrote {out_path} ({len(blob)} bytes)\n")

    if "--publish" in argv:
        # A SEPARATE REPO from the round trail. Every put is a commit, and at this cadence the
        # trail's history — the audit surface — would be buried under a hundred thousand status
        # commits a year. The status repo is explicitly not part of the audit surface.
        repo = os.environ.get("RALPH_STATUS_REPO", "RalphLabsAI/ralph-v2-status")
        from .publish import HFSink
        HFSink(repo).put("status.json", blob.encode())
        sys.stdout.write(f"  published -> {repo}/status.json\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
