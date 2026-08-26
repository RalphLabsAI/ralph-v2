"""Regression tests for `eval/status.py` — the public snapshot miners read.

    python -m tests.test_status      # self-running harness, no deps
    pytest tests/test_status.py      # if pytest is installed

THE THING UNDER TEST IS AN ABSENCE. Zero rounds have completed, so every retention, crown and
weight is a number that does not exist. The failure mode of every dashboard ever written is to
render that as `0.00`, which a miner reads as "I scored zero" rather than "nothing has been
measured". So most of what follows asserts that a value is NOT there, and that the document refuses
to be published when it is.

The second theme is disclosure: the exam is a pure function of `commit_root ‖ round_nonce`, and
publishing it while a round is still fetching artifacts hands every miner the exam with minutes to
spare.
"""
from __future__ import annotations

import json
import time

from eval.status import SCHEMA, build, m, unmeasured, validate
from eval.watchdog import LogView, UnitView

NOW = 1_786_030_000.0


def _unit(running=False, pid=4242, exec_start="python -u -m eval.run_orchestrated"):
    return UnitView(load_state="loaded", active_state="activating" if running else "failed",
                    sub_state="start" if running else "failed", pid=pid if running else 0,
                    start_t=NOW - 600, invocation="abc", exec_start=exec_start,
                    pid_alive=running)


def _build(**kw):
    d = dict(now=NOW, unit=_unit(), log=LogView(mtime=NOW - 50_000, trusted=False),
             watchdog_state={"last_pass": NOW - 60, "last_state": "IDLE", "last_alarms": []},
             rentals=[], commitments=[], chain_err="", trail_index={}, live=False)
    d.update(kw)
    return build(**d)


# --- the envelope -----------------------------------------------------------------------------

def test_an_unmeasured_quantity_carries_no_value_at_all():
    """Not zero, not null, not an em-dash — ABSENT. A renderer that forgets to branch gets a
    KeyError, which someone fixes, instead of a plausible number, which nobody notices."""
    doc = _build()
    for path in (("trail", "crowns"), ("trail", "weights")):
        node = doc[path[0]][path[1]]
        assert node["state"] == "NOT_YET_MEASURED", node
        assert "value" not in node, f"{path} smuggled a value into an unmeasured field: {node}"
        assert node["because"], f"{path} has no sentence a miner can read"


def test_validate_rejects_every_shape_that_could_fabricate_a_score():
    cases = (
        ({"state": "NOT_YET_MEASURED", "because": "x", "value": 0.0}, "value on an unmeasured node"),
        ({"state": "MEASURED"}, "MEASURED with nothing in it"),
        ({"state": "MEASURED", "value": None}, "null, which renders as 0 downstream"),
        ({"state": "UNKNOWN"}, "no `because`"),
        ({"state": "INVENTED", "because": "x"}, "a state no renderer knows"),
    )
    for node, why in cases:
        try:
            validate({"x": node})
        except ValueError:
            continue
        raise AssertionError(f"validate() accepted {why}: {node}")


def test_a_measured_zero_is_not_a_placeholder():
    """`rounds_published: 0` is a real count — we looked, and it was zero. The schema has to keep
    that distinguishable from "not measured", or the honest half becomes unreadable too."""
    doc = _build()
    node = doc["trail"]["rounds_published"]
    assert node["state"] == "MEASURED" and node["value"] == 0, node


def test_m_and_unmeasured_cannot_be_misused():
    try:
        unmeasured("x", state="MEASURED")
    except ValueError:
        pass
    else:
        raise AssertionError("unmeasured() minted a MEASURED node with no value")
    assert m(0.5)["value"] == 0.5


# --- disclosure -------------------------------------------------------------------------------

def test_the_exam_is_never_published_while_a_round_is_in_flight():
    """The 72-of-900 item draw and the observer derive from `commit_root ‖ round_nonce`. Attempt 8
    had ~7 minutes between writing the nonce and fetching the first artifact; attempt 9 had eleven.
    A dashboard polling at 30 s would have handed every miner the exam inside that window."""
    doc = _build(unit=_unit(running=True),
                 log=LogView(mtime=NOW - 30, milestone="scoring", banner_pid=4242, trusted=True))
    rnd = doc["round"]
    assert rnd["in_flight"]["value"] is True
    assert "value" not in rnd["exam"], f"the exam leaked while the round was live: {rnd['exam']}"
    blob = json.dumps(doc)
    for forbidden in ("round_nonce", "commit_root", "item_indices", "observer_drawn"):
        assert forbidden not in blob, f"{forbidden} appears in a document published mid-round"


def test_an_unattributable_log_does_not_publish_a_dead_rounds_stage():
    """The watchdog's own rule. If no line can be attributed to the running process, the stage is
    unknown — showing the last matched milestone would be showing the PREVIOUS round's."""
    doc = _build(unit=_unit(running=True),
                 log=LogView(mtime=NOW - 30, milestone="scoring", banner_pid=999, trusted=False))
    assert doc["round"]["stage"]["state"] == "UNKNOWN", doc["round"]["stage"]


# --- failing closed ---------------------------------------------------------------------------

def test_a_chain_read_failure_never_becomes_an_empty_miner_list():
    """A miner missing from a short list reads it as "I was not seen" — a personalised falsehood,
    published to everyone at once. The chain section fails closed as a unit."""
    doc = _build(commitments=None, chain_err="WebsocketConnectionError: timed out")
    for key in ("miners", "cohort"):
        node = doc["chain"][key]
        assert node["state"] == "UNKNOWN", node
        assert "value" not in node
        assert "timed out" in node["because"]


def test_an_unreachable_provider_is_not_nothing_is_running():
    doc = _build(rentals=None)
    assert doc["rental"]["state"] == "UNKNOWN", doc["rental"]
    assert "NOT" in doc["rental"]["because"]


def test_an_empty_rental_list_is_still_a_measurement():
    """An empty collection is a measurement and needs the same envelope a number does."""
    doc = _build(rentals=[])
    assert doc["rental"]["state"] == "MEASURED" and doc["rental"]["value"] == []


# --- freshness --------------------------------------------------------------------------------

def test_the_writer_stamps_expired_because_only_it_knows_the_age_at_push_time():
    """A renderer learns a section's age at FETCH time; the writer knows it at PUSH time. A
    section that silently goes stale looks exactly like a section that is fine."""
    doc = _build(watchdog_state={"last_pass": NOW - 5000, "last_state": "IDLE",
                                 "last_alarms": []})
    assert doc["validator"].get("expired") is True, doc["validator"]
    fresh = _build()
    assert "expired" not in fresh["validator"]


def test_a_quiet_but_healthy_round_is_not_marked_stale():
    """A round generating 72 items writes nothing for minutes and is entirely healthy. Stamping
    `expired` on it is the false-stale twin of a false kill: it teaches a reader to ignore the one
    flag that means something. The section's freshness is the STAGE's budget."""
    log = LogView(mtime=NOW - 400, milestone="scoring", banner_pid=4242, trusted=True)
    doc = _build(unit=_unit(running=True), log=log)
    assert "expired" not in doc["round"], doc["round"]
    assert doc["round"]["max_age_s"] > 400

    # ...and it IS marked once the stage is genuinely past its budget
    stale = _build(unit=_unit(running=True),
                   log=LogView(mtime=NOW - 9000, milestone="scoring", banner_pid=4242,
                               trusted=True))
    assert stale["round"].get("expired") is True


_PROGRESS = (
    (30.0, 'pool', '900 trajectories'),
    (112.0, 'fetch', '[1/8] 5ERWJp4StMcQ… hf://Jordun01/qwen3-8b-sn40-tern-mix-tB@8428903e'),
    (140.0, 'hash', 'Jordun01/qwen3-8b-sn40-tern-mix-tB 4.1 GB'),
    (197.0, 'intake', '5ERWJp4StMcQ… tier=ternary'),
    (269.0, 'generate', 'Qwen/Qwen3-8B 8/72 ctx=3910 @256 tok'),
)


def _live(progress=_PROGRESS):
    return LogView(mtime=NOW - 20, milestone='scoring', banner_pid=4242, trusted=True,
                   progress=progress)


def test_a_live_round_shows_what_it_is_actually_doing():
    """"scoring" is one word for the entire expensive leg. A miner watching a live round needs to
    see which of eight artifacts is downloading and how far generation has got, or the page looks
    frozen for forty minutes while the GPU is busy."""
    doc = _build(unit=_unit(running=True), log=_live())
    sub = doc['round']['substage']
    assert sub['state'] == 'MEASURED', sub
    assert sub['value']['stage'] == 'generate'
    assert '8/72' in sub['value']['detail']
    # both clocks, because they answer different questions and neither may be extrapolated
    assert sub['value']['remote_elapsed_s'] == 269.0
    assert sub['value']['seen_at'] == NOW - 20


def test_a_submission_shows_how_far_the_round_has_got_with_it():
    """The scorer names a truncated hotkey in its heartbeats; matching is by prefix so a miner can
    find their own row and see their artifact was fetched, hashed, and entered intake."""
    doc = _build(unit=_unit(running=True), log=_live(),
                 commitments=[{'hotkey': '5ERWJp4StMcQQBNgxxxxxxxx', 'tier': 'ternary',
                               'artifact_uri': 'hf://Jordun01/x@1'},
                              {'hotkey': '5ZZZneverTouchedYetxxxxx', 'tier': 'sub4',
                               'artifact_uri': 'hf://other/y@1'}])
    rows = doc['chain']['miners']['value']
    assert rows[0]['handling']['state'] == 'MEASURED'
    assert rows[0]['handling']['value']['step'] == 'intake'
    # the one the round has not reached says so, rather than silently looking identical
    assert rows[1]['handling']['state'] == 'NOT_YET_MEASURED'
    assert 'not reached' in rows[1]['handling']['because']


def test_a_rejected_submission_stops_reading_as_in_progress():
    """The round announced 11 accepted artifacts and then scored 10, and for the whole expensive
    leg nothing said which miner fell out. `intake` is a step the round ENTERS, so a rejected
    submission's last word was "intake" — indistinguishable from one still being worked on, for
    the next two hours. `rejected` is the only terminal step the live page can show, and the miner
    it concerns is the one who most needs it."""
    prog = _PROGRESS + ((201.0, 'rejected', '5ERWJp4StMcQ… bit budget exceeded: 4.7 > 4.0'),)
    doc = _build(unit=_unit(running=True), log=_live(prog),
                 commitments=[{'hotkey': '5ERWJp4StMcQQBNgxxxxxxxx', 'tier': 'ternary',
                               'artifact_uri': 'hf://Jordun01/x@1'}])
    row = doc['chain']['miners']['value'][0]
    assert row['handling']['state'] == 'MEASURED', row['handling']
    assert row['handling']['value']['step'] == 'rejected', row['handling']['value']
    # and it must WIN over the earlier intake line rather than being ordered away
    assert row['handling']['value']['remote_elapsed_s'] == 201.0
    # rejection is still not an OUTCOME — only a published record can report that
    assert row['outcome']['state'] != 'MEASURED', row['outcome']


def test_being_intaken_is_never_reported_as_being_accepted():
    """`intake` is logged when intake BEGINS. The gates decide acceptance and only a published
    record can report it — the single most tempting false claim on this page, because a miner
    watching their artifact get fetched will read any positive word as 'I passed'."""
    doc = _build(unit=_unit(running=True), log=_live(),
                 commitments=[{'hotkey': '5ERWJp4StMcQQBNgxxxxxxxx', 'tier': 'ternary',
                               'artifact_uri': 'hf://Jordun01/x@1'}])
    row = doc['chain']['miners']['value'][0]
    assert row['outcome']['state'] == 'NOT_YET_MEASURED', row['outcome']
    assert 'value' not in row['outcome']
    blob = json.dumps(row).lower()
    for word in ('accepted', 'passed', 'approved', 'valid'):
        assert word not in blob, f'the row implies {word!r} from a log line that only means started'


def test_substage_is_absent_before_the_scoring_leg():
    doc = _build(unit=_unit(running=True),
                 log=LogView(mtime=NOW - 20, milestone='installing', banner_pid=4242,
                             trusted=True, progress=()))
    assert doc['round']['substage']['state'] == 'NOT_YET_MEASURED'
    assert 'value' not in doc['round']['substage']


def test_every_section_dates_itself():
    doc = _build()
    for name in ("validator", "round", "chain", "trail"):
        assert "as_of" in doc[name] and "max_age_s" in doc[name], name
    assert doc["schema"] == SCHEMA and doc["stale_after_s"] > 0


def test_live_is_read_from_the_unit_that_runs():
    """One drop-in file separates a dry round from one that sets weights on chain, so this is read
    from the effective ExecStart every pass rather than from anything in the repo."""
    from eval.status import _effective_live
    assert _effective_live(_unit(exec_start="python -m eval.run_orchestrated --live")) is True
    assert _effective_live(_unit(exec_start="python -m eval.run_orchestrated")) is False


def test_the_whole_document_serialises_without_nan_or_infinity():
    """`Infinity` is not JSON and nothing downstream can read it back."""
    json.dumps(_build(), allow_nan=False)




def test_a_live_retention_is_provisional_and_never_an_outcome():
    """The operator asked to see scores during the ~2h a round takes, rather than only at publish.
    That is a real need — the page read as "nothing is happening" while the GPU worked — but the
    number is unaudited, unsigned, unanchored, and the round can still be WITHHELD after it. So it
    gets a state of its own that CARRIES a value and is FORCED to carry the sentence saying what
    the value is not, and it lives in its own field so no renderer can promote it into `outcome`."""
    prog = _PROGRESS + ((402.0, 'retention', '5ERWJp4StMcQ… tier=ternary retention=0.9131 gates=ok'),)
    doc = _build(unit=_unit(running=True), log=_live(prog),
                 commitments=[{'hotkey': '5ERWJp4StMcQQBNgxxxxxxxx', 'tier': 'ternary',
                               'artifact_uri': 'hf://Jordun01/x@1'},
                              {'hotkey': '5ZZZneverTouchedYetxxxxx', 'tier': 'sub4',
                               'artifact_uri': 'hf://other/y@1'}])
    rows = doc['chain']['miners']['value']
    pr = rows[0]['provisional_retention']
    assert pr['state'] == 'PROVISIONAL', pr
    assert pr['value'] == 0.9131, pr
    assert pr['because'], "a PROVISIONAL number must say what it is not"
    assert 'not been audited' in pr['because'], pr['because']
    # it must NOT become an outcome, and must NOT displace the step
    assert rows[0]['outcome']['state'] != 'MEASURED', rows[0]['outcome']
    assert rows[0]['handling']['value']['step'] == 'intake', rows[0]['handling']
    # and a miner the round has not scored says so rather than showing someone else's number
    assert rows[1]['provisional_retention']['state'] == 'NOT_YET_MEASURED'
    assert 'value' not in rows[1]['provisional_retention']


def test_a_reworded_retention_line_degrades_to_no_number():
    """The grammar being parsed lives in another module and WILL be reworded. A lenient parse would
    put a wrong number next to somebody's hotkey; a miss just leaves it unmeasured."""
    from eval.status import _parse_retention
    assert _parse_retention('5ERWJp… tier=ternary retention=0.9131 gates=ok') == 0.9131
    assert _parse_retention('5ERWJp… scored 0.9131') is None
    assert _parse_retention('') is None
    assert _parse_retention('retention=nonsense') is None
    # out-of-band values mean the grammar moved under us, not that a miner scored 4000
    assert _parse_retention('retention=4000.0') is None


def _rec(**kw):
    d = dict(round=1, weights={"5AAA": 0.5, "5BBB": 0.5},
             submissions=[{"miner": "5ERWJp4StMcQQBNgxxxxxxxx", "tier": "ternary",
                           "retention": 0.3142, "retention_lb": 0.3142, "gates_ok": True,
                           "reasons": []}],
             events=[{"action": "crown", "tier": "ternary", "miner": "5ERWJp4StMcQQBNgxxxxxxxx",
                      "king": "abc123", "retention": 0.3142}])
    d.update(kw)
    return d


def test_a_published_record_becomes_the_score_a_miner_reads():
    """Round 1 anchored and the page still said `0 rounds completed`, because nothing ever read the
    record. `outcome` is the only field allowed to carry a MEASURED score, and it comes from bytes
    that were audited, signed, published and anchored."""
    doc = _build(trail_index={"rounds": [{"round": 1}]}, latest_record=_rec(),
                 commitments=[{'hotkey': '5ERWJp4StMcQQBNgxxxxxxxx', 'tier': 'ternary',
                               'artifact_uri': 'hf://x/y@1'}])
    assert doc['trail']['rounds_published']['value'] == 1
    out = doc['chain']['miners']['value'][0]['outcome']
    assert out['state'] == 'MEASURED', out
    assert out['value']['retention'] == 0.3142
    assert out['round'] == 1, "a score with no round attached cannot be checked against the trail"
    crowns = doc['trail']['crowns']
    assert crowns['state'] == 'MEASURED' and crowns['value']['ternary']['retention'] == 0.3142
    assert doc['trail']['weights']['state'] == 'MEASURED'


def test_a_held_throne_is_a_crown_not_an_empty_tier():
    """Round 2 held both tiers, so it carries `hold` events and no `crown` events, and this said
    "published but crowned no tier" — which to a miner reads as an empty throne. A crown minted in
    an earlier round is still a crown; a hold is a reign, not an absence. Same shape as the audit
    bug that rejected that round outright."""
    rec = _rec(round=2,
               events=[{"action": "hold", "tier": "ternary", "king": "KT", "margin_lcb": 0.0},
                       {"action": "none", "tier": "binary", "king": None}],
               submissions=[{"miner": "5KING", "tier": "ternary", "model_id": "KT",
                             "retention": 0.2406, "retention_lb": 0.2406, "gates_ok": True,
                             "reasons": [], "role": "incumbent"}])
    doc = _build(trail_index={"rounds": [{"round": 2}]}, latest_record=rec, commitments=[])
    cr = doc['trail']['crowns']
    assert cr['state'] == 'MEASURED', cr
    assert 'ternary' in cr['value'], cr['value']
    got = cr['value']['ternary']
    # the miner and the number come from the SUBMISSION, not from the event
    assert got['miner'] == '5KING' and got['retention'] == 0.2406, got
    assert got['held'] is True, "a defended crown must be distinguishable from a fresh one"
    # a tier nobody has ever won stays absent rather than rendering as a blank throne
    assert 'binary' not in cr['value'], cr['value']


def test_one_miner_two_entries_is_not_two_different_answers():
    """A miner who holds a crown appears TWICE: their challenger submission, and the re-scored
    incumbent. The crowns panel read the first match and the submissions table read the last, so
    the page showed 0.3030 and 0.2875 for the same miner and the same model_id at the same time.

    The split has to be a decision: the table answers "what did YOUR SUBMISSION score", the crown
    answers "what did the throne DEFEND with"."""
    rec = _rec(round=2,
               events=[{"action": "hold", "tier": "sub4", "king": "KS", "margin_lcb": -0.0637}],
               submissions=[{"miner": "5KING", "tier": "sub4", "model_id": "KS",
                             "retention": 0.3030, "retention_lb": 0.3030, "gates_ok": True,
                             "reasons": [], "role": "challenger"},
                            {"miner": "5KING", "tier": "sub4", "model_id": "KS",
                             "retention": 0.2875, "retention_lb": 0.2875, "gates_ok": True,
                             "reasons": [], "role": "incumbent"}])
    doc = _build(trail_index={"rounds": [{"round": 2}]}, latest_record=rec,
                 commitments=[{"hotkey": "5KING", "tier": "sub4", "artifact_uri": "hf://k/k@1"}])

    row = doc["chain"]["miners"]["value"][0]["outcome"]
    assert row["state"] == "MEASURED"
    assert row["value"]["retention"] == 0.3030, "the table must show the miner's own submission"

    crown = doc["trail"]["crowns"]["value"]["sub4"]
    assert crown["retention"] == 0.2875, "the crown must show what it DEFENDED with"
    assert crown["held"] is True
    # and the number that actually decided the crown has to be on the page
    assert crown["margin_lcb"] == -0.0637, crown


def test_a_rejected_miner_reads_the_reason_from_the_signed_record():
    """Round 1 dropped two miners: one never revealed, one shipped a GGUF llama.cpp cannot open.
    Both reasons existed only in a summary file on the orchestrator — a box the miner cannot see —
    so from their side they simply vanished from a round they had been accepted into."""
    rec = _rec(rejected=[["5ZZZnotInTheRecordxxxxxx",
                          ["committed but not revealed: the sealed value cannot be checked"]]])
    doc = _build(trail_index={"rounds": [{"round": 1}]}, latest_record=rec,
                 commitments=[{'hotkey': '5ZZZnotInTheRecordxxxxxx', 'tier': 'sub4',
                               'artifact_uri': 'hf://x/z@1'}])
    out = doc['chain']['miners']['value'][0]['outcome']
    # NOT_APPLICABLE, not NOT_YET_MEASURED: nothing is still coming for this miner this round
    assert out['state'] == 'NOT_APPLICABLE', out
    assert 'committed but not revealed' in out['because'], out['because']
    assert 'value' not in out, "a rejection is not a score"


def test_a_committed_miner_absent_from_the_record_is_told_so():
    """Eleven cleared intake, nine reached the record. A miner missing from a published round must
    read a reason, not an empty cell that looks identical to 'still running'."""
    doc = _build(trail_index={"rounds": [{"round": 1}]}, latest_record=_rec(),
                 commitments=[{'hotkey': '5ZZZnotInTheRecordxxxxxx', 'tier': 'sub4',
                               'artifact_uri': 'hf://x/z@1'}])
    out = doc['chain']['miners']['value'][0]['outcome']
    assert out['state'] == 'NOT_YET_MEASURED', out
    assert 'published without this submission' in out['because'], out['because']
    assert 'without a recorded reason' in out['because'], 'silence about the silence'
    assert 'value' not in out


def test_an_unreadable_trail_is_never_a_confident_zero():
    """THE BUG THIS FILE EXISTS FOR, one layer up. `RecordPublisher` refuses to construct without a
    state_path, so the trail read raised on every pass and a bare `except` turned it into an empty
    index — the page published `rounds_published: {MEASURED, 0}` for an hour after round 1 anchored.
    A failed read is not a measurement of zero."""
    doc = _build(trail_index={}, trail_err="PublishError: state_path is required")
    rp = doc['trail']['rounds_published']
    assert rp['state'] == 'UNKNOWN', rp
    assert 'value' not in rp, "a failed read published a number"
    assert 'state_path' in rp['because']
    for k in ('crowns', 'weights'):
        assert doc['trail'][k]['state'] == 'UNKNOWN', doc['trail'][k]


# --- the round timeline, and what arrived after it -----------------------------------------------

def _tl_rec(n, subs=(), events=(), **kw):
    d = dict(round=n, submissions=list(subs), events=list(events), teacher="Qwen/Qwen3-8B",
             manifest={"observer": "obs", "item_indices": list(range(72))}, weights={})
    d.update(kw)
    return d


def _tl_sub(hk, tier="sub4", role="challenger", uri="", **kw):
    d = {"miner": hk, "tier": tier, "role": role, "artifact_uri": uri or f"hf://{hk}/m@r1",
         "retention": 0.0, "retention_lb": 0.0, "code_bits": 0.0, "container_bits": 0.0,
         "gates_ok": True, "reasons": [], "model_id": hk}
    d.update(kw)
    return d


def test_a_rescored_incumbent_is_not_counted_as_another_submission():
    """A HELD CROWN APPEARS TWICE in a record — once as the artifact its miner submitted, once as
    the incumbent re-scored on this round's exam. `len(submissions)` therefore says 13 for a round
    that scored 11 miners, and a miner checking whether they made it in gets a number matching
    nothing they can see."""
    from eval.status import _round_summary
    r = _round_summary(_tl_rec(2, subs=[_tl_sub("a"), _tl_sub("b"), _tl_sub("b", role="incumbent")]))
    assert r["scored"] == 2, r
    assert r["rescored_incumbents"] == 1, r
    assert r["by_tier"] == {"sub4": 2}, r["by_tier"]


def test_a_round_with_no_recorded_clock_reports_unknown_not_zero():
    """Rounds 1-2 ran before the record carried timestamps. A 0 renders as `instant`; None renders
    as `not recorded`, which is the true statement."""
    from eval.status import _round_summary
    assert _round_summary(_tl_rec(1))["duration_s"] is None
    assert _round_summary(_tl_rec(1))["published_at"] is None
    timed = _round_summary(_tl_rec(3, started_at=1000.0, published_at=1600.0))
    assert timed["duration_s"] == 600.0, timed


def test_shakedown_and_live_are_declared_never_inferred():
    """A shakedown round computes a weight vector exactly like a live one — it simply is not
    written — so nothing IN a record can tell them apart. The boundary is an operator statement,
    and the default must classify everything as shakedown rather than imply someone was paid."""
    from eval.status import FIRST_LIVE_ROUND, _round_summary
    assert FIRST_LIVE_ROUND > 1000, "the default must not silently mark rounds as live"
    assert _round_summary(_tl_rec(4), first_live=7)["phase"] == "shakedown"
    # "paid", NOT "live": the word rendered as a LIVE badge on every FINISHED round, so the
    # dashboard showed three completed rounds all claiming to be running while the panel beside
    # them said "no round in flight". The distinction being drawn is emission, not liveness.
    assert _round_summary(_tl_rec(7), first_live=7)["phase"] == "paid"
    assert _round_summary(_tl_rec(9), first_live=7)["phase"] == "paid"


def test_a_held_crown_still_shows_the_tier_as_occupied():
    """`hold` is the action a round emits when the incumbent survives. A timeline that only counted
    `crown` would show a tier going empty on every successful defence."""
    from eval.status import _round_summary
    r = _round_summary(_tl_rec(2, subs=[_tl_sub("a")],
                            events=[{"tier": "ternary", "action": "hold", "king": "K"},
                                    {"tier": "binary", "action": "none", "king": None}]))
    assert r["crowns"] == {"ternary": "K"}, r["crowns"]


def test_pending_separates_a_new_miner_from_one_who_resubmitted():
    """Different miners, different meaning. This field iterates on recipe far more often than it
    grows, so collapsing both into "pending" hides what is actually happening."""
    from eval.status import _pending_from
    last = _tl_rec(2, subs=[_tl_sub("alice", uri="hf://alice/m@v1"),
                         _tl_sub("bob", uri="hf://bob/m@v1")])
    p = _pending_from([{"hotkey": "alice", "tier": "sub4", "artifact_uri": "hf://alice/m@v1"},
                       {"hotkey": "bob", "tier": "sub4", "artifact_uri": "hf://bob/m@V2"},
                       {"hotkey": "carol", "tier": "binary", "artifact_uri": "hf://carol/m@v1"}],
                      last)
    assert [r["hotkey"] for r in p["new_entrants"]] == ["carol"], p["new_entrants"]
    assert [r["hotkey"] for r in p["resubmitted"]] == ["bob"], p["resubmitted"]
    assert p["unchanged"] == 1 and p["awaiting_scoring"] == 2, p
    assert p["by_tier"] == {"binary": 1, "sub4": 1}, p["by_tier"]


def test_a_miner_rejected_last_round_is_not_reported_as_a_new_entrant():
    """They were seen, judged and told why. Calling them new would erase the rejection — the exact
    thing `rejected` was added to the signed record to stop.

    THIS TEST USED TO ALSO ASSERT `awaiting_scoring == 0`, which encoded the bug it now pins the
    fix for: a rejected miner IS awaiting scoring — the next round reads every commitment, and
    their artifact hash is not in the trail's submissions so skip-already-scored will not skip it.
    Filing them under `unchanged` ("unchanged since scoring" — they were never scored) is how a
    miner who fixed their commit-reveal and re-committed a real QAT artifact stayed invisible."""
    from eval.status import _pending_from
    last = _tl_rec(2, subs=[], rejected=[["dave", ["committed but never revealed"]]])
    p = _pending_from([{"hotkey": "dave", "tier": "sub4", "artifact_uri": "hf://dave/m@v1"}], last)
    assert p["new_entrants"] == [], p
    assert [r["hotkey"] for r in p["resubmitted"]] == ["dave"], p
    assert p["awaiting_scoring"] == 1, p


def test_an_unreadable_trail_yields_no_timeline_rather_than_an_empty_one():
    """An empty list of rounds says "this subnet has never run one". That is a very different
    claim from "the history could not be read", and it is the same failure that once printed a
    MEASURED zero next to `rounds published` for an hour after round 1 anchored."""
    doc = _build(trail_index={"rounds": [{"round": 1}]}, records=None, trail_err="HTTP 502")
    assert doc["trail"]["rounds"]["state"] != "MEASURED"
    assert doc["trail"]["rounds"]["state"] == "UNKNOWN"
    assert "value" not in doc["trail"]["rounds"], "an unread history must carry no value at all"


def test_the_timeline_is_capped_and_says_so():
    """A bounded list that does not declare its bound reads as the whole history."""
    from eval.status import ROUND_WINDOW
    recs = [_tl_rec(n) for n in range(1, ROUND_WINDOW + 6)]
    doc = _build(trail_index={"rounds": [{"round": r["round"]} for r in recs]}, records=recs)
    v = doc["trail"]["rounds"]["value"]
    assert len(v["items"]) == ROUND_WINDOW
    assert v["total"] == len(recs) and v["window"] == ROUND_WINDOW
    assert [i["round"] for i in v["items"]][:2] == [len(recs), len(recs) - 1], "newest first"


def test_a_round_carries_its_own_field_because_the_cohort_moves_on():
    """WHY THE FIELD LIVES WITH THE ROUND. The Submissions panel shows the CURRENT cohort, and a
    commitment slot holds only its latest value — so once a miner resubmits, nothing anywhere can
    still say who was in round 1. The signed record is the only durable answer, so the field is
    carried per round."""
    from eval.status import _round_summary
    r = _round_summary(_tl_rec(
        2,
        subs=[_tl_sub("alice", retention=0.30, code_bits=4.0, model_id="M"),
              _tl_sub("alice", role="incumbent", retention=0.2875, code_bits=0.0, model_id="M"),
              _tl_sub("bob", retention=0.21, code_bits=4.0, model_id="N")],
        events=[{"tier": "sub4", "action": "hold", "king": "M"}]))
    rows = r["submissions"]
    assert [x["miner"] for x in rows] == ["alice", "alice", "bob"], "best retention first"
    assert rows[0]["role"] == "challenger" and rows[1]["role"] == "incumbent"
    assert rows[0]["crowned"] and rows[1]["crowned"] and not rows[2]["crowned"]
    # the incumbent carries NO bit measurement — it is re-scored, not re-ingested
    assert rows[1]["code_bits"] == 0.0 and rows[0]["code_bits"] == 4.0


def test_the_field_never_carries_the_per_sample_measurement_blob():
    """The record's submissions hold `steps`, `effects`, `slices` and `per_point` — megabytes that
    exist so an auditor can recompute a score, and that have no business in a snapshot a browser
    polls every few seconds."""
    from eval.status import _round_summary
    fat = _tl_sub("alice", retention=0.3)
    fat.update(steps=["x" * 5000] * 72, effects=[[1, 2, 3, 4]] * 72,
               slices={"a": [0.1] * 72}, per_point=[1] * 72)
    row = _round_summary(_tl_rec(1, subs=[fat]))["submissions"][0]
    for heavy in ("steps", "effects", "slices", "per_point"):
        assert heavy not in row, f"{heavy} leaked into the status document"
    assert len(json.dumps(row)) < 600, len(json.dumps(row))


def test_a_gated_submission_shows_why_it_did_not_count():
    """Scored-but-gated and simply-low-scoring look identical from a number alone, and only the
    first has anything its miner can act on."""
    from eval.status import _round_summary
    bad = _tl_sub("carol", retention=0.19)
    bad.update(gates_ok=False, reasons=["degenerate output: 82% subword salad"])
    row = _round_summary(_tl_rec(3, subs=[bad]))["submissions"][0]
    assert row["gates_ok"] is False
    assert "salad" in " ".join(row["reasons"])


# COLLECTED AFTER EVERY TEST IS DEFINED. This used to sit above the last few tests, which
# meant a test appended to the end of the file was silently never run — it does not fail,
# it does not appear, and the count just does not go up. Keep this immediately above main().
TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    bad = 0
    for t in TESTS:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except Exception as e:
            bad += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - bad}/{len(TESTS)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

def test_a_rejected_miner_who_recommits_is_not_filed_as_unchanged():
    """Rejected rows carry no artifact_uri, so the classifier could not compare bytes and fell
    through to `unchanged` — false twice over (never scored, possibly new bytes). A miner who fixed
    their commit-reveal and re-committed a real QAT artifact stayed invisible exactly here."""
    from eval.status import _pending_from
    record = {"round": 2,
              "submissions": [{"miner": "hk_scored", "artifact_uri": "hf://a@1"}],
              "rejected": [["hk_rejected", ["commit-reveal: bytes are not what was committed"]]]}
    commitments = [
        {"hotkey": "hk_rejected", "tier": "binary", "artifact_uri": "hf://new-qat@2"},
        {"hotkey": "hk_scored", "tier": "sub4", "artifact_uri": "hf://a@1"},
    ]
    p = _pending_from(commitments, record)
    assert [r["hotkey"] for r in p["resubmitted"]] == ["hk_rejected"]
    assert p["unchanged"] == 1, "the genuinely unchanged scored miner stays unchanged"
    assert p["awaiting_scoring"] == 1


def test_a_skipped_unchanged_row_is_not_counted_as_a_rejection():
    """Round 3 published twelve `rejected` rows: four real refusals and seven skip-already-scored
    notes. The dashboard printed "12 rejected" in red for a round that refused four."""
    from eval.status import _round_summary
    rec = _tl_rec(3, subs=[], rejected=[
        ["hk_fmt", ["unrunnable format TQ1_0: mainline llama.cpp has no Metal kernels"]],
        ["hk_old", ["unchanged since round 1 — the same bytes were already scored there."]],
        ["hk_old2", ["unchanged since round 1 — the same bytes were already scored there."]],
    ])
    r = _round_summary(rec)
    assert r["rejected"] == 1, r
    assert r["skipped_unchanged"] == 2, r
    assert [x["hotkey"] for x in r["rejected_detail"]] == ["hk_fmt"]
    assert len(r["unchanged_detail"]) == 2


def test_an_unchanged_miner_is_not_awaiting_scoring():
    """THE TWO FIXES MEETING BADLY. skip-already-scored writes an "unchanged" row so nobody is
    silently absent; the pending classifier then read every `rejected` row as re-entering. Result:
    "12 awaiting scoring" for a field where eleven could never be scored without new bytes — and it
    would have said so again every round, forever."""
    from eval.status import _pending_from
    rec = _tl_rec(3, subs=[], rejected=[
        ["hk_old", ["unchanged since round 1 — the same bytes were already scored there."]],
        ["hk_bad", ["commit-reveal: the bytes are not what was committed"]],
    ])
    p = _pending_from([{"hotkey": "hk_old", "tier": "sub4", "artifact_uri": "hf://a@1"},
                       {"hotkey": "hk_bad", "tier": "binary", "artifact_uri": "hf://b@2"}], rec)
    assert [r["hotkey"] for r in p["resubmitted"]] == ["hk_bad"], p
    assert p["awaiting_scoring"] == 1, p
    assert p["unchanged"] == 1, p


def test_a_miner_rejected_last_round_carries_its_reason_forward():
    """Labelling them `resubmitted` implies new bytes. Four miners sat in the round-5 queue that
    way while holding the same TQ1_0 and the same commit-reveal mismatch round 4 refused — round 5
    would refuse identically. The reason is the only thing they can act on."""
    from eval.status import _pending_from
    rec = _tl_rec(4, subs=[], rejected=[
        ["hk_fmt", ["unrunnable format TQ1_0: mainline llama.cpp has no Metal kernels"]],
    ])
    p = _pending_from([{"hotkey": "hk_fmt", "tier": "ternary", "artifact_uri": "hf://a@1"}], rec)
    row = p["resubmitted"][0]
    assert row["hotkey"] == "hk_fmt"
    assert "TQ1_0" in row["rejected_last_round"][0]


def test_a_clean_re_entrant_carries_no_rejection_reason():
    """Absence has to mean something: a miner who was never refused must not be labelled as one."""
    from eval.status import _pending_from
    rec = _tl_rec(4, subs=[{"miner": "hk_ok", "artifact_uri": "hf://old@1"}], rejected=[])
    p = _pending_from([{"hotkey": "hk_ok", "tier": "sub4", "artifact_uri": "hf://new@2"}], rec)
    assert p["resubmitted"][0].get("rejected_last_round") is None


def test_a_crown_carries_the_artifact_not_only_a_score():
    """A hotkey and a retention number is a leaderboard. The size and bit budget are what the
    subnet actually produces, and the page could not show them."""
    from eval.status import _crowns_from
    rec = {"round": 4,
           "events": [{"tier": "binary", "action": "hold", "king": "m1", "margin_lcb": -0.041}],
           "submissions": [{"model_id": "m1", "miner": "hk", "role": "incumbent",
                            "retention": 0.1886, "code_bits": 1.1477, "container_bits": 1.8681,
                            "params": 8190427136,
                            "artifact_uri": "hf://tensor-tailor/ralph-qwen3-8b-binary@abc"}]}
    m = _crowns_from(rec, 4, 4, "hf://rec", True, "")
    k = m["value"]["binary"]
    assert k["code_bits"] == 1.1477 and k["container_bits"] == 1.8681
    assert k["params"] == 8190427136
    assert "tensor-tailor" in k["artifact_uri"]
