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