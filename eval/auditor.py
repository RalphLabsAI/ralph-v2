"""The auditor role — follow the published trail, re-run every round, publish a signed verdict.

WHAT THIS IS FOR. Ralph v2 needs one GPU scorer, not many: the expensive work is scoring, and the
mechanism is designed so CHECKING that work is cheap. But "one validator publishes, everyone else
copies its weights" is weight-copying with extra steps — it manufactures the APPEARANCE of
independent agreement while adding no safety at all, because a copier sets identical weights
whether the round reproduces or not. This is the other version: verify first, and diverge when
verification fails.

THE VERDICT IS THE PRODUCT, NOT THE WEIGHT. Yuma consensus penalises divergence from the
stake-weighted median, so an auditor who correctly catches the operator cheating and diverges
on-chain pays a vtrust cost for doing its job. Enforcement through weights is therefore backwards.
What an auditor produces here is a SIGNED, PUBLISHED VERDICT: "I fetched round N, its digest was
X, I ran L0 and L1, they passed, L2 and L3 I did not run." Reputational, checkable by anyone, and
free of the perverse incentive. Weight-setting is a separate, opt-in consequence.

FOUR PROPERTIES THAT MAKE A VERDICT WORTH ANYTHING:

  1. IT SAYS WHAT IT DID NOT DO. `levels_run` and `levels_required` are in the signed body. An
     auditor that only ran the free arithmetic check cannot be mistaken for one that reloaded the
     checkpoints, and nobody has to take its word for which it was.

  2. ANY FAILURE IS A FAILURE. A FAIL at a level the auditor did not declare as required still
     rejects the round. You do not get to run a check, see it fail, and discount it because it was
     optional. `required` only decides whether a SKIP makes the verdict INCOMPLETE.

  3. THE SIGNER IS PINNED. `rerun` checks that a record's signature is VALID; it cannot know whose
     signature it should be. An auditor following a sink without pinning the expected signer will
     happily verify records signed by whoever holds the repo — so an unpinned signer is itself an
     INCOMPLETE verdict, loudly.

  4. SILENCE IS A FINDING. v1's failure was not a wrong number, it was a validator that stopped
     publishing on 7 July while continuing to set weights, and nothing alarmed for four days. A
     trail with no new round inside `stale_after_s` produces a STALE verdict rather than nothing.

WHAT AN AUDITOR CANNOT DO. It cannot prove the operator scored a submission that was never
published — that is what `--history` (gap detection against the on-chain anchor chain) is for, and
it runs on every pass. And an auditor running L0 alone cannot detect a rigged SCORE, only a rigged
SUM; that is not a flaw to hide, it is why the verdict names its levels.

TWO IDENTITIES, AND THEY ARE NOT THE SAME STRING. `--signer` is the key the operator SIGNS
RECORDS with; `--validator-hotkey` is the ss58 whose on-chain commitment slot holds the anchor.
The first version passed the record signer to the metagraph lookup, so every round read INCOMPLETE
and a daemon with `--set-weights` verified everything correctly and then wrote nothing, forever.
That is why preflight refuses to start rather than letting a misconfiguration look like a quiet
subnet.

    python -m eval.auditor --once --signer <record key>            # L0, no weights
    python -m eval.auditor --follow --require L0,L1 --signer <k> --interval 600
    python -m eval.auditor --follow --require L0,L1,L2 --signer <k> \
        --observer HuggingFaceTB/SmolLM2-1.7B-Instruct,google/gemma-2-2b-it
    python -m eval.auditor --follow --signer <k> --validator-hotkey <ss58> --set-weights
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field

ACCEPT, REJECT, INCOMPLETE, STALE = "ACCEPT", "REJECT", "INCOMPLETE", "STALE"
LEVELS = ("L0", "L1", "L2", "L3")


@dataclass
class AuditorConfig:
    # what to follow
    records_repo: str = "RalphLabsAI/ralph-v2-rounds"
    # TWO DIFFERENT IDENTITIES, and conflating them was a blocker.
    #   expected_signer  -- the key the operator SIGNS RECORDS with (ed25519 hex, or an ss58 if the
    #                       operator signs with its hotkey). Pins whose trail this is.
    #   validator_hotkey -- the ss58 whose on-chain COMMITMENT SLOT holds the anchor. Resolved
    #                       against the metagraph.
    # The first version passed expected_signer to the metagraph lookup, so an ed25519 hex string was
    # searched for among ss58 addresses, never found, and every round read INCOMPLETE forever.
    expected_signer: str = ""
    validator_hotkey: str = ""
    netuid: int = 40
    network: str = "finney"

    # how deep to check. Levels not listed are still RUN if the inputs are present, and a failure
    # at any level rejects — `require` only decides which SKIPs make a verdict INCOMPLETE.
    require: tuple = ("L0",)
    pool_path: str = ""                # local corpus JSONL -> enables L1
    pool_from_trail: bool = True       # or fetch the pool the record pins, from the trail
    # A POOL, NOT ONE MODEL. The observer rotates per round from the nonce, so an auditor pinned to
    # a single model is auditing with the wrong grader on most rounds. Give it the same pool the
    # operator declares and it loads whichever one that round drew.
    observers: tuple = ()
    artifacts_dir: str = ""            # local checkpoints -> enables L3
    l3_items: int = 0                  # 0 = every item

    # consequences
    set_weights: bool = False          # off by default: the verdict is the product
    on_incomplete: str = "hold"        # "hold" (do not follow what you did not check) | "follow"
    burn_uid: int = 0                  # where weight goes when there is nothing verified to pay
    wallet: str = "ralph"
    hotkey: str = "auditor"

    # liveness
    stale_after_s: float = 6 * 3600.0
    # how many consecutive passes a record may be unfetchable before it counts as gone
    unfetchable_passes: int = 3
    interval_s: float = 600.0

    work_dir: str = os.path.join(os.path.expanduser("~"), ".ralph-v2-audit")

    @property
    def state_path(self) -> str:
        return os.path.join(self.work_dir, "auditor-state.json")

    @property
    def verdict_dir(self) -> str:
        return os.path.join(self.work_dir, "verdicts")

    @classmethod
    def from_env(cls) -> "AuditorConfig":
        e = os.environ.get
        req = tuple(x.strip() for x in e("RALPH_AUDIT_REQUIRE", "L0").split(",") if x.strip())
        obs = tuple(x.strip() for x in e("RALPH_AUDIT_OBSERVER", "").split(",") if x.strip())
        return cls(
            records_repo=e("RALPH_HF_REPO", "RalphLabsAI/ralph-v2-rounds"),
            expected_signer=e("RALPH_EXPECTED_SIGNER", ""),
            validator_hotkey=e("RALPH_VALIDATOR_HOTKEY", ""),
            netuid=int(e("RALPH_NETUID", "40")),
            network=e("RALPH_NETWORK", "finney"),
            require=req or ("L0",),
            pool_path=e("RALPH_AUDIT_POOL", ""),
            observers=obs,
            artifacts_dir=e("RALPH_AUDIT_ARTIFACTS", ""),
            burn_uid=int(e("RALPH_BURN_UID", "0")),
            wallet=e("RALPH_WALLET", "ralph"),
            hotkey=e("RALPH_AUDIT_HOTKEY", "auditor"),
            work_dir=e("RALPH_AUDIT_WORK_DIR",
                       os.path.join(os.path.expanduser("~"), ".ralph-v2-audit")),
        )


@dataclass
class Verdict:
    """One auditor's signed statement about one round.

    `prev` chains verdicts the same way the operator's anchors chain records, and for the same
    reason: without it an auditor can quietly delete the verdict it later regrets, and "auditor X
    verified round N" becomes deniable in both directions."""
    round: int = -1
    verdict: str = INCOMPLETE
    record_sha256: str = ""
    record_anchor: str = ""
    record_signer: str = ""
    expected_signer: str = ""
    levels_run: list = field(default_factory=list)
    levels_required: list = field(default_factory=list)
    level_status: dict = field(default_factory=dict)
    failed: list = field(default_factory=list)      # [(level, name, detail)]
    skipped: list = field(default_factory=list)
    trail: str = ""                                 # COMPLETE / TRAIL BROKEN / INCOMPLETE
    trail_detail: dict = field(default_factory=dict)
    chain_head: str = ""                            # the anchor this auditor read FROM CHAIN
    weights: dict = field(default_factory=dict)     # what the auditor would set, given this round
    followed_round: int = -1                        # which round those weights actually came from
    note: str = ""
    # Provisional: this verdict was driven by trail state (an unreachable sink, a broken history)
    # rather than by the record, so the round must be re-audited rather than written off.
    retry: bool = False
    ts: float = 0.0
    auditor: str = ""
    sig_scheme: str = ""
    signature: str = ""
    prev: str = ""

    def canonical(self) -> str:
        d = {k: v for k, v in asdict(self).items() if k not in ("signature",)}
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    def sign(self, signer) -> "Verdict":
        self.auditor = signer.public_id()
        self.sig_scheme = signer.scheme
        self.signature = signer.sign(self.canonical().encode())
        return self

    def verify_signature(self) -> bool:
        from .signing import verify
        if not (self.signature and self.auditor and self.sig_scheme):
            return False
        return verify(self.canonical().encode(), self.signature, self.auditor, self.sig_scheme)


def level_status(a, level: str) -> str:
    """PASS / FAIL / SKIP for a whole level.

    A single FAIL sinks it. A REQUIRED skip also sinks it to SKIP even when other checks at that
    level passed — "I ran four of the five L0 checks" is not L0, and the case that forces this is
    the signer pin: an auditor that could not confirm WHOSE record it verified has not verified the
    validator, however cleanly the arithmetic reproduced. Optional skips (no dethrone this round,
    no weights this round) are not absences of evidence and do not count."""
    from .rerun import FAIL, PASS, SKIP
    cs = [c for c in a.checks if c.level == level]
    if any(c.status == FAIL for c in cs):
        return FAIL
    if any(c.status == SKIP and c.required for c in cs):
        return SKIP
    return PASS if any(c.status == PASS for c in cs) else SKIP


def level_ran(a, level: str) -> bool:
    """Did anything at this level actually execute? Distinct from `level_status`: a level can have
    run and still be SKIP because one required check inside it did not."""
    from .rerun import SKIP
    return any(c.level == level and c.status != SKIP for c in a.checks)


class Auditor:
    """Stateless per round, stateful across them: it remembers what it has already ruled on and
    which round it is currently willing to follow."""

    def __init__(self, cfg: AuditorConfig, sink=None, signer=None, head_anchor_fn=None,
                 set_weights_fn=None, set_burn_fn=None, out=sys.stdout, now_fn=time.time):
        self.cfg = cfg
        self.out = out
        self.now = now_fn
        self.signer = signer
        self.head_anchor_fn = head_anchor_fn
        self.set_weights_fn = set_weights_fn
        self.set_burn_fn = set_burn_fn
        os.makedirs(cfg.work_dir, exist_ok=True)
        os.makedirs(cfg.verdict_dir, exist_ok=True)

        from .publish import HFReadSink, RecordPublisher
        self.sink = sink if sink is not None else HFReadSink(cfg.records_repo)
        # A REAL state path, not allow_no_state. The auditor never publishes, so it inherits the
        # never-shrink guard only if it feeds the high-water mark from its reads — see note_seen.
        # Without it the operator could serve a truncated index to an auditor that had already seen
        # the full one, which is v1's lineage bug pointed the other way.
        self.pub = RecordPublisher(self.sink, window=8,
                                   state_path=os.path.join(cfg.work_dir, "auditor-hwm.json"))
        self.state = self._load_state()
        self._pool = None
        self._pool_sha = ""
        self._observers = {}
        self._last_logged = None

    # ---- state -------------------------------------------------------------

    def _load_state(self) -> dict:
        try:
            with open(self.cfg.state_path) as fh:
                s = json.load(fh)
        except Exception:
            s = {}
        s.setdefault("last_round", -1)       # highest round we have ruled on
        s.setdefault("followed_round", -1)   # highest round we ACCEPTED and would pay
        s.setdefault("followed_weights", {})
        s.setdefault("last_verdict_sha", "")
        s.setdefault("last_seen_ts", 0.0)
        s.setdefault("stale_reported", False)
        s.setdefault("unfetchable", {})
        return s

    def _save_state(self) -> None:
        tmp = self.cfg.state_path + ".part"
        with open(tmp, "w") as fh:
            json.dump(self.state, fh, sort_keys=True, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.cfg.state_path)

    # ---- inputs, loaded once and reused across rounds -----------------------

    def _load_pool(self, rec):
        """The corpus for L1. Prefer the local file an operator configured; otherwise fetch the
        pool the RECORD pins from the trail. Either way `rerun` re-digests it against the signed
        manifest, so fetching it from the operator's own sink is safe: a substituted pool changes
        the digest and fails L1 rather than passing quietly."""
        want = (rec.manifest or {}).get("pool_sha256") or ""
        if self.cfg.pool_path:
            if self._pool is None:
                from .rerun import load_pool
                self._pool = load_pool(self.cfg.pool_path)
            return self._pool
        if not (self.cfg.pool_from_trail and want):
            return None
        if self._pool is not None and self._pool_sha == want:
            return self._pool
        from .publish import pool_name
        from .rerun import load_pool
        blob = self.sink.get(pool_name(want))
        if blob is None:
            return None
        path = os.path.join(self.cfg.work_dir, f"pool-{want[:16]}.jsonl")
        with open(path, "wb") as fh:
            fh.write(blob)
        self._pool, self._pool_sha = load_pool(path), want
        return self._pool

    def _load_observer(self, rec):
        """The observer THIS ROUND drew, from the pool this auditor was configured with.

        The observer rotates per round out of `commit_root ‖ round_nonce`, so a daemon pinned to a
        single model would be re-deriving the grades with the wrong grader on roughly half of all
        rounds — and, because any FAIL rejects, would publish a signed accusation each time. If the
        round's observer is not in our pool we simply cannot run L2, which is a skip, not a fault
        of the record.

        Returns (runner, name). Runners are cached: an auditor following a live subnet loads each
        observer once, not once per round."""
        want = (rec.manifest or {}).get("observer") or ""
        if not self.cfg.observers or not want:
            return None, want
        from .rerun import _obs_key
        match = [o for o in self.cfg.observers if _obs_key(o) == _obs_key(want)]
        if not match:
            return None, want
        repo = match[0]
        if repo not in self._observers:
            from .runners import HFRunner
            self._observers[repo] = HFRunner(repo)
        return self._observers[repo], repo

    def _make_runner(self):
        if not self.cfg.artifacts_dir:
            return None

        def make_runner(model_id, artifact_uri, _root=self.cfg.artifacts_dir):
            d = os.path.join(_root, model_id)
            if not os.path.isdir(d):
                return None
            from .runners import SafeStudentRunner
            return SafeStudentRunner(d)

        return make_runner

    # ---- the pass ----------------------------------------------------------

    def check_trail(self) -> tuple[str, dict, str]:
        """Is the published history complete, and does it agree with the chain?

        Runs every pass, not once: a round that was scored, paid out and never published leaves no
        record to audit, so only walking the index against the on-chain anchor finds it."""
        from .publish import verify_history
        head = ""
        if self.head_anchor_fn is not None:
            try:
                head = self.head_anchor_fn() or ""
            except Exception as e:
                return "INCOMPLETE", {"chain": f"unreadable: {type(e).__name__}: {e}"}, ""
        h = verify_history(self.pub, head_anchor_fn=(lambda: head) if head else None)

        # A NETWORK BLIP IS NOT TAMPERING — BUT A PERSISTENT ONE IS.
        #
        # HFReadSink.get returns None for a 429, a DNS failure and a deleted file alike, so one
        # rate-limited fetch anywhere in the history used to land in `broken`, read as TRAIL BROKEN,
        # and REJECT every round in the pass — permanently, since last_round advances and `todo` is
        # strictly-greater. An honest auditor would have signed a public accusation over a timeout,
        # and five auditors hitting blips at different moments would hold different rounds from
        # identical honest input.
        #
        # The signal that separates the two is PERSISTENCE, and nothing visible in a single fetch.
        # So an unreadable record reads INCOMPLETE for `unfetchable_passes` passes and only then
        # escalates to tampering — the same conclusion a human reaches, on the same evidence.
        soft = [b for b in h.broken if isinstance(b, dict) and b.get("why") == "not fetchable"]
        tampered = [b for b in h.broken if b not in soft]
        seen = {str(b["round"]) for b in soft}
        counts = {k: v for k, v in (self.state.get("unfetchable") or {}).items() if k in seen}
        for k in seen:
            counts[k] = int(counts.get(k, 0)) + 1
        self.state["unfetchable"] = counts
        persistent, transient = [], []
        for b in soft:
            n = counts.get(str(b["round"]), 0)
            if n >= self.cfg.unfetchable_passes:
                persistent.append({**b, "why": f"still unfetchable after {n} passes — a record "
                                               f"that stays gone is a deletion, not a blip"})
            else:
                transient.append({**b, "passes": n})
        tampered += persistent

        detail = {k: v for k, v in (("gaps", h.gaps), ("broken", tampered),
                                    ("chain_breaks", h.chain_breaks),
                                    ("mismatched", h.mismatched),
                                    ("unanchored", h.unanchored)) if v}
        if detail:
            return "TRAIL BROKEN", detail, head
        if transient:
            return "INCOMPLETE", {"unfetchable": transient}, head
        return ("COMPLETE" if h.ok else "INCOMPLETE"), {}, head

    def verify_round(self, entry: dict, trail: str, trail_detail: dict,
                     head: str) -> Verdict:
        """Fetch one round's record and rule on it."""
        from .rerun import FAIL, PASS, SKIP, audit_loaded, record_from_blob

        v = Verdict(round=int(entry["round"]), record_sha256=entry.get("sha256", ""),
                    record_anchor=entry.get("anchor", ""),
                    expected_signer=self.cfg.expected_signer,
                    levels_required=list(self.cfg.require),
                    trail=trail, trail_detail=trail_detail, chain_head=head,
                    ts=float(self.now()), prev=self.state.get("last_verdict_sha", ""))

        blob = self.sink.get(entry["name"])
        if blob is None:
            v.verdict = REJECT
            # Not a REJECT on its own: the sink may simply be unreachable, and an auditor that
            # signs a public accusation over a 429 is an auditor nobody trusts. It becomes a
            # rejection once check_trail has watched it stay gone across passes — at which point
            # the deletion IS the finding, and reporting it as merely "incomplete" would understate
            # the one tamper that per-record checks are structurally blind to.
            v.verdict = REJECT if trail == "TRAIL BROKEN" else INCOMPLETE
            v.retry = True
            v.note = (f"{entry['name']} is indexed but could not be fetched — retrying rather than "
                      f"treating an unreachable sink as a deleted record")
            v.skipped = [["L0", "record fetchable", v.note]]
            return self._finish(v)
        # Ask the SAME question the publisher's heartbeat asks, through the same function. The
        # digest is over canonical(), not over the stored bytes — hashing the blob here rejected
        # every honest record, which is the failure mode of re-deriving a rule instead of calling it.
        from .publish import roundtrip_reason
        ok, why = roundtrip_reason(blob, entry.get("sha256", ""),
                                   expect_round=int(entry.get("round", -1)),
                                   expect_signature=entry.get("signature", ""))
        if not ok:
            v.verdict = REJECT
            v.failed = [["L0", "record matches the index", why]]
            return self._finish(v)
        try:
            rec = record_from_blob(blob)
        except Exception as e:
            v.verdict = REJECT
            v.failed = [["L0", "record parses", f"{type(e).__name__}: {e}"]]
            return self._finish(v)

        v.record_signer = getattr(rec, "signer", "")
        obs, obs_name = self._load_observer(rec)
        a = audit_loaded(rec, pool=self._load_pool(rec), observer=obs,
                         observer_name=obs_name, make_runner=self._make_runner(),
                         max_l3_items=self.cfg.l3_items)

        # WHOSE RECORD IS THIS. rerun proves the signature is valid over the body; it has no idea
        # whose signature it ought to be. Following a sink without pinning the signer means
        # verifying whatever the repo holder signs, which is not an audit of the validator.
        if not self.cfg.expected_signer:
            a.add("L0", "record signed by the expected validator", SKIP,
                  "no expected_signer configured — this verdict says the record is internally "
                  "consistent, NOT that the subnet's validator produced it")
        else:
            a.ok("L0", "record signed by the expected validator",
                 v.record_signer == self.cfg.expected_signer,
                 fail=f"signed by {v.record_signer[:16] or '(nobody)'}… but this auditor follows "
                      f"{self.cfg.expected_signer[:16]}… — either the key rotated or this is not "
                      f"the validator's trail",
                 info=f"{self.cfg.expected_signer[:16]}…")

        v.level_status = {lv: level_status(a, lv) for lv in LEVELS}
        v.levels_run = [lv for lv in LEVELS if level_ran(a, lv)]
        v.failed = [[c.level, c.name, c.detail] for c in a.checks if c.status == FAIL]
        v.skipped = [[c.level, c.name, c.detail] for c in a.checks
                     if c.status == SKIP and c.required]

        # ANY failure rejects, including at a level this auditor did not declare as required. You
        # do not get to run a check, watch it fail, and discount it as optional.
        if v.failed:
            v.verdict = REJECT
        elif trail == "TRAIL BROKEN":
            v.verdict = REJECT
            v.retry = True          # the record may be fine; re-audit once the history is repaired
            v.note = "the round itself reproduces, but the published history does not"
        elif any(v.level_status.get(lv) != PASS for lv in self.cfg.require):
            v.verdict = INCOMPLETE
            miss = [lv for lv in self.cfg.require if v.level_status.get(lv) != PASS]
            v.note = f"required level(s) {miss} did not run"
        elif trail != "COMPLETE":
            v.verdict = INCOMPLETE
            v.retry = True
            v.note = ("the round reproduces, but the trail was not checked against the chain — "
                      "without the on-chain head this compares the operator's records to the "
                      "operator's index")
        else:
            v.verdict = ACCEPT
        v.weights = dict(getattr(rec, "weights", {}) or {})
        return self._finish(v)

    def _finish(self, v: Verdict) -> Verdict:
        """Apply the follow policy, sign, and record the verdict's own hash for the next link."""
        follow = v.verdict == ACCEPT or (v.verdict == INCOMPLETE
                                         and self.cfg.on_incomplete == "follow")
        if follow:
            v.followed_round = v.round
        else:
            # HOLD, NOT HALT — the same direction the operator's own publish gate fails. Weighting
            # the last round this auditor actually verified means a rejected crown does not get
            # paid, while emission continues to the last one that did check out.
            v.followed_round = int(self.state.get("followed_round", -1))
            v.weights = dict(self.state.get("followed_weights", {}) or {})
        if self.signer is not None:
            v.sign(self.signer)
        return v

    def publish_verdict(self, v: Verdict) -> str:
        path = os.path.join(self.cfg.verdict_dir, f"verdict-{v.round:08d}.json")
        blob = json.dumps(asdict(v), sort_keys=True, indent=1).encode()
        tmp = path + ".part"
        with open(tmp, "wb") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        # Append only when the RULING changed. A provisional verdict is re-emitted every pass
        # while the condition lasts, and an append-per-pass would bury the trail in duplicates.
        key = (v.round, v.verdict, v.record_sha256, v.trail)
        if key != self._last_logged:
            with open(os.path.join(self.cfg.verdict_dir, "verdicts.jsonl"), "a") as fh:
                fh.write(json.dumps(asdict(v), sort_keys=True, separators=(",", ":")) + "\n")
            self._last_logged = key
        return path

    def once(self) -> list:
        """One pass: check the trail, rule on every round we have not ruled on, act."""
        w = self.out.write
        trail, detail, head = self.check_trail()
        idx = self.pub.load_index()
        rounds = sorted(idx.get("rounds", []), key=lambda r: r["round"])
        self.pub.note_seen(len(rounds))

        # SILENCE IS A FINDING. A validator that stops publishing while continuing to set weights
        # is exactly what happened to v1 on 7 July, and nothing alarmed for four days. The clock
        # starts on the FIRST pass, not on the first record — a trail that never publishes anything
        # at all has to go stale too, and keying off "the last round we saw" would never fire.
        if not self.state.get("last_seen_ts"):
            self.state["last_seen_ts"] = float(self.now())
        if rounds and int(rounds[-1]["round"]) > int(self.state["last_round"]):
            self.state["last_seen_ts"] = float(self.now())
            self.state["stale_reported"] = False
        age = float(self.now()) - float(self.state["last_seen_ts"])
        if age > self.cfg.stale_after_s and not self.state.get("stale_reported"):
            v = Verdict(round=int(rounds[-1]["round"]) if rounds else -1, verdict=STALE,
                        trail=trail, trail_detail=detail, chain_head=head, ts=float(self.now()),
                        levels_required=list(self.cfg.require),
                        followed_round=int(self.state.get("followed_round", -1)),
                        weights=dict(self.state.get("followed_weights", {}) or {}),
                        note=f"no new round in {age / 3600:.1f}h (threshold "
                             f"{self.cfg.stale_after_s / 3600:.1f}h) — the trail has gone quiet "
                             f"while emission continues",
                        prev=self.state.get("last_verdict_sha", ""))
            if self.signer is not None:
                v.sign(self.signer)
            w(f"  STALE   trail silent for {age / 3600:.1f}h\n")
            # One verdict per stale EPISODE, not one per pass. An alarm that repeats every
            # interval is an alarm people filter, and the condition is already in the record.
            self.state["stale_reported"] = True
            self._commit(v)
            self._apply_weights()
            return [v]

        todo = [r for r in rounds if int(r["round"]) > int(self.state["last_round"])]
        if not todo:
            w(f"  up to date at round {self.state['last_round']} "
              f"(trail {trail}, following round {self.state['followed_round']})\n")
            self._save_state()      # the staleness clock lives here; an early return must persist it
            self._apply_weights()   # every pass sets weights, or the auditor goes inactive on chain
            return []

        out = []
        for entry in todo:
            v = self.verify_round(entry, trail, detail, head)
            mark = {ACCEPT: "ok  ", REJECT: "FAIL", INCOMPLETE: "part", STALE: "----"}[v.verdict]
            w(f"  {mark}  round {v.round}: {v.verdict}  "
              f"[{'+'.join(v.levels_run) or 'nothing ran'}]"
              f"{'  -> follow ' + str(v.followed_round) if v.followed_round != v.round else ''}\n")
            for lv, name, det in v.failed[:4]:
                w(f"          [{lv}] {name}: {det}\n")
            if v.note:
                w(f"          {v.note}\n")
            self._commit(v)
            out.append(v)

        # THE WATERMARK ADVANCES OVER THE SETTLED PREFIX ONLY. A verdict marked `retry` is
        # provisional — the record may be fine and the sink merely unreachable — so everything from
        # the first provisional round onward stays in `todo` and is re-audited next pass. Taking
        # max(round) instead would settle a round we explicitly said we could not settle.
        settled = int(self.state["last_round"])
        for v in out:
            if v.retry:
                break
            settled = max(settled, v.round)
        self.state["last_round"] = settled
        self._save_state()
        self._apply_weights()
        return out

    def _commit(self, v: Verdict) -> None:
        """Record the verdict and advance state. DOES NOT touch the chain — see `_apply_weights`.

        Weight-setting used to live here, i.e. once per ROUND. On a cold start `todo` is the entire
        published history, so a fresh auditor fired one extrinsic per historical round in a tight
        loop; `weights_rate_limit` rejected everything after the first, and the daemon discarded the
        False return because the state write had already happened. Net effect: an auditor starting
        at round 80 landed round 1's weights on chain and kept them, and two auditors starting on
        different days ended up with different vectors from identical honest input. The chain is
        touched once per pass now, from the final state."""
        self.publish_verdict(v)
        self.state["last_verdict_sha"] = v.sha256()
        # NOTE: last_round is NOT advanced here. `todo` is strictly-greater, so advancing per
        # round would step straight over a provisional verdict lower down and make it permanent.
        # The watermark is computed once per pass, over the contiguous settled prefix — see once().
        if v.followed_round >= int(self.state.get("followed_round", -1)):
            self.state["followed_round"] = v.followed_round
            self.state["followed_weights"] = dict(v.weights)
        self._save_state()

    def _apply_weights(self) -> None:
        """Set weights ONCE per pass, from whatever the auditor is currently willing to pay.

        The fallback when there is nothing verified is the BURN vector, not silence. "Hold" is
        sound advice for the incumbent operator — its previous weights persist on chain — but a
        third-party auditor that has never set any has no previous weights to persist. Setting
        nothing means contributing nothing to consensus, earning nothing, and eventually crossing
        `activity_cutoff` into inactive while believing it is validating. v1's chain layer already
        made this call for the same reason (`set_burn_weights`: "so the validator (and the auditor)
        STILL sets weights every epoch — this keeps the validator's vTrust alive")."""
        if not (self.cfg.set_weights and self.set_weights_fn is not None):
            return
        weights = dict(self.state.get("followed_weights") or {})
        src = self.state.get("followed_round", -1)
        if not weights:
            if self.set_burn_fn is None:
                self.out.write("  WARN  nothing verified to pay and no burn setter configured — "
                               "setting no weights, which reads on chain as an absent validator\n")
                return
            try:
                ok = self.set_burn_fn()
                self.out.write(f"  burn  no verified round to pay yet -> uid {self.cfg.burn_uid}: "
                               f"{ok}\n")
            except Exception as e:
                self.out.write(f"  ERROR set_burn_weights: {type(e).__name__}: {e}\n")
            return
        try:
            ok = self.set_weights_fn(weights)
            # A False return means rate-limited or refused, NOT written. Reporting it as a
            # success is how the backlog bug stayed invisible.
            self.out.write(f"  {'wrote' if ok else 'DEFER'} weights from round {src}: "
                           f"{ok and 'on chain' or 'not accepted this block — will retry'}\n")
        except Exception as e:
            # Never let a chain error rewrite the verdict — the verdict is already signed and
            # written, and it is the part that has to survive.
            self.out.write(f"  ERROR set_weights: {type(e).__name__}: {e}\n")

    def follow(self, interval_s: float = 0.0, max_passes: int = 0) -> int:
        n = 0
        while True:
            try:
                self.once()
            except Exception as e:
                # A pass that throws must not kill the daemon: an operator serving a corrupt index
                # would otherwise be able to switch its auditors off by publishing bad bytes once.
                self.out.write(f"  ERROR pass failed: {type(e).__name__}: {e}\n")
            n += 1
            if max_passes and n >= max_passes:
                return 0
            time.sleep(interval_s or self.cfg.interval_s)


# ---------------------------------------------------------------- chain wiring

def chain_head_anchor(netuid: int, network: str, hotkey: str):
    """Read the operator's anchor FROM THE CHAIN. Injectable, and the default is the real thing.

    This is the only value in the whole loop the operator cannot edit, which is what makes an
    auditor's trail check different from re-reading the operator's own index back to them."""
    def read() -> str:
        import bittensor as bt
        st = bt.Subtensor(network=network)
        mg = st.metagraph(netuid=netuid)
        try:
            uid = list(mg.hotkeys).index(hotkey)
        except ValueError:
            raise RuntimeError(f"{hotkey[:16]}… is not registered on netuid {netuid}")
        return str(st.get_commitment(netuid=netuid, uid=uid) or "")

    return read


def bittensor_chain(cfg: AuditorConfig):
    """One chain object, built lazily and REUSED. Constructing it per weight-set meant a fresh
    Subtensor connection plus two full metagraph loads on every round of a backlog."""
    holder: dict = {}

    def get():
        if "c" not in holder:
            from karpa.chain_layer.bittensor_chain import BittensorChain  # type: ignore
            holder["c"] = BittensorChain(network=cfg.network, netuid=cfg.netuid,
                                         wallet_name=cfg.wallet, wallet_hotkey=cfg.hotkey)
        return holder["c"]

    return get


def bittensor_weight_setter(cfg: AuditorConfig, get_chain=None):
    """Set weights with the AUDITOR's own wallet, via the same chain layer the validator uses.

    Returns (set_weights, set_burn). The second is not optional garnish: a validator that sets
    nothing contributes nothing to consensus and crosses `activity_cutoff` into inactive, so an
    auditor with no verified round to pay must still write SOMETHING every epoch."""
    get = get_chain or bittensor_chain(cfg)

    def setter(weights: dict) -> bool:
        return bool(get().set_weights(weights))

    def burn() -> bool:
        os.environ.setdefault("RALPH_BURN_UID", str(cfg.burn_uid))
        return bool(get().set_burn_weights())

    return setter, burn


def preflight(cfg: AuditorConfig, out=sys.stdout) -> list:
    """Everything that can be checked before the first fetch. Returns a list of fatal reasons.

    The failure this exists for: the daemon is designed to fail closed, so a misconfiguration looks
    exactly like a quiet subnet — it prints a verdict every interval and never writes a weight. An
    operator can run that for weeks believing they are validating."""
    bad = []
    unknown = [lv for lv in cfg.require if lv not in LEVELS]
    if unknown:
        bad.append(f"unknown level(s) {unknown}; valid: {list(LEVELS)}")
    if "L2" in cfg.require and not cfg.observers:
        bad.append("--require L2 needs --observer <hf id>[,<hf id>…] (the round's observer pool)")
    if "L3" in cfg.require and not cfg.artifacts_dir:
        bad.append("--require L3 needs --artifacts <dir of checkpoints>")
    if cfg.set_weights:
        # WEIGHT-SETTING NEEDS THE CHAIN, and the chain needs an identity that is not the record
        # signer. Refusing here is the difference between "this cannot work" and "this silently
        # does nothing".
        if not cfg.validator_hotkey:
            bad.append("--set-weights needs --validator-hotkey <ss58 of the validator whose "
                       "commitment holds the anchor>: without the on-chain head the trail can "
                       "only be compared against the operator's own index, so no round can ACCEPT")
        if not cfg.expected_signer:
            bad.append("--set-weights needs --signer <the validator's record-signing public id>: "
                       "an unpinned signer means following whoever holds the repo")
        try:
            from karpa.chain_layer.bittensor_chain import BittensorChain  # noqa: F401
        except Exception as e:
            bad.append(f"--set-weights needs the v1 chain layer (karpa.chain_layer): {e}")
    try:
        import huggingface_hub  # noqa: F401
    except Exception:
        bad.append("huggingface_hub is required to fetch the trail: pip install huggingface_hub")
    return bad


def _signer():
    """ed25519 from RALPH_AUDITOR_SEED, or the auditor's hotkey when a wallet is configured.

    Unsigned verdicts are allowed but nearly useless — anyone can mint one — so this warns rather
    than failing, and the missing signature is visible in every verdict it writes."""
    seed = os.environ.get("RALPH_AUDITOR_SEED", "")
    if seed:
        from .signing import Ed25519Signer
        return Ed25519Signer(seed=bytes.fromhex(seed)[:32])
    return None


def main(argv: list) -> int:
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        return 0
    cfg = AuditorConfig.from_env()

    def opt(flag, cast=str, default=None):
        return cast(argv[argv.index(flag) + 1]) if flag in argv else default

    cfg.require = tuple(x.strip() for x in (opt("--require") or ",".join(cfg.require)).split(",")
                        if x.strip())
    cfg.pool_path = opt("--pool") or cfg.pool_path
    obs = opt("--observer")
    if obs:
        cfg.observers = tuple(x.strip() for x in obs.split(",") if x.strip())
    cfg.artifacts_dir = opt("--artifacts") or cfg.artifacts_dir
    cfg.records_repo = opt("--repo") or cfg.records_repo
    cfg.expected_signer = opt("--signer") or cfg.expected_signer
    cfg.validator_hotkey = opt("--validator-hotkey") or cfg.validator_hotkey
    cfg.wallet = opt("--wallet") or cfg.wallet
    cfg.hotkey = opt("--hotkey") or cfg.hotkey
    cfg.work_dir = opt("--work-dir") or cfg.work_dir
    cfg.l3_items = opt("--l3-items", int, 0) or 0
    cfg.interval_s = opt("--interval", float, 0.0) or cfg.interval_s
    cfg.set_weights = "--set-weights" in argv
    if "--follow-incomplete" in argv:
        cfg.on_incomplete = "follow"

    bad = preflight(cfg)
    if bad:
        for r in bad:
            print(f"  BLOCKED  {r}")
        print("\n  refusing to start — a daemon that cannot set weights looks identical to a "
              "quiet subnet, so this is a startup error rather than a silent no-op\n")
        return 2

    print(f"\n  auditor: {cfg.records_repo}")
    print(f"  require: {'+'.join(cfg.require)}   "
          f"observers: {','.join(o.split('/')[-1] for o in cfg.observers) or '(none)'}   "
          f"artifacts: {cfg.artifacts_dir or '(none)'}")
    if not cfg.expected_signer:
        print("  WARN   no --signer pinned: verdicts will say the record is internally "
              "consistent, not that the subnet's validator wrote it")
    if not cfg.validator_hotkey:
        print("  WARN   no --validator-hotkey: the on-chain anchor cannot be read, so every "
              "verdict reads INCOMPLETE (the trail is only checkable against the operator's index)")
    if cfg.set_weights:
        print(f"  LIVE   will set weights as {cfg.wallet}/{cfg.hotkey}; with nothing verified "
              f"yet it burns to uid {cfg.burn_uid} rather than setting nothing")
    print()

    set_w, set_burn = (bittensor_weight_setter(cfg) if cfg.set_weights else (None, None))
    a = Auditor(cfg, signer=_signer(),
                head_anchor_fn=(chain_head_anchor(cfg.netuid, cfg.network, cfg.validator_hotkey)
                                if cfg.validator_hotkey else None),
                set_weights_fn=set_w, set_burn_fn=set_burn)
    if "--follow" in argv:
        return a.follow()
    vs = a.once()
    # Same convention as eval/rerun: 1 means something DEMONSTRABLY did not reproduce, 2 means
    # nothing contradicted the trail but the checks that would matter were not run.
    if any(v.verdict in (REJECT, STALE) for v in vs):
        return 1
    if any(v.verdict == INCOMPLETE for v in vs):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
