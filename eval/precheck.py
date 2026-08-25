"""What the next round will decide about each pending commitment — before renting anything.

    python -m eval.precheck            # reads the public status snapshot, prints one row per
                                       # pending artifact: formats, measured bits, tier verdict

Runs anywhere with internet: no GPU, no chain client, no secrets, and nothing is downloaded except
each GGUF's HEADER. `read_gguf` never reads tensor data, so a range request for the first ~32 MB
plus a sparse `truncate` to the true size measures a multi-GB artifact in seconds. The same
`measure_gguf_dir`/`bit_tier_gate` code intake runs makes the verdicts — a pre-check that can
disagree with intake would be worse than none, because it would be believed.

WHAT THIS CANNOT TELL YOU: coherence. The bit gates are a pure function of the header; retention
and the degeneracy gate need a GPU and the round's sealed exam. A row that passes here can still
score near zero — two of the artifacts that motivated this tool do exactly that.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import urllib.request

STATUS_URL = os.environ.get(
    "RALPH_STATUS_URL",
    "https://huggingface.co/datasets/RalphLabsAI/ralph-v2-status/resolve/main/status.json")
HEADER_BYTES = 32 * 1024 * 1024


def _get(url: str, rng: str = "") -> bytes:
    req = urllib.request.Request(url, headers={"Range": rng} if rng else {})
    return urllib.request.urlopen(req, timeout=90).read()


def _repo_rev(uri: str) -> tuple[str, str]:
    body = uri.split("://", 1)[1]
    repo, _, rev = body.partition("@")
    return repo, rev or "main"


def _resolve_main(repo: str) -> str:
    d = json.loads(_get(f"https://huggingface.co/api/models/{repo}"))
    return str(d.get("sha", "main"))


def check_one(uri: str, declared: str, out=sys.stdout) -> dict:
    from .bitrate import TIERS, bit_tier_gate, measure_gguf_dir

    repo, rev = _repo_rev(uri)
    note = ""
    if rev == "main":
        # a moving reference: measure what main points at NOW and say so — by round time the miner
        # may have pushed different bytes, and then commit-reveal decides, not this
        rev = _resolve_main(repo)
        note = "committed @main (moving ref) — measured its current target"
    tree = json.loads(_get(f"https://huggingface.co/api/models/{repo}/tree/{rev}"))
    ggufs = sorted(((f.get("size", 0), f["path"]) for f in tree
                    if f["path"].lower().endswith(".gguf")), reverse=True)
    if not ggufs:
        return {"uri": uri, "declared": declared, "verdict": "NO GGUF IN REPO", "note": note}
    size, path = ggufs[0]
    with tempfile.TemporaryDirectory() as td:
        local = os.path.join(td, "m.gguf")
        with open(local, "wb") as fh:
            fh.write(_get(f"https://huggingface.co/{repo}/resolve/{rev}/{path}",
                          rng=f"bytes=0-{HEADER_BYTES - 1}"))
        os.truncate(local, size)
        rep = measure_gguf_dir([local])
    tier = next((t for t in TIERS if t.name == declared), None)
    ok, why = bit_tier_gate(rep, tier) if tier else (False, [f"unknown tier {declared!r}"])
    fits = [t.name for t in TIERS if bit_tier_gate(rep, t)[0]]
    parent = 8_190_427_136
    if rep.params and abs(rep.params - parent) / parent > 0.02:
        note = (note + "; " if note else "") + f"params {rep.params:,} off the parent by >2%"
    fmts = " ".join(f"{k}:{v // 10**6}M" for k, v in
                    sorted(rep.formats.items(), key=lambda x: -x[1])[:3])
    return {"uri": uri, "declared": declared, "code": rep.code_bits,
            "container": rep.container_bits, "formats": fmts,
            "verdict": "PASS" if ok else f"REJECT — {why[0][:60]}",
            "fits": fits, "note": note}


def main(argv=None) -> int:
    doc = json.loads(_get(STATUS_URL))
    p = doc["chain"]["cohort"]["value"]["pending"]
    rows = ([dict(r, kind="new") for r in p.get("new_entrants", [])]
            + [dict(r, kind="re") for r in p.get("resubmitted", [])])
    if not rows:
        sys.stdout.write("nothing awaiting scoring — the next round would rent nothing\n")
        return 0
    sys.stdout.write(f"{len(rows)} pending (snapshot: unchanged={p.get('unchanged')})\n\n")
    bad = 0
    for r in rows:
        try:
            v = check_one(r.get("artifact_uri", ""), str(r.get("tier", "")))
        except Exception as e:
            v = {"verdict": f"UNREADABLE — {type(e).__name__}: {e}", "note": ""}
            bad += 1
        sys.stdout.write(f"{r['kind']:3s} {r['hotkey'][:12]}… {str(r.get('tier')):8s} "
                         f"{v.get('code', '—'):>7} code {v.get('container', '—'):>7} cont  "
                         f"{v['verdict']}\n")
        if v.get("formats"):
            sys.stdout.write(f"      {v['formats']}\n")
        if v.get("note"):
            sys.stdout.write(f"      note: {v['note']}\n")
    sys.stdout.write("\nBit gates only. Coherence (degeneracy + retention) needs the round.\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
