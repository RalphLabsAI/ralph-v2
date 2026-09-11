"""A busy Hugging Face must not take the round down.

One 429 ("maximum queue size reached") while listing a single miner's repo used to unwind the whole
round: only FetchRefused was caught, so the error escaped before anyone was scored.
Transient errors are now retried with backoff; past that, the one artifact is skipped with its
reason. Exercised through the real fetch/resolver with injected network legs."""
from __future__ import annotations

import pytest

from eval import fetch as F


class _Resp:
    def __init__(self, code):
        self.status_code = code


class HTTPErr(Exception):
    """Shaped like huggingface_hub's HfHubHTTPError: carries `.response.status_code`."""

    def __init__(self, code):
        super().__init__(f"{code} Too Many Requests" if code == 429 else f"{code} error")
        self.response = _Resp(code)


def _files(repo, rev):
    return [("model.gguf", 10), ("config.json", 1)]


def _dl_ok(repo, rev, name, out):
    with open(out, "wb") as fh:
        fh.write(b"x" * (10 if name.endswith(".gguf") else 1))


def test_a_429_is_retried_and_the_fetch_succeeds(monkeypatch, tmp_path):
    slept = []
    monkeypatch.setattr(F, "_sleep", slept.append)
    calls = {"n": 0}

    def flaky(repo, rev):
        calls["n"] += 1
        if calls["n"] < 3:
            raise HTTPErr(429)
        return _files(repo, rev)

    r = F.fetch("hf://org/model@abc", str(tmp_path), lister=flaky, downloader=_dl_ok)
    assert calls["n"] == 3 and r.files == 2
    assert slept == list(F.RETRY_DELAYS_S[:2])


def test_a_permanent_error_is_raised_at_once(monkeypatch):
    slept = []
    monkeypatch.setattr(F, "_sleep", slept.append)

    def gone(repo, rev):
        raise HTTPErr(404)

    with pytest.raises(HTTPErr):
        F.plan("hf://org/model@abc", lister=gone)
    assert slept == [], "a 404 is an answer, not a busy moment — no backoff"


def test_a_flaky_download_retries_without_leaving_a_partial_file(monkeypatch, tmp_path):
    monkeypatch.setattr(F, "_sleep", lambda s: None)
    seen = {"n": 0}

    def dl(repo, rev, name, out):
        seen["n"] += 1
        if name.endswith(".gguf") and seen["n"] == 1:
            with open(out, "wb") as fh:
                fh.write(b"partial")
            raise HTTPErr(503)
        _dl_ok(repo, rev, name, out)

    r = F.fetch("hf://org/model@abc", str(tmp_path), lister=_files, downloader=dl)
    assert r.bytes == 11, "the partial bytes from the failed attempt must not be counted or kept"


def test_past_the_retries_the_miner_is_skipped_not_the_round(monkeypatch, tmp_path):
    monkeypatch.setattr(F, "_sleep", lambda s: None)

    def always_busy(repo, rev):
        raise HTTPErr(429)

    monkeypatch.setattr(F, "_hf_list", always_busy)
    log = []
    fetch_dir_for = F.resolver(str(tmp_path), log=log)
    assert fetch_dir_for("hot0", "hf://org/model@abc") == ""
    assert log[-1][0] == "hot0" and log[-1][1] == "failed" and "429" in log[-1][-1], log


def test_a_refusal_is_still_a_refusal(monkeypatch, tmp_path):
    slept = []
    monkeypatch.setattr(F, "_sleep", slept.append)
    log = []
    fetch_dir_for = F.resolver(str(tmp_path), log=log)
    assert fetch_dir_for("hot1", "file:///etc/passwd") == ""
    assert log[-1][1] == "refused" and slept == []
