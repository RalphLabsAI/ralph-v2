"""Regression coverage for the miner commit/reveal command-line path."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.chain_bittensor import build_commitment_envelope
from eval.identity import commit_value, content_hash
from miner import submit


def _write_reveal_state(tmp_path: Path) -> tuple[Path, dict]:
    ckpt = tmp_path / "model"
    ckpt.mkdir()
    (ckpt / "config.json").write_text('{"model_type":"qwen3"}')

    salt = "test-salt"
    digest = content_hash(ckpt)
    committed = commit_value(digest, salt)
    base_envelope = build_commitment_envelope(
        "sub2",
        committed,
        "hf://example/model@revision",
        declared_compute_h100h=1.25,
        bond_posted=0.5,
    )
    state = {
        "content_hash": digest,
        "salt": salt,
        "commit_value": committed,
        "tier": "sub2",
        "artifact_uri": "hf://example/model@revision",
        "envelope": base_envelope,
    }
    submit._state_path(ckpt).write_text(json.dumps(state))
    return ckpt, state


def test_reveal_cli_uses_chain_arguments_and_publishes_complete_envelope(tmp_path, monkeypatch):
    ckpt, state = _write_reveal_state(tmp_path)
    seen: dict = {}

    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5TestHotkey"))

    def fake_wallet(*, name, hotkey):
        seen["wallet"] = (name, hotkey)
        return wallet

    class FakeSubtensor:
        def set_commitment(self, *, wallet, netuid, data):
            seen["write"] = (wallet, netuid, data)

    def fake_subtensor(*, network):
        seen["network"] = network
        return FakeSubtensor()

    monkeypatch.setitem(
        sys.modules,
        "bittensor",
        SimpleNamespace(wallet=fake_wallet, subtensor=fake_subtensor),
    )

    rc = submit.main([
        "reveal",
        "--ckpt", str(ckpt),
        "--wallet", "alice",
        "--hotkey", "miner",
        "--netuid", "40",
        "--network", "finney",
    ])

    assert rc == 0
    assert seen["wallet"] == ("alice", "miner")
    assert seen["network"] == "finney"
    written_wallet, netuid, raw = seen["write"]
    assert written_wallet is wallet
    assert netuid == 40
    body = json.loads(raw)
    assert body == {
        "v": 2,
        "tier": "sub2",
        "cv": state["commit_value"],
        "uri": "hf://example/model@revision",
        "h100h": 1.25,
        "bond": 0.5,
        "ch": state["content_hash"],
        "salt": state["salt"],
    }


def test_reveal_cli_dry_run_accepts_chain_arguments_without_writing(tmp_path, monkeypatch, capsys):
    ckpt, state = _write_reveal_state(tmp_path)

    def should_not_run(**_kwargs):
        raise AssertionError("dry-run attempted a chain call")

    monkeypatch.setitem(
        sys.modules,
        "bittensor",
        SimpleNamespace(wallet=should_not_run, subtensor=should_not_run),
    )

    rc = submit.main([
        "reveal",
        "--ckpt", str(ckpt),
        "--wallet", "alice",
        "--hotkey", "miner",
        "--netuid", "40",
        "--network", "finney",
        "--dry-run",
    ])

    out = capsys.readouterr().out
    assert rc == 0
    assert "--dry-run: nothing written to chain" in out
    assert state["content_hash"] in out


def test_reveal_cli_writes_chunked_fields_through_the_11x_call_path(tmp_path, monkeypatch):
    """bittensor >= 11 has no `Subtensor.set_commitment`: the slot is written as a generated
    call, in the 128-byte Raw fields every reveal on netuid 40 already uses, signed by the hotkey."""
    ckpt, state = _write_reveal_state(tmp_path)
    seen: dict = {}
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5TestHotkey"))

    class FakeSubtensor:
        def __init__(self, *, network):
            seen["network"] = network

        def submit_call(self, call, w, **kwargs):
            seen["call"], seen["wallet"], seen["kwargs"] = call, w, kwargs
            return SimpleNamespace(success=True)

    def should_not_run(*_a, **_k):
        raise AssertionError("the legacy lowercase entry points must not be used on 11.x")

    fake = SimpleNamespace(
        Wallet=lambda *, name, hotkey: wallet, Subtensor=FakeSubtensor,
        wallet=should_not_run, subtensor=should_not_run,
        calls=SimpleNamespace(Commitments=SimpleNamespace(
            set_commitment=lambda netuid, info: {"netuid": netuid, "info": info})))
    monkeypatch.setitem(sys.modules, "bittensor", fake)

    rc = submit.main(["reveal", "--ckpt", str(ckpt), "--wallet", "alice", "--hotkey", "miner"])
    assert rc == 0
    assert seen["wallet"] is wallet and seen["kwargs"] == {"signer": "hotkey"}
    assert seen["call"]["netuid"] == 40
    fields = seen["call"]["info"]["fields"]
    chunks = [next(iter(f.values())) for f in fields]
    assert 1 < len(fields) <= submit.MAX_FIELDS
    assert all(list(f)[0] == f"Raw{len(c)}" for f, c in zip(fields, chunks))
    assert all(len(c) == 128 for c in chunks[:-1]) and 0 < len(chunks[-1]) <= 128
    body = json.loads(b"".join(chunks))
    assert body["ch"] == state["content_hash"] and body["salt"] == state["salt"]
    assert body["cv"] == state["commit_value"]


def test_a_rejected_extrinsic_is_reported_not_swallowed(tmp_path, monkeypatch, capsys):
    ckpt, _ = _write_reveal_state(tmp_path)

    class FakeSubtensor:
        def __init__(self, *, network):
            pass

        def submit_call(self, call, w, **kwargs):
            return SimpleNamespace(success=False, error="SpaceLimitExceeded")

    fake = SimpleNamespace(
        Wallet=lambda *, name, hotkey: SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5T")),
        Subtensor=FakeSubtensor,
        calls=SimpleNamespace(Commitments=SimpleNamespace(set_commitment=lambda n, i: (n, i))))
    monkeypatch.setitem(sys.modules, "bittensor", fake)
    rc = submit.main(["reveal", "--ckpt", str(ckpt)])
    out = capsys.readouterr().out
    assert rc == 3 and "not accepted" in out and "SpaceLimitExceeded" in out


def test_the_slot_holds_three_fields_of_128_bytes():
    assert submit.commitment_fields("x" * 384) == [{"Raw128": b"x" * 128}] * 3
    assert submit.commitment_fields("y" * 129) == [{"Raw128": b"y" * 128}, {"Raw1": b"y"}]
    with pytest.raises(ValueError, match="384"):
        submit.commitment_fields("z" * 385)
