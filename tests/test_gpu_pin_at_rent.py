"""The pinned GPU must be checked when the box is rented, not after the round is scored.

`require_gpu` was enforced only in `verify_returned_record`, which runs once the remote has
installed, scored every submission and returned. Live round 2 (2026-08-18) rented an `H100_sxm5`
against a round pinned to `NVIDIA H100 PCIe`: Shadeform sells both under `gpu_type=H100`, and the
SXM variant was simply the cheapest available that hour. Left alone it would have cost the full
install, the full scoring run and the full bill, and then been thrown away at audit.

`assert_gpu_present` runs inside the rent-retry loop, so failing there turns a dead round into a
dud region.
"""
import pytest

from eval.orchestrator import GpuSpec, Instance, RemoteRoundError, assert_gpu_present

PCIE = "GPU 0: NVIDIA H100 PCIe (UUID: GPU-1111)"
SXM = "GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-2222)"


def _inst():
    return Instance(id="i", cloud="c", region="r", instance_type="H100",
                    price_per_hour=1.0, status="active")


def test_the_pinned_card_is_accepted(monkeypatch):
    monkeypatch.setattr("eval.orchestrator.gpu_devices", lambda i, s: [PCIE])
    assert assert_gpu_present(_inst(), GpuSpec(require_gpu="NVIDIA H100 PCIe")) == [PCIE]


def test_the_wrong_card_is_refused_before_the_install(monkeypatch):
    """The exact case: an SXM box against a PCIe pin."""
    monkeypatch.setattr("eval.orchestrator.gpu_devices", lambda i, s: [SXM])
    with pytest.raises(RemoteRoundError, match="pinned to"):
        assert_gpu_present(_inst(), GpuSpec(require_gpu="NVIDIA H100 PCIe"))


def test_no_gpu_still_refused(monkeypatch):
    monkeypatch.setattr("eval.orchestrator.gpu_devices", lambda i, s: [])
    with pytest.raises(RemoteRoundError, match="NO GPU"):
        assert_gpu_present(_inst(), GpuSpec(require_gpu="NVIDIA H100 PCIe"))


def test_an_unpinned_round_accepts_anything(monkeypatch):
    """The FIRST round has no pin yet — it learns the name and pins it afterwards. Refusing here
    would make the very round that establishes the pin impossible to run."""
    monkeypatch.setattr("eval.orchestrator.gpu_devices", lambda i, s: [SXM])
    assert assert_gpu_present(_inst(), GpuSpec(require_gpu="")) == [SXM]


def test_matches_by_substring_not_equality(monkeypatch):
    """`require_gpu` is torch's device name; these come from `nvidia-smi -L`, which wraps it in
    "GPU 0: ... (UUID: ...)". Demanding equality would reject the correct box too."""
    monkeypatch.setattr("eval.orchestrator.gpu_devices", lambda i, s: [PCIE])
    assert assert_gpu_present(_inst(), GpuSpec(require_gpu="NVIDIA H100 PCIe"))
