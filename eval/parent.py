"""Pinned parents — what makes this compression-distillation rather than a model beauty contest.

THE TASK, AS CONST FRAMED IT. Bonsai does not distil a big teacher into a different small
model. It takes ONE pinned pretrained model, leaves the architecture completely untouched, and
re-stores its weights at 1.125 or 1.71 bits. So the subnet's task is: given THIS parent and THIS
bit budget, keep as much of its capability as you can.

That framing fixes the scoring question that three different normalizations could not:

  * scoring against the MINER'S DECLARED parent invites weak-parent farming — declare a
    lobotomised parent and retention goes to 1.0. The degenerate limit is parent == student.
  * scoring absolute capability rewards whoever submits the strongest artifact from anywhere,
    which is the clone attack wearing a tie.
  * scoring against a PINNED parent is well-defined, identical for every miner in the tier, and
    un-gameable, because the miner does not choose the denominator.

WHAT THIS GATE CAN AND CANNOT DO. It verifies the artifact is SHAPE-COMPATIBLE with the pinned
parent — same architecture family, same weight-element count, same config essentials — using
headers only, no weight load. That is an admission gate, not a provenance proof: nothing here
shows the weights descend from the parent, and no behavioural test can (a different model of
identical shape passes the same checkers). What it does buy is concrete and worth having:
submitting Bonsai-27B into a Qwen3-8B tier fails at the door, `Manifest.student_base` stops
being decorative, and "retention vs the parent" becomes a number that means the same thing for
every submission in the tier.

Entries must be VERIFIED against the real artifact before use — a wrong param count silently
rejects every honest submission. The shipped entry was measured from a real GGUF release.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ParentSpec:
    """A pinned parent. `weight_params` is the count of weight-bearing elements as BOTH readers
    report it (safetensors headers and GGUF tensor tables agree, since architecture is
    unchanged); `tol` absorbs the small differences in how formats treat the 1-D tail."""
    name: str                      # canonical id, e.g. "Qwen/Qwen2.5-0.5B-Instruct"
    arch: str                      # model_type / general.architecture, e.g. "qwen2"
    weight_params: int             # measured, not guessed
    hidden_size: int = 0
    num_layers: int = 0
    vocab_size: int = 0
    tol: float = 0.02              # fractional tolerance on the param count


# VERIFIED entries only. Values below were measured from the real artifact this session:
# Qwen2.5-0.5B-Instruct GGUF reports 630,095,872 weight elements across 291 tensors, arch
# "qwen2". Add a parent only after reading its real header — a guessed number rejects every
# honest submission to that tier and is indistinguishable from the tier being broken.
PARENTS: dict[str, ParentSpec] = {
    "qwen2.5-0.5b-instruct": ParentSpec(
        name="Qwen/Qwen2.5-0.5B-Instruct", arch="qwen2",
        weight_params=630_095_872, hidden_size=896, vocab_size=151936,
    ),
}


@dataclass
class ParentCheck:
    ok: bool = False
    parent: str = ""
    measured_params: int = 0
    reasons: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _config_fields(ckpt_dir: Path) -> dict:
    cp = ckpt_dir / "config.json"
    if not cp.exists():
        return {}
    try:
        j = json.loads(cp.read_text())
        return j if isinstance(j, dict) else {}
    except Exception:
        return {}


def parent_compat(ckpt_dir, spec: ParentSpec, inspection=None) -> ParentCheck:
    """Header-only compatibility of an artifact with its pinned parent.

    `inspection` is a gates.Inspection if already computed (avoids re-reading headers). Works
    for both safetensors and GGUF submissions, comparing quantities both formats agree on:
    architecture string, weight-element count, and — when a config.json ships — the shape
    essentials. Deliberately does NOT compare tensor NAMES: GGUF renames everything
    (`blk.0.attn_q.weight` vs `model.layers.0.self_attn.q_proj.weight`), so a name check would
    reject the very format the product ships in."""
    d = Path(ckpt_dir)
    chk = ParentCheck(parent=spec.name)

    if inspection is None:
        from .gates import inspect_checkpoint
        inspection = inspect_checkpoint(d)
    if not inspection.ok:
        chk.reasons = list(inspection.reasons) or ["artifact failed inspection"]
        return chk

    chk.measured_params = inspection.params
    lo = spec.weight_params * (1 - spec.tol)
    hi = spec.weight_params * (1 + spec.tol)
    if not (lo <= inspection.params <= hi):
        chk.reasons.append(
            f"weight params {inspection.params:,} outside {spec.name}'s "
            f"{spec.weight_params:,} +/-{spec.tol:.0%} — architecture is not unchanged")

    # architecture string, from whichever side declares it
    cfg = _config_fields(d)
    declared = (cfg.get("model_type") or "").lower()
    if not declared:
        gg = sorted(d.glob("*.gguf"))
        if gg:
            from .gguf import read_gguf
            declared = (read_gguf(gg[0]).arch or "").lower()
    if declared and spec.arch and declared != spec.arch.lower():
        chk.reasons.append(f"architecture {declared!r} != pinned parent {spec.arch!r}")
    elif not declared:
        chk.notes.append("no architecture string in artifact; matched on shape alone")

    for field_name, want in (("hidden_size", spec.hidden_size),
                             ("num_hidden_layers", spec.num_layers),
                             ("vocab_size", spec.vocab_size)):
        got = cfg.get(field_name)
        if want and got is not None and int(got) != int(want):
            chk.reasons.append(f"{field_name} {got} != pinned parent {want}")

    chk.ok = not chk.reasons
    return chk
