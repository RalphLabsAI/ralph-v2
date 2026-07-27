"""Pre-scoring gates — the fail-closed checks before a checkpoint is ever loaded or run.

The validator loads and RUNS untrusted miner weights, so this is the security boundary,
not a nicety. Two classes of check:

  SAFETY (refuse to even load):
    * safetensors only — a pickle (pytorch_model.bin / .pt) is arbitrary code execution
      on load (the exact teacher-as-student vector the prior subnet was hit with)
    * no miner *.py and no tokenizer auto_map — trust_remote_code stays False, always
    * file allowlist + size cap — no decompression bombs, no smuggled payloads

  INTEGRITY (derive the truth from the artifact, never the miner's declaration):
    * params / effective-bits / dtype recomputed from the LOADED state dict, never from
      config.json or the miner's claimed number — a "small" model that is actually the
      full teacher behind a decoy fails here
    * degeneracy probe on outputs — a looping / length-blowup student is a free DoS and
      is disqualified rather than scored

Everything here operates on files or already-produced outputs, so it runs on CPU with no
model download. `inspect_checkpoint` reads only the safetensors HEADER (dtype+shape per
tensor), which is bytes, not weights — cheap and load-free.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_FILES = {
    ".safetensors", ".json", ".txt", ".model", ".tiktoken", ".vocab", ".merges",
}
FORBIDDEN_FILES = {".py", ".bin", ".pt", ".pth", ".pkl", ".ckpt", ".h5", ".msgpack", ".so", ".sh"}
MAX_TOTAL_BYTES = 400 * 1024**3   # 400 GB hard cap (decompression-bomb / smuggle guard)

# bytes per element by safetensors dtype string
_DTYPE_BITS = {
    "F64": 64, "F32": 32, "F16": 16, "BF16": 16, "F8_E4M3": 8, "F8_E5M2": 8,
    "I64": 64, "I32": 32, "I16": 16, "I8": 8, "U8": 8, "BOOL": 8,
    "I4": 4, "U4": 4,
    # unsigned packing widths used by real quantized artifacts (MLX packs weights into U32,
    # GPTQ/AWQ into I32/U16). Their absence made `.get(dt, 16)` fail OPEN: an MLX U32 pack
    # reported 16.0 bpw and sailed through a 16-bit tier, while a GPTQ I32 pack reported 32
    # and was rejected — the gate punished the deployable form and rewarded the container.
    "U16": 16, "U32": 32, "U64": 64,
}


@dataclass
class Inspection:
    ok: bool
    params: int = 0
    serialized_bytes: int = 0
    effective_bits_per_param: float = 0.0
    dtype_hist: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)


def _read_safetensors_header(path: Path) -> dict:
    """Read only the JSON header of a .safetensors file (no tensor data)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > 100 * 1024 * 1024:  # header should be < 100MB
            raise ValueError("implausible safetensors header length")
        return json.loads(f.read(n))


def inspect_checkpoint(ckpt_dir: str | Path) -> Inspection:
    """Fail-closed safety + integrity inspection of an unpacked checkpoint directory.

    Derives params / effective-bits / dtype from the actual tensor headers. Never trusts
    config.json or any declared value.
    """
    d = Path(ckpt_dir)
    reasons: list[str] = []
    if not d.is_dir():
        return Inspection(False, reasons=[f"not a directory: {d}"])

    files = [p for p in d.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    if total > MAX_TOTAL_BYTES:
        reasons.append(f"checkpoint {total/1024**3:.0f}GB exceeds cap {MAX_TOTAL_BYTES/1024**3:.0f}GB")

    st_files = []
    for p in files:
        suf = p.suffix.lower()
        if suf in FORBIDDEN_FILES:
            reasons.append(f"forbidden file (code / pickle): {p.name}")
        elif suf == ".safetensors":
            st_files.append(p)
        elif suf and suf not in ALLOWED_FILES:
            reasons.append(f"file type not on allowlist: {p.name}")

    if not st_files:
        reasons.append("no .safetensors weights found (safetensors-only)")

    # tokenizer must not carry remote code
    for cfg in ("tokenizer_config.json", "config.json"):
        cp = d / cfg
        if cp.exists():
            try:
                j = json.loads(cp.read_text())
                if isinstance(j, dict) and (j.get("auto_map") or j.get("custom_pipelines")):
                    reasons.append(f"{cfg} carries auto_map/custom code (trust_remote_code path)")
            except Exception:
                reasons.append(f"unparseable {cfg}")

    # integrity: recompute params + effective bits from the tensor headers
    params, bits_total, hist = 0, 0, {}
    for p in st_files:
        try:
            hdr = _read_safetensors_header(p)
        except Exception as e:
            reasons.append(f"bad safetensors header {p.name}: {e}")
            continue
        for name, meta in hdr.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            shape = meta.get("shape", [])
            dt = meta.get("dtype", "?")
            n = 1
            for s in shape:
                n *= s
            params += n
            if dt not in _DTYPE_BITS:
                # HARD REJECT on an unknown dtype rather than assuming 16. Guessing here is how
                # an unrecognized packing silently understates its own bit budget.
                reasons.append(f"unknown dtype {dt!r} in {p.name} — cannot verify bit budget")
                continue
            bits = _DTYPE_BITS[dt]
            bits_total += n * bits
            hist[dt] = hist.get(dt, 0) + n

    eff_bits = (bits_total / params) if params else 0.0
    ok = not reasons and params > 0
    return Inspection(ok=ok, params=params, serialized_bytes=total,
                      effective_bits_per_param=round(eff_bits, 3), dtype_hist=hist,
                      reasons=reasons)


@dataclass
class TierBudget:
    name: str
    max_params: int
    max_effective_bits: float = 16.0   # e.g. 4.5 for a 4-bit tier, 2.0 for a 2-bit tier


def tier_gate(insp: Inspection, tier: TierBudget) -> tuple[bool, list[str]]:
    """Does the artifact actually fit the tier it was submitted to? Uses recomputed
    values, so declaring 'sub-1B' while shipping the teacher fails here."""
    r = []
    if not insp.ok:
        return False, insp.reasons
    if insp.params > tier.max_params:
        r.append(f"params {insp.params:,} exceed tier {tier.name} cap {tier.max_params:,}")
    if insp.effective_bits_per_param > tier.max_effective_bits + 1e-6:
        r.append(f"effective bits {insp.effective_bits_per_param} exceed tier cap {tier.max_effective_bits}")
    return (not r), r


def degeneracy_flags(outputs: list[str], max_loop_ratio: float = 0.5,
                     max_len_chars: int = 8000) -> tuple[bool, list[str]]:
    """A looping / runaway student is a DoS and is disqualified, not merely low-scored.

    Flags: high repeated-n-gram ratio, or a large fraction of responses hitting the
    length ceiling on ordinary prompts.
    """
    if not outputs:
        return True, ["no outputs"]
    reasons = []
    long_frac = sum(1 for o in outputs if len(o) >= max_len_chars) / len(outputs)
    if long_frac > 0.5:
        reasons.append(f"{long_frac:.0%} of responses hit the length ceiling (runaway)")

    def loop_ratio(text: str, n: int = 6) -> float:
        toks = text.split()
        if len(toks) < 2 * n:
            return 0.0
        grams = [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        return 1.0 - len(set(grams)) / len(grams)

    loopy = sum(1 for o in outputs if loop_ratio(o) > max_loop_ratio) / len(outputs)
    if loopy > 0.3:
        reasons.append(f"{loopy:.0%} of responses are highly repetitive (looping)")
    return (not reasons), reasons
