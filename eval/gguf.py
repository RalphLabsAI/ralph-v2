"""GGUF header reader — accept the artifact the product actually is.

WHY. The subnet's deliverable is a checkpoint someone can run on a phone, and the format that
ships is GGUF: PrismML's Bonsai releases are `Q1_0` (1.125 bpw) and `Q2_0` (ternary), and the
1-bit format is merged upstream in llama.cpp. Intake was safetensors-only, so the only sub-2-bit
form that survived the pipeline was PrismML's own BF16 `-unpacked` container — which is exactly
the free clone vector. The subnet structurally could not receive its own product.

WHY THIS IS SAFE. We read the HEADER only, in pure Python: magic, version, counts, the metadata
key/value block, and the per-tensor (name, shape, ggml_type, offset) table. No llama.cpp, no
native parser, no weight data touched — the same discipline as gates._read_safetensors_header.
Running such a model still needs a runtime and must go through the sandbox; that is a separate
concern from admitting it.

WHY IT IS BETTER THAN THE SAFETENSORS PATH. GGUF states the quantization type PER TENSOR, and
each ggml type has a fixed (block_size, bytes_per_block). So true bits-per-weight is exact
arithmetic rather than inference from the values — no sampling, no level counting, no
affine-vs-symmetric guesswork. `Q2_0` reports 2.0625 and `TQ1_0` 1.6875 because that is what the
format costs, which is also why PrismML's shipped ternary is 2.125 bpw against a marketed 1.71.

Unknown types are a HARD REJECT. Guessing a width is how an unrecognized packing silently
understates its own budget — the same failure the safetensors dtype table had.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

MAGIC = b"GGUF"

# ggml type -> (block_size, bytes_per_block). bits/weight = bytes_per_block * 8 / block_size.
# Only types whose layout is fixed and known are listed; anything else is rejected rather than
# assumed. Sources: ggml type table (llama.cpp), which is also where the Bonsai formats live.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0:  ("F32",     1, 4),
    1:  ("F16",     1, 2),
    2:  ("Q4_0",   32, 18),
    3:  ("Q4_1",   32, 20),
    6:  ("Q5_0",   32, 22),
    7:  ("Q5_1",   32, 24),
    8:  ("Q8_0",   32, 34),
    9:  ("Q8_1",   32, 36),
    10: ("Q2_K",  256, 82),
    11: ("Q3_K",  256, 110),
    12: ("Q4_K",  256, 144),
    13: ("Q5_K",  256, 176),
    14: ("Q6_K",  256, 210),
    15: ("Q8_K",  256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS",  256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S",   256, 50),
    20: ("IQ4_NL",   32, 18),
    21: ("IQ3_S",   256, 110),
    22: ("IQ2_S",   256, 82),
    23: ("IQ4_XS",  256, 136),
    24: ("I8",       1, 1),
    25: ("I16",      1, 2),
    26: ("I32",      1, 4),
    27: ("I64",      1, 8),
    28: ("F64",      1, 8),
    29: ("IQ1_M",  256, 56),
    30: ("BF16",     1, 2),
    34: ("TQ1_0",  256, 54),      # ternary packing, 1.6875 bpw
    35: ("TQ2_0",  256, 66),      # ternary packing, 2.0625 bpw
    # THE TWO LOW-BIT FORMATS THAT POST-DATE THE REST OF THIS TABLE. Without them `read_gguf`
    # returns None for the type and the artifact is refused — so the sub2 tier could not have
    # accepted a submission in the one format it exists for. Ids and block layouts taken from
    # llama.cpp's own `ggml.h` / `ggml-common.h`, not inferred from the names:
    #   block_q1_0 = ggml_half + QK1_0/8 bytes over QK1_0=128 elems -> 18*8/128 = 1.125 bpw
    #   block_q2_0 = ggml_half + QK2_0/4 bytes over QK2_0=64  elems -> 18*8/64  = 2.25  bpw
    41: ("Q1_0",   128, 18),      # 1.125 bpw — PrismML's format; needs THEIR fork to run
    42: ("Q2_0",    64, 18),      # 2.25 bpw — mainline llama.cpp + Metal. The phone-native one.
}

_NON_WEIGHT = ("norm", "bias", "_scale", "rope_freqs")


def type_bits(ggml_type: int) -> float | None:
    """CONTAINER bits per weight: the whole block, scales and all."""
    t = GGML_TYPES.get(ggml_type)
    return None if t is None else t[2] * 8.0 / t[1]


def code_bits(type_name: str) -> float:
    """ACHIEVEMENT bits per weight: the index width the format name asserts.

    `Q4_K` -> 4, `IQ2_XXS` -> 2, `F16` -> 16. A block holds N packed indices plus scales and mins;
    the indices are the information that survived quantization, and the rest is container overhead.
    That is exactly the code/container split the tiers are built on — and unlike safetensors,
    nothing needs sampling or level-counting, because the block layout is fixed and published, so
    an auditor recomputing this cannot disagree. An unrecognised name returns 0 and the caller
    refuses the artifact rather than guessing a budget."""
    import re
    n = (type_name or "").upper()
    if n.startswith("BF"):
        return float(n[2:] or 0)
    if n.startswith("F") and n[1:].isdigit():
        return float(n[1:])
    # Any letter prefix before the Q: Q4_K, IQ2_XXS, and TQ2_0 — the TERNARY format, which is
    # this subnet's headline tier and which an `^I?Q` pattern silently failed to match.
    m = re.match(r"^[A-Z]*Q(\d+)", n)
    return float(m.group(1)) if m else 0.0


@dataclass
class GGUFInfo:
    ok: bool = False
    version: int = 0
    n_tensors: int = 0
    params: int = 0              # weight-bearing elements
    bits_per_weight: float = 0.0  # CONTAINER bits/weight — what you download, exact
    # ACHIEVEMENT bits/weight: the per-weight index width, excluding the block's scales and mins.
    # The tiers gate on the container and RANK on this, so both have to come out of the same walk
    # — deriving one of them in another module means two parsers that can disagree about the same
    # artifact, and the bit budget is the number the whole tier system rests on.
    code_bits_per_weight: float = 0.0
    weight_type_hist: dict = field(default_factory=dict)   # weight-bearing elements per ggml type
    file_bytes: int = 0
    arch: str = ""
    type_hist: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)


def _read_str(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    if n > 1 << 20:
        raise ValueError("absurd string length in header")
    return f.read(n).decode("utf-8", "replace")


def _skip_value(f, vtype: int) -> object:
    """Advance past one metadata value, returning it for the few scalar types we care about."""
    simple = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
              7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    if vtype in simple:
        fmt = simple[vtype]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if vtype == 8:                                   # string
        return _read_str(f)
    if vtype == 9:                                   # array
        (etype,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            _skip_value(f, etype)
        return None
    raise ValueError(f"unknown metadata value type {vtype}")


def read_gguf(path) -> GGUFInfo:
    """Parse a GGUF header. Never reads tensor data."""
    from pathlib import Path
    p = Path(path)
    info = GGUFInfo(file_bytes=p.stat().st_size)
    try:
        with open(p, "rb") as f:
            if f.read(4) != MAGIC:
                info.reasons.append("not a GGUF file (bad magic)")
                return info
            info.version, = struct.unpack("<I", f.read(4))
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            if n_tensors > 1 << 20 or n_kv > 1 << 20:
                info.reasons.append("absurd tensor/metadata count")
                return info
            info.n_tensors = n_tensors
            for _ in range(n_kv):
                key = _read_str(f)
                (vtype,) = struct.unpack("<I", f.read(4))
                val = _skip_value(f, vtype)
                if key == "general.architecture" and isinstance(val, str):
                    info.arch = val

            bits_total, code_total, params = 0.0, 0.0, 0
            for _ in range(n_tensors):
                name = _read_str(f)
                (ndim,) = struct.unpack("<I", f.read(4))
                if ndim > 8:
                    info.reasons.append(f"absurd tensor rank in {name}")
                    return info
                dims = struct.unpack(f"<{ndim}Q", f.read(8 * ndim))
                (tt,) = struct.unpack("<I", f.read(4))
                struct.unpack("<Q", f.read(8))            # data offset, unused here
                n = 1
                for d in dims:
                    n *= d
                tname = GGML_TYPES.get(tt, (f"UNKNOWN_{tt}", 0, 0))[0]
                info.type_hist[tname] = info.type_hist.get(tname, 0) + n
                if ndim < 2 or any(t in name.lower() for t in _NON_WEIGHT):
                    continue                              # the negligible tail, as elsewhere
                b = type_bits(tt)
                if b is None:
                    # HARD REJECT: an unrecognized packing whose width we would have to guess.
                    info.reasons.append(f"unknown ggml type {tt} in {name}")
                    return info
                c = code_bits(tname)
                if c <= 0:
                    info.reasons.append(f"no code width for ggml type {tname!r} in {name}")
                    return info
                bits_total += b * n
                code_total += c * n
                info.weight_type_hist[tname] = info.weight_type_hist.get(tname, 0) + n
                params += n
            if params == 0:
                info.reasons.append("no weight tensors found")
                return info
            info.params = params
            info.bits_per_weight = round(bits_total / params, 4)
            info.code_bits_per_weight = round(code_total / params, 4)
            info.ok = True
    except Exception as e:                                # malformed header -> fail closed
        info.reasons.append(f"gguf parse error: {e}")
    return info
