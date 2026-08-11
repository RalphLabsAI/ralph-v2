"""True bits-per-weight of a submitted artifact — the measurement a quantization crown needs.

WHY THE OBVIOUS MEASUREMENT IS WRONG. `gates.inspect_checkpoint` derives bits from safetensors
DTYPE headers. That is right for "is this miner smuggling a bigger model", and wrong for "how
compressed is this really", in both directions:

  * PrismML publish `Bonsai-27B-unpacked` — a genuinely 1.125-bit model stored in a BF16
    container. Header arithmetic calls it 16.0 bpw. It is not; the tensors hold two distinct
    values per group of 128.
  * Conversely the field routinely ships "2-bit" builds that are really 2.8-3.0 bpw once the
    higher-precision escape hatches (first layers, `down_proj`, LM head) are counted. Trusting
    the advertised quant name pays for a label.

So this module measures the INFORMATION actually present in the weights, reproducing the
arithmetic PrismML publish for their own format:

    binary  : 1 code bit          + 16/128 scale = 1.125 bpw
    ternary : log2(3) = 1.58496   + 16/128       = 1.70996 bpw

Method (deterministic, CPU-only, no torch): per group of G weights, divide by the group's
max-abs to recover the code, round to a tolerance, and collect the GLOBAL codebook. A genuinely
ternary tensor yields {-1, 0, +1} in every group -> k=3 -> log2(3)+16/G. A real FP16 tensor
yields a distinct code almost everywhere -> log2(k) explodes -> we cap at the container width.
Both numbers are reported, because they answer different questions and the gap between them is
itself informative (it is exactly the ideal-vs-shipped gap PrismML's marketing elides):

    code_bits      what the miner ACHIEVED   -> the compression axis, what the crown pays for
    container_bits what the miner SHIPPED    -> bytes actually downloaded and run

Rules that follow, and that the tier gate enforces:
  * Rank on `code_bits` (achievement), gate on BOTH (a 1.1-bit model in a 16-bit container is a
    real compression but an unshippable artifact — it must still declare its container).
  * Measure over ALL weight tensors including embeddings and the LM head. Excluding them is the
    "higher-precision escape hatch" that makes sub-2-bit claims meaningless.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# a code is "the same" if it matches to this tolerance after group normalization. Loose enough
# to survive bf16 round-trip of an exactly-ternary tensor, tight enough that genuinely distinct
# fp16 weights do not collapse into one code.
CODE_TOL = 1e-3


@dataclass
class BitReport:
    params: int = 0
    code_bits: float = 0.0          # information actually in the weights (the achievement)
    container_bits: float = 0.0     # bits per weight as stored/shipped (what you download)
    codebook: int = 0               # distinct codes found (2 = binary, 3 = ternary)
    group_size: int = 128
    per_tensor: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def compression_x(self) -> float:
        """vs an FP16 baseline, on the achieved code bits."""
        return 16.0 / self.code_bits if self.code_bits > 0 else 0.0


def tensor_code_bits(values, group_size: int = 128, scale_bits: int = 16,
                     tol: float = CODE_TOL, max_codebook: int = 4096) -> tuple[float, int]:
    """(bits_per_weight, mean_levels_per_group) for one flat weight tensor.

    Counts DISTINCT LEVELS WITHIN EACH GROUP, not a global codebook. That distinction is
    load-bearing: a global codebook normalized by group max-abs only works for SYMMETRIC
    quantization (one scale per group). Every mainstream 4-bit scheme — GPTQ, AWQ, NF4,
    bitsandbytes — is AFFINE, carrying a per-group zero-point as well, so the normalized codes
    differ from group to group and the global set explodes. Measured: affine int4 read 9.06 bpw
    with a codebook of 490 instead of ~4.25, i.e. an honest 4-bit artifact would have been
    rejected from the 4-bit tier. Per-group counting is invariant to ANY per-group affine
    transform, so symmetric and affine schemes both measure correctly.

    Scale overhead is inferred per group: a level set spanning zero symmetrically implies one
    scale; an asymmetric one implies scale + zero-point (two). Dense tensors saturate at
    group_size levels and the caller falls back to the container width."""
    n = len(values)
    if n == 0:
        return 0.0, 0
    inv_tol = 1.0 / tol
    tot_bits, tot_w, tot_levels, n_groups = 0.0, 0, 0, 0
    for start in range(0, n, group_size):
        g = values[start:start + group_size]
        w = len(g)
        s = max((abs(v) for v in g), default=0.0)
        if s <= 0.0:
            levels, sym = {0}, True             # an all-zero group carries one level
        else:
            inv_s = 1.0 / s
            levels = {int(round(v * inv_s * inv_tol)) for v in g}
            lo, hi = min(levels), max(levels)
            sym = abs(lo + hi) <= max(1, int(0.05 * inv_tol))   # spans zero symmetrically
        k = max(1, len(levels))
        n_scales = 1 if sym else 2              # symmetric: scale. affine: scale + zero-point.
        tot_bits += (math.log2(k) + n_scales * scale_bits / group_size) * w
        tot_w += w
        tot_levels += k
        n_groups += 1
    return tot_bits / tot_w, (tot_levels // max(1, n_groups))


def measure_state_dict(tensors: dict, group_size: int = 128, scale_bits: int = 16,
                       container_bits_of=None) -> BitReport:
    """`tensors`: name -> (flat sequence of floats, container_bits_for_that_tensor).

    Weight-bearing tensors only — norms/biases/scales are the "negligible tail" every scheme
    keeps in higher precision; they are counted in the container total but never allowed to
    define the code width."""
    rep = BitReport(group_size=group_size)
    total_code, total_container, total_n = 0.0, 0.0, 0
    all_codes = 0
    for name, item in sorted(tensors.items()):
        vals, cbits = item if isinstance(item, tuple) else (item, 16)
        n = len(vals)
        if n == 0:
            continue
        bits, k = tensor_code_bits(vals, group_size, scale_bits)
        if not math.isfinite(bits) or bits > cbits:
            bits = float(cbits)                # not quantized (or worse than its container)
        rep.per_tensor[name] = {"n": n, "code_bits": round(bits, 4), "codebook": k,
                                "container_bits": cbits}
        all_codes = max(all_codes, k)
        total_code += bits * n
        total_container += cbits * n
        total_n += n
    if total_n:
        rep.params = total_n
        rep.code_bits = round(total_code / total_n, 4)
        rep.container_bits = round(total_container / total_n, 4)
        rep.codebook = all_codes
    return rep


# ---- the producer: read real checkpoint bytes -------------------------------------------
# Without this, everything above is library code with no input. Pure struct/mmap, no torch and
# no safetensors dependency, mirroring gates._read_safetensors_header — the validator must be
# able to measure an artifact on a CPU box with nothing installed.

_FLOAT_DTYPES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}
# name fragments for tensors that are NOT weights: the "negligible tail" every scheme keeps in
# higher precision. Counted in the container total, never allowed to define the code width.
_NON_WEIGHT = ("norm", "bias", "scale", "zero_point", "zeros", "g_idx", "embed_positions")


def _decode(buf: bytes, dt: str) -> list[float]:
    import struct
    if dt == "F32":
        return list(struct.unpack(f"<{len(buf)//4}f", buf[:len(buf)//4*4]))
    if dt == "F64":
        return list(struct.unpack(f"<{len(buf)//8}d", buf[:len(buf)//8*8]))
    if dt == "F16":
        return list(struct.unpack(f"<{len(buf)//2}e", buf[:len(buf)//2*2]))
    if dt == "BF16":
        n = len(buf) // 2
        hi = struct.unpack(f"<{n}H", buf[:n * 2])
        # bf16 is the top 16 bits of an f32
        return list(struct.unpack(f"<{n}f", struct.pack(f"<{n}I", *(h << 16 for h in hi))))
    return []


def measure_checkpoint(ckpt_dir, group_size: int = 128, max_groups_per_tensor: int = 64,
                       max_tensors: int = 256) -> BitReport:
    """Measure a real checkpoint directory. Deterministic and bounded.

    Sampling is by whole GROUPS at evenly spaced offsets — never random elements, because level
    counting needs intact groups, and never the whole file, because a 27B artifact is 54 GB.
    Even spacing keeps it reproducible: an auditor measuring the same bytes gets the same
    number, which is the property that lets a bit budget be part of a published verdict.

    Integer dtypes are ALREADY-PACKED codes whose meaning depends on a quantization scheme we
    would have to trust the miner's config to interpret. For those the shipped container IS the
    achievement, so they contribute their container width and are not level-counted."""
    import json
    import struct
    from pathlib import Path

    rep = BitReport(group_size=group_size)
    # GGUF FIRST, because it was invisible here. This function globbed *.safetensors only, so a
    # GGUF artifact measured params=0 and the bit-tier gate failed closed with "no weights
    # measured" — safe, but it meant NO GGUF SUBMISSION COULD EVER BE ACCEPTED, while the README
    # advertises GGUF, fetch.py allows it, and parent_compat goes out of its way to support it.
    # Found by quantizing the real parent and running it through the real gates.
    gg = sorted(Path(ckpt_dir).glob("*.gguf"))
    if gg:
        return measure_gguf_dir(gg, group_size=group_size)
    files = sorted(Path(ckpt_dir).glob("*.safetensors"))
    tot_code, tot_container, tot_n, max_levels = 0.0, 0.0, 0, 0
    seen = 0
    for f in files:
        with open(f, "rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            hdr = json.loads(fh.read(hlen))
            base = 8 + hlen
            for name, meta in sorted(hdr.items()):
                if name == "__metadata__" or not isinstance(meta, dict) or seen >= max_tensors:
                    continue
                dt, shape = meta.get("dtype", "?"), meta.get("shape", [])
                if len(shape) < 2:
                    continue                              # 1-D: norms/biases, not weights
                if any(t in name.lower() for t in _NON_WEIGHT):
                    continue
                n = 1
                for s in shape:
                    n *= s
                if n < group_size:
                    continue
                off = meta.get("data_offsets", [0, 0])
                cbits = (off[1] - off[0]) * 8.0 / n if n else 0.0
                if dt in _FLOAT_DTYPES:
                    w = _FLOAT_DTYPES[dt]
                    n_groups = min(max_groups_per_tensor, n // group_size)
                    stride = max(1, (n // group_size) // max(1, n_groups))
                    vals: list[float] = []
                    for gi in range(n_groups):
                        start = base + off[0] + gi * stride * group_size * w
                        fh.seek(start)
                        vals += _decode(fh.read(group_size * w), dt)
                    bits, levels = tensor_code_bits(vals, group_size) if vals else (cbits, 0)
                    # SATURATED = not quantized. Level counting can only observe group_size
                    # distinct values, so a dense fp16 tensor bottoms out at log2(128)=7 and
                    # would otherwise claim to be a 7-bit achievement. If the groups are
                    # essentially all-distinct, the container IS the truth.
                    if levels >= 0.9 * group_size:
                        bits = cbits
                    bits = min(bits, cbits)               # never claim less than you ship
                    max_levels = max(max_levels, levels)
                else:
                    bits = cbits                          # packed codes: container is the truth
                rep.per_tensor[name] = {"n": n, "dtype": dt, "code_bits": round(bits, 4),
                                        "container_bits": round(cbits, 4)}
                tot_code += bits * n
                tot_container += cbits * n
                tot_n += n
                seen += 1
    if tot_n:
        rep.params = tot_n
        rep.code_bits = round(tot_code / tot_n, 4)
        rep.container_bits = round(tot_container / tot_n, 4)
        rep.codebook = max_levels
    else:
        rep.notes.append("no weight tensors measured")
    return rep


from .gguf import code_bits as _gguf_code   # ONE definition of the code width


def measure_gguf_dir(paths, group_size: int = 128) -> BitReport:
    """Exact bit accounting for a GGUF artifact, straight out of `read_gguf`'s own walk.

    No sampling and no second parser: `eval.gguf` already visits every tensor to compute container
    bits, so it computes the code bits in the same pass and this just reports them. Writing the
    walk again here is how two modules end up disagreeing about one artifact's bit budget."""
    from .gguf import read_gguf
    rep = BitReport(group_size=group_size)
    tot_code = tot_container = 0.0
    tot_n = 0
    widths = set()
    for path in paths:
        info = read_gguf(path)
        if not info.ok:
            rep.notes.extend(info.reasons[:2])
            continue
        tot_code += info.code_bits_per_weight * info.params
        tot_container += info.bits_per_weight * info.params
        tot_n += info.params
        for tname, n in (info.weight_type_hist or {}).items():
            rep.per_tensor[tname] = rep.per_tensor.get(tname, 0) + n
            widths.add(int(_gguf_code(tname)))
    if tot_n:
        rep.params = tot_n
        rep.code_bits = round(tot_code / tot_n, 4)
        rep.container_bits = round(tot_container / tot_n, 4)
        # the NARROWEST code present is what the artifact can honestly claim: a model with one
        # 2-bit tensor and the rest at 8 is not a 2-bit model, and the mean above says so, but
        # the codebook field should not overstate it either
        rep.codebook = 2 ** min(widths) if widths else 0
    else:
        rep.notes.append("no weight tensors measured")
    return rep


# ---- bit tiers: rank on what was ACHIEVED, gate on what was SHIPPED ---------------------

@dataclass(frozen=True)
class BitTier:
    """A compression tier. `max_code_bits` is the real budget (PrismML's own operating points:
    1.125 binary, 1.71 ternary); `max_container_bits` stops a 1-bit model being shipped as an
    undeployable 16-bit file and still winning the small-model tier."""
    name: str
    max_code_bits: float
    max_container_bits: float = 4.5


TIERS = (
    BitTier("binary", max_code_bits=1.15, max_container_bits=2.5),    # Bonsai 1-bit = 1.125
    BitTier("ternary", max_code_bits=1.75, max_container_bits=2.5),   # Bonsai ternary = 1.71
    BitTier("sub2", max_code_bits=2.0, max_container_bits=3.0),
    BitTier("sub4", max_code_bits=4.0, max_container_bits=5.0),
)

# Share of emission per tier. NOT UNIFORM, because the tiers are not equally hard and miners price
# effort correctly: the first two rounds drew 9 sub4 and 6 ternary submissions and nothing at all in
# binary or sub2, which is exactly what an equal split should be expected to produce. Sub-2-bit is a
# research problem; 4-bit is one `llama-quantize` invocation, and the sub4 crown came in at 4.61 GB
# against a 5.0 GB cap — a stock Q4_K_M, of which thousands already exist.
#
# The numbers below are a judgement, not a measurement. What they encode: binary and ternary are
# where an artifact small enough to run on a phone lives (2.5 container bits x 8.19B params = 2.56
# GB), and that is the only place a compression subnet can produce something nobody can already
# download. Paired with `Tournament.weights` no longer redistributing an empty tier's share, this
# makes an unentered binary tier a standing bounty rather than a bonus for whoever entered sub4.
TIER_EMISSION_WEIGHT = {
    "binary": 0.40,
    "ternary": 0.30,
    "sub2": 0.20,
    "sub4": 0.10,
}


def emission_weight(tier_name: str, n_tiers: int = 4) -> float:
    """Emission share for a tier, falling back to an equal split for tiers not in the table
    (`rehearsal`, `open`, and the simulation tiers, which run one tier at a time anyway)."""
    return TIER_EMISSION_WEIGHT.get(tier_name, 1.0 / max(1, n_tiers))


def bit_tier_gate(rep: BitReport, tier: BitTier) -> tuple[bool, list[str]]:
    """Fail-closed. Both budgets bind: the achievement AND the artifact."""
    r: list[str] = []
    if rep.params <= 0:
        r.append("no weights measured")
    if rep.code_bits > tier.max_code_bits + 1e-6:
        r.append(f"code bits {rep.code_bits} exceed {tier.name} budget {tier.max_code_bits}")
    if rep.container_bits > tier.max_container_bits + 1e-6:
        r.append(f"container bits {rep.container_bits} exceed {tier.name} cap "
                 f"{tier.max_container_bits} (unshippable artifact)")
    return (not r), r
