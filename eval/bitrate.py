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
    """(bits_per_weight, codebook_size) for one flat weight tensor.

    Per group, normalize by max-abs to recover the code, then count DISTINCT codes globally.
    Returns bits = log2(k) + scale_bits/group_size, where k is the codebook size. Bails out to
    (inf, max_codebook) once the codebook is clearly dense — that tensor is not quantized and
    the caller falls back to the container width."""
    n = len(values)
    if n == 0:
        return 0.0, 0
    codes: set[int] = set()
    inv_tol = 1.0 / tol
    for start in range(0, n, group_size):
        g = values[start:start + group_size]
        s = max((abs(v) for v in g), default=0.0)
        if s <= 0.0:
            codes.add(0)                       # an all-zero group carries one code
            continue
        inv_s = 1.0 / s
        for v in g:
            codes.add(int(round(v * inv_s * inv_tol)))
            if len(codes) > max_codebook:
                return float("inf"), max_codebook
    k = max(1, len(codes))
    return math.log2(k) + scale_bits / group_size, k


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
