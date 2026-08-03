"""Intelligence density — capability per gigabyte, the unit this market is actually competing on.

PrismML report a single number for a single model and keep the method private. The unit is the
smart part of their messaging and it is worth adopting outright: "how much model do you get per
byte" is the question a person deciding what to download is actually asking, and neither raw
capability nor raw size answers it alone.

The spin is that Ralph does not assert a density, it RANKS them. Both halves are already measured
on every submission — retention from observer-KL, and true bits per weight from tensor data rather
than dtype headers — so density is a derived quantity, not a claim. It goes in the record as its
INPUTS (params, code bits, retention) so anyone can recompute it, which is the same rule the rest
of the record follows.

ONE HONEST CAVEAT, and it has to travel with the number. Ralph's capability term is RETENTION
AGAINST THE PINNED PARENT, not an average of absolute benchmarks. So:

  * `retention_per_gb` is comparable ACROSS SUBMISSIONS ON THE SAME PARENT — which is exactly what
    a market needs to rank entries, and what the crown is decided on.
  * it is NOT interchangeable with a published benchmark-average-per-GB figure. A model retaining
    95% of a weak parent is not the same product as one retaining 95% of a strong one, and
    retention says nothing about the parent's absolute score.

Reporting the two as if they were the same number would be the exact dishonest comparison this
project spends its credibility avoiding. When absolute benchmark numbers exist for a crowned
artifact, report BOTH and label which is which.
"""
from __future__ import annotations

from dataclasses import dataclass

BF16_BITS = 16.0


def size_bytes(params: int, bits_per_weight: float) -> float:
    return params * bits_per_weight / 8.0


def size_gb(params: int, bits_per_weight: float) -> float:
    return size_bytes(params, bits_per_weight) / 1e9


def compression_ratio(bits_per_weight: float, baseline_bits: float = BF16_BITS) -> float:
    """How many times smaller than the bf16 parent. The headline ratio everyone quotes."""
    return baseline_bits / bits_per_weight if bits_per_weight > 0 else 0.0


@dataclass
class Density:
    params: int = 0
    code_bits: float = 0.0          # information actually in the weights
    container_bits: float = 0.0     # what you download
    retention: float = 0.0          # observer-KL score vs the pinned parent, 0..1

    @property
    def code_gb(self) -> float:
        return size_gb(self.params, self.code_bits)

    @property
    def download_gb(self) -> float:
        """What a user actually pulls. Rank on the achievement, quote the shipped size — a 1-bit
        model in a 16-bit container is a real compression and an undeployable file."""
        return size_gb(self.params, self.container_bits or self.code_bits)

    @property
    def shrink(self) -> float:
        return compression_ratio(self.container_bits or self.code_bits)

    @property
    def retention_per_gb(self) -> float:
        """THE UNIT. Retention percentage per gigabyte downloaded.

        Uses the CONTAINER size, not the code size: density is a claim about what you get per byte
        you actually store, and the byte you store is the shipped one."""
        gb = self.download_gb
        return (self.retention * 100.0 / gb) if gb > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "params": self.params,
            "code_bits": round(self.code_bits, 4),
            "container_bits": round(self.container_bits, 4),
            "retention": round(self.retention, 6),
            "download_gb": round(self.download_gb, 3),
            "shrink_vs_bf16": round(self.shrink, 2),
            "retention_per_gb": round(self.retention_per_gb, 2),
        }


def from_record(sub, parent_params: int = 0) -> Density:
    """Build a Density from a SubmissionRecord. Everything here is a published field, so the number
    is recomputable by anyone holding the record rather than taken on the operator's word."""
    return Density(
        params=getattr(sub, "params", 0) or parent_params,
        code_bits=getattr(sub, "code_bits", 0.0) or 0.0,
        container_bits=getattr(sub, "container_bits", 0.0) or 0.0,
        retention=getattr(sub, "retention", 0.0) or 0.0,
    )


def rank(subs, parent_params: int = 0) -> list:
    """Rank submissions by density. Returns [(sub, Density)] best first.

    This is the leaderboard PrismML cannot publish, because they have one entry."""
    scored = [(s, from_record(s, parent_params)) for s in subs]
    return sorted(scored, key=lambda p: p[1].retention_per_gb, reverse=True)
