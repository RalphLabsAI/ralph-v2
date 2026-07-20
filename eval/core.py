"""Core types for the SN40 v2 capability-retention harness.

Design invariants (see ../README.md):

  * Items are GENERATED, never loaded from a static file — freshness comes from a
    seed derived from the on-chain commit hash, drawn only AFTER checkpoints are
    committed. Generators and checkers are public; seeds are not.
  * Checkers are DETERMINISTIC. No LLM judge is ever in the scoring path.
  * A model is a black box: give it a prompt, get text back. Nothing about how the
    student was trained is inspected or trusted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Item:
    """One generated eval task.

    `answer` is whatever the axis's checker needs to grade the response — a number,
    a string, a list of unit tests. It never leaves the validator.
    """

    axis: str
    prompt: str
    answer: object
    # provenance so a verdict can be re-derived: which template/params produced it
    meta: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.axis}:{self.meta.get('template', '?')}:{self.meta.get('idx', 0)}"


@dataclass
class ItemResult:
    item: Item
    output: str
    passed: bool


@runtime_checkable
class Axis(Protocol):
    """A capability axis: generates fresh items and grades them deterministically."""

    name: str
    weight: float

    def generate(self, seed: int, n: int, difficulty: int = 1) -> list[Item]: ...

    def check(self, item: Item, output: str) -> bool: ...


@runtime_checkable
class ModelRunner(Protocol):
    """Anything that turns prompts into text.

    Implementations: StubRunner (tests, no GPU), HFRunner (transformers), and later
    a vLLM/server runner. The harness never assumes anything else about the model.
    """

    name: str

    def generate(self, prompts: Sequence[str], max_new_tokens: int = 512) -> list[str]: ...


def run_axis(
    axis: Axis,
    runner: ModelRunner,
    items: Sequence[Item],
    max_new_tokens: int = 512,
) -> list[ItemResult]:
    """Run one model over one axis's items and grade every response."""
    outputs = runner.generate([it.prompt for it in items], max_new_tokens=max_new_tokens)
    if len(outputs) != len(items):
        raise ValueError(f"runner returned {len(outputs)} outputs for {len(items)} items")
    return [ItemResult(item=it, output=out, passed=bool(axis.check(it, out)))
            for it, out in zip(items, outputs)]
