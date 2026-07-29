"""Trajectory sources and step extraction — the input side of observer-KL scoring.

The crown needs (prefix K, step K->K+1) pairs, which raw web text cannot provide: fineweb has
no step boundaries. This module holds VERIFIED trajectory sources and the boundary rule for each.

SCALE IS THE ANTI-OVERFIT LEVER HERE (const: "just have a large enough dataset"), and it works
for a reason specific to this mechanism: the pre-fit attack is "memorize the teacher's step at
every K", which requires running the teacher across the corpus and training the student to match
— that IS distillation. The cheat and the job are the same activity. But that argument only
holds if the corpus is large in UNIQUE CONTENT, which is why the dedup rule below is not
optional.

TWO FAILURE MODES THAT PRODUCE SILENT GARBAGE, both found while verifying:

  1. ROW COUNT != UNIQUE CONTENT. SWE-ZERO advertises 12.29M rows, but sampling 160 rows across
     8 offsets spanning the whole set returned only 10 distinct `instance_id` — contiguous
     blocks are ~100 rollouts of the SAME pull request, over 122,908 unique tasks. Sampling rows
     uniformly means a miner who memorizes 122,908 PRs has memorized what looks like a 12M-row
     corpus. `dedup_key` + `max_per_key` is what makes the scale claim honest.
  2. A DELIMITER THAT DOES NOT EXIST. A `<think>`-only parser returns ZERO steps on
     OpenThoughts-114k, which uses `<|begin_of_thought|>`. An empty corpus is the default
     failure mode of a wrong field/delimiter guess, and it looks exactly like "the source is
     quiet today". `verify_source` exists to make that loud, and callers must run it before
     ingest.

Every entry below was verified live against datasets-server (row counts, EXACT field names,
licence, gating). All are ungated and apache-2.0 / mit / cc-by-4.0 / odc-by, so a permissionless
validator can pull them with no token and no click-through.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


@dataclass(frozen=True)
class StepSpec:
    """How to recover step boundaries from one source.

    kind:
      "message"   — text_field is a LIST of role/content structs; a step is one assistant turn.
                    Unambiguous: the boundary is a list index, contestable by nobody.
      "tool_call" — a step is one <tool_call>..</tool_call> plus its ```output execution result.
                    Also unambiguous, and the observable at the boundary is a real interpreter
                    result, so K->K+1 is objectively checkable as well as distributionally.
      "para"      — a step is one blank-line block inside <think>..</think>. HEURISTIC: keep it
                    off the differentiating crown axis and use it as scale ballast.
      "none"      — no step structure. Script/language coverage only, K=0.
    """
    kind: str
    msg_keys: tuple = ()
    role_key: str = "role"
    content_key: str = "content"
    wrap: tuple = (THINK_OPEN, THINK_CLOSE)
    in_role: str = "assistant"


# VERIFIED live 2026-07-29. Do not add an entry without reading its real schema — a guessed
# field name yields an empty corpus that is indistinguishable from a quiet source.
TRAJECTORY_SOURCES: dict[str, dict] = {
    # --- HARD boundaries: use these on the crown path ---
    "omr_tir": {
        "dataset": "nvidia/OpenMathReasoning", "split": "tir", "config": "default",
        "text_field": "generated_solution", "pool_rows": 1_718_466, "license": "cc-by-4.0",
        "step": StepSpec(kind="tool_call"),
    },
    "nemotron_tools": {
        "dataset": "nvidia/Nemotron-Post-Training-Dataset-v1", "split": "tool_calling",
        "config": "default", "text_field": "messages", "pool_rows": 310_051,
        "license": "cc-by-4.0",
        "step": StepSpec(kind="message", msg_keys=("role", "content", "tool_calls")),
    },
    "mini_coder": {
        "dataset": "ricdomolm/mini-coder-trajs-400k", "split": "train", "config": "default",
        "text_field": "messages", "pool_rows": 396_515, "license": "mit",
        "step": StepSpec(kind="message", msg_keys=("role", "content")),
    },
    "swe_zero": {
        "dataset": "AlienKevin/SWE-ZERO-12M-trajectories", "split": "train", "config": "default",
        "text_field": "messages", "pool_rows": 12_290_800, "license": "apache-2.0",
        # MANDATORY: 12.29M rows are only 122,908 unique PRs (~100 rollouts each). Without this
        # the corpus is ~100x smaller than it looks and the scale argument collapses.
        "dedup_key": "instance_id", "max_per_key": 5,
        "step": StepSpec(kind="message", msg_keys=("role", "content")),
    },
    "agenttrove": {
        "dataset": "open-thoughts/AgentTrove", "split": "train", "config": "default",
        "text_field": "conversations", "pool_rows": 1_696_847, "license": "apache-2.0",
        "step": StepSpec(kind="message", msg_keys=("content", "role")),
    },
    # --- SOFT boundaries: scale ballast, not the differentiating axis ---
    "omr_cot": {
        "dataset": "nvidia/OpenMathReasoning", "split": "cot", "config": "default",
        "text_field": "generated_solution", "pool_rows": 3_201_061, "license": "cc-by-4.0",
        "step": StepSpec(kind="para"),
    },
    "glaive_r1": {
        "dataset": "glaiveai/reasoning-v1-20m", "split": "train", "config": "default",
        "text_field": "response", "pool_rows": 22_199_375, "license": "apache-2.0",
        "step": StepSpec(kind="para"),
    },
    # --- non-Latin: the axis a cloned artifact loses on ---
    "samvaad_hi": {
        "dataset": "sarvamai/samvaad-hi-v1", "split": "train", "config": "default",
        "text_field": "messages", "pool_rows": 101_476, "license": "apache-2.0",
        "step": StepSpec(kind="message", msg_keys=("content", "role")),
    },
    "zh_reasoning": {
        "dataset": "zake7749/OpenScience-Chinese-Reasoning-SFT", "split": "train",
        "config": "default", "text_field": "messages", "pool_rows": 50_000,
        "license": "cc-by-4.0",
        "step": StepSpec(kind="para", msg_keys=("content", "role")),
    },
}

# Rejected after verification, recorded so nobody re-adds them:
#   lmsys/lmsys-chat-1m      GATED, and its licence forbids transferring the dataset to a third
#                            party — every validator fetching the same corpus IS that transfer.
#   teknium/OpenHermes-2.5   no licence declared at all, and 100% single-turn.
#   allenai/tulu-3-sft       ~1.03 assistant turns/row: effectively single-turn, no prefix depth.
#   OpenThoughts-114k        uses <|begin_of_thought|>, so a <think> parser silently yields 0.
#   SWE-bench / SWE-smith / SWE-rebench / R2E-Gym
#                            TASK INSTANCES, not trajectories — no messages column exists. The
#                            single most likely thing to be mistaken for step data.

_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)


@dataclass
class Trajectory:
    """One trajectory reduced to the (prefix, step) pairs the scorer consumes."""
    id: str = ""
    source: str = ""
    prefix: str = ""                 # K
    step: str = ""                   # the teacher's K -> K+1
    index: int = 0                   # which step this is
    meta: dict = field(default_factory=dict)


def _msg_text(m, spec: StepSpec) -> tuple[str, str]:
    if not isinstance(m, dict):
        return "", ""
    return str(m.get(spec.role_key, "")), str(m.get(spec.content_key, "") or "")


def extract_steps(row: dict, spec: StepSpec, text_field: str, source: str = "",
                  row_id: str = "", max_steps: int = 8) -> list[Trajectory]:
    """(prefix, step) pairs from one row. Empty list when the row carries no usable boundary —
    callers must treat a source that returns all-empty as a configuration error, not a quiet day."""
    body = row.get(text_field)
    out: list[Trajectory] = []

    if spec.kind == "message":
        if not isinstance(body, list):
            return []
        rendered: list[str] = []
        for i, m in enumerate(body):
            role, content = _msg_text(m, spec)
            if role == spec.in_role and content.strip() and rendered:
                # prefix is everything BEFORE this turn; the step is this turn
                out.append(Trajectory(id=f"{row_id}#{i}", source=source,
                                      prefix="\n".join(rendered), step=content, index=i,
                                      meta={"role": role, "n_msgs": len(body)}))
                if len(out) >= max_steps:
                    break
            rendered.append(f"{role}: {content}" if role else content)
        return out

    if spec.kind == "tool_call":
        text = str(body or "")
        calls = list(_TOOL_CALL.finditer(text))
        for i, m in enumerate(calls[:max_steps]):
            out.append(Trajectory(id=f"{row_id}#{i}", source=source,
                                  prefix=text[:m.start()], step=m.group(0), index=i,
                                  meta={"n_calls": len(calls)}))
        return out

    if spec.kind == "para":
        text = str(body or "")
        if isinstance(body, list):                     # messages column: take the assistant turn
            for m in body:
                role, content = _msg_text(m, spec)
                if role == spec.in_role and spec.wrap[0] in content:
                    text = content
                    break
        o, c = spec.wrap
        if o not in text or c not in text:
            return []                                   # wrong delimiter for this source
        inner = text.split(o, 1)[1].split(c, 1)[0]
        blocks = [b.strip() for b in inner.split("\n\n") if b.strip()]
        for i in range(1, min(len(blocks), max_steps + 1)):
            out.append(Trajectory(id=f"{row_id}#{i}", source=source,
                                  prefix=o + "\n\n".join(blocks[:i]), step=blocks[i], index=i,
                                  meta={"n_blocks": len(blocks)}))
        return out

    return []                                           # kind == "none": no step structure


def dedup_rows(rows, key: str, max_per_key: int):
    """Cap rollouts per unique task. Without this, SWE-ZERO's 12.29M rows are 122,908 tasks at
    ~100 rollouts each, and 'we have 12M trajectories' is false by two orders of magnitude."""
    seen: dict = {}
    for r in rows:
        k = str(r.get(key, ""))
        if seen.get(k, 0) >= max_per_key:
            continue
        seen[k] = seen.get(k, 0) + 1
        yield r


def verify_source(name: str, rows: list[dict]) -> tuple[bool, str]:
    """Assert a source actually yields steps. Run this before ingest, every round.

    A wrong field name or delimiter produces an EMPTY corpus, which is indistinguishable from a
    quiet source and will silently shrink the eval instead of failing it."""
    spec_d = TRAJECTORY_SOURCES.get(name)
    if spec_d is None:
        return False, f"unknown source {name!r}"
    n = sum(len(extract_steps(r, spec_d["step"], spec_d["text_field"], name)) for r in rows)
    if n == 0:
        return False, (f"{name}: extracted 0 steps from {len(rows)} rows — wrong field "
                       f"({spec_d['text_field']!r}) or wrong delimiter for kind "
                       f"{spec_d['step'].kind!r}")
    return True, f"{name}: {n} steps from {len(rows)} rows"


def pool_rows(names=None) -> int:
    names = names or list(TRAJECTORY_SOURCES)
    return sum(TRAJECTORY_SOURCES[n]["pool_rows"] for n in names)
