"""Experience pile — load agentic trajectories into Rollouts with NATIVE segmentation.

The key architectural fact: in teacher-state scoring the pile supplies only STATES
(prefixes 0..K); GLM regenerates its genuine next step fresh at scoring time. So the
trajectories can be authored by ANY model — the pile need not be GLM-generated — and
agentic traces bring their own step boundaries (one assistant turn = one step, its
following environment observation folded in). That sidesteps the CoT-segmentation
failure that pinned retention to ~0 on the first real run: agentic turns are already
semantic steps, no chunker needed.

Unresolved / failed trajectories are kept on purpose — they are the "degraded states"
that exercise recovery (and feed the provenance gate's need for >=25 states where the
pinned teacher itself fails).

Primary launch source: nebius/SWE-agent-trajectories (CC-BY-4.0, 80k traj, resolved +
unresolved labeled). Loaders are format-specific; each turns one raw record into a
Rollout(context, steps, success).
"""
from __future__ import annotations

from typing import Iterable

from .trajectory import Rollout


def _fold_observation(step_text: str, obs: str | None, max_obs: int = 1500) -> str:
    """Attach the environment observation that followed an action to that step, so
    prefix(k) reconstructs the real state the next action was taken from."""
    if not obs:
        return step_text
    return f"{step_text}\n[observation]\n{obs[:max_obs]}"


def rollouts_from_role_turns(record: dict, *, msgs_key: str, role_key: str, text_key: str,
                             actor_roles: set[str], obs_roles: set[str],
                             system_roles: set[str] = frozenset({"system"}),
                             success_key: str | None = None, min_steps: int = 3,
                             rid: str = "r") -> Rollout | None:
    """Generic role-tagged-turn -> Rollout.

    Each ACTOR turn (assistant / ai / gpt) becomes a step; the OBSERVATION turn that
    immediately follows it (user / tool) is folded into that step. Leading SYSTEM turn
    becomes the context.
    """
    msgs = record.get(msgs_key) or []
    if not isinstance(msgs, list) or not msgs:
        return None
    context, steps, pending = "", [], None
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get(role_key, "")).lower()
        text = m.get(text_key)
        text = text if isinstance(text, str) else ""
        if role in system_roles and not context:
            context = text[:4000]
        elif role in actor_roles:
            if pending is not None:
                steps.append(pending)
            pending = text
        elif role in obs_roles and pending is not None:
            pending = _fold_observation(pending, text)
    if pending is not None:
        steps.append(pending)
    steps = [s for s in steps if s and s.strip()]
    if len(steps) < min_steps:
        return None
    success = True
    if success_key is not None:
        v = record.get(success_key)
        success = bool(v) if not isinstance(v, str) else v.lower() in ("1", "true", "resolved", "yes")
    return Rollout(id=rid, context=context or "(task)", steps=steps, success=success)


def load_swe_agent_trajectories(limit: int = 3000, resolved_frac: float = 0.66,
                                min_steps: int = 4, streaming: bool = True) -> list[Rollout]:
    """nebius/SWE-agent-trajectories -> list[Rollout], keeping a mix of resolved and
    unresolved (the unresolved are the degraded-state salt).

    Field `trajectory` is a role-tagged list; `target` (bool) is resolved/unresolved.
    Uses streaming so it does not download the whole 1.1GB set.
    """
    from datasets import load_dataset

    ds = load_dataset("nebius/SWE-agent-trajectories", split="train", streaming=streaming)
    want_resolved = int(limit * resolved_frac)
    out: list[Rollout] = []
    n_res = n_unres = 0
    for i, rec in enumerate(ds):
        r = rollouts_from_role_turns(
            rec, msgs_key="trajectory", role_key="role", text_key="content",
            actor_roles={"ai", "assistant"}, obs_roles={"user", "tool"},
            success_key="target", min_steps=min_steps, rid=f"swe{i}")
        if r is None:
            continue
        if r.success and n_res >= want_resolved:
            continue
        if not r.success and n_unres >= (limit - want_resolved):
            continue
        out.append(r)
        n_res += r.success
        n_unres += (not r.success)
        if len(out) >= limit:
            break
    return out


def summarize(rollouts: Iterable[Rollout]) -> dict:
    rs = list(rollouts)
    steps = [len(r.steps) for r in rs]
    return {
        "n": len(rs),
        "resolved": sum(r.success for r in rs),
        "unresolved": sum(not r.success for r in rs),
        "steps_min": min(steps) if steps else 0,
        "steps_max": max(steps) if steps else 0,
        "steps_mean": round(sum(steps) / len(steps), 1) if steps else 0,
    }
