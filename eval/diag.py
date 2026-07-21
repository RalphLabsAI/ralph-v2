"""Diagnostic: why is retention ~0? Print raw (prefix, teacher step, student step,
judge verdict) tuples so we can see whether it's a judge-parsing bug, a step-granularity
problem, or a genuine 'many valid continuations' concern."""
from __future__ import annotations

import os

from .rollouts_gen import generate_experience
from .runners import HFRunner
from .trajectory import GroundedJudge

T = os.environ.get("RALPH_TEACHER", "Qwen/Qwen2.5-7B-Instruct")
S = os.environ.get("RALPH_STUDENT", "Qwen/Qwen2.5-3B-Instruct")
B = os.environ.get("RALPH_BASE", "Qwen/Qwen2.5-0.5B-Instruct")

teacher = HFRunner(T, name="teacher")
student = HFRunner(S, name="student")
base = HFRunner(B, name="base")
judge = GroundedJudge(HFRunner(T, name="judge"))  # reuse teacher weights via same id (cached)

exp = generate_experience(teacher, n=4)
print(f"{len(exp)} rollouts\n")

n_examples, s_hits, b_hits, tot = 0, 0, 0, 0
for r in exp:
    for k in range(1, len(r.steps)):
        prefix = r.prefix(k)
        glm_step = r.steps[k] if k < len(r.steps) else teacher.generate([prefix], 160)[0]
        s_step = student.generate([prefix], 160)[0]
        b_step = base.generate([prefix], 160)[0]
        s_agree = judge.agreement(prefix, glm_step, s_step)
        b_agree = judge.agreement(prefix, glm_step, b_step)
        s_hits += s_agree; b_hits += b_agree; tot += 1
        if n_examples < 6:
            print("=" * 70)
            print(f"TASK: {r.context[:80]}")
            print(f"GLM step[{k}]: {glm_step[:160]}")
            print(f"STUDENT-3B : {s_step[:160]}")
            print(f"  judge(GLM,student) = {s_agree}")
            print(f"BASE-0.5B  : {b_step[:160]}")
            print(f"  judge(GLM,base)    = {b_agree}\n")
            n_examples += 1
    if tot >= 20:
        break

print("=" * 70)
print(f"over {tot} points: student agreement rate {s_hits/tot:.3f}, base {b_hits/tot:.3f}")
print("if both near 0 -> judge too strict / step too specific (many valid continuations)")
print("if student >> base -> mechanism discriminates; retention bug is elsewhere")
