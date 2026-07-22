"""Model runners.

`HFRunner` is the real one (transformers, needs a GPU for a 9B teacher).

The `Sim*` runners exist so the MECHANISM can be validated with no GPU at all: they
are not models, they are adversaries. Each one simulates a way a miner could try to
win without doing real compression work, and the harness must rank every one of them
below an honest student. That experiment is the whole point of Phase 2 — find the
hole before miners do.
"""
from __future__ import annotations

import random
import re
from typing import Sequence

from .axes.math_gsm import extract_answer


class HFRunner:
    """Real inference via transformers. Lazy-imports so the harness runs without torch."""

    def __init__(self, model_id: str, device: str = "auto", dtype: str = "bfloat16",
                 revision: str | None = None, name: str | None = None,
                 trust_remote_code: bool = False, batch_size: int = 8):
        self.model_id = model_id
        self.revision = revision
        self.name = name or model_id
        self._device, self._dtype = device, dtype
        self._trust = trust_remote_code   # OK for OUR pinned teacher/judge; never for miner output
        self._batch_size = batch_size
        self._model = None
        self._tok = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kw = {"trust_remote_code": self._trust}
        if self.revision:
            kw["revision"] = self.revision
        self._tok = AutoTokenizer.from_pretrained(self.model_id, **kw)
        # decoder-only generation requires LEFT padding so the newest token is at the
        # right edge for every row; batched greedy decode is otherwise misaligned.
        self._tok.padding_side = "left"
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=getattr(torch, self._dtype), device_map=self._device, **kw
        )
        self._model.eval()

    def _render(self, prompt: str) -> str:
        try:
            return self._tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt

    def generate_chat(self, message_lists: Sequence[list], max_new_tokens: int = 32) -> list[str]:
        """One assistant turn per conversation. `message_lists` is a batch of
        [{role, content}, ...]; the chat template + add_generation_prompt make the model
        CONTINUE the conversation (produce the next turn), never restart. This is the
        multi-turn path — a genuine agent turn, used by the gridworld ModelAgent."""
        self._load()
        import torch

        texts = []
        for msgs in message_lists:
            try:
                texts.append(self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            except Exception:
                texts.append("\n".join(m.get("content", "") for m in msgs))
        outs: list[str] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            enc = self._tok(batch, return_tensors="pt", padding=True, truncation=False).to(self._model.device)
            with torch.no_grad():
                gen = self._model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                           pad_token_id=self._tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            outs.extend(self._tok.batch_decode(new, skip_special_tokens=True))
        return outs

    def generate(self, prompts: Sequence[str], max_new_tokens: int = 512) -> list[str]:
        self._load()
        import torch

        texts = [self._render(p) for p in prompts]
        outs: list[str] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            enc = self._tok(batch, return_tensors="pt", padding=True, truncation=False).to(self._model.device)
            with torch.no_grad():
                # greedy: scoring must be reproducible
                gen = self._model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                           pad_token_id=self._tok.pad_token_id)
            # left-padded -> every row's prompt ends at the same column; slice uniformly
            new = gen[:, enc["input_ids"].shape[1]:]
            outs.extend(self._tok.batch_decode(new, skip_special_tokens=True))
        return outs


# --------------------------------------------------------------------------
# Simulated students — the adversarial cast for Phase 2
# --------------------------------------------------------------------------

class SimOracle:
    """Stands in for the pinned TEACHER: solves at a fixed competence rate."""

    def __init__(self, items_by_prompt: dict[str, str], accuracy: float = 0.85, seed: int = 0,
                 name: str = "teacher(sim)"):
        self.name = name
        self._ans = items_by_prompt
        self.accuracy = accuracy
        self._rng = random.Random(seed)

    def generate(self, prompts: Sequence[str], max_new_tokens: int = 512) -> list[str]:
        out = []
        for p in prompts:
            true = self._ans.get(p)
            if true is not None and self._rng.random() < self.accuracy:
                out.append(f"Working through it step by step.\nAnswer: {true}")
            else:
                out.append("Let me think about this.\nAnswer: 0")
        return out


class SimHonestStudent(SimOracle):
    """An honest compression: uniformly a bit worse than the teacher, everywhere."""

    def __init__(self, items_by_prompt, retention: float = 0.8, teacher_acc: float = 0.85,
                 seed: int = 1, name: str = "student-honest(sim)"):
        super().__init__(items_by_prompt, accuracy=teacher_acc * retention, seed=seed, name=name)


class SimStyleOnly:
    """Learned the teacher's MANNERISMS, lost the reasoning.

    Produces fluent, teacher-shaped chain-of-thought and a wrong number. This is the
    failure that killed earlier distillation contests: it would score extremely well
    on any imitation/KL metric and near-zero here, which is the entire point.
    """

    name = "student-style-only(sim)"

    def __init__(self, seed: int = 2):
        self._rng = random.Random(seed)

    def generate(self, prompts: Sequence[str], max_new_tokens: int = 512) -> list[str]:
        filler = ("Let me reconsider. Wait — I should double-check that step. "
                  "Breaking it down carefully, step by step. Hmm, let me verify.")
        return [f"{filler}\nAnswer: {self._rng.randint(1, 400)}" for _ in prompts]


class SimNarrow:
    """Trained hard on ONE template family, sacrificed the rest.

    Catches the capability-laundering case: strong where it studied, collapsed
    elsewhere. Worst-domain aggregation must drag this below an honest student even
    though its headline average can look respectable.
    """

    name = "student-narrow(sim)"

    def __init__(self, items_by_prompt: dict[str, str], strong_template: str = "rate_total",
                 template_of: dict[str, str] | None = None, strong_acc: float = 0.98,
                 weak_acc: float = 0.15, seed: int = 3):
        self._ans = items_by_prompt
        self._tpl = template_of or {}
        self.strong_template, self.strong_acc, self.weak_acc = strong_template, strong_acc, weak_acc
        self._rng = random.Random(seed)

    def generate(self, prompts: Sequence[str], max_new_tokens: int = 512) -> list[str]:
        out = []
        for p in prompts:
            acc = self.strong_acc if self._tpl.get(p) == self.strong_template else self.weak_acc
            true = self._ans.get(p)
            if true is not None and self._rng.random() < acc:
                out.append(f"Answer: {true}")
            else:
                out.append(f"Answer: {self._rng.randint(1, 400)}")
        return out


class SimNoOpBrittle:
    """Pattern-matcher: fine normally, falls apart when a NoOp distractor appears.

    This is the specific signature of a student that memorized surface form instead
    of learning the procedure (GSM-Symbolic's core result).
    """

    name = "student-noop-brittle(sim)"

    def __init__(self, items_by_prompt: dict[str, str], noop_of: dict[str, bool],
                 clean_acc: float = 0.82, noop_acc: float = 0.10, seed: int = 4):
        self._ans, self._noop = items_by_prompt, noop_of
        self.clean_acc, self.noop_acc = clean_acc, noop_acc
        self._rng = random.Random(seed)

    def generate(self, prompts: Sequence[str], max_new_tokens: int = 512) -> list[str]:
        out = []
        for p in prompts:
            acc = self.noop_acc if self._noop.get(p) else self.clean_acc
            true = self._ans.get(p)
            ok = true is not None and self._rng.random() < acc
            out.append(f"Answer: {true if ok else self._rng.randint(1, 400)}")
        return out


class SimNaiveQuant(SimOracle):
    """The free baseline (round-to-nearest int4). Must earn ~zero margin."""

    def __init__(self, items_by_prompt, teacher_acc: float = 0.85, ratio: float = 0.55,
                 seed: int = 5, name: str = "naive-quant-control(sim)"):
        super().__init__(items_by_prompt, accuracy=teacher_acc * ratio, seed=seed, name=name)


class SafeStudentRunner(HFRunner):
    """HFRunner locked down for UNTRUSTED miner checkpoints — defense in depth beyond
    the intake file-gate. Refuses to load anything but safetensors, forces
    trust_remote_code=False, and rejects a checkpoint dir containing forbidden files at
    construction time (never reaches from_pretrained if a pickle/.py is present)."""

    def __init__(self, ckpt_dir: str, device: str = "auto", dtype: str = "bfloat16",
                 name: str | None = None, batch_size: int = 8):
        from pathlib import Path
        from .gates import FORBIDDEN_FILES
        d = Path(ckpt_dir)
        bad = [p.name for p in d.rglob("*") if p.is_file() and p.suffix.lower() in FORBIDDEN_FILES]
        if bad:
            raise ValueError(f"refusing to load: forbidden files present {bad}")
        if not list(d.rglob("*.safetensors")):
            raise ValueError("refusing to load: no safetensors weights")
        super().__init__(str(d), device=device, dtype=dtype,
                         name=name or f"student:{d.name}", trust_remote_code=False,
                         batch_size=batch_size)

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # use_safetensors=True forces the safetensors path even if other files exist
        self._tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=False)
        self._tok.padding_side = "left"   # batched decoder generation, see HFRunner._load
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=getattr(torch, self._dtype), device_map=self._device,
            trust_remote_code=False, use_safetensors=True)
        self._model.eval()
