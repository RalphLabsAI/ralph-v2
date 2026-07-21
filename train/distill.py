"""Distill a real student from a pinned teacher on the trajectory substrate.

The eval scores a student by whether it reproduces the teacher's next STEP from a
prefix (trajectory.py). So the natural training target is the same object: given
prefix(k), produce step k. The pile's steps ARE the teacher's steps (we generate the
pile with the pinned teacher), so training targets come free — no separate labeling.

This is deliberately plain SFT distillation (completion-only cross-entropy on the
teacher's step), not on-policy/logit distillation. It is enough to manufacture the two
students the experiment needs:

  * S_broad  — trained on ALL (prefix, step) points across every domain and every k.
               A genuine compression: it should retain the teacher's step behavior both
               from teacher states AND from its own states.
  * S_drift  — trained ONLY on short prefixes (k <= drift_max_k). It never sees its own
               long-horizon states, so it does the right next step when handed the
               teacher's prefix but compounds error on its own rollout. This is the
               exposure-bias failure the self_state slice exists to catch, made real.

Both save as safetensors, so each is a VALID miner submission and flows through the
exact same intake gates + scoring as a real one.
"""
from __future__ import annotations

from dataclasses import dataclass

from eval.trajectory import Rollout


@dataclass
class SFTExample:
    prompt: str      # prefix(k) rendered through the chat template at train time
    target: str      # the teacher's step k (the only tokens we compute loss on)
    domain: str
    k: int


def build_sft_examples(experience: list[Rollout], rollout_idx: list[int],
                       min_k: int = 1, max_k: int | None = None,
                       domains: set[str] | None = None) -> list[SFTExample]:
    """(prefix(k) -> step[k]) SFT pairs from the chosen train rollouts.

    `max_k` caps horizon (the drifter uses a small cap); `domains` restricts to a
    subset (a narrow-domain drifter). Both default to "use everything".
    """
    out: list[SFTExample] = []
    for ri in rollout_idx:
        r = experience[ri]
        if domains is not None and r.domain not in domains:
            continue
        hi = len(r.steps) if max_k is None else min(len(r.steps), max_k + 1)
        for k in range(min_k, hi):
            out.append(SFTExample(prompt=r.prefix(k), target=r.steps[k], domain=r.domain, k=k))
    return out


def _collate(tok, examples, max_len: int):
    """Tokenize prompt+target, masking prompt tokens so loss is completion-only."""
    import torch

    input_ids, labels = [], []
    for ex in examples:
        try:
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": ex.prompt}], tokenize=False, add_generation_prompt=True)
        except Exception:
            rendered = ex.prompt
        p_ids = tok(rendered, add_special_tokens=False)["input_ids"]
        t_ids = tok(ex.target + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p_ids + t_ids)[:max_len]
        lab = ([-100] * len(p_ids) + t_ids)[:max_len]
        input_ids.append(ids)
        labels.append(lab)
    width = max(len(x) for x in input_ids)
    pad = tok.pad_token_id
    attn = [[1] * len(x) + [0] * (width - len(x)) for x in input_ids]
    input_ids = [x + [pad] * (width - len(x)) for x in input_ids]
    labels = [x + [-100] * (width - len(x)) for x in labels]
    return (torch.tensor(input_ids), torch.tensor(attn), torch.tensor(labels))


def distill_student(student_base: str, examples: list[SFTExample], output_dir: str,
                    epochs: int = 3, lr: float = 1e-5, batch_size: int = 8,
                    max_len: int = 1024, dtype: str = "bfloat16", seed: int = 0,
                    log=print) -> str:
    """Full-parameter SFT of `student_base` on the teacher-step targets; save safetensors.

    Full FT (not LoRA) so the saved checkpoint is a standalone model that loads through
    SafeStudentRunner with no adapter merge step — exactly what a miner submits.
    """
    import random
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = AutoTokenizer.from_pretrained(student_base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        student_base, dtype=getattr(torch, dtype), device_map="cuda")
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    ex = list(examples)
    steps_per_epoch = (len(ex) + batch_size - 1) // batch_size
    log(f"distill {student_base}: {len(ex)} examples, {epochs} epochs, {steps_per_epoch} steps/epoch")
    for ep in range(epochs):
        rng.shuffle(ex)
        running = 0.0
        for bi in range(0, len(ex), batch_size):
            batch = ex[bi:bi + batch_size]
            ids, attn, labels = _collate(tok, batch, max_len)
            ids, attn, labels = ids.cuda(), attn.cuda(), labels.cuda()
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            running += out.loss.item()
        log(f"  epoch {ep+1}/{epochs}  mean_loss={running / max(1, steps_per_epoch):.4f}")

    model.save_pretrained(output_dir, safe_serialization=True)
    tok.save_pretrained(output_dir)
    log(f"  saved -> {output_dir}")
    return output_dir
