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

    # BATCH SIZE IS 1 BECAUSE OF CORRECTNESS, AND ONLY INCIDENTALLY BECAUSE OF MEMORY.
    #
    # `build_shared` generates the parent's step over ALL trajectories; `score_submission` and the
    # identity canary generate over the USABLE subset only (MIN_TEACHER_EFFECT and the empty-step
    # discards make that strictly smaller in the normal case). Left padding pads each item to the
    # longest in ITS window — so at bs=2 a measured 32 of 64 trajectories land beside DIFFERENT
    # neighbours in the two calls, get a different number of pad tokens, and decode differently.
    # The identity canary therefore is NOT 1.000 by construction above bs=1, contrary to its own
    # docstring: it can pass or fail depending on whether the neighbours happened to match, which
    # is worse than failing outright. batch_invariance.py already measured that mechanism at
    # 0.046-0.097 with the step text moving every time.
    #
    # And the fairness half a miner would hold us to: `GGUFStudentRunner.generate` runs one prompt
    # at a time, unpadded, and every submission this round was GGUF. At bs>1 the REFERENCE answer
    # is produced under padding that no submission experiences. At bs=1 both sides are unpadded and
    # the paired comparison is like-for-like.
    #
    # Memory is the lesser reason, though it also binds: `eager` is pinned for determinism and
    # materialises the full batch x heads x q_len x k_len score matrix — quadratic in prompt
    # length, linear in batch. At batch_size=8 a real round asked for one 64.12 GiB allocation on
    # an 80 GB H100 and died.
    #
    # Pinned into every record (`_versions()` reads this default BY NAME) because it changes the
    # answer: measured spread across batch_size in {1,2,3,5,8} is 0.050 retention on an H100 — the
    # dethrone margin itself. Safe to set now only because no round has ever published a number.
    def __init__(self, model_id: str, device: str = "auto", dtype: str = "bfloat16",
                 revision: str | None = None, name: str | None = None,
                 trust_remote_code: bool = False, batch_size: int = 1,
                 load_in_4bit: bool = False, load_in_8bit: bool = False):
        self.model_id = model_id
        self.revision = revision
        self.name = name or model_id
        self._device, self._dtype = device, dtype
        self._trust = trust_remote_code   # OK for OUR pinned teacher/judge; never for miner output
        self._batch_size = batch_size
        # Load the SAME checkpoint at a narrower storage width. Needed because the shakedown's
        # student is a genuine compression of the pinned parent — same architecture, fewer bits —
        # which is the actual task, and there was no way to construct one without it.
        self._load_in_4bit = load_in_4bit
        self._load_in_8bit = load_in_8bit
        self._model = None
        self._tok = None

    # DETERMINISM IS A PROPERTY OF THE BOX, NOT OF THE DESIGN — measured, not assumed.
    #
    # On an H100 PCIe (torch 2.11+cu128), four identical generate() calls over the SAME batch of
    # the same 24 prefixes returned FOUR DIFFERENT outputs. On an A100 and an L40S (torch
    # 2.13+cu130) the same code was bit-exact. The A100/L40S "noise floor is 0.0" result was a
    # property of those two boxes, and would have shipped as if it were a property of the
    # mechanism.
    #
    # It is LENGTH-DEPENDENT, which is what identifies the cause: deterministic at 32 and 64 new
    # tokens, nondeterministic at 128 and 256. That is a KV-length kernel heuristic switching to a
    # split-K attention reduction whose atomics accumulate in nondeterministic order. Confirming
    # it: torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG=:4096:8 + TF32 off +
    # reduced-precision-reduction off did NOT fix it, while pinning the attention implementation
    # DID — both eager and the SDPA MATH backend are bit-exact at 256 on the same box.
    #
    # Why this matters more than its size: the crown is decided by greedy decoding, so a flipped
    # near-tie changes the STEP TEXT, not just a logit. The parent scored against itself must be
    # exactly 1.0 by construction (s = 0, |d_A - d_G| = 0); on the H100 it scored 0.8754. A
    # 12-point self-inconsistency is not noise to be tolerated inside a margin, it means the round
    # was not scoring what it believed it was scoring.
    ATTN_IMPLEMENTATION = "eager"

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # A COLD BOX DOWNLOADS 16 GB HERE, and nothing downstream of `from_pretrained` reports
        # anything the orchestrator can see. Announced on both sides so the silence detector reads
        # a slow download as a slow download.
        from .progress import tick
        tick("load", self.model_id, force=True)
        kw = {"trust_remote_code": self._trust}
        if self.ATTN_IMPLEMENTATION:
            kw["attn_implementation"] = self.ATTN_IMPLEMENTATION
        if self.revision:
            kw["revision"] = self.revision
        self._tok = AutoTokenizer.from_pretrained(self.model_id, **kw)
        # decoder-only generation requires LEFT padding so the newest token is at the
        # right edge for every row; batched greedy decode is otherwise misaligned.
        self._tok.padding_side = "left"
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        if self._load_in_8bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self._load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, self._dtype),
                # no double quant: the point is a clean, reportable bit width, not the last
                # fraction of a bit, and double quantisation makes the effective width harder
                # to state honestly
                bnb_4bit_use_double_quant=False)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=getattr(torch, self._dtype), device_map=self._device, **kw
        )
        self._model.eval()
        tick("loaded", self.model_id, force=True)

    def _gen_config(self, max_new_tokens: int):
        """A VALIDATOR-OWNED decode config, fully specified.

        `generation_config.json` is an allowlisted `.json`, so it is hashed into the committed
        identity — but hashing is not freezing. `from_pretrained` loads it into
        `model.generation_config`, and passing only max_new_tokens/do_sample/pad_token_id left
        `num_beams`, `bad_words_ids`, `sequence_bias`, `suppress_tokens`,
        `no_repeat_ngram_size`, `renormalize_logits` and friends under miner control. That is a
        $0 copy-the-king exploit: download the crowned bytes, add {"num_beams": 8}, and the new
        content hash is a legitimately different artifact that beats the king on exact-match
        span tasks with no training at all — while multiplying validator cost 8x.

        Passing an explicit GenerationConfig makes the model's own config inert for the call, so
        every submission is decoded identically. Greedy, because scoring must be reproducible."""
        from transformers import GenerationConfig
        return GenerationConfig(
            max_new_tokens=max_new_tokens, min_new_tokens=0,
            do_sample=False, num_beams=1, num_beam_groups=1,
            temperature=1.0, top_k=0, top_p=1.0, typical_p=1.0,
            repetition_penalty=1.0, encoder_repetition_penalty=1.0, length_penalty=1.0,
            no_repeat_ngram_size=0, bad_words_ids=None, force_words_ids=None,
            sequence_bias=None, suppress_tokens=None, begin_suppress_tokens=None,
            forced_bos_token_id=None, forced_eos_token_id=None, forced_decoder_ids=None,
            exponential_decay_length_penalty=None, renormalize_logits=False,
            early_stopping=False, use_cache=True,
            pad_token_id=self._tok.pad_token_id, eos_token_id=self._tok.eos_token_id,
        )

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
                gen = self._model.generate(**enc, generation_config=self._gen_config(max_new_tokens))
            new = gen[:, enc["input_ids"].shape[1]:]
            outs.extend(self._tok.batch_decode(new, skip_special_tokens=True))
        return outs

    # Top-k support for observer distributions. Full-vocab KL over a 150k vocabulary would be
    # ~600KB per position and unrecordable; SN97 ran top-128 for the same reason. The KL is
    # taken over the union of the two supports, so a token one side proposes and the other never
    # did still costs (see observer_kl.kl).
    TOPK = 128
    # Probabilities are ROUNDED before they leave this function. Logit arithmetic is not
    # bit-stable across batch composition or kernel choice, and an unrounded low-order bit turns
    # into a KL difference that looks like a real score difference. Rounding is the cheapest part
    # of making a crown decision reproducible; the harness measures what is left
    # (eval/determinism.py).
    PROB_DECIMALS = 6

    def distributions(self, prefix: str, continuation: str) -> list[dict]:
        """The observer's per-position next-token distribution over `continuation`, given
        `prefix`. Teacher-forced: no sampling, one forward pass, so it is a pure function of
        (prefix, continuation) up to kernel noise.

        Position alignment is the subtle part. With P prefix tokens and C continuation tokens,
        the logits at index P-1 predict continuation[0], index P predicts continuation[1], and so
        on — so the C distributions wanted are at indices P-1 .. P+C-2. Off by one here silently
        compares the wrong positions and every KL becomes noise."""
        self._load()
        import torch

        pre = self._tok(prefix, return_tensors="pt", truncation=False)["input_ids"]
        full = self._tok(prefix + continuation, return_tensors="pt",
                         truncation=False)["input_ids"]
        n_pre, n_full = pre.shape[1], full.shape[1]
        if n_full <= n_pre:
            return []
        full = full.to(self._model.device)
        with torch.no_grad():
            logits = self._model(input_ids=full).logits[0]          # [n_full, vocab]
        out: list[dict] = []
        k = min(self.TOPK, logits.shape[-1])
        for pos in range(n_pre - 1, n_full - 1):
            probs = torch.softmax(logits[pos].float(), dim=-1)
            vals, idx = torch.topk(probs, k)
            out.append({int(i): round(float(v), self.PROB_DECIMALS)
                        for v, i in zip(vals.tolist(), idx.tolist())})
        return out

    def generate(self, prompts: Sequence[str], max_new_tokens: int = 512) -> list[str]:
        self._load()
        import torch
        from .progress import tick

        texts = [self._render(p) for p in prompts]
        outs: list[str] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            enc = self._tok(batch, return_tensors="pt", padding=True, truncation=False).to(self._model.device)
            # THE CONTEXT LENGTH IS REPORTED, not just the item index. Eager attention allocates
            # batch x heads x ctx x ctx, so an OOM is a fact about the LONGEST prompt in the pool
            # and the log needs to say which one — otherwise the next failure is another guess.
            # The batch loop is also the only seam inside a `generate` call, and one round's worth
            # of them is minutes of GPU with nothing else to say for itself.
            ctx = int(enc["input_ids"].shape[1])
            tick("generate", f"{self.name} {i}/{len(texts)} ctx={ctx} @{max_new_tokens} tok")
            with torch.no_grad():
                # validator-owned decode: the miner's generation_config.json cannot influence
                # scoring (see _gen_config — hashed is not frozen).
                gen = self._model.generate(**enc, generation_config=self._gen_config(max_new_tokens))
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
                 name: str | None = None, batch_size: int = 1):
        from pathlib import Path
        from .gates import FORBIDDEN_FILES
        d = Path(ckpt_dir)
        bad = [p.name for p in d.rglob("*") if p.is_file() and p.suffix.lower() in FORBIDDEN_FILES]
        if bad:
            raise ValueError(f"refusing to load: forbidden files present {bad}")
        if not list(d.rglob("*.safetensors")):
            # NAME THE FORMAT. "no safetensors weights" was the message a GGUF submission got after
            # clearing all six gates — a gate allowlists .gguf, bitrate measures it, and
            # parent_compat goes out of its way to match its architecture, so the miner has every
            # reason to think it is supported. Telling them WHICH format they shipped and what to
            # use instead is the difference between a bug report and a wasted round.
            if list(d.rglob("*.gguf")):
                raise ValueError(
                    "GGUF is not loadable by this runner: the gates accept it but there is no GGUF "
                    "inference path, so it cannot be scored. Use GGUFStudentRunner, or submit "
                    "safetensors.")
            raise ValueError("refusing to load: no safetensors weights")
        super().__init__(str(d), device=device, dtype=dtype,
                         name=name or f"student:{d.name}", trust_remote_code=False,
                         batch_size=batch_size)

    # DETERMINISM IS A PROPERTY OF THE BOX, NOT OF THE DESIGN — measured, not assumed.
    #
    # On an H100 PCIe (torch 2.11+cu128), four identical generate() calls over the SAME batch of
    # the same 24 prefixes returned FOUR DIFFERENT outputs. On an A100 and an L40S (torch
    # 2.13+cu130) the same code was bit-exact. The A100/L40S "noise floor is 0.0" result was a
    # property of those two boxes, and would have shipped as if it were a property of the
    # mechanism.
    #
    # It is LENGTH-DEPENDENT, which is what identifies the cause: deterministic at 32 and 64 new
    # tokens, nondeterministic at 128 and 256. That is a KV-length kernel heuristic switching to a
    # split-K attention reduction whose atomics accumulate in nondeterministic order. Confirming
    # it: torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG=:4096:8 + TF32 off +
    # reduced-precision-reduction off did NOT fix it, while pinning the attention implementation
    # DID — both eager and the SDPA MATH backend are bit-exact at 256 on the same box.
    #
    # Why this matters more than its size: the crown is decided by greedy decoding, so a flipped
    # near-tie changes the STEP TEXT, not just a logit. The parent scored against itself must be
    # exactly 1.0 by construction (s = 0, |d_A - d_G| = 0); on the H100 it scored 0.8754. A
    # 12-point self-inconsistency is not noise to be tolerated inside a margin, it means the round
    # was not scoring what it believed it was scoring.
    ATTN_IMPLEMENTATION = "eager"

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


class GGUFStudentRunner:
    """Generate from an UNTRUSTED GGUF checkpoint. The only student format that can pass the tiers.

    WHY THIS HAS TO EXIST. The bit tiers are expressed in bits per weight — sub4 allows 4.0 code /
    5.0 container — and no safetensors artifact can reach them. `measure_checkpoint` treats integer
    dtypes as already-packed codes and credits them their full container width, so an int8 file
    measures 8.0/8.0 and clears nothing; and a genuinely 4-bit-packed safetensors would carry half
    the element count, failing the pinned-parent gate instead. GGUF is the only format that can
    express sub-8-bit storage AND keep an honest weight-element count — and it is what the field
    actually ships (PrismML's Bonsai, every llama.cpp quant). Until this existed the gates accepted
    GGUF and the loader refused it, so the subnet had NO submittable format at all.

    A STUDENT ONLY HAS TO GENERATE. The observer computes distributions and stays an HF model; the
    submission is asked for one thing, `generate(prompts, max_new_tokens) -> list[str]`. That is
    why this is tractable at all — no logits, no tokenizer surgery, no KV plumbing.

    DETERMINISM IS PINNED THE SAME WAY THE HF PATH PINS IT, and for the same measured reason: a
    round that is not bit-exact on repeat cannot be audited. Greedy only (temperature 0, no
    sampling), a fixed thread count and batch size, and a fixed context — thread count changes
    reduction order in llama.cpp exactly as batch size does in torch, so it belongs in the record's
    pinned versions rather than in whatever the box happened to default to.

    UNTRUSTED, so the same refusals as SafeStudentRunner: no forbidden files, and nothing is
    imported or executed out of the artifact — a GGUF is data to llama.cpp, never code.
    """

    N_THREADS = 8            # pinned: thread count changes reduction order, hence the output

    @staticmethod
    def backend() -> str:
        """"cuda" or "cpu" — WHERE SUBMISSIONS ACTUALLY RAN, which belongs in the record.

        The parent's device is pinned and audited (`require_gpu`, and an absent name is fatal);
        the students' was not recorded at all. It should be, for the same reason: llama.cpp on CPU
        and on CUDA do not produce identical tokens, so a crown measured one way is not comparable
        with one defended the other.

        And it cannot be inferred from configuration. The pip wheel is built CPU-only, in which
        case `n_gpu_layers` is ACCEPTED AND SILENTLY IGNORED — a parameter that lies. The only
        honest source is llama.cpp's own capability flag."""
        # ONE PROBE, SHARED WITH THE INSTALL GUARD. This used to ask only for
        # `llama_supports_gpu_offload`, which recent bindings no longer export — so a genuine CUDA
        # build reported "cpu" into the signed record, and the install guard built on the same
        # question refused a working round outright.
        from .gpu_check import probe
        ok, _why = probe()
        return "cuda" if ok else "cpu"
    # 8192, NOT 4096, AND THIS IS A FAIRNESS FIX. llama.cpp silently rewrites
    # `max_tokens = n_ctx - len(prompt_tokens)` rather than erroring, so at n_ctx=4096 a miner with
    # a 3,900-token prefix gets 196 step tokens where the parent got its full 256 — a
    # prefix-length-dependent handicap on one side of a paired comparison, invisible in the record.
    # 8192 clears the clamped 4096 prefix plus 256 step with slack, and sits far below the GGUF's
    # native window so no RoPE scaling is triggered.
    N_GPU_LAYERS = -1        # all layers; see _load. Pinned into the record via backend().
    N_CTX = 8192
    N_BATCH = 512

    def __init__(self, ckpt_dir: str, name: str | None = None, backend=None):
        from pathlib import Path
        from .gates import FORBIDDEN_FILES
        d = Path(ckpt_dir)
        bad = [p.name for p in d.rglob("*") if p.is_file() and p.suffix.lower() in FORBIDDEN_FILES]
        if bad:
            raise ValueError(f"refusing to load: forbidden files present {bad}")
        gg = sorted(d.rglob("*.gguf"))
        if not gg:
            raise ValueError("refusing to load: no gguf weights")
        if len(gg) > 1:
            # Which file is the model would be a guess, and a guess here silently scores something
            # other than what was committed.
            raise ValueError(f"refusing to load: {len(gg)} gguf files, expected exactly one")
        self.path = str(gg[0])
        self.name = name or f"student:{d.name}"
        self._backend = backend          # injectable: the whole path is testable without llama.cpp
        self._llm = None

    def _load(self):
        if self._llm is not None:
            return self._llm
        if self._backend is not None:
            self._llm = self._backend(self.path)
            return self._llm
        from llama_cpp import Llama       # local import: nothing else here needs llama.cpp
        # OFFLOAD EVERY LAYER. Measured on a real submission: 7.9s/prompt on GPU regardless of
        # prefix length, against 11-27s on CPU where prefill dominates. Ten miners go from ~3.5 h —
        # which does not fit the 2.5 h rental ceiling — to 1.59 h, and the cost stops depending on
        # which items the nonce happened to draw.
        #
        # -1 is a REQUEST, not a guarantee: on a CPU-only build llama.cpp accepts it and ignores
        # it. `backend()` reports what actually happened and `_versions()` records it, because CPU
        # and GPU llama.cpp do not emit identical tokens — a crown measured one way is not
        # comparable with one defended the other.
        self._llm = Llama(model_path=self.path, n_ctx=self.N_CTX, n_threads=self.N_THREADS,
                          n_batch=self.N_BATCH, n_gpu_layers=self.N_GPU_LAYERS,
                          logits_all=False, verbose=False, seed=0)
        return self._llm

    def close(self) -> None:
        """Free the llama.cpp context and its VRAM. IDEMPOTENT, and never raises.

        Reported by a miner on 2026-08-07 against round 1 and confirmed: the scoring loop kept
        every challenger's runner alive in `registry` for the whole round, and nothing here ever
        released one. Nine retained students were ~30 GB of weights plus ~1.7 GB each of KV and
        compute buffer at n_ctx=8192 — roughly 45 GB held for models that would never be used
        again, next to the bf16 parent and observer on an 80 GB card. The tenth load then failed
        with a bare `ValueError: Failed to load model from file`, which reads exactly like a
        corrupt artifact and is not one. Python's GC would not have saved us: `registry` held a
        strong reference deliberately.

        Nothing must escape from here. A cleanup that raises would convert a scored submission
        into a rejected one — the very failure this is fixing."""
        llm, self._llm = self._llm, None
        if llm is None:
            return
        try:
            close = getattr(llm, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
        try:
            del llm
            import gc
            gc.collect()
        except Exception:
            pass

    def generate(self, prompts, max_new_tokens: int = 512) -> list[str]:
        """Greedy continuation per prompt. Same contract as HFRunner.generate."""
        from .progress import tick

        llm = self._load()
        out = []
        for n, p in enumerate(prompts, 1):
            # THE MINER LEG IS THE LONGEST SILENT BLOCK IN THE ROUND, and until this line it was
            # completely silent. `HFRunner.generate` ticks per batch; this path — one prompt at a
            # time under llama.cpp, on CPU, over prefixes up to the context ceiling — did not, and
            # `score_submission` only ticks AFTER this call returns. So a legitimately long
            # submission looked identical to a hang, and the orchestrator's 20-minute silence
            # budget killed a healthy round at $2.75.
            #
            # That is precisely the failure eval/progress.py's own docstring warns about: a silence
            # timeout over a silent-by-design scorer is not a safety net, it is a way to kill the
            # rounds that were going to succeed. Instrumenting the fast path and forgetting the
            # slow one is how you build exactly that.
            tick("student", f"{self.name} {n}/{len(prompts)} @{max_new_tokens} tok")
            r = llm(p, max_tokens=int(max_new_tokens), temperature=0.0, top_k=1, top_p=1.0,
                    repeat_penalty=1.0, echo=False)
            try:
                out.append(r["choices"][0]["text"])
            except Exception:
                # A backend that returns an unexpected shape must not be guessed at: an empty step
                # scores as inert, which is the honest reading of "this model produced nothing".
                out.append("")
        return out


def student_runner(ckpt_dir: str, **kw):
    """Pick the loader by what the artifact actually IS.

    Miners choose their format and both are legitimate, so dispatch belongs here rather than in
    every call site — the round should not have to know, and the previous arrangement (one loader
    that raised on the only format the tiers admit) was invisible until a real submission arrived.
    """
    from pathlib import Path
    d = Path(ckpt_dir)
    if list(d.rglob("*.safetensors")):
        return SafeStudentRunner(str(d), **{k: v for k, v in kw.items() if k != "backend"})
    if list(d.rglob("*.gguf")):
        return GGUFStudentRunner(str(d), **{k: v for k, v in kw.items()
                                            if k in ("name", "backend")})
    raise ValueError("refusing to load: artifact holds neither safetensors nor gguf weights")
