"""Benchmark Ralph's crowned models against PrismML's Bonsai, across four axes.

DELIBERATELY NOT THE SUBNET'S SCORER. Retention is a token-level KL against a pinned parent — the
right measure for ranking compressions of one model, the wrong one for "is this any good". This
asks the plain question: given the same problems, how many does each model get right.

Each model runs in its NATIVE runtime (GGUF on llama.cpp, safetensors on transformers) and gets ITS
OWN chat template, applied by its own tokenizer. That is the only way "identical treatment" means
anything across models trained by different people — the first version fed everything a bare
completion prompt, scored the parent at 35.5% on GSM8K and one crown at 0/200, and measured the
prompt rather than any model.

    python -m bench.compare
    python -m bench.compare --tasks gsm8k,mgsm --only ralph-ternary,bonsai-ternary
    python -m bench.compare --limit-scale 0.05        # smoke run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import bench.tasks as tasks
from bench.tasks import DATASET_IDS, TASKS, score, select_mgsm_langs

# label -> spec. `gguf` runs on llama.cpp, `hf` on transformers.
#
# `dtype` AND `config` ARE NOT INCIDENTAL — they are where the first run's confounds lived, and they
# are spelled out per model so the next reader can see the treatment rather than infer it:
#
#   dtype   the previous run hard-coded torch_dtype=bfloat16 for every transformers model. The
#           parent is natively BF16 so it was untouched; both Bonsai checkpoints ship F16, so they
#           alone were re-rounded to a coarser mantissa before being scored. A precision penalty
#           applied only to the competitor, in a benchmark whose conclusion was about the
#           competitor. Each model now loads in the dtype it was published in.
#
#   config  both Bonsai repos ship rope_scaling {yarn, factor 4.0, original 16384}, which
#           transformers applies STATICALLY to every prompt; the parent ships rope_scaling null.
#           Qwen's own guidance is that static YaRN costs short-sequence quality, and every prompt
#           here is short. Left as shipped, it depresses all four of their axes for a reason that is
#           not compression — which is the only thing this benchmark is trying to measure. It is
#           disabled, explicitly and on the record, rather than silently.
MODELS = {
    "parent-qwen3-8b":  {"kind": "hf", "repo": "Qwen/Qwen3-8B",
                         "rev": "b968826d9c46dd6066d109eabc6255188de91218",
                         "dtype": "bfloat16"},

    # Ralph crowns — exactly the bytes that were scored and anchored
    # THE REIGNING CROWNS, as of round 2 — not whoever held them when this file was written.
    # These were pinned to tensor-tailor, which in round 2 scored 0.2062 ternary and 0.1341 sub4:
    # neither is a king. Benchmarking a model the subnet did not crown, and publishing it as "our
    # crown", is the same class of error as the three confounds this harness was rebuilt to fix.
    #
    # Revisions are COMMIT SHAS, never `main`. The record pins ternary at `@main`, which is a
    # moving reference — fine for a round that re-fetches and hash-verifies every time, useless for
    # a benchmark whose whole value is that someone else can re-run it and get the same number.
    "ralph-ternary":    {"kind": "gguf", "repo": "crazy-m1ner/ralph-qwen3-8b-ternary",
                         "rev": "319aa2d43933931a2e9d831166e545eb187a248f"},
    "ralph-sub4":       {"kind": "gguf", "repo": "andreas11112/qwen3-8b-sn40-sub4",
                         "rev": "8c8cfa61be18005fb859fd58b5c8d943a26ac1dd"},

    # PrismML. Their COMPRESSED ggufs advertise file_type 141 / 41 — a custom fork's types that
    # stock llama.cpp cannot load — so the unpacked safetensors are the only artifact of theirs
    # runnable without adopting their toolchain. Verified genuinely ternary: exactly 3 distinct
    # values in every 128-weight window across three tensors.
    "bonsai-ternary":   {"kind": "hf", "repo": "prism-ml/Ternary-Bonsai-8B-unpacked",
                         "rev": "ac20f03fc62e872399218b659c8e949dfca05769",
                         "dtype": "float16", "config": {"rope_scaling": None}},
    "bonsai-1bit":      {"kind": "hf", "repo": "prism-ml/Bonsai-8B-unpacked",
                         "rev": "376f381570d6115bc03f82adcfa4af0c7672ae54",
                         "dtype": "float16", "config": {"rope_scaling": None}},
}

# Filled in as models load: the settings that actually took effect, written into the results file.
# The first run recorded none of this, so establishing what it had done took an audit against
# library source and HF metadata rather than a look at its own output.
RUNTIME: dict = {}

# Set when any model had to fall back to a raw prompt. A silent fallback produces numbers that look
# fine and measure the prompt; if it fires, every result in the run is suspect and must say so.
TEMPLATE_FALLBACKS: dict = {}

# label -> count of items that took run_gguf's untemplated fallback. Separate from the above because
# it is a separate path: TEMPLATE_FALLBACKS only ever covered transformers, so its emptiness was
# read as a clean bill of health for a fleet half of which it never inspected.
RAW_FALLBACKS: dict = {}

# label -> the kwargs that were in force when a model was found to be reasoning anyway. Non-empty
# means the run is not like-for-like and its cross-model numbers must not be published.
THINKING_ON: dict = {}


def _chat_text(tok, content, label=""):
    """Render this model's own chat template with reasoning suppressed, and CHECK THAT IT WORKED.

    Asking whether the call raised is the wrong question, and asking it is how the asymmetry
    survived a whole run. Two ways this silently fails and neither throws:

      - `{"enable_thinking": False}` raises, the bare `{}` retry succeeds, and the model runs
        thinking-ON. The old loop only recorded a fallback when BOTH attempts failed, so this middle
        branch wrote nothing at all.
      - the template simply has no `enable_thinking` branch and ignores the kwarg. Both Bonsai repos
        are in this case: they suppress reasoning unconditionally in their own template, so the flag
        is inert and the result happens to be right — but a model whose template ignored the flag
        and defaulted ON would look identical from the call site.

    So the guarantee is taken from the rendered string, which is the thing that actually reaches the
    model. Same check the GGUF path makes, so both paths are held to one standard."""
    from bench.tasks import THINK_CLOSE
    msgs = [{"role": "user", "content": content}]
    last = ""
    for kwargs in ({"enable_thinking": False}, {}):
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           **kwargs)
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            continue
        if THINK_CLOSE not in text and label not in THINKING_ON:
            THINKING_ON[label] = kwargs
            sys.stderr.write(
                f"  !! {label}: REASONING IS STILL ENABLED (rendered with {kwargs or 'no kwargs'}; "
                f"no {THINK_CLOSE} in the prompt). This model gets test-time compute the others do "
                f"not — its scores are not comparable with theirs.\n")
        return text
    # LOUD, ONCE PER MODEL. `except: continue` used to swallow this, and the first time it fired
    # for real the cause was a missing jinja2 — which no result would have hinted at.
    if label not in TEMPLATE_FALLBACKS:
        TEMPLATE_FALLBACKS[label] = last
        sys.stderr.write(f"  !! {label}: NO CHAT TEMPLATE APPLIED ({last}). This model's scores "
                         f"measure the prompt, not the model.\n")
    return content


def preflight() -> list:
    """Everything checkable WITHOUT a GPU, checked before one is rented. Three launches died on
    things a HEAD request knew — an ssh line, a card that was not attached, a legacy dataset id."""
    import urllib.error
    import urllib.request

    def head(url):
        try:
            urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "ralph-bench-preflight"}),
                timeout=45).read(1)
            return ""
        except urllib.error.HTTPError as e:
            return f"HTTP {e.code}"
        except Exception as e:
            return f"{type(e).__name__}"

    bad = []
    for ds in DATASET_IDS:
        why = head(f"https://huggingface.co/api/datasets/{ds}")
        if why:
            bad.append(f"dataset {ds} does not resolve ({why})")
    for label, spec in MODELS.items():
        why = head(f"https://huggingface.co/api/models/{spec['repo']}/tree/{spec['rev']}")
        if why:
            bad.append(f"{label}: {spec['repo']}@{spec['rev'][:12]} does not resolve ({why})")
    return bad


def wilson(k, n, z=1.96):
    """95% interval on a proportion. n=200 gave ±4-6%, which could not separate 86.5% from 88.5% —
    the exact comparison the first run existed to make. Every number is reported with its interval,
    so a reader is never invited to rank two results that overlap."""
    import math
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# --- runners. Both take READY user-message strings and return raw completions --------------------

_GGUF: dict = {}
_HF: dict = {}


def _no_think_handler(llm, label):
    """Force this GGUF's own chat template into Qwen3's NON-thinking mode.

    THIS IS THE FIX FOR THE RUN'S CENTRAL DEFECT. The template embedded in these GGUFs is
    byte-identical to stock Qwen3-8B's, and it emits the empty `<think></think>` prefill only when
    `enable_thinking` is explicitly false — undefined means thinking is ON. llama-cpp-python's
    `create_chat_completion` has no `**kwargs` and forwards a fixed argument list, so there is no
    call-site spelling that reaches the template variable. The result was a benchmark in which our
    two GGUF models reasoned before answering and the parent and both competitor models did not: our
    models got test-time compute the models they were being compared against were denied.

    `Jinja2ChatFormatter.__call__` does forward `**kwargs` into the jinja render, so a subclass that
    injects the variable reaches it. The formatter is then installed as `llm.chat_handler`, which
    `create_chat_completion` consults ahead of the format registry.

    Raises rather than degrades. A silent fallback here would restore the exact defect, and this
    function exists because that defect survived a whole run undetected."""
    from llama_cpp import llama_chat_format
    from bench.tasks import THINK_CLOSE

    template = llm.metadata.get("tokenizer.chat_template")
    if not template:
        raise RuntimeError(f"{label}: GGUF carries no tokenizer.chat_template — cannot guarantee "
                           f"the same prompt treatment as the transformers models")
    eos_id, bos_id = llm.token_eos(), llm.token_bos()

    class _NoThink(llama_chat_format.Jinja2ChatFormatter):
        def __call__(self, **kwargs):
            kwargs.setdefault("enable_thinking", False)
            return super().__call__(**kwargs)

    fmt = _NoThink(template=template,
                   eos_token=llm._model.token_get_text(eos_id),
                   bos_token=llm._model.token_get_text(bos_id) if bos_id != -1 else "",
                   stop_token_ids=[eos_id])

    # PROVE IT TOOK. Rendering one probe prompt costs nothing and is the difference between
    # believing thinking is off and knowing it. If a future llama-cpp-python changes the formatter's
    # internals, this aborts the run instead of quietly producing another confounded table.
    probe = fmt(messages=[{"role": "user", "content": "probe"}]).prompt
    if THINK_CLOSE not in probe:
        raise RuntimeError(f"{label}: enable_thinking=False did not reach the template — the "
                           f"rendered prompt has no {THINK_CLOSE}. Refusing to score.")
    RUNTIME.setdefault(label, {})["prompt_tail"] = probe[-60:]
    return fmt.to_chat_handler()


def run_gguf(label, spec, contents, max_new):
    from huggingface_hub import snapshot_download
    from llama_cpp import Llama
    import llama_cpp
    repo, rev = spec["repo"], spec["rev"]
    key = (repo, rev)
    if key not in _GGUF:
        d = snapshot_download(repo_id=repo, revision=rev, allow_patterns=["*.gguf"])
        path = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".gguf")][0]
        # ONE MODEL RESIDENT AT A TIME. Keeping every scored model loaded is what starved the tenth
        # submission of an 80 GB card during round 1 and read as a corrupt artifact.
        for k in list(_GGUF):
            try:
                _GGUF.pop(k).close()
            except Exception:
                pass
        # seed=1 because llama-cpp-python treats 0 as falsy and silently substitutes its own
        # default — a pin that was never applied. Greedy decoding makes it moot, but a line that
        # claims a pin it did not get is worse than no line.
        llm = Llama(model_path=path, n_ctx=8192, n_gpu_layers=-1, logits_all=False,
                    verbose=False, seed=1)
        llm.chat_handler = _no_think_handler(llm, label)
        RUNTIME.setdefault(label, {}).update(
            {"runtime": "llama.cpp", "llama_cpp": llama_cpp.__version__,
             "n_ctx": 8192, "chat_format": llm.chat_format, "thinking": False})
        _GGUF[key] = llm
    llm = _GGUF[key]
    out = []
    for i, c in enumerate(contents, 1):
        # llama.cpp reuses the longest common prefix of the previous request's KV cache, which makes
        # a result depend on what was scored before it. The transformers path evaluates every prompt
        # from scratch; this makes the two comparable and the GGUF numbers order-independent.
        llm.reset()
        try:
            r = llm.create_chat_completion(messages=[{"role": "user", "content": c}],
                                           max_tokens=max_new, temperature=0.0)
            out.append(r["choices"][0]["message"]["content"] or "")
        except Exception as e:
            # LOUD, AND COUNTED. This fallback drops to a bare untemplated prompt — precisely the
            # defect the module docstring says invalidated the first version of this harness — and
            # it used to fire silently, per item, leaving nothing in the results file. An empty
            # `template_fallbacks` was being read as proof the prompts were fine; it never covered
            # this path at all.
            RAW_FALLBACKS[label] = RAW_FALLBACKS.get(label, 0) + 1
            if RAW_FALLBACKS[label] == 1:
                sys.stderr.write(f"  !! {label}: FELL BACK TO AN UNTEMPLATED PROMPT "
                                 f"({type(e).__name__}: {e}). These scores measure the prompt.\n")
            out.append(llm(c, max_tokens=max_new, temperature=0.0,
                           echo=False)["choices"][0]["text"])
        if i % 100 == 0:
            sys.stderr.write(f"    {i}/{len(contents)}\n")
    return out


def run_hf(label, spec, contents, max_new):
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    repo, rev = spec["repo"], spec["rev"]
    key = (repo, rev)
    if key not in _HF:
        for k in list(_HF):
            del _HF[k]
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        cfg = AutoConfig.from_pretrained(repo, revision=rev)
        applied = {}
        for k, v in (spec.get("config") or {}).items():
            applied[k] = {"was": getattr(cfg, k, None), "now": v}
            setattr(cfg, k, v)
        # THE DTYPE THE CHECKPOINT WAS PUBLISHED IN, not one dtype for everybody. Casting an F16
        # release to BF16 trades three mantissa bits for nothing, and doing it to some models and
        # not others puts a precision difference inside a comparison that is supposed to isolate
        # compression.
        dtype = getattr(torch, spec.get("dtype", "bfloat16"))
        # `torch_dtype` was renamed to `dtype` in transformers 5, and the runner installs
        # transformers unpinned — the 2026-08-11 run got 5.15.0. An unrecognised keyword here is
        # absorbed into the config kwargs rather than raising, so the wrong spelling does not fail,
        # it loads in float32 and prints a table. Try the current spelling, fall back to the old one,
        # and let the assertion below be what actually decides.
        base = dict(revision=rev, config=cfg, device_map="auto", attn_implementation="eager")
        try:
            model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype, **base)
        except (TypeError, ValueError):
            model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype, **base)
        model.eval()
        # ASSERT THE DTYPE RATHER THAN ASSUME IT. `torch_dtype` is being renamed to `dtype` in
        # transformers, and an unrecognised keyword here is absorbed into the config kwargs instead
        # of raising — so the day the old spelling is dropped, every model silently loads in float32
        # and the table still prints. Nothing else in this harness would notice.
        got = next(model.parameters()).dtype
        if got != dtype:
            raise RuntimeError(f"{label}: asked for {dtype}, model loaded as {got} under "
                               f"transformers {transformers.__version__}. Neither dtype spelling "
                               f"took; refusing to score at a precision nobody chose.")
        RUNTIME.setdefault(label, {}).update(
            {"runtime": "transformers", "transformers": transformers.__version__,
             "torch": torch.__version__, "dtype": str(dtype),
             "attn": "eager", "config_overrides": applied})
        _HF[key] = (tok, model)
    tok, model = _HF[key]
    out = []
    for i, c in enumerate(contents, 1):
        text = _chat_text(tok, c, label)
        if i == 1:
            RUNTIME.setdefault(label, {})["prompt_tail"] = text[-60:]
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        out.append(tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
        if i % 100 == 0:
            sys.stderr.write(f"    {i}/{len(contents)}\n")
    return out


# How many raw completions to keep per (model, task), and how much of each.
SAMPLES_PER_TASK = 8
SAMPLE_CHARS = 1500


def _samples(items, texts, scored):
    """Keep a few raw completions, FAILURES FIRST.

    The last run produced 86 unexplained format-failures on the parent's MMLU and 800 on each GGUF
    model, and not one character of what those models actually emitted was written down. Diagnosing
    them means renting an H100 to reproduce something we already paid to observe. A format-failure
    is the one outcome whose cause is unguessable from the score alone, so failures are sampled
    first and successes only fill the remainder — the file stays small and the next question about
    a strange number is answerable for free."""
    pools = ([i for i, s in enumerate(scored) if s[1]],                       # format-failures
             [i for i, s in enumerate(scored) if s[2]],                       # truncations
             [i for i, s in enumerate(scored) if not any(s)],                 # plain wrong answers
             [i for i, s in enumerate(scored) if s[0]])                       # a couple that worked
    seen: set = set()

    def take(pool, want):
        """A strided walk, so the sample spans the run rather than its first few items — with 800
        items ordered by language, taking the head would sample Spanish and call it a suite."""
        step = max(1, len(pool) // max(1, want))
        for i in pool[::step] + pool:      # strided first, then anything left to top up
            if len(seen) >= SAMPLES_PER_TASK or want <= 0:
                return
            if i not in seen:
                seen.add(i)
                want -= 1

    for pool in pools:                     # a fair share of each, priority order
        take(pool, SAMPLES_PER_TASK // len(pools))
    for pool in pools:                     # then fill the slack the empty pools left behind
        take(pool, SAMPLES_PER_TASK)
    picked = sorted(seen)
    return [{"slice": items[i]["slice"], "gold": items[i]["gold"],
             "correct": bool(scored[i][0]), "format_fail": bool(scored[i][1]),
             "truncated": bool(scored[i][2]),
             "text": (texts[i] or "")[:SAMPLE_CHARS]} for i in sorted(picked)]


def _save(path, payload):
    """Write after EVERY completed (model, task), not once at the end.

    A run that hits its deadline mid-suite used to lose every result it had already produced —
    hours of GPU, recoverable only by scraping the log. The file is written to a temp path and
    renamed, so a kill during the write cannot leave a truncated JSON behind either."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def _assert_gpu() -> None:
    """Refuse to benchmark on CPU. The round path has had this since the CPU-fallback hole; the
    BENCHMARK path never did, and on 2026-08-18 it quietly spent 31 minutes and ~$3.60 running a
    rented H100 SXM box at 1252% CPU and 0% GPU.

    The cause was `CUDA error 802: system not yet initialized` — an H100 SXM board whose
    `nvidia-fabricmanager` would not start, so the NVLink fabric was never brought up and
    `cudaGetDeviceCount()` failed. torch was the correct `2.11.0+cu128` and the driver
    (570.148.08) supported it; nothing about the install was wrong, and nothing said so.

    It is not only a money failure. CPU and GPU do not produce identical logits, so a benchmark
    that silently changes device is a fourth confound in a harness rebuilt specifically to remove
    three of them — and it would have been the kind that favours nobody predictably, which is
    worse than one that favours us.

    `RALPH_ALLOW_CPU_BENCH=1` to accept it deliberately."""
    import os

    if os.environ.get("RALPH_ALLOW_CPU_BENCH") == "1":
        return
    try:
        import torch
    except Exception:
        return                      # no torch at all: the GGUF-only path is llama.cpp's problem
    if torch.cuda.is_available():
        print(f"  gpu: {torch.cuda.get_device_name(0)}")
        return
    hint = ""
    try:
        import subprocess
        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        if smi.returncode == 0 and smi.stdout.strip():
            hint = (f" `nvidia-smi` DOES see {smi.stdout.strip().splitlines()[0][:60]}, so the card "
                    f"is attached and the CUDA runtime cannot reach it — on an SXM board that is "
                    f"usually nvidia-fabricmanager failing to start.")
    except Exception:
        pass
    raise SystemExit(
        f"refusing to benchmark: torch reports no CUDA device, so every model would run on CPU."
        f"{hint} The numbers would not be comparable with any GPU run and the box bills either "
        f"way. Rent elsewhere, or set RALPH_ALLOW_CPU_BENCH=1 to accept it.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit-scale", type=float, default=1.0,
                    help="multiply every task's default n (0.05 for a smoke run)")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="bench-results-v2.json")
    ap.add_argument("--resume", action="store_true",
                    help="skip (model, task) pairs already present in --out")
    ap.add_argument("--langs", default="",
                    help="restrict MGSM to a subset, e.g. bn,te,th. Without this every language in "
                         "MGSM_LANGS runs, so adding three costs a full re-run of all nine.")
    a = ap.parse_args(argv)

    if a.langs:
        print(f"  mgsm restricted to: {', '.join(select_mgsm_langs(a.langs))}")

    _assert_gpu()

    for p in preflight():
        print(f"  PREFLIGHT: {p}")

    want_models = [x.strip() for x in a.only.split(",") if x.strip()] or list(MODELS)
    want_tasks = [t.strip() for t in a.tasks.split(",") if t.strip() in TASKS]

    loaded = {}
    for t in want_tasks:
        n = max(1, int(TASKS[t]["default_n"] * a.limit_scale))
        loaded[t] = TASKS[t]["load"](n)
        sl = sorted({it["slice"] for it in loaded[t]})
        extra = f"  slices: {','.join(sl[:8])}" + ("…" if len(sl) > 8 else "")
        print(f"  {t:<10} {len(loaded[t]):>5} items{extra if len(sl) > 1 else ''}")
    print()

    results: dict = {}
    if a.resume and os.path.exists(a.out):
        try:
            results = json.load(open(a.out)).get("results", {})
            done = sum(len(v.get("tasks", {})) for v in results.values())
            print(f"  resuming: {done} (model, task) pair(s) already done\n")
        except Exception as e:
            print(f"  could not read {a.out} to resume ({type(e).__name__}); starting clean\n")
            results = {}

    def payload():
        return {"limit_scale": a.limit_scale, "tasks": want_tasks, "results": results,
                "template_fallbacks": TEMPLATE_FALLBACKS, "raw_fallbacks": RAW_FALLBACKS,
                "thinking_on": THINKING_ON,
                # read through the module, not a from-import: select_mgsm_langs rebinds it, and a
                # from-import would freeze the nine-language default into every restricted run's
                # provenance record.
                "mgsm_langs": list(tasks.MGSM_SELECTED), "runtime": RUNTIME}

    for label in want_models:
        if label not in MODELS:
            print(f"  ! unknown model {label!r}")
            continue
        spec = MODELS[label]
        kind, repo, rev = spec["kind"], spec["repo"], spec["rev"]
        results.setdefault(label, {"repo": repo, "revision": rev, "tasks": {}})
        for t in want_tasks:
            prev = results[label]["tasks"].get(t) or {}
            # A CELL THAT SCORED ZERO ON EVERY ITEM IS NOT "DONE". --resume used to accept
            # 0/800 with format_fail=800 as a completed result, so pointing a fixed run at an old
            # results file would skip exactly the cells the fix existed to repair and report a clean
            # re-run that changed nothing.
            unusable = prev and prev.get("n") and (
                prev.get("format_fail", 0) + prev.get("truncated", 0) >= prev["n"])
            if a.resume and prev and "error" not in prev and not unusable:
                print(f"== {label} / {t}  (already done, skipping)")
                continue
            if a.resume and unusable:
                print(f"== {label} / {t}  (previous result was 100% unscoreable — RE-RUNNING)")
            items = loaded[t]
            print(f"== {label} / {t}  ({len(items)} items)")
            t0 = time.time()
            try:
                texts = (run_gguf if kind == "gguf" else run_hf)(
                    label, spec, [it["prompt"] for it in items], TASKS[t]["max_new"])
            except Exception as e:
                print(f"   FAILED: {type(e).__name__}: {e}\n")
                results[label]["tasks"][t] = {"error": f"{type(e).__name__}: {e}"}
                _save(a.out, payload())
                continue
            ok = fmt = trunc = 0
            per_slice: dict = {}
            scored = []
            for it, tx in zip(items, texts):
                c, f, tr = score(it, tx)
                scored.append((c, f, tr))
                ok += c
                fmt += f
                trunc += tr
                s = per_slice.setdefault(it["slice"], [0, 0])
                s[0] += c
                s[1] += 1
            lo, hi = wilson(ok, len(items))
            results[label]["tasks"][t] = {
                "correct": ok, "n": len(items), "accuracy": ok / len(items),
                "ci95": [lo, hi], "format_fail": fmt, "truncated": trunc,
                "per_slice": {k: {"correct": v[0], "n": v[1]} for k, v in per_slice.items()},
                "samples": _samples(items, texts, scored),
                "seconds": round(time.time() - t0, 1)}
            _save(a.out, payload())        # survive a deadline, by design rather than by luck
            print(f"   {ok}/{len(items)} = {ok / len(items):.1%}  [{lo:.1%}, {hi:.1%}]   "
                  f"format-fail {fmt}   truncated {trunc}   "
                  f"({(time.time() - t0) / 60:.1f} min)\n")

    # --- summary
    print("\n  model               " + "".join(f"{t:>15}" for t in want_tasks))
    print("  " + "-" * (20 + 15 * len(want_tasks)))
    for label in want_models:
        if label not in results:
            continue
        row = f"  {label:<20}"
        for t in want_tasks:
            r = results[label]["tasks"].get(t) or {}
            if not r:
                row += f"{'-':>15}"
            elif "error" in r:
                row += f"{'ERR':>15}"
            else:
                mark = "*" if r["format_fail"] else " "
                row += f"{r['accuracy']:>14.1%}{mark}"
        print(row)
    print("\n  * = some completions ignored the answer format; see format_fail. A low score with a "
          "high\n    format-fail is an instruction-following collapse, not lost knowledge.")

    # multilingual detail — the axis GSM8K cannot see, and the one worst-slice claims to protect
    if "mgsm" in want_tasks:
        langs = sorted({s for lbl in results for s in
                        (results[lbl]["tasks"].get("mgsm") or {}).get("per_slice", {})})
        if langs:
            print("\n  mgsm by language" + "".join(f"{l:>8}" for l in langs))
            print("  " + "-" * (18 + 8 * len(langs)))
            for label in want_models:
                ps = (results.get(label, {}).get("tasks", {}).get("mgsm") or {}).get("per_slice")
                if not ps:
                    continue
                row = f"  {label:<18}"
                for l in langs:
                    v = ps.get(l)
                    row += f"{(v['correct'] / v['n']):>7.0%} " if v and v["n"] else f"{'-':>8}"
                print(row)

    if TEMPLATE_FALLBACKS:
        print("\n  !! MODELS WITHOUT THEIR CHAT TEMPLATE — those scores measure the prompt:")
        for k, v in TEMPLATE_FALLBACKS.items():
            print(f"     {k}: {v}")
    if RAW_FALLBACKS:
        print("\n  !! GGUF ITEMS SCORED ON A BARE, UNTEMPLATED PROMPT:")
        for k, v in RAW_FALLBACKS.items():
            print(f"     {k}: {v} item(s)")
    if THINKING_ON:
        print("\n  !! THESE MODELS REASONED BEFORE ANSWERING AND THE OTHERS DID NOT.")
        print("     The run is not like-for-like. Do not publish any cross-model comparison.")
        for k, v in THINKING_ON.items():
            print(f"     {k}: rendered with {v or 'no kwargs'}")

    # The prompt each model was actually sent, side by side. Every model here shares one parent and
    # one template, so these tails should be identical; when they were not, it took an audit of
    # library source to find out, because nothing in the run said so.
    tails = {k: v.get("prompt_tail") for k, v in RUNTIME.items() if v.get("prompt_tail")}
    if len(set(tails.values())) > 1:
        print("\n  !! MODELS WERE SENT DIFFERENT PROMPT ENDINGS — treatment is NOT identical:")
        for k, v in tails.items():
            print(f"     {k:<20} …{v!r}")

    _save(a.out, payload())
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
