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

from bench.tasks import DATASET_IDS, TASKS, score

# label -> (kind, repo, revision). `gguf` runs on llama.cpp, `hf` on transformers.
MODELS = {
    "parent-qwen3-8b":  ("hf", "Qwen/Qwen3-8B", "main"),

    # Ralph crowns — exactly the bytes that were scored and anchored
    "ralph-ternary":    ("gguf", "tensor-tailor/ralph-qwen3-8b-ternary",
                         "301e4db4e8889831856eabd58e0824034d4ed236"),
    "ralph-sub4":       ("gguf", "tensor-tailor/ralph-qwen3-8b-sub4",
                         "2a2e1bcf9fa9e53165c78be66c9be14d3f9cc1c7"),

    # PrismML. Their COMPRESSED ggufs advertise file_type 141 / 41 — a custom fork's types that
    # stock llama.cpp cannot load — so the unpacked safetensors are the only artifact of theirs
    # runnable without adopting their toolchain. Verified genuinely ternary: exactly 3 distinct
    # values in every 128-weight window across three tensors.
    "bonsai-ternary":   ("hf", "prism-ml/Ternary-Bonsai-8B-unpacked", "main"),
    "bonsai-1bit":      ("hf", "prism-ml/Bonsai-8B-unpacked", "main"),
}

# Set when any model had to fall back to a raw prompt. A silent fallback produces numbers that look
# fine and measure the prompt; if it fires, every result in the run is suspect and must say so.
TEMPLATE_FALLBACKS: dict = {}


def _chat_text(tok, content, label=""):
    msgs = [{"role": "user", "content": content}]
    last = ""
    for kwargs in ({"enable_thinking": False}, {}):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           **kwargs)
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
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
    for label, (kind, repo, rev) in MODELS.items():
        why = head(f"https://huggingface.co/api/models/{repo}/tree/{rev}")
        if why:
            bad.append(f"{label}: {repo}@{rev[:12]} does not resolve ({why})")
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


def run_gguf(repo, rev, contents, max_new):
    from huggingface_hub import snapshot_download
    from llama_cpp import Llama
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
        _GGUF[key] = Llama(model_path=path, n_ctx=8192, n_gpu_layers=-1, logits_all=False,
                           verbose=False, seed=0)
    llm = _GGUF[key]
    out = []
    for i, c in enumerate(contents, 1):
        try:
            r = llm.create_chat_completion(messages=[{"role": "user", "content": c}],
                                           max_tokens=max_new, temperature=0.0)
            out.append(r["choices"][0]["message"]["content"] or "")
        except Exception:
            out.append(llm(c, max_tokens=max_new, temperature=0.0,
                           echo=False)["choices"][0]["text"])
        if i % 100 == 0:
            sys.stderr.write(f"    {i}/{len(contents)}\n")
    return out


def run_hf(repo, rev, contents, max_new):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    key = (repo, rev)
    if key not in _HF:
        for k in list(_HF):
            del _HF[k]
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        model = AutoModelForCausalLM.from_pretrained(repo, revision=rev,
                                                     torch_dtype=torch.bfloat16,
                                                     device_map="auto",
                                                     attn_implementation="eager")
        model.eval()
        _HF[key] = (tok, model)
    tok, model = _HF[key]
    out = []
    for i, c in enumerate(contents, 1):
        ids = tok(_chat_text(tok, c, repo), return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        out.append(tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
        if i % 100 == 0:
            sys.stderr.write(f"    {i}/{len(contents)}\n")
    return out


def _save(path, payload):
    """Write after EVERY completed (model, task), not once at the end.

    A run that hits its deadline mid-suite used to lose every result it had already produced —
    hours of GPU, recoverable only by scraping the log. The file is written to a temp path and
    renamed, so a kill during the write cannot leave a truncated JSON behind either."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit-scale", type=float, default=1.0,
                    help="multiply every task's default n (0.05 for a smoke run)")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="bench-results-v2.json")
    ap.add_argument("--resume", action="store_true",
                    help="skip (model, task) pairs already present in --out")
    a = ap.parse_args(argv)

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
                "template_fallbacks": TEMPLATE_FALLBACKS}

    for label in want_models:
        if label not in MODELS:
            print(f"  ! unknown model {label!r}")
            continue
        kind, repo, rev = MODELS[label]
        results.setdefault(label, {"repo": repo, "revision": rev, "tasks": {}})
        for t in want_tasks:
            if a.resume and t in results[label]["tasks"] \
                    and "error" not in results[label]["tasks"][t]:
                print(f"== {label} / {t}  (already done, skipping)")
                continue
            items = loaded[t]
            print(f"== {label} / {t}  ({len(items)} items)")
            t0 = time.time()
            try:
                texts = (run_gguf if kind == "gguf" else run_hf)(
                    repo, rev, [it["prompt"] for it in items], TASKS[t]["max_new"])
            except Exception as e:
                print(f"   FAILED: {type(e).__name__}: {e}\n")
                results[label]["tasks"][t] = {"error": f"{type(e).__name__}: {e}"}
                _save(a.out, payload())
                continue
            ok = fmt = 0
            per_slice: dict = {}
            for it, tx in zip(items, texts):
                c, f = score(it, tx)
                ok += c
                fmt += f
                s = per_slice.setdefault(it["slice"], [0, 0])
                s[0] += c
                s[1] += 1
            lo, hi = wilson(ok, len(items))
            results[label]["tasks"][t] = {
                "correct": ok, "n": len(items), "accuracy": ok / len(items),
                "ci95": [lo, hi], "format_fail": fmt,
                "per_slice": {k: {"correct": v[0], "n": v[1]} for k, v in per_slice.items()},
                "seconds": round(time.time() - t0, 1)}
            _save(a.out, payload())        # survive a deadline, by design rather than by luck
            print(f"   {ok}/{len(items)} = {ok / len(items):.1%}  [{lo:.1%}, {hi:.1%}]   "
                  f"format-fail {fmt}   ({(time.time() - t0) / 60:.1f} min)\n")

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

    _save(a.out, payload())
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
