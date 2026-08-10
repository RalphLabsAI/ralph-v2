"""Benchmark Ralph's crowned models against PrismML's Bonsai, on the same questions.

DELIBERATELY NOT THE SUBNET'S SCORER. Retention is a token-level KL against a pinned parent; it is
the right measure for ranking compressions of one model and the wrong one for answering "is this
any good". This asks the plain question instead — given the same problems, how many does each model
get right — which is the question anyone comparing two compressed models actually has.

Each model runs in its NATIVE runtime: GGUF through llama.cpp, safetensors through transformers.
For a KL that would matter; for "did it reach the right number" it does not, and forcing one runtime
would mean re-quantising somebody's model and measuring our conversion instead of their work.

    python -m bench.compare --limit 200
    python -m bench.compare --limit 200 --only ralph-ternary,bonsai-ternary
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

# label -> (kind, repo, revision-or-file). `gguf` runs on llama.cpp, `hf` on transformers.
MODELS = {
    # the pinned parent: the ceiling everything here is compressed from
    "parent-qwen3-8b":  ("hf", "Qwen/Qwen3-8B", "main"),

    # Ralph round-1 crowns, exactly the bytes that were scored and anchored
    "ralph-ternary":    ("gguf", "tensor-tailor/ralph-qwen3-8b-ternary",
                         "301e4db4e8889831856eabd58e0824034d4ed236"),
    "ralph-sub4":       ("gguf", "tensor-tailor/ralph-qwen3-8b-sub4",
                         "2a2e1bcf9fa9e53165c78be66c9be14d3f9cc1c7"),

    # PrismML. Their COMPRESSED ggufs advertise file_type 141 / 41, which stock llama.cpp cannot
    # load — a custom fork's types — so the unpacked safetensors are the only artifact of theirs we
    # can run without adopting their toolchain.
    "bonsai-ternary":   ("hf", "prism-ml/Ternary-Bonsai-8B-unpacked", "main"),
    "bonsai-1bit":      ("hf", "prism-ml/Bonsai-8B-unpacked", "main"),
}

ANS = re.compile(r"(-?[\d,]*\.?\d+)")


def _gold(a: str) -> str:
    return a.split("####")[-1].strip().replace(",", "")


def _pred(text: str) -> str:
    """Last number in the completion. Crude and applied identically to every model, which is what
    makes it fair — a per-model parser is where benchmark results quietly become authored."""
    hits = ANS.findall(text.replace(",", ""))
    return hits[-1].strip(".") if hits else ""


# `openai/gsm8k`, NOT the bare `gsm8k`. The legacy unnamespaced id no longer resolves — the hub
# rejects it before any download with `Repository id must be 'namespace/name'` — so the benchmark
# died on its first line after a clean fifteen-minute install.
GSM8K = "openai/gsm8k"


def preflight() -> list:
    """Everything checkable WITHOUT a GPU, checked before one is rented.

    Three launches died on things a HEAD request knew: a legacy dataset id the hub no longer
    resolves, and before that an ssh line and a card that was not there. Each cost a rental and
    fifteen minutes to discover something that takes two seconds to ask. `run_orchestrated` already
    HEADs every observer before renting for exactly this reason; this file simply did not.

    Returns a list of problems. Empty means every repo and revision named here exists."""
    import urllib.error
    import urllib.request

    def head(url):
        try:
            urllib.request.urlopen(
                urllib.request.Request(url, method="GET",
                                       headers={"User-Agent": "ralph-bench-preflight"}),
                timeout=45).read(1)
            return ""
        except urllib.error.HTTPError as e:
            return f"HTTP {e.code}"
        except Exception as e:
            return f"{type(e).__name__}"

    bad = []
    why = head(f"https://huggingface.co/api/datasets/{GSM8K}")
    if why:
        bad.append(f"dataset {GSM8K} does not resolve ({why})")
    for label, (kind, repo, rev) in MODELS.items():
        why = head(f"https://huggingface.co/api/models/{repo}/tree/{rev}")
        if why:
            bad.append(f"{label}: {repo}@{rev[:12]} does not resolve ({why})")
    return bad


def load_questions(limit: int):
    from datasets import load_dataset
    ds = load_dataset(GSM8K, "main", split="test")
    return [(ds[i]["question"], _gold(ds[i]["answer"])) for i in range(min(limit, len(ds)))]


def run_gguf(repo, rev, questions, max_new):
    from huggingface_hub import snapshot_download
    from llama_cpp import Llama
    d = snapshot_download(repo_id=repo, revision=rev, allow_patterns=["*.gguf"])
    path = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".gguf")][0]
    llm = Llama(model_path=path, n_ctx=8192, n_gpu_layers=-1, logits_all=False,
                verbose=False, seed=0)
    out = []
    for i, q in enumerate(questions, 1):
        msgs = [{"role": "user", "content": INSTRUCTION.format(q=q)}]
        try:
            # the GGUF carries its own chat template; llama.cpp applies it
            r = llm.create_chat_completion(messages=msgs, max_tokens=max_new, temperature=0.0)
            out.append(r["choices"][0]["message"]["content"] or "")
        except Exception:
            r = llm(PROMPT.format(q=q), max_tokens=max_new, temperature=0.0, echo=False)
            out.append(r["choices"][0]["text"])
        if i % 20 == 0:
            sys.stderr.write(f"    {i}/{len(questions)}\n")
    try:
        llm.close()
    except Exception:
        pass
    return out


def run_hf(repo, rev, questions, max_new):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    prompts = [_chat_text(tok, q) for q in questions]
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, torch_dtype=torch.bfloat16,
                                                 device_map="auto",
                                                 attn_implementation="eager")
    model.eval()
    out = []
    for i, p in enumerate(prompts, 1):
        ids = tok(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        out.append(tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
        if i % 20 == 0:
            sys.stderr.write(f"    {i}/{len(prompts)}\n")
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return out


INSTRUCTION = ("Solve the problem. Reason briefly, then give the final numeric answer on its own "
               "line after '####'.\n\nProblem: {q}")

# THE RAW-COMPLETION VERSION SCORED THE PARENT AT 35.5%. Qwen3-8B does not do that badly on GSM8K —
# it is a hybrid-reasoning model that expects its own chat format, and fed a bare completion prompt
# it never reliably enters the mode where it does the arithmetic. Every model here therefore gets
# ITS OWN template, applied by its own tokenizer, which is the only way "same treatment" means
# anything across models trained by different people.
#
# Thinking is disabled where the template supports it. That is a fairness decision, not a
# performance one: a reasoning trace that overruns max_new leaves the parser reading whatever
# number came last, and it would penalise whichever model reasons longest rather than whichever
# reasons better.
PROMPT = INSTRUCTION + "\n\nSolution:"     # fallback for a tokenizer with no chat template


def _chat_text(tok, q):
    msgs = [{"role": "user", "content": INSTRUCTION.format(q=q)}]
    for kwargs in ({"enable_thinking": False}, {}):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           **kwargs)
        except Exception:
            continue
    return PROMPT.format(q=q)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=200)
    # 320 TRUNCATED THE REASONING and the parser then read whichever number came last. Qwen3
    # reasons at length; the fix is to give it room rather than to make the parser cleverer.
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="bench-results.json")
    a = ap.parse_args(argv)

    want = [x.strip() for x in a.only.split(",") if x.strip()] or list(MODELS)
    for problem in preflight():
        print(f"  PREFLIGHT: {problem}")
    qs = load_questions(a.limit)
    prompts = [PROMPT.format(q=q) for q, _ in qs]
    golds = [g for _, g in qs]
    print(f"{len(qs)} GSM8K questions, max_new={a.max_new}\n")

    results = {}
    for label in want:
        if label not in MODELS:
            print(f"  ! unknown model {label!r}, skipping")
            continue
        kind, repo, rev = MODELS[label]
        print(f"== {label}  ({kind}) {repo}@{rev[:12]}")
        t0 = time.time()
        try:
            texts = (run_gguf if kind == "gguf" else run_hf)(repo, rev, prompts, a.max_new)
        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {e}\n")
            results[label] = {"error": f"{type(e).__name__}: {e}"}
            continue
        hits = sum(1 for t, g in zip(texts, golds) if _pred(t) == g)
        dt = time.time() - t0
        results[label] = {"correct": hits, "n": len(qs), "accuracy": hits / len(qs),
                          "seconds": round(dt, 1), "repo": repo, "revision": rev}
        print(f"   {hits}/{len(qs)} = {hits / len(qs):.1%}   ({dt / 60:.1f} min)\n")

    print("\n  model                accuracy   correct   time")
    print("  " + "-" * 52)
    for label in want:
        r = results.get(label) or {}
        if "error" in r:
            print(f"  {label:<20} FAILED     {r['error'][:40]}")
        elif r:
            print(f"  {label:<20} {r['accuracy']:>7.1%}   {r['correct']:>3}/{r['n']:<5} "
                  f"{r['seconds'] / 60:>5.1f}m")
    with open(a.out, "w") as fh:
        json.dump({"limit": a.limit, "max_new": a.max_new, "results": results}, fh, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
