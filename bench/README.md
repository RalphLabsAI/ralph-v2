# bench/ — capability benchmarks

Deliberately **not** the subnet's scorer. Retention is a token-level KL against a pinned parent: the
right measure for ranking compressions of one model, the wrong one for "is this any good". This asks
the plain question — given the same problems, how many does each model get right.

---

## ⛔ The results files in this directory are WITHDRAWN

**`results-2026-08-11-multitask.json` and `results-2026-08-10-gsm8k.json` are not valid comparisons
between models. Do not cite any cross-model number from either.**

They are left in place, unedited, because they are the honest record of what was measured. Deleting a
published result because it turned out to be wrong is the opposite of the discipline this subnet is
built on. But they must be read with the following.

### What went wrong

An audit of the harness on 2026-08-11, **before** the results were published anywhere, found three
independent defects. Every one of them favours our own models and penalises the competitor's:

| # | defect | verified against |
|---|---|---|
| 1 | The two GGUF models ran with **Qwen3's reasoning mode ON**; the parent and both Bonsai models ran with it **OFF**. Our models were allowed to think before answering; the models they were compared against were not. | The template embedded in the GGUFs is byte-identical to `Qwen/Qwen3-8B`'s (sha256 `a55ee1b1…`, 4168 bytes), which enables reasoning unless `enable_thinking` is explicitly `false`. `llama_cpp.Llama.create_chat_completion` has no `**kwargs` and forwards a fixed argument list, so that variable was unreachable from the call site. |
| 2 | Both Bonsai checkpoints ship **F16** and were hard-cast to **BF16** on load — three mantissa bits discarded on the competitor's weights only. The parent is natively BF16 and was untouched. | safetensors headers: `F16`×235 on both Bonsai repos, `BF16`×81 on the parent. |
| 3 | Both Bonsai repos ship `rope_scaling: {yarn, factor 4.0}`, which transformers applies **statically to every prompt**. The parent ships `rope_scaling: null`. Qwen's guidance is that static YaRN costs short-sequence quality, and every prompt here is short. | `config.json` in all three repos. |

Additionally, the multiple-choice columns (`mmlu`, `hellaswag`) are void for **all five** models:
`max_new=16` was too small even without a reasoning block, which is what produced 86/800
format-failures on the parent and 786–800/800 on the two GGUF models.

### What still stands

- Bonsai is **genuinely ternary** — exactly three distinct values in every 128-weight window across
  three tensors (`tools/bitcheck.py`). No inference involved, so no confound.
- Their compressed GGUFs advertise a custom `file_type` that stock llama.cpp cannot load, which is
  why the `-unpacked` safetensors were benchmarked instead.
- Bonsai's own chat template suppresses reasoning **unconditionally** — their models were genuinely
  non-thinking, by their choice rather than by our harness.
- Sampling was genuinely greedy and identical on both paths; `n_ctx` was never approached.

### What is being done

The harness is fixed on branch `fix/bench-treatment-parity`:

- the GGUF path is forced into non-thinking mode through a chat handler and **aborts** if a probe
  prompt comes back without the empty think block;
- each checkpoint loads in the dtype it was published in, asserted after load;
- Bonsai's static YaRN is disabled explicitly and recorded in the results, rather than silently;
- scoring reports truncation as a third outcome, so an unfinished reasoning block is no longer
  counted as a confident wrong answer;
- `tests/test_bench_prompt_parity.py` renders the real template bytes offline and fails if the two
  paths would send different prompts. One of its cases reproduces the treatment that shipped here and
  asserts it is detectably non-identical.

A clean re-run replaces these files. Until then there is no published capability claim from this
subnet, and the numbers in this directory are not one.

---

## Running it

```bash
python -m bench.compare                                   # everything
python -m bench.compare --tasks mgsm --langs bn,te,th     # one task, three languages
python -m bench.compare --limit-scale 0.05                # smoke run
python -m bench.run_remote --tasks mgsm --langs sw         # rent a GPU, run, destroy it
```

`--langs` exists because the language list is not free: without it, adding three languages to
`MGSM_LANGS` re-runs all nine for every model.
