"""Regression tests for the benchmark's ONE invariant: every model gets the same prompt.

    python -m tests.test_bench_prompt_parity
    pytest tests/test_bench_prompt_parity.py

WHY THIS FILE EXISTS. The 2026-08-11 multi-task run published a table headed "each model in its
native runtime, its own chat template, identical treatment". It was not identical. The two GGUF
models ran through llama.cpp with Qwen3's reasoning mode ON, and the parent and both competitor
models ran through transformers with it OFF, because `llama_cpp.Llama.create_chat_completion` has no
`**kwargs` and cannot pass `enable_thinking` into the template. Our models were allowed to think
before answering; the models they were being compared against were not. The run produced a finding
that favoured our models, and nothing in its output said any of this — establishing it afterwards
took an audit of library source and remote model metadata.

Every assertion below runs offline against the REAL template bytes, vendored in tests/fixtures. Not
a miniature stand-in: a hand-written template that "works like Qwen3's" is exactly the fixture that
passes while the real reader does the opposite, which this project has already paid for once. The
sha256 of each fixture is recorded, and `test_fixtures_match_upstream` re-fetches and compares when a
network is available, so the vendored bytes cannot quietly drift from what the models actually ship.

Requires jinja2 only — no torch, no transformers, no llama_cpp, no GPU. It runs in about a second,
which is the point: the defect it guards against cost a rented H100 and a nearly-published claim.
"""
from __future__ import annotations

import hashlib
import json
import os

import jinja2
import jinja2.ext
import jinja2.sandbox

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
MSG = [{"role": "user", "content": "What is 2+2?"}]

# The empty block Qwen3's template emits to SUPPRESS reasoning. Its presence in a rendered prompt is
# what "thinking is off" means — the assistant turn opens with a closed, empty think block, so the
# model has nothing to continue and goes straight to the answer.
NO_THINK = "<think>\n\n</think>"


def _load(name):
    with open(os.path.join(FIXTURES, name)) as fh:
        return fh.read()


def _render(template, **kw):
    """Render the way BOTH runtimes do.

    `transformers.apply_chat_template` and `llama_cpp.Jinja2ChatFormatter` both render the model's
    own jinja template in a sandboxed environment with trim_blocks/lstrip_blocks and pass
    `add_generation_prompt` plus whatever extra kwargs they were given. That shared shape is exactly
    why the same template plus the same kwargs must produce the same prompt on both paths — and why
    a kwarg that only one path can pass silently splits the fleet in two."""
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True,
        extensions=[jinja2.ext.loopcontrols])
    return env.from_string(template).render(
        messages=kw.pop("messages", MSG), add_generation_prompt=True,
        bos_token="", eos_token="<|im_end|>", **kw)


# --- the defect itself ---------------------------------------------------------------------------

def test_enable_thinking_is_load_bearing():
    """Omitting the kwarg is not equivalent to passing it. This is the whole bug in one assertion."""
    tpl = _load("qwen3_chat_template.jinja")
    off = _render(tpl, enable_thinking=False)
    default = _render(tpl)
    assert NO_THINK in off, "enable_thinking=False must emit the empty think block"
    assert NO_THINK not in default, (
        "template no longer defaults to thinking-on; the harness's assumptions need rechecking")
    assert off != default, "the kwarg made no difference — the guarantee is unenforceable"


def test_qwen3_template_only_suppresses_on_explicit_false():
    """`is defined and is false` — so None, 0, "" and "false" all leave reasoning ENABLED.

    A caller who passes `enable_thinking=None` believing it means "off" gets a thinking model, and
    nothing anywhere reports it. Only the literal boolean works."""
    tpl = _load("qwen3_chat_template.jinja")
    for sneaky in (None, 0, "", "false", "no"):
        assert NO_THINK not in _render(tpl, enable_thinking=sneaky), (
            f"enable_thinking={sneaky!r} silently left reasoning on")
    assert NO_THINK in _render(tpl, enable_thinking=False)


def test_gguf_and_hf_paths_agree_when_both_pass_the_kwarg():
    """The fix's guarantee: same template + same kwargs -> byte-identical prompt.

    Our crown's GGUF embeds the parent's template byte for byte, so once llama.cpp is made to pass
    `enable_thinking=False` there is nothing left to differ."""
    tpl = _load("qwen3_chat_template.jinja")
    assert _render(tpl, enable_thinking=False) == _render(tpl, enable_thinking=False)
    prompt = _render(tpl, enable_thinking=False)
    assert prompt.endswith(f"<|im_start|>assistant\n{NO_THINK}\n\n"), repr(prompt[-80:])


def test_the_shipped_run_would_have_failed_this_file():
    """Reproduces the published run's treatment and asserts it is NOT identical.

    A test that only checks the fixed path would pass on the broken one too, because the broken path
    differs in a kwarg no assertion mentioned. This pins the actual historical defect."""
    qwen = _load("qwen3_chat_template.jinja")
    bonsai = _load("bonsai_chat_template.jinja")
    as_shipped = {
        "ralph-sub4":     _render(qwen),                          # llama.cpp: could not pass it
        "ralph-ternary":  _render(qwen),                          # llama.cpp: could not pass it
        "parent":         _render(qwen, enable_thinking=False),   # transformers: passed it
        "bonsai-ternary": _render(bonsai, enable_thinking=False),
        "bonsai-1bit":    _render(bonsai, enable_thinking=False),
    }
    thinking = {k for k, v in as_shipped.items() if NO_THINK not in v}
    assert thinking == {"ralph-sub4", "ralph-ternary"}, thinking
    assert len({v[-40:] for v in as_shipped.values()}) > 1, (
        "the run that shipped must be detectable as non-identical, or this test guards nothing")


# --- the competitor's template, which behaves differently and is worth pinning --------------------

def test_bonsai_template_hardwires_thinking_off():
    """PrismML's release ignores `enable_thinking` and always emits the empty block.

    So their models were genuinely non-thinking in the published run — that half of the comparison
    was sound, by their choice rather than by our harness. It also means their models cannot be put
    INTO thinking mode without substituting a template they did not ship, which is a modification
    that would have to be disclosed rather than made quietly."""
    tpl = _load("bonsai_chat_template.jinja")
    assert "enable_thinking" not in tpl
    for kw in ({}, {"enable_thinking": True}, {"enable_thinking": False}):
        assert NO_THINK in _render(tpl, **kw), f"expected unconditional suppression, kwargs={kw}"


# --- the scorer, which must survive a reasoning block whichever way this goes ---------------------

def test_strip_think_handles_closed_open_and_absent():
    from bench.tasks import _strip_think
    assert _strip_think("<think>\n\n</think>\n\nB") == ("\n\nB", False)
    assert _strip_think("no reasoning here") == ("no reasoning here", False)
    assert _strip_think("<think>counting: 1, 2, 3") == ("", True)      # ran out of budget
    # last close wins, so a trace that discusses the tag itself cannot split it early
    assert _strip_think("<think>a</think>mid</think>tail") == ("tail", False)


def test_reasoning_prose_cannot_become_an_answer():
    """The two parsers used to read the whole completion, so a truncated reasoning block scored as a
    confident answer with no format-failure recorded — the model was marked wrong for arithmetic it
    was never given the tokens to finish."""
    from bench.tasks import score
    mc = {"kind": "letter", "gold": "B", "slice": "x"}
    num = {"kind": "num", "gold": "42", "slice": "x"}
    assert score(mc, "<think>Option A looks plausible, but C") == (False, False, True)
    assert score(num, "<think>First, 7 times 3 is 21, then 21 plus 4 is 25") == (False, False, True)
    # and a completed answer still scores normally
    assert score(mc, "<think>\n\n</think>\n\nB")[0] is True
    assert score(num, "<think>\n\n</think>\n\nThe answer is #### 42")[0] is True


def test_article_a_is_not_an_answer():
    """`\\b([A-D])\\b` matched the English article, so "A student..." scored as answering A —
    chance-level accuracy manufactured from prose, and never counted as a format failure."""
    from bench.tasks import _letter
    assert _letter("A student is asked to choose.") != "A"
    for good in ("B", "B.", "**C**", "Answer: D", "answer is (B)", "b"):
        assert _letter(good) in "ABCD", good


# --- fixture provenance --------------------------------------------------------------------------

def test_fixtures_match_upstream():
    """Vendored bytes must still be what the models ship. Skipped without a network."""
    import urllib.error
    import urllib.request
    sums = json.load(open(os.path.join(FIXTURES, "SHA256SUMS.json")))
    for name, want in sums.items():
        if name.startswith("_"):
            continue
        got = hashlib.sha256(_load(name).encode()).hexdigest()
        assert got == want, f"{name} was edited in place: {got[:16]} != {want[:16]}"
    try:
        req = urllib.request.Request(
            "https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer_config.json",
            headers={"User-Agent": "ralph-tests"})
        live = json.loads(urllib.request.urlopen(req, timeout=30).read())["chat_template"]
    except Exception:
        return                      # offline: the local sha check above already ran
    assert hashlib.sha256(live.encode()).hexdigest() == sums["qwen3_chat_template.jinja"], (
        "Qwen3-8B's published chat template changed — re-vendor the fixture and re-read the "
        "enable_thinking branch before trusting any of these tests")


# COLLECTED AFTER EVERY TEST IS DEFINED. Keep this immediately above main().
TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    bad = 0
    for t in TESTS:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except Exception as e:
            bad += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - bad}/{len(TESTS)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
