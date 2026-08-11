# tools/

Read-only investigators that earned their keep. None of them touch a round.

| tool | what it answers |
|---|---|
| `gguf_probe.py` | what quantiser and which imatrix calibration produced a GGUF — read from the header over HTTP, no download. Every submission advertises both in plaintext. |
| `bitcheck.py` | is an "unpacked" safetensors model REALLY low-bit, or the full-precision original? Counts distinct weight values INSIDE a group; across a whole sample ternary-with-scales looks like a 6-bit codebook. |
| `finish_round.py` | finish a round whose scoring completed but whose tail did not. Re-audits, checks `prev_anchor` against the published head, then signs/publishes/anchors from disk. No GPU. |

`finish_round.py` recovered round 2 after the audit rejected it with the rental already destroyed —
`run_remote_round` writes `record.json` and `pool.jsonl` BEFORE auditing, so the scoring survives a
failed audit and only the tail needs re-running.

    python -m tools.gguf_probe "label=<hf-user>/<repo>@<rev>"
    python -m tools.bitcheck prism-ml/Ternary-Bonsai-8B-unpacked
    python -m tools.finish_round --round 2            # dry
    python -m tools.finish_round --round 2 --commit
