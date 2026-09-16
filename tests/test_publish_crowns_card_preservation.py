"""The crowns card belongs to the maintainer; the publisher owns one delimited block of it.

The first publisher replaced README.md with its template on every crown change — one dethrone
away from deleting the launch copy, the device receipts and the reviewed license section that no
round produced. These tests pin the contract that replaced it: every byte outside the managed
block survives a publish, the legacy table section is adopted into the block exactly once, the
front matter is merged rather than rewritten, and a license declaration is bound to the bytes it
was reviewed for."""
from __future__ import annotations

import pytest

import eval.publish_crowns as pc

LIVE = """---
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
base_model: Qwen/Qwen3-8B
tags: [gguf, conversational, qwen3, quantized, compression, bittensor]
---

<p align="center"><img src="banner.png" alt="Ralph"></p>

# The same model. Small enough for your phone.

## Start here: the 2.94 GB sub2 crown

Hand-written launch copy.

<!-- ralph:managed:physical-iphone-receipts:begin -->
## Physical-iPhone device lab

| iPhone 17 Pro Max | 18.54 tok/s |
<!-- ralph:managed:physical-iphone-receipts:end -->

## Current crowns — through Round 7

| File | Tier |
|---|---|
| `ralph-qwen3-8b-sub2.gguf` | sub2 |

### Exact checksums

```text
deadbeef  ralph-qwen3-8b-sub2.gguf
```

The same filenames are in crowns.json.

## What the crown means

Prose about the crown.

## License

Each exact GGUF listed above is Apache-2.0.
"""

META = {"filename": "ralph-qwen3-8b-sub2.gguf", "file_bytes": 2_937_263_168,
        "file_sha256": "9" * 64, "chat_template_present": True}


def _entry(model_id: str = "same", **over) -> dict:
    e = {"model_id": model_id, "tier": "sub2", "round": 7, "retention": 0.282457,
         "miner": "builder-hotkey", "source_repo": "builder/model", "source_revision": "abc123",
         "code_bits": 2.2626, "container_bits": 2.81, "record": "https://example.test/round-7.json",
         **META}
    e.update(over)
    return e


def test_hand_written_card_survives_and_only_the_crowns_section_is_replaced():
    m = {"sub2": pc.declare_license(_entry(), "Apache-2.0")}
    card = pc.readme(m, "https://example.test/round-7.json", "org/crowns",
                     record={"round": 7}, existing_card=LIVE)

    for kept in ('<p align="center">', "# The same model.", "Hand-written launch copy.",
                 "ralph:managed:physical-iphone-receipts:begin", "18.54 tok/s",
                 "## What the crown means", "Prose about the crown.", "## License",
                 "Each exact GGUF listed above is Apache-2.0."):
        assert kept in card, kept
    # the legacy section is gone, the block stands in its place, exactly once
    assert "deadbeef  ralph" not in card and "The same filenames are in crowns.json." not in card
    assert card.count(pc.CROWNS_BEGIN) == 1 and card.count(pc.CROWNS_END) == 1
    assert ("9" * 64 + "  ralph-qwen3-8b-sub2.gguf") in card
    assert card.index("physical-iphone-receipts:end") < card.index(pc.CROWNS_BEGIN)
    assert card.index(pc.CROWNS_END) < card.index("## What the crown means")
    assert "**Apache-2.0**, declared by the source owner for exactly these bytes" in card
    assert "license: apache-2.0" in card and "license_name" not in card

    # a later publish replaces between the markers and never duplicates them
    again = pc.readme(m, "https://example.test/round-8.json", "org/crowns",
                      record={"round": 8}, existing_card=card)
    assert again.count(pc.CROWNS_BEGIN) == 1 and again.count(pc.CROWNS_END) == 1
    assert "through Round 8" in again and "held through Round 8" in again
    assert "Prose about the crown." in again and "Hand-written launch copy." in again
    assert "https://example.test/round-7.json" not in again.split(pc.CROWNS_BEGIN)[1]


def test_a_crown_change_voids_the_license_and_flips_the_badge_to_pending():
    prev = pc.declare_license(_entry(), "Apache-2.0")
    assert pc.licensed(prev) == "Apache-2.0"
    # same id, different bytes: not covered — the declaration was for a sha, not a tier
    assert pc.licensed(dict(prev, file_sha256="7" * 64)) == ""

    new_meta = dict(META, file_sha256="8" * 64)
    new = pc.manifest_entry(tier="sub2",
                            sub={"model_id": "new", "artifact_uri": "hf://b/n@def",
                                 "retention": 0.3, "miner": "m", "code_bits": 2.2,
                                 "container_bits": 2.8},
                            round_no=8, record_url="u", metadata=new_meta, previous=prev,
                            adopt_license="apache-2.0")
    assert "license" not in new and pc.licensed(new) == ""
    assert new["round"] == 8 and new["source_repo"] == "b/n"

    card = pc.readme({"sub2": new}, "u", "org/crowns", record={"round": 8}, existing_card=LIVE)
    assert "license: other" in card and "license_name: Artifact license pending" in card
    assert "license: apache-2.0" not in card
    assert "**pending review**: crowned in Round 8" in card
    assert "Each exact GGUF listed above is Apache-2.0." in card, "the maintainer's prose is theirs"

    # an unchanged crown keeps its facts, and a backfill adopts the badge the card already wears
    kept = pc.manifest_entry(tier="sub2", sub={"model_id": "same"}, round_no=8, record_url="u",
                             metadata=META, previous=_entry(), adopt_license="apache-2.0")
    assert kept["round"] == 7 and kept["retention"] == 0.282457
    assert pc.licensed(kept) == "apache-2.0"
    assert pc.repo_license({"sub2": kept}) == ("apache-2.0", "")
    assert pc.repo_license({"sub2": kept, "sub4": _entry("x")}) == ("other", "Artifact license pending")
    assert pc.repo_license({}) == ("other", "Artifact license pending")


def test_front_matter_is_merged_not_replaced():
    existing = ("---\nlicense: apache-2.0\nlanguage:\n  - en\ntags: [custom, gguf]\n"
                "extra_key: kept\n---\n\n# Title\n\nbody text\n")
    card = pc.readme({"sub2": pc.declare_license(_entry(), "Apache-2.0")}, "u", "org/crowns",
                     record={"round": 7}, existing_card=existing)
    fm = card.split("---")[1]
    assert "extra_key: kept" in fm and "language:\n  - en" in fm
    assert "tags: [custom, gguf, conversational, qwen3, quantized, compression, bittensor]" in fm
    assert "pipeline_tag: text-generation" in fm and "base_model: Qwen/Qwen3-8B" in fm
    assert "library_name: gguf" in fm and fm.count("license:") == 1
    # no legacy table anywhere: the block lands right after the title, and the body survives
    assert card.index("# Title") < card.index(pc.CROWNS_BEGIN) < card.index("body text")

    # a block-style tag list is the maintainer's and is left exactly as written
    block_tags = "---\ntags:\n  - custom\n---\n\n# T\n"
    card = pc.readme({"sub2": _entry()}, "u", "org/crowns", record={"round": 7},
                     existing_card=block_tags)
    assert "tags:\n  - custom\n" in card and "tags: [" not in card

    # prose never counts as metadata, and a placeholder is not a declaration
    assert pc.front_matter_license("---\nlicense: other\n---\n`license: apache-2.0` in prose") == ""
    assert pc.front_matter_license("no front matter\nlicense: apache-2.0") == ""
    assert pc.front_matter_license("---\nlicense: apache-2.0\n---\n") == "apache-2.0"


@pytest.mark.parametrize("card", [
    pc.CROWNS_BEGIN,
    pc.CROWNS_END,
    f"{pc.CROWNS_END}\n{pc.CROWNS_BEGIN}",
    f"{pc.CROWNS_BEGIN}\na\n{pc.CROWNS_END}\n{pc.CROWNS_BEGIN}\nb\n{pc.CROWNS_END}",
])
def test_ambiguous_markers_refuse_to_guess(card: str):
    with pytest.raises(ValueError):
        pc.replace_managed_block(card, "block")


def test_fresh_repo_gets_the_full_template():
    record = {"round": 7, "manifest": {"observers_scored": ["a", "b", "c"]}}
    card = pc.readme({"sub2": _entry()}, "u", "org/crowns", record=record)
    assert card.startswith("---\n") and "license: other" in card
    assert card.count(pc.CROWNS_BEGIN) == 1
    assert "## Running them" in card and "hf download org/crowns ralph-qwen3-8b-sub2.gguf" in card
    assert "All 3 recorded judges" in card
    # and the fresh card is itself a valid base for the next publish
    again = pc.readme({"sub2": _entry()}, "u", "org/crowns", record=record, existing_card=card)
    assert again.count(pc.CROWNS_BEGIN) == 1 and "## Running them" in again


def test_published_card_reads_the_commit_guard_revision(monkeypatch):
    seen = []

    def fake_get_text(url: str) -> str:
        seen.append(url)
        return "---\nlicense: apache-2.0\n---\n"

    monkeypatch.setattr(pc, "_get_text", fake_get_text)
    card = pc.published_card("RalphLabsAI/ralph-crowns", "guarded-head")

    assert "license: apache-2.0" in card
    assert seen == [
        "https://huggingface.co/RalphLabsAI/ralph-crowns/resolve/guarded-head/README.md"
    ]
