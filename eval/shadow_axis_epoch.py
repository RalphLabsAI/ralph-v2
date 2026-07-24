"""Operator-runnable shadow on the AXIS substrate — the full v2 epoch, chain faked.

This is what an operator runs to see the REAL crown substrate end to end: a FakeChain
supplies committed submissions and the round nonce, and everything downstream is the
production path — intake (economics + safety + tier + commit-reveal), axis_round over the
deterministic-checker cover, the genre-overfit crown precondition, bond settlement, a
SIGNED record, and weights. Swapping FakeChain for the karpa validator's ChainIO is the
only step to a live shadow; nothing else changes.

    python -m eval.shadow_axis_epoch      # real HF models on a GPU; env-configurable

Env: RALPH_TEACHER, RALPH_BASE, RALPH_STUDENTS (comma-sep), RALPH_ITEMS.
"""
from __future__ import annotations

import os

from .axes.code_exec import CodeExec
from .axes.instruction import InstructionFollowing
from .axes.long_context import LongContext
from .axes.math_gsm import MathGSM
from .axes.multihop import MultiHop
from .axis_round import AxisSpec
from .chain import Commitment, run_v2_axis_epoch
from .economics import RegistrationLedger
from .gates import TierBudget
from .identity import commit_value, content_hash
from .koth import Tier, Tournament
from .signing import Ed25519Signer

TEACHER = os.environ.get("RALPH_TEACHER", "THUDM/glm-4-9b-chat-hf")
BASE = os.environ.get("RALPH_BASE", "Qwen/Qwen2.5-0.5B-Instruct")
STUDENTS = os.environ.get(
    "RALPH_STUDENTS", "Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-1.5B-Instruct"
).split(",")
ITEMS = int(os.environ.get("RALPH_ITEMS", "64"))


class FakeChain:
    """Minimal ChainIO: hands back the committed submissions + a fixed nonce, records the
    write-back so the operator can inspect the crown/weights/record. NO real chain."""

    def __init__(self, commits):
        self._commits = commits
        self.weights = None
        self.kings = {}
        self.record = None

    def current_block(self):
        return 1000

    def block_hash(self, block):
        return f"nonce-block-{block}"

    def read_commitments(self, lo, hi):
        return self._commits

    def commit_root(self, lo, hi):
        return "commit-root"

    def set_weights(self, weights):
        self.weights = weights
        return True

    def set_king(self, tier, hotkey, model_id):
        self.kings[tier] = (hotkey, model_id)

    def get_king(self, tier):
        return self.kings.get(tier)

    def blacklist(self, hotkey, reason):
        pass

    def publish_record(self, record):
        self.record = record

    def settle_bonds(self, refunds):
        pass


def _tier_specs():
    # code axis grades untrusted student output by executing it -> hard sandbox REQUIRED in
    # production (fail closed if bwrap is absent). Set RALPH_CODE_SANDBOX=auto only on a
    # trusted dev box without bubblewrap.
    code_sandbox = os.environ.get("RALPH_CODE_SANDBOX", "require")
    specs = [
        AxisSpec(MathGSM(), "math", 1.0, difficulty=3),
        AxisSpec(CodeExec(sandbox=code_sandbox), "code", 1.0, difficulty=2, items=160),
        AxisSpec(InstructionFollowing(), "instruction", 1.0, difficulty=1),
        AxisSpec(LongContext(base_facts=12), "long_context", 1.0, difficulty=1),
        AxisSpec(MultiHop(), "multihop", 1.0, difficulty=1),
    ]
    return specs


def main() -> int:
    from .runners import HFRunner, SafeStudentRunner

    # miners commit real checkpoints; here they are off-the-shelf HF ids resolved to a local
    # dir with a proper commit-reveal so the epoch's fetch-verify runs for real.
    from huggingface_hub import snapshot_download
    commits, runners = [], {}
    for i, mid in enumerate(STUDENTS):
        d = snapshot_download(mid, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])
        h, salt = content_hash(d), f"salt{i}"
        commits.append(Commitment(hotkey=f"hot{i}", coldkey=f"cold{i}", tier="open",
                                   ckpt_dir=d, declared_compute_h100h=1.0,
                                   revealed_hash=h, salt=salt, committed_value=commit_value(h, salt)))
        runners[d] = mid

    glm = HFRunner(TEACHER, name="glm", trust_remote_code=True, batch_size=16)
    base = HFRunner(BASE, name="base", batch_size=16)
    tiers = [Tier("open", 10 ** 12, 1.0)]
    budgets = {"open": TierBudget(name="open", max_params=10 ** 12, max_effective_bits=32.0)}
    chain = FakeChain(commits)

    # optional: enable the LIVE genre-overfit crown precondition on a real corpus
    # (RALPH_OVERFIT_CORPUS=cc_news). Off by default because it scores every student on the
    # corpus too — a real per-round cost the operator opts into.
    overfit_check = None
    src = os.environ.get("RALPH_OVERFIT_CORPUS")
    if src:
        from .corpus_hf import load_hf_timestamped, median_commit_ts
        from .overfit_gate import make_overfit_check
        docs = load_hf_timestamped(src, n=int(os.environ.get("RALPH_CORPUS_N", "300")))
        cut = median_commit_ts(docs)
        overfit_check = make_overfit_check(docs, cut, glm, base, min_n=20)
        print(f"overfit gate LIVE on {len(docs)} {src} docs (cut={cut})")

    result = run_v2_axis_epoch(
        chain, 1, _tier_specs(), glm, base, tiers, budgets,
        Tournament(tiers, margin=0.03), RegistrationLedger(), {},
        make_safe_runner=lambda cd: SafeStudentRunner(cd, name=runners[cd].split("/")[-1]),
        items_per_axis=ITEMS, overfit_check=overfit_check,
        signer=Ed25519Signer(seed=b"shadow-operator-key-000000000000"[:32]),
    )

    print("\n===== SHADOW AXIS EPOCH =====")
    print("accepted:", result.outcome.accepted)
    print("rejected:", result.outcome.rejected)
    print("weights:", chain.weights)
    print("crown:", chain.kings)
    rec = chain.record
    print(f"record signed+valid: {rec is not None and rec.verify_signature()} "
          f"(signer {rec.signer[:12] if rec else '-'}..., sha {result.record_sha256[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
