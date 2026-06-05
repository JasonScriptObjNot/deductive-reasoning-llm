"""
FOLIO → train.jsonl / eval.jsonl

Run once:  python scripts/run_preprocess.py

Each row is {"text": "<|goal|>...<|premises|>...<|next_step|>...<|end|>"}.
The completion-only collator masks everything up to and including <|next_step|>,
so only the generated step is in the loss.

Positive examples (label == "True"):
    goal      = conclusion
    premises  = all premises joined by "  " (double space separates them)
    next_step = conclusion  (1-step proof: premises directly support the goal)

Refusal examples (~20% of positives, sampled from label != "True"):
    goal      = conclusion from a True example
    premises  = premises borrowed from a *different* example (cannot support goal)
    next_step = REFUSAL_TEXT
"""

import os
import random

from datasets import load_dataset

from dras.config import Config
from dras.utils import get_logger, save_jsonl

GOAL_TOKEN = "<|goal|>"
PREMISES_TOKEN = "<|premises|>"
NEXT_STEP_TOKEN = "<|next_step|>"
END_TOKEN = "<|end|>"
REFUSAL_TEXT = "No valid inference follows from these premises toward the stated goal."


def _format(goal: str, premises: list[str], next_step: str) -> str:
    premises_str = "  ".join(p.strip() for p in premises)
    return (
        f"{GOAL_TOKEN}{goal}\n"
        f"{PREMISES_TOKEN}{premises_str}\n"
        f"{NEXT_STEP_TOKEN}{next_step}{END_TOKEN}"
    )


def _load_folio() -> tuple[list[dict], list[dict]]:
    """Returns (true_examples, other_examples) as lists of dicts with keys
    'premises' (list[str]) and 'conclusion' (str)."""
    log = get_logger(__name__)
    log.info("Loading FOLIO from HuggingFace (tasksource/folio)…")
    ds = load_dataset("tasksource/folio")

    true_ex, other_ex = [], []
    for split in ("train", "validation"):
        if split not in ds:
            continue
        for row in ds[split]:
            # premises is a newline-separated string in tasksource/folio
            raw = row["premises"]
            premises = [p.strip() for p in raw.split("\n") if p.strip()] if isinstance(raw, str) else raw
            entry = {"premises": premises, "conclusion": row["conclusion"]}
            if row["label"] == "True":
                true_ex.append(entry)
            else:
                other_ex.append(entry)

    log.info(f"FOLIO loaded: {len(true_ex)} true, {len(other_ex)} other")
    return true_ex, other_ex


def _build_positives(true_ex: list[dict]) -> list[dict]:
    return [
        {"text": _format(ex["conclusion"], ex["premises"], ex["conclusion"])}
        for ex in true_ex
    ]


def _build_refusals(true_ex: list[dict], other_ex: list[dict], n: int) -> list[dict]:
    """n refusal examples — each uses a true goal with premises from a different example."""
    pool = other_ex + true_ex   # borrow premises from any example
    refusals = []
    goals = random.sample(true_ex, min(n, len(true_ex)))
    for ex in goals:
        donor = random.choice([p for p in pool if p is not ex])
        refusals.append({"text": _format(ex["conclusion"], donor["premises"], REFUSAL_TEXT)})
    return refusals


def build_dataset(cfg: Config, seed: int = 42, force: bool = False) -> None:  # noqa: C901
    log = get_logger(__name__)
    train_path = os.path.join(cfg.data_dir, "train.jsonl")
    eval_path = os.path.join(cfg.data_dir, "eval.jsonl")

    if not force and os.path.exists(train_path) and os.path.exists(eval_path):
        log.info("Dataset already exists. Use --force to rebuild.")
        return

    os.makedirs(cfg.data_dir, exist_ok=True)
    random.seed(seed)

    true_ex, other_ex = _load_folio()

    positives = _build_positives(true_ex)
    n_refusals = max(1, int(len(positives) * 0.20))
    refusals = _build_refusals(true_ex, other_ex, n_refusals)

    # synthetic multi-step chains — core of complex reasoning capability
    from dras.synth import build_chains
    chain_examples = build_chains()
    log.info(f"Synthetic chains: {len(chain_examples)} training examples")

    all_rows = positives + refusals + chain_examples
    random.shuffle(all_rows)

    split = int(len(all_rows) * 0.85)
    train_rows, eval_rows = all_rows[:split], all_rows[split:]

    save_jsonl(train_path, train_rows)
    save_jsonl(eval_path, eval_rows)
    log.info(
        f"Saved {len(train_rows)} train, {len(eval_rows)} eval rows  "
        f"({len(positives)} FOLIO positives, {len(refusals)} refusals, "
        f"{len(chain_examples)} synthetic chain steps)"
    )
