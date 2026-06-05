"""
Pure metric functions — no model weights loaded here.
Called by retriever.py, reasoner.py, and run_eval.py.
"""

from __future__ import annotations

import numpy as np

from dras.preprocess import REFUSAL_TEXT


# ---------------------------------------------------------------------------
# Semantic similarity (shared embedding model instance passed by caller)
# ---------------------------------------------------------------------------

def semantic_sim(a: str, b: str, embed_fn) -> float:
    """Cosine similarity in [0, 1]. embed_fn(text) → 1-D numpy array."""
    va = np.array(embed_fn(a), dtype=float)
    vb = np.array(embed_fn(b), dtype=float)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


# ---------------------------------------------------------------------------
# LLM-as-judge (uses base model via a separate forward pass)
# ---------------------------------------------------------------------------

VALIDITY_PROMPT = (
    "You are a strict logic judge. Given a set of premises and a proposed inference "
    "step, answer ONLY 'valid' or 'invalid'.\n\n"
    "Premises:\n{premises}\n\n"
    "Proposed step: {step}\n\n"
    "Does this step logically follow from the premises? Answer:"
)


def step_validity_judge(premises: list[str], step: str, infer_fn) -> bool:
    """infer_fn(prompt: str) → str.  Returns True if the step is judged valid."""
    if step.strip().startswith(REFUSAL_TEXT[:20]):
        return True   # refusals are always structurally valid outputs
    prompt = VALIDITY_PROMPT.format(
        premises="\n".join(f"- {p}" for p in premises),
        step=step,
    )
    answer = infer_fn(prompt).strip().lower()
    return answer.startswith("valid")


# ---------------------------------------------------------------------------
# Retrieval metrics (Section 6.1)
# ---------------------------------------------------------------------------

def retrieval_metrics(retrieved: list[str], ground_truth: list[str]) -> dict:
    """
    retrieved    : ordered list returned by retriever.query()
    ground_truth : the premises that were actually used to derive the next step
    """
    if not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "mrr": 0.0}

    gt_set = set(ground_truth)
    hits = [r for r in retrieved if r in gt_set]

    precision = len(hits) / len(retrieved) if retrieved else 0.0
    recall = len(hits) / len(gt_set)

    mrr = 0.0
    for rank, r in enumerate(retrieved, start=1):
        if r in gt_set:
            mrr = 1.0 / rank
            break

    return {"precision": round(precision, 4), "recall": round(recall, 4), "mrr": round(mrr, 4)}


def step_blocking_rate(blocking_flags: list[bool]) -> float:
    """Fraction of steps where at least one essential premise was outside top-k."""
    if not blocking_flags:
        return 0.0
    return round(sum(blocking_flags) / len(blocking_flags), 4)


# ---------------------------------------------------------------------------
# End-to-end proof metrics (Section 5)
# ---------------------------------------------------------------------------

def proof_completion_rate(results: list[dict]) -> float:
    """Fraction of loop runs that ended with status PROOF_FOUND."""
    if not results:
        return 0.0
    found = sum(1 for r in results if r.get("status") == "PROOF_FOUND")
    return round(found / len(results), 4)


def stall_rate(results: list[dict]) -> float:
    """Fraction of loop runs that stalled (no new premises, not PROOF_FOUND)."""
    if not results:
        return 0.0
    stalled = sum(1 for r in results if r.get("status") == "STALLED")
    return round(stalled / len(results), 4)


# ---------------------------------------------------------------------------
# Reasoner isolated metrics (Section 6.2)
# ---------------------------------------------------------------------------

def reasoner_metrics(
    generated_steps: list[str],
    premises_lists: list[list[str]],
    infer_fn,
) -> dict:
    """
    generated_steps  : model output for each eval example
    premises_lists   : the oracle premises supplied for each example
    infer_fn         : judge forward pass  (same signature as step_validity_judge)
    """
    n = len(generated_steps)
    if n == 0:
        return {"step_accuracy": 0.0, "hallucination_rate": 0.0, "refusal_rate": 0.0}

    valid_count = 0
    halluc_count = 0
    refusal_count = 0

    for step, premises in zip(generated_steps, premises_lists):
        if step.strip().startswith(REFUSAL_TEXT[:20]):
            refusal_count += 1
            valid_count += 1   # correct refusal counts as valid
            continue

        if step_validity_judge(premises, step, infer_fn):
            valid_count += 1

        # hallucination: step introduces a word that appears in neither premises nor goal
        premise_words = set(" ".join(premises).lower().split())
        step_words = set(step.lower().split())
        novel = step_words - premise_words
        # filter short function words
        novel = {w for w in novel if len(w) > 3}
        if len(novel) > 5:   # threshold: more than 5 completely novel content words
            halluc_count += 1

    return {
        "step_accuracy": round(valid_count / n, 4),
        "hallucination_rate": round(halluc_count / n, 4),
        "refusal_rate": round(refusal_count / n, 4),
    }
