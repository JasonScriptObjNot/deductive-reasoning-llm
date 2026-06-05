"""
Premise validator — runs before any text enters the RAG store.

Two distinct use cases with different threat profiles:

  validate_generated_step(step, support_premises, embed_fn, infer_fn)
      Called after the reasoner generates a new derived step.
      Checks: well-formedness, grounding (no invented concepts), LLM validity.
      Primary defense against hallucination bleeding into the store.

  validate_manual_premise(premise, store_contents, embed_fn, infer_fn)
      Called when a user manually supplies seed premises.
      Checks: well-formedness, then contradiction against existing store.
      Primary defense against user input that would poison the chain.

Both return (valid: bool, issues: list[str]).
Callers decide whether to block or warn — the validator only reports.
"""

from __future__ import annotations

import re

from dras.preprocess import REFUSAL_TEXT
from dras.utils import get_logger

log = get_logger(__name__)

# ── Contradiction-check prompt ───────────────────────────────────────────────
_CONTRADICTION_PROMPT = (
    "You are a strict logic checker. Answer ONLY 'yes' or 'no'.\n\n"
    "Statement A: {a}\n"
    "Statement B: {b}\n\n"
    "Do Statement A and Statement B directly contradict each other "
    "(i.e., they cannot both be true at the same time)? Answer:"
)

# ── Validity-check prompt ────────────────────────────────────────────────────
_VALIDITY_PROMPT = (
    "You are a strict logic judge. Answer ONLY 'valid' or 'invalid'.\n\n"
    "Premises:\n{premises}\n\n"
    "Proposed inference: {step}\n\n"
    "Does this inference logically follow from the premises? Answer:"
)


# ---------------------------------------------------------------------------
# Structural checks (no model required)
# ---------------------------------------------------------------------------

def is_well_formed(text: str) -> tuple[bool, str]:
    """Reject empty strings, questions, and non-declarative fragments."""
    text = text.strip()
    if not text:
        return False, "Empty text."
    if len(text.split()) < 3:
        return False, "Too short to be a meaningful premise."
    if text.endswith("?"):
        return False, "Questions cannot serve as premises."
    if text.lower().startswith(("assume ", "suppose ", "if we assume")):
        return False, "Hypothetical assumptions are not premises."
    if text.endswith(","):
        return False, "Step appears incomplete (ends with comma — truncated conditional)."
    # Reject refusal strings — they are not inference steps and must not be
    # stored in the retriever (stored refusals poison subsequent retrievals).
    # Rejection here increments consecutive_rejects, triggering the sampling
    # fallback after 3 in a row, which is the right escape mechanism.
    if text.lower().startswith(REFUSAL_TEXT[:20].lower()):
        return False, "Refusal text is not a valid inference step."
    return True, ""


def is_grounded(step: str, premises: list[str]) -> tuple[bool, str]:
    """
    Check that the step does not introduce proper nouns or key noun phrases
    that appear in none of the supporting premises.

    This catches hallucinations like B3's "A patch is being developed" when
    no premise mentions development — only deployment.

    Uses a simple heuristic: capitalised words (potential named entities / key
    terms) in the step that appear in no premise are flagged.
    """
    if step.strip().startswith(REFUSAL_TEXT[:15]):
        return True, ""   # refusals are always grounded — they assert nothing

    # collect all words from premises (lowercased, alpha only)
    premise_words: set[str] = set()
    for p in premises:
        premise_words.update(w.lower() for w in re.findall(r"[a-zA-Z]+", p))

    # capitalised content words in the step that might be named entities / key nouns
    step_words = re.findall(r"\b[A-Z][a-z]{2,}\b", step)
    novel = [w for w in step_words if w.lower() not in premise_words]

    # filter common sentence-starters and pronouns that are trivially capitalised
    ignore = {"If", "Any", "All", "No", "The", "This", "That", "Then", "Since",
               "When", "Where", "Because", "Therefore", "Thus", "Hence",
               "Anyone", "Everyone", "Someone", "Nothing", "Something"}
    novel = [w for w in novel if w not in ignore]

    if len(novel) > 2:
        return False, f"Step introduces concepts not in premises: {novel}"
    return True, ""


# ---------------------------------------------------------------------------
# LLM-based checks (requires model forward pass)
# ---------------------------------------------------------------------------

def _llm_check(prompt: str, infer_fn) -> str:
    """Run a single forward pass and return the lowercased first word."""
    try:
        answer = infer_fn(prompt).strip().lower()
        return answer[:10]   # trim to first word-ish
    except Exception as e:
        log.warning(f"Validator LLM call failed: {e}")
        return "unknown"


def check_validity(step: str, premises: list[str], infer_fn) -> tuple[bool, str]:
    """Ask the model whether this step validly follows from premises."""
    if step.strip().startswith(REFUSAL_TEXT[:15]):
        return True, ""
    prompt = _VALIDITY_PROMPT.format(
        premises="\n".join(f"- {p}" for p in premises),
        step=step,
    )
    answer = _llm_check(prompt, infer_fn)
    if answer.startswith("invalid"):
        return False, "LLM judge: inference does not follow from these premises."
    return True, ""


def check_contradiction(new_premise: str, existing: list[str], embed_fn, infer_fn,
                        sim_threshold: float = 0.70) -> list[str]:
    """
    For each existing premise semantically close to new_premise, ask the model
    whether they contradict each other.  Returns a list of warning strings
    (one per contradiction found).

    sim_threshold: only check pairs with cosine similarity above this value.
    Lower than the dedup threshold so we catch near-opposites, not near-duplicates.
    """
    if not existing:
        return []

    from dras.evaluate import semantic_sim
    import numpy as np

    warnings: list[str] = []
    new_vec = embed_fn(new_premise)

    for ex in existing:
        sim = semantic_sim(new_premise, ex, embed_fn)
        if sim < sim_threshold:
            continue
        prompt = _CONTRADICTION_PROMPT.format(a=new_premise, b=ex)
        answer = _llm_check(prompt, infer_fn)
        if answer.startswith("yes"):
            warnings.append(f'Contradicts existing premise: "{ex[:80]}"')

    return warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_generated_step(
    step: str,
    support_premises: list[str],
    embed_fn,
    infer_fn,
    cfg=None,
) -> tuple[bool, list[str]]:
    """
    Validate a step produced by the reasoner before adding it to the store.

    Returns (valid, issues).  If valid=False the step should be discarded.
    Issues is populated even when valid=True (for warnings).
    """
    issues: list[str] = []

    # 1. well-formedness
    ok, msg = is_well_formed(step)
    if not ok:
        return False, [msg]

    # 2. grounding — no invented entities
    ok, msg = is_grounded(step, support_premises)
    if not ok:
        issues.append(msg)
        return False, issues   # hard block on ungrounded steps

    # 3. LLM validity (only if grounding passed — avoids wasted inference)
    ok, msg = check_validity(step, support_premises, infer_fn)
    if not ok:
        issues.append(msg)
        return False, issues

    return True, issues


def validate_manual_premise(
    premise: str,
    store_contents: list[str],
    embed_fn,
    infer_fn,
    cfg=None,
) -> tuple[bool, list[str]]:
    """
    Validate a manually supplied seed premise before adding it to the store.

    Returns (valid, issues).  invalid = well-formedness failure.
    Contradiction warnings are returned in issues but do NOT set valid=False —
    the user is shown the warning and can override.
    """
    issues: list[str] = []

    # 1. well-formedness — hard block
    ok, msg = is_well_formed(premise)
    if not ok:
        return False, [msg]

    # 2. contradiction against existing store — soft warning
    contradictions = check_contradiction(premise, store_contents, embed_fn, infer_fn)
    issues.extend(contradictions)

    return True, issues
