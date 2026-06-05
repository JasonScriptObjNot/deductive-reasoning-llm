"""
Inference loop controller.

The only module that imports both retriever and reasoner.
Stateless: takes already-constructed instances as arguments.

Usage:
    retriever = make_retriever(cfg)
    model, tokenizer = load_model(cfg)
    result = run_loop(goal, seed_premises, retriever, model, tokenizer, cfg)

result keys:
    status      : "PROOF_FOUND" | "MAX_ITER" | "STALLED"
    steps       : list of ProofStep (all intermediate + final steps)
    proof_tree  : ordered list of step texts from seeds to goal (only if PROOF_FOUND)
    iterations  : number of loop iterations executed
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from dras.config import Config
from dras.evaluate import semantic_sim
from dras.retriever import BaseRetriever
from dras.utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_conditional(text: str) -> bool:
    """True for universal / conditional rules or disjunctive premises.

    Bridge rules (If X then Y, Any X does Y) are the targets for backward
    chaining — their antecedents match derived facts, their consequents are
    the next intermediate steps we want to derive.

    Disjunctions ("...either A or B...") are also bridge rules: combined with
    evidence for "not A" (the anchor), they enable modus tollendo ponens
    (disjunctive syllogism) to derive "B" in a focused 2-premise context.
    """
    lowered = text.lower().strip()
    if lowered.startswith(("if ", "any ", "all ", "every ", "no ", "when ")):
        return True
    # Disjunctive premises ("either A or B") serve as DS bridge rules
    if " either " in lowered:
        return True
    return False


def _lookahead_score(
    candidate: str,
    retrieved: list[str],
    goal: str,
    model,
    tokenizer,
    cfg: Config,
    embed_fn,
) -> float:
    """
    Simulate one additional step from `candidate` and return how close that
    next step is to the goal.  A low score means `candidate` leads to a dead
    end; a high score means it opens a productive path.
    """
    from dras.reasoner import infer as _infer
    la_premises = (retrieved + [candidate])[-cfg.retrieval_k :]
    la_step = _infer(model, tokenizer, goal, la_premises, cfg)
    if la_step.endswith("<|"):
        la_step = la_step[:-2].strip()
    return float(semantic_sim(la_step, goal, embed_fn))


def _backward_chain_step(
    goal: str,
    anchor: str,
    retriever: BaseRetriever,
    model,
    tokenizer,
    cfg: Config,
    embed_fn,
) -> str | None:
    """
    Backward-chain recovery: find the nearest applicable bridge rule and
    derive the missing intermediate step via focused two-premise inference.

    When the main loop is stuck (the model refuses with the full k-premise
    context), this function reduces the inference problem to a single modus
    ponens by isolating the fact-rule pair that is immediately applicable:

        anchor  (the last derived fact, or the chain entry point)
      + pivot   (the conditional rule whose antecedent matches anchor)
      ──────────────────────────────────────────────────────────────
      ∴  next step  (the rule's consequent instantiated for this entity)

    This mirrors the mathematical strategy of targeting a known intermediate
    lemma rather than trying to bridge directly to the final theorem.

    The anchor is selected by the caller:
      - mid-chain: last_derived (the most recent accepted step)
      - first iteration: the seed fact most distant from the goal (entry point)

    Returns the derived step text, or None if no valid step could be found.
    """
    from dras.reasoner import infer as _infer
    from dras.preprocess import REFUSAL_TEXT

    # Fetch step-proximate candidates from the store; the bridge rule (whose
    # antecedent matches anchor vocabulary) will rank highest in this query.
    k_fetch = min(cfg.retrieval_k * 3, 12)
    candidates = retriever.query(anchor, k=k_fetch)

    conditionals = [p for p in candidates if _is_conditional(p)]
    if not conditionals:
        log.debug("backchain: no conditional rules found near anchor")
        return None

    # Sort by similarity to anchor descending: the rule whose antecedent
    # shares vocabulary with anchor scores highest and is tried first.
    # Rules that were already applied (whose consequent ≈ anchor) will
    # also score high but get filtered by the near-duplicate check below.
    conditionals.sort(
        key=lambda r: float(semantic_sim(r, anchor, embed_fn)),
        reverse=True,
    )

    for pivot in conditionals:
        focused = [anchor, pivot]
        candidate = _infer(model, tokenizer, goal, focused, cfg)
        if candidate.endswith("<|"):
            candidate = candidate[:-2].strip()

        # Reject refusals — the 2-premise context should be unambiguous;
        # if the model still refuses, this pivot doesn't apply to anchor.
        from dras.preprocess import REFUSAL_TEXT
        if candidate.lower().startswith(REFUSAL_TEXT[:20].lower()):
            continue

        # Reject conditional outputs — backchain derives specific facts by
        # instantiating a rule's consequent for the anchor entity.  A conditional
        # output means the model generated a new rule instead of a fact, which
        # always indicates hallucination and would poison last_derived.
        if _is_conditional(candidate):
            continue

        # Reject near-duplicates of the anchor — this pivot was the rule
        # that derived anchor, not the rule that extends from it.
        if float(semantic_sim(candidate, anchor, embed_fn)) >= cfg.dedup_threshold:
            continue

        log.info(f"backchain: anchor='{anchor[:50]}' pivot='{pivot[:50]}' → '{candidate[:80]}'")
        return candidate

    return None


# ---------------------------------------------------------------------------
# Goal projection (backward sub-goal synthesis)
# ---------------------------------------------------------------------------

def _extract_rule_parts(rule_text: str) -> tuple[str, str] | None:
    """
    Parse a universal rule into (antecedent_predicate, consequent_phrase).

    Handles "Any/All/Every [noun phrase] that/with [antecedent] [consequent]".
    Returns None when the rule doesn't match this pattern.
    """
    import re
    text = rule_text.strip().rstrip(".")
    m = re.match(
        r"^(?:any|all|every)\s+\w+(?:\s+\w+)?\s+(?:that\s+|with\s+)(.*)",
        text, re.IGNORECASE,
    )
    if not m:
        return None
    rest = m.group(1)

    # Find the split between antecedent and consequent: the LATEST positional
    # occurrence of a consequent-introducing verb.  "Latest" ensures that when
    # the antecedent contains a verb (e.g. "funds renewable projects sees..."),
    # we split at the main consequent verb ("sees") not the antecedent one.
    split_pos: int | None = None
    for pattern in [
        r"\bmust\s+", r"\bcan\s+", r"\bwill\s+", r"\bshall\s+",
        r"\bqualifies\b", r"\bgains\b", r"\battracts\b", r"\bsees\b",
        r"\bbecomes\b", r"\bfunds\b",
    ]:
        for m2 in re.finditer(pattern, rest, re.IGNORECASE):
            if split_pos is None or m2.start() > split_pos:
                split_pos = m2.start()  # keep the LATEST match in the string

    if split_pos is None:
        return None

    antecedent = rest[:split_pos].strip().rstrip(",")
    consequent = rest[split_pos:].strip()
    return antecedent, consequent


def _try_direct_rule_application(
    last_derived: str,
    steps: list,
    embed_fn,
    dedup_threshold: float,
    antecedent_sim_threshold: float = 0.68,
) -> str | None:
    """
    Deterministic modus ponens fallback.

    When the inference model is stuck generating conditionals instead of
    grounding a universal rule to the current entity, we apply the rule
    ourselves: find the rule whose antecedent best matches last_derived,
    extract its consequent, and prepend the entity name.

    Only fires when the antecedent similarity exceeds the threshold, ensuring
    we only apply rules whose preconditions are actually satisfied.

    Returns an instantiated conclusion string, or None if nothing applies.
    """
    import re

    if last_derived is None:
        return None

    rule_texts = [s.text for s in steps if _is_conditional(s.text)]
    fact_texts = [s.text for s in steps if not _is_conditional(s.text)]

    # Extract entity: longest run of capitalised words at the start of last_derived.
    entity_m = re.match(r"^([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)", last_derived)
    entity = entity_m.group(1) if entity_m else last_derived.split()[0]

    best: tuple[float, str] | None = None
    for rule in rule_texts:
        parts = _extract_rule_parts(rule)
        if parts is None:
            continue
        antecedent, consequent = parts

        ante_sim = float(semantic_sim(last_derived, antecedent, embed_fn))
        if ante_sim < antecedent_sim_threshold:
            continue

        conclusion = f"{entity} {consequent}."
        # Skip if this conclusion is already in the factual store.
        if any(
            float(semantic_sim(conclusion, f, embed_fn)) >= dedup_threshold
            for f in fact_texts
        ):
            continue

        if best is None or ante_sim > best[0]:
            best = (ante_sim, conclusion)

    if best is None:
        return None

    sim, conclusion = best
    log.info(
        f"direct rule application: '{conclusion[:80]}' "
        f"(antecedent_sim={sim:.3f})"
    )
    return conclusion

def _project_subgoals(
    goal: str,
    steps: list,
    model,
    tokenizer,
    cfg: Config,
    embed_fn,
    n_rules: int = 3,
) -> list[str]:
    """
    Project intermediate sub-goals backward from the final goal.

    For each goal-proximate conditional rule R in the store, invokes
    infer(model, goal, [R]).  The model was trained to produce the next valid
    inference step toward a goal; given only one rule and the goal, the only
    coherent answer is the rule's antecedent instantiated for the relevant
    entity.  That antecedent is a concrete, closer waypoint on the path to
    the goal.

    Returns sub-goals sorted by estimated reachability from the current
    factual store (highest first), filtered to exclude refusals, new rules,
    and conclusions already in the store or too close to the goal itself.
    """
    from dras.reasoner import infer as _infer
    from dras.preprocess import REFUSAL_TEXT

    fact_texts = [s.text for s in steps if not _is_conditional(s.text)]
    rule_texts = [s.text for s in steps if _is_conditional(s.text)]
    if not rule_texts:
        return []

    # Score rules by similarity to goal — goal-proximate rules are candidates
    # whose consequents link to the goal; their antecedents become sub-goals.
    rule_sims = sorted(
        ((r, float(semantic_sim(r, goal, embed_fn))) for r in rule_texts),
        key=lambda x: -x[1],
    )
    top_rules = [r for r, _ in rule_sims[: n_rules * 2]]

    known = set(fact_texts)
    subgoals: list[tuple[str, float]] = []

    for rule in top_rules:
        candidate = _infer(model, tokenizer, goal, [rule], cfg)
        if candidate.endswith("<|"):
            candidate = candidate[:-2].strip()

        if candidate.lower().startswith(REFUSAL_TEXT[:20].lower()):
            continue
        if _is_conditional(candidate):
            continue
        if float(semantic_sim(candidate, goal, embed_fn)) >= cfg.goal_sim_threshold:
            continue  # too close to goal — not a useful intermediate
        if candidate in known:
            continue  # already derived
        if any(float(semantic_sim(candidate, f, embed_fn)) >= cfg.dedup_threshold
               for f in fact_texts):
            continue

        # Reachability: how close is this sub-goal to what we already have?
        # Higher means the current store can probably derive it in fewer steps.
        reachability = max(
            (float(semantic_sim(candidate, f, embed_fn)) for f in fact_texts),
            default=0.0,
        )
        subgoals.append((candidate, reachability))
        log.info(
            f"goal projection: sub-goal '{candidate[:70]}' "
            f"(reachability={reachability:.3f}, rule='{rule[:50]}')"
        )

    subgoals.sort(key=lambda x: -x[1])
    return [sg for sg, _ in subgoals[:n_rules]]


# ---------------------------------------------------------------------------
# Exhaustive pairwise inference
# ---------------------------------------------------------------------------

def _exhaustive_pair_search(
    goal: str,
    steps: list,
    model,
    tokenizer,
    cfg: Config,
    embed_fn,
    verbose: bool = False,
) -> tuple[list[tuple[str, list[str]]], bool]:
    """
    BFS over all premise pairs to discover new valid inference steps.

    Tries every ordered pair (P_i, P_j) from the current store via focused
    2-premise inference.  Valid new steps are added to the frontier so they
    can immediately be paired with everything else (BFS expansion).  Returns
    all newly found steps and a bool indicating whether the goal was reached.

    Fires as a last resort after all other recovery mechanisms have failed.
    O(N²) inference calls per round, but N is typically small at trigger time.
    """
    from dras.reasoner import infer as _infer
    from dras.reasoner import _raw_generate as _raw_infer
    from dras.validator import validate_generated_step
    from dras.preprocess import REFUSAL_TEXT

    all_texts = [
        s.text for s in steps
        if not s.text.lower().startswith(REFUSAL_TEXT[:20].lower())
    ]
    if len(all_texts) < 2:
        return [], False

    known: set[str] = set(all_texts)
    new_found: list[tuple[str, list[str]]] = []
    existing: list[str] = list(all_texts)
    # First round: every item is in the frontier so all pairs are tried.
    frontier: list[str] = list(all_texts)

    log.info(
        f"exhaustive pair search: {len(existing)} premises → "
        f"up to {len(existing) * (len(existing) - 1)} initial ordered pairs"
    )

    round_num = 0
    while frontier:
        round_num += 1
        frontier_set = set(frontier)

        # Build ordered pairs where at least one element is new (from frontier).
        # Pairs (A, B) where both A and B were in prior rounds are already done.
        pairs: list[tuple[str, str]] = []
        for fi in frontier:
            for ex in existing:
                if fi == ex:
                    continue
                pairs.append((fi, ex))
                if ex not in frontier_set:
                    # ex is an old item — add the reverse order too
                    pairs.append((ex, fi))
        # Cross-pairs within frontier (both new)
        for idx, fi in enumerate(frontier):
            for fj in frontier[idx + 1:]:
                pairs.append((fi, fj))
                pairs.append((fj, fi))

        # Deduplicate (can arise from frontier cross-pair logic above)
        seen_pair_set: set[tuple[str, str]] = set()
        unique_pairs: list[tuple[str, str]] = []
        for p in pairs:
            if p not in seen_pair_set:
                seen_pair_set.add(p)
                unique_pairs.append(p)
        pairs = unique_pairs

        # Pre-compute goal-similarity for sorting (one embed call per unique text)
        goal_sims: dict[str, float] = {}
        for t in {item for pair in pairs for item in pair}:
            if t not in goal_sims:
                goal_sims[t] = float(semantic_sim(t, goal, embed_fn))

        # Fact-rule pairs first (p0=factual claim, p1=conditional rule) because
        # that's the canonical modus ponens shape the model was trained on.
        # Within that group, sort by descending combined goal-similarity.
        pairs.sort(key=lambda p: (
            0 if (_is_conditional(p[1]) and not _is_conditional(p[0])) else 1,
            -(goal_sims.get(p[0], 0.0) + goal_sims.get(p[1], 0.0)),
        ))

        next_frontier: list[str] = []
        log.info(f"pair search round {round_num}: {len(pairs)} ordered pairs to try")

        for p1, p2 in pairs:
            candidate = _infer(model, tokenizer, goal, [p1, p2], cfg)
            if candidate.endswith("<|"):
                candidate = candidate[:-2].strip()

            from dras.preprocess import REFUSAL_TEXT
            if candidate.lower().startswith(REFUSAL_TEXT[:20].lower()):
                continue
            # Pairwise search derives facts, not rules. Conditional outputs are
            # always hallucinations here — the rules are already in the premise
            # set and generating new rules would poison the BFS frontier.
            if _is_conditional(candidate):
                continue
            if candidate in known:
                continue
            if any(
                float(semantic_sim(candidate, t, embed_fn)) >= cfg.dedup_threshold
                for t in existing
            ):
                continue

            valid, _ = validate_generated_step(
                candidate, [p1, p2], embed_fn,
                lambda prompt: _raw_infer(model, tokenizer, prompt),
            )
            if not valid:
                continue

            goal_sim = float(semantic_sim(candidate, goal, embed_fn))
            known.add(candidate)
            existing.append(candidate)
            next_frontier.append(candidate)
            new_found.append((candidate, [p1, p2]))
            log.info(f"pair search found: '{candidate[:80]}' (goal_sim={goal_sim:.3f})")
            if verbose:
                print(f"  [pair] {candidate[:80]}  (goal_sim={goal_sim:.3f})")

            if goal_sim >= cfg.goal_sim_threshold:
                return new_found, True

            # Cap BFS expansion to prevent explosion on unprovable problems
            # where hallucinated steps can compound into arbitrarily large frontiers.
            if len(new_found) >= 10:
                log.info("pair search: new-step cap reached, stopping BFS")
                return new_found, False

        frontier = next_frontier

    return new_found, False


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ProofStep:
    id: str
    text: str
    parent_ids: list[str]
    iteration: int
    is_seed: bool = False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(
    goal: str,
    seed_premises: list[str],
    retriever: BaseRetriever,
    model,
    tokenizer,
    cfg: Config,
    embed_fn=None,
    verbose: bool = False,
) -> dict:
    """
    embed_fn(text) → numpy array.  If None, imports SentenceTransformer on first call
    using cfg.embed_model (shares the DenseRetriever's model when backend is 'dense').
    """
    if embed_fn is None:
        from sentence_transformers import SentenceTransformer
        _st = SentenceTransformer(cfg.embed_model)
        embed_fn = lambda t: _st.encode(t, normalize_embeddings=True)

    retriever.reset()

    steps: list[ProofStep] = []
    id_map: dict[str, ProofStep] = {}

    for text in seed_premises:
        pid = str(uuid.uuid4())
        retriever.add(text)
        step = ProofStep(id=pid, text=text, parent_ids=[], iteration=0, is_seed=True)
        steps.append(step)
        id_map[pid] = step

    status = "MAX_ITER"
    last_derived: str | None = None
    consecutive_rejects: int = 0
    dedup_hit_count: int = 0   # cumulative near-dedup hits; doesn't reset on paraphrase acceptance
    subgoal_stack: list[str] = []  # projected sub-goals (most reachable on top)
    events: list[dict] = []        # structured trace for offline inspection

    from dras.reasoner import infer as _infer_step, _raw_generate as _raw_infer
    from dras.preprocess import REFUSAL_TEXT

    events.append({"type": "seeds", "texts": list(seed_premises)})

    for iteration in range(1, cfg.max_iterations + 1):
        # When a sub-goal is active, anchor retrieval toward it rather than the
        # distant final goal — bidirectional search meets the forward chain halfway.
        effective_goal = subgoal_stack[-1] if subgoal_stack else goal

        if last_derived is None and len(seed_premises) > cfg.retrieval_k:
            retrieved_texts = list(seed_premises)
        else:
            retrieved_texts = retriever.query_multi(effective_goal, last_derived, cfg.retrieval_k)
        if not retrieved_texts:
            log.warning("Retriever returned no results — terminating.")
            status = "STALLED"
            break

        events.append({
            "type": "iteration",
            "iteration": iteration,
            "store_size": len(steps),
            "effective_goal": effective_goal,
            "retrieved": list(retrieved_texts),
        })
        if verbose:
            store_size = len(steps)
            print(f"\n  [iter {iteration}]  store size: {store_size} premises")
            print(f"  retrieved ({len(retrieved_texts)}):")
            for r in retrieved_texts:
                print(f"    • {r}")

        use_sampling = consecutive_rejects >= 3
        next_step_text = _infer_step(
            model, tokenizer, goal, retrieved_texts, cfg,
            do_sample=use_sampling, temperature=0.7,
        )
        if next_step_text.endswith("<|"):
            next_step_text = next_step_text[:-2].strip()
        log.info(f"[iter {iteration}] → {next_step_text[:120]}{' [sampled]' if use_sampling else ''}")
        if verbose:
            print(f"  generated{'  [sampled]' if use_sampling else ''}: {next_step_text}")
        events.append({"type": "generated", "text": next_step_text, "sampled": use_sampling})

        # Premises that justify next_step_text for parent-pointer tracing.
        # May be overridden by backward-chain recovery below.
        justifying_premises = retrieved_texts

        if getattr(cfg, "validate_steps", True):
            from dras.validator import validate_generated_step
            valid, issues = validate_generated_step(
                next_step_text, retrieved_texts, embed_fn,
                lambda prompt: _raw_infer(model, tokenizer, prompt),
            )
            if not valid:
                consecutive_rejects += 1
                log.warning(f"[iter {iteration}] Step rejected by validator: {issues}")
                if verbose:
                    print(f"  REJECTED: {issues}")
                events.append({"type": "rejected", "text": next_step_text, "issues": list(issues)})

                # ── Backward-chain recovery ──────────────────────────────
                # When the main loop is stuck, identify the nearest applicable
                # bridge rule and rederive the missing step using only that
                # rule and the chain's current anchor (last derived fact, or
                # the most entry-proximate seed fact on the first iteration).
                # This targets an intermediate lemma rather than forcing the
                # model to bridge the full distance to the goal in one step.
                backchain_k = getattr(cfg, "backchain_k", 0)
                recovered_step: str | None = None
                recovered_premises: list[str] | None = None

                if (backchain_k > 0
                        and (
                            (consecutive_rejects >= backchain_k
                             and consecutive_rejects % backchain_k == 0)
                            or (dedup_hit_count > 0
                                and dedup_hit_count % backchain_k == 0)
                        )):

                    # Determine anchor: last derived fact, or the seed fact
                    # most distant from the goal when no steps yet exist.
                    if last_derived is not None:
                        bc_anchor = last_derived
                    else:
                        non_rule_seeds = [s for s in seed_premises if not _is_conditional(s)]
                        if non_rule_seeds:
                            bc_anchor = min(
                                non_rule_seeds,
                                key=lambda f: float(semantic_sim(f, goal, embed_fn)),
                            )
                        else:
                            bc_anchor = None

                    if bc_anchor is not None:
                        log.info(
                            f"[iter {iteration}] backchain triggered "
                            f"(consecutive_rejects={consecutive_rejects}, "
                            f"anchor='{bc_anchor[:60]}')"
                        )
                        events.append({
                            "type": "backchain_trigger",
                            "iteration": iteration,
                            "consecutive_rejects": consecutive_rejects,
                            "anchor": bc_anchor,
                        })
                        bc_raw = _backward_chain_step(
                            goal, bc_anchor, retriever, model, tokenizer, cfg, embed_fn
                        )
                        if bc_raw is not None:
                            bc_premises = [bc_anchor] + [
                                p for p in retrieved_texts if p != bc_anchor
                            ][:cfg.retrieval_k - 1]
                            bc_valid, _ = validate_generated_step(
                                bc_raw, bc_premises, embed_fn,
                                lambda prompt: _raw_infer(model, tokenizer, prompt),
                            )
                            events.append({"type": "backchain_result", "text": bc_raw, "valid": bc_valid})
                            if bc_valid:
                                recovered_step = bc_raw
                                recovered_premises = bc_premises
                                dedup_hit_count = 0  # reset after successful recovery

                if recovered_step is None:
                    # ── Goal projection ──────────────────────────────────
                    # Switches the retrieval anchor to a closer intermediate
                    # waypoint, giving the forward chain a reachable sub-goal
                    # instead of trying to bridge the full distance to the
                    # final goal in one step.  Fires before pairwise so the
                    # enriched context helps both the main loop and pairwise.
                    goal_proj_k = getattr(cfg, "goal_proj_k", 0)
                    if (goal_proj_k > 0
                            and consecutive_rejects >= goal_proj_k
                            and consecutive_rejects % goal_proj_k == 0):
                        # Abandon the current sub-goal if it isn't making
                        # progress (we're still accumulating rejects with it).
                        if subgoal_stack:
                            old = subgoal_stack.pop()
                            log.info(f"[iter {iteration}] sub-goal abandoned (no progress): '{old[:60]}'")
                        log.info(
                            f"[iter {iteration}] goal projection triggered "
                            f"(consecutive_rejects={consecutive_rejects})"
                        )
                        events.append({
                            "type": "goal_proj_trigger",
                            "iteration": iteration,
                            "consecutive_rejects": consecutive_rejects,
                        })

                        # Primary: model-based backward sub-goal projection.
                        # Asks infer(goal, [rule]) for each goal-proximate rule —
                        # the model generates the rule's antecedent as a waypoint.
                        new_subgoals = _project_subgoals(
                            goal, steps, model, tokenizer, cfg, embed_fn
                        )
                        if new_subgoals:
                            # Push in reverse so the most-reachable is on top
                            subgoal_stack.extend(reversed(new_subgoals))
                            log.info(
                                f"goal projection: pushed {len(new_subgoals)} sub-goals; "
                                f"active='{subgoal_stack[-1][:60]}'"
                            )
                            events.append({"type": "goal_proj_subgoals", "subgoals": list(new_subgoals)})
                            if verbose:
                                print(f"  [goal projection] {len(new_subgoals)} sub-goals:")
                                for sg in reversed(subgoal_stack[-len(new_subgoals):]):
                                    print(f"    → {sg[:80]}")
                        else:
                            # Fallback: deterministic modus ponens.  When the model
                            # can't generate the antecedent backward (generates
                            # conditionals or refuses), instantiate the applicable
                            # rule directly using entity + consequent extraction.
                            # After each successful application, immediately try
                            # again — the new step may unlock another rule, so we
                            # chain forward until no more rules apply or the goal
                            # is reached.  This completes multi-step blocked chains
                            # in a single goal projection trigger.
                            from dras.validator import validate_generated_step
                            _chain_cap = 10  # hard stop against theoretical cycles
                            _chain_count = 0
                            while _chain_count < _chain_cap:
                                direct = _try_direct_rule_application(
                                    last_derived, steps, embed_fn, cfg.dedup_threshold
                                )
                                if direct is None:
                                    break
                                direct_premises = [last_derived] + [
                                    s.text for s in steps
                                    if _is_conditional(s.text)
                                    and float(semantic_sim(direct, s.text, embed_fn))
                                    >= 0.5
                                ][:cfg.retrieval_k - 1]
                                direct_valid, _ = validate_generated_step(
                                    direct, direct_premises, embed_fn,
                                    lambda prompt: _raw_infer(model, tokenizer, prompt),
                                )
                                if not direct_valid:
                                    break
                                log.info(f"goal projection (direct): '{direct[:80]}'")
                                if verbose:
                                    print(f"  [goal projection direct]: {direct[:80]}")
                                events.append({"type": "goal_proj_direct", "text": direct})
                                pid = str(uuid.uuid4())
                                par_ids = [s.id for s in steps if s.text in direct_premises]
                                new_ps = ProofStep(
                                    id=pid, text=direct,
                                    parent_ids=par_ids, iteration=iteration,
                                )
                                steps.append(new_ps)
                                id_map[pid] = new_ps
                                retriever.add(direct)
                                last_derived = direct
                                consecutive_rejects = 0
                                dedup_hit_count = 0
                                _chain_count += 1
                                if subgoal_stack:
                                    sg_sim = float(semantic_sim(direct, subgoal_stack[-1], embed_fn))
                                    if sg_sim >= cfg.goal_sim_threshold:
                                        achieved = subgoal_stack.pop()
                                        log.info(f"sub-goal achieved by direct step: '{achieved[:60]}'")
                                direct_goal_sim = float(semantic_sim(direct, goal, embed_fn))
                                if direct_goal_sim >= cfg.goal_sim_threshold:
                                    status = "PROOF_FOUND"
                                    break  # exits while loop
                            if status == "PROOF_FOUND":
                                break  # exits for loop

                    # ── Exhaustive pairwise search ───────────────────────
                    # Last resort: try all ordered premise pairs when every
                    # other recovery mechanism has been exhausted.  BFS-
                    # expands valid new steps so compound inferences are
                    # reachable within a single trigger.
                    pair_search_k = getattr(cfg, "pair_search_k", 0)
                    if (pair_search_k > 0
                            and consecutive_rejects >= pair_search_k
                            and consecutive_rejects % pair_search_k == 0):
                        log.info(
                            f"[iter {iteration}] pairwise search triggered "
                            f"(consecutive_rejects={consecutive_rejects})"
                        )
                        events.append({
                            "type": "pairwise_trigger",
                            "iteration": iteration,
                            "consecutive_rejects": consecutive_rejects,
                        })
                        ps_steps, ps_proof = _exhaustive_pair_search(
                            goal, steps, model, tokenizer, cfg, embed_fn,
                            verbose=verbose,
                        )
                        if ps_steps:
                            for step_text, step_prems in ps_steps:
                                events.append({"type": "pairwise_found", "text": step_text, "premises": list(step_prems)})
                                pid = str(uuid.uuid4())
                                par_ids = [s.id for s in steps if s.text in step_prems]
                                new_ps = ProofStep(
                                    id=pid, text=step_text,
                                    parent_ids=par_ids, iteration=iteration,
                                )
                                steps.append(new_ps)
                                id_map[pid] = new_ps
                                retriever.add(step_text)
                                if not step_text.startswith(REFUSAL_TEXT[:20]):
                                    last_derived = step_text
                            consecutive_rejects = 0
                            dedup_hit_count = 0
                            if ps_proof:
                                status = "PROOF_FOUND"
                                break
                    continue  # resume with enriched store (or genuine failure)

                next_step_text = recovered_step
                justifying_premises = recovered_premises
                if verbose:
                    print(f"  backward chain recovered: {next_step_text[:80]}")

        # lookahead: if a step's 1-step future is a dead end, try sampled alternatives
        if getattr(cfg, "lookahead_k", 0) > 0 and not use_sampling:
            from dras.validator import validate_generated_step
            greedy_la = _lookahead_score(next_step_text, retrieved_texts, goal, model, tokenizer, cfg, embed_fn)
            log.info(f"[iter {iteration}] lookahead(greedy)={greedy_la:.3f}")
            if greedy_la < cfg.lookahead_threshold:
                best_text = next_step_text
                best_la = greedy_la
                for _ in range(cfg.lookahead_k):
                    alt = _infer_step(model, tokenizer, goal, retrieved_texts, cfg, do_sample=True, temperature=0.7)
                    if alt.endswith("<|"):
                        alt = alt[:-2].strip()
                    alt_valid, _ = validate_generated_step(
                        alt, retrieved_texts, embed_fn,
                        lambda prompt: _raw_infer(model, tokenizer, prompt),
                    )
                    if alt_valid:
                        alt_la = _lookahead_score(alt, retrieved_texts, goal, model, tokenizer, cfg, embed_fn)
                        log.info(f"[iter {iteration}] lookahead(alt)={alt_la:.3f}: {alt[:80]}")
                        if alt_la > best_la:
                            best_la = alt_la
                            best_text = alt
                if best_text is not next_step_text:
                    log.info(f"[iter {iteration}] Lookahead replaced greedy step (la {greedy_la:.3f}→{best_la:.3f})")
                    if verbose:
                        print(f"  lookahead replaced greedy: {best_text[:80]}")
                next_step_text = best_text

        # deduplication check — soft reject so the sampling fallback can recover.
        # A near-duplicate increments consecutive_rejects (enabling temperature
        # sampling on the next iteration) rather than hard-terminating the loop.
        dedup_hit = False
        for existing in steps:
            if not existing.is_seed:
                sim = semantic_sim(next_step_text, existing.text, embed_fn)
                if sim >= cfg.dedup_threshold:
                    consecutive_rejects += 1
                    dedup_hit_count += 1
                    log.info(
                        f"Dedup threshold hit (sim={sim:.3f}) — skipping "
                        f"(consecutive_rejects={consecutive_rejects}, "
                        f"dedup_hit_count={dedup_hit_count})"
                    )
                    if verbose:
                        print(f"  NEAR-DUPLICATE (sim={sim:.3f}), skipping")
                    events.append({"type": "dedup", "text": next_step_text, "sim": float(sim)})
                    dedup_hit = True
                    break

        if dedup_hit:
            # Attempt backward-chain recovery immediately on first dedup hit, then
            # every backchain_k hits.  When the model re-derives the same step
            # (near-dup), the disjunction/bridge rule is the pivot that produces
            # the NEXT step in the chain via focused 2-premise inference.
            backchain_k = getattr(cfg, "backchain_k", 0)
            if (backchain_k > 0 and last_derived is not None
                    and (dedup_hit_count == 1
                         or (dedup_hit_count > 1
                             and dedup_hit_count % backchain_k == 0))):
                log.info(
                    f"backchain triggered by dedup "
                    f"(dedup_hit_count={dedup_hit_count}, "
                    f"anchor='{last_derived[:60]}')"
                )
                from dras.validator import validate_generated_step
                bc_raw = _backward_chain_step(
                    goal, last_derived, retriever, model, tokenizer, cfg, embed_fn
                )
                if bc_raw is not None:
                    bc_premises = [last_derived] + [
                        p for p in retrieved_texts if p != last_derived
                    ][:cfg.retrieval_k - 1]
                    bc_valid, _ = validate_generated_step(
                        bc_raw, bc_premises, embed_fn,
                        lambda prompt: _raw_infer(model, tokenizer, prompt),
                    )
                    events.append({"type": "backchain_dedup_result", "text": bc_raw, "valid": bc_valid})
                    if bc_valid:
                        log.info(f"backchain dedup recovery: '{bc_raw[:80]}'")
                        if verbose:
                            print(f"  backward chain (dedup recovery): {bc_raw[:80]}")
                        next_step_text = bc_raw
                        justifying_premises = bc_premises
                        dedup_hit_count = 0
                        dedup_hit = False  # fall through to acceptance

            if dedup_hit:
                continue  # no recovery

        # Step accepted — reset counter and advance the chain.
        consecutive_rejects = 0
        goal_sim = semantic_sim(next_step_text, goal, embed_fn)
        events.append({"type": "accepted", "text": next_step_text, "goal_sim": float(goal_sim)})
        pid = str(uuid.uuid4())
        parent_ids = [
            s.id for s in steps
            if s.text in justifying_premises
        ]
        new_step = ProofStep(id=pid, text=next_step_text, parent_ids=parent_ids, iteration=iteration)
        steps.append(new_step)
        id_map[pid] = new_step
        retriever.add(next_step_text)
        if not next_step_text.startswith(REFUSAL_TEXT[:20]):
            last_derived = next_step_text

        # Check if this step satisfies the current sub-goal; if so, pop it
        # and let the loop resume toward the next waypoint (or the final goal).
        if subgoal_stack:
            sg_sim = float(semantic_sim(next_step_text, subgoal_stack[-1], embed_fn))
            if sg_sim >= cfg.goal_sim_threshold:
                achieved = subgoal_stack.pop()
                log.info(f"sub-goal achieved (sim={sg_sim:.3f}): '{achieved[:60]}'")
                if verbose:
                    print(f"  sub-goal achieved (sim={sg_sim:.3f}): {achieved[:80]}")
                events.append({"type": "subgoal_achieved", "subgoal": achieved, "sim": sg_sim})

        if goal_sim >= cfg.goal_sim_threshold:
            log.info(f"Goal reached (sim={goal_sim:.3f}) after {iteration} iterations.")
            if verbose:
                print(f"  ✓ goal similarity {goal_sim:.3f} — PROOF_FOUND")
            events.append({"type": "proof_found", "iteration": iteration, "goal_sim": float(goal_sim)})
            status = "PROOF_FOUND"
            break
        if verbose:
            print(f"  goal similarity {goal_sim:.3f} — continuing")

    proof_tree: list[str] = []
    proof_steps_detail: list[dict] = []
    if status == "PROOF_FOUND":
        proof_tree = _trace_proof(steps[-1], id_map, seed_premises)
        proof_tree_order = {t: i for i, t in enumerate(proof_tree)}
        for ps in steps:
            if ps.text not in proof_tree_order:
                continue
            parent_texts = [id_map[pid].text for pid in ps.parent_ids if pid in id_map]
            proof_steps_detail.append({
                "text": ps.text,
                "is_seed": ps.is_seed,
                "parent_texts": parent_texts,
                "iteration": ps.iteration,
            })
        proof_steps_detail.sort(key=lambda x: proof_tree_order.get(x["text"], 999))
    else:
        events.append({"type": "max_iter", "iterations": cfg.max_iterations})

    return {
        "status": status,
        "steps": steps,
        "proof_tree": proof_tree,
        "proof_steps_detail": proof_steps_detail,
        "events": events,
        "iterations": iteration if status != "MAX_ITER" else cfg.max_iterations,
    }


def _trace_proof(final: ProofStep, id_map: dict[str, ProofStep], seed_texts: list[str]) -> list[str]:
    """BFS back-trace from goal step to seeds, then reverse for forward order."""
    ordered: list[str] = []
    visited: set[str] = set()
    queue = [final]
    while queue:
        node = queue.pop(0)
        if node.id in visited:
            continue
        visited.add(node.id)
        ordered.append(node.text)
        for pid in node.parent_ids:
            if pid in id_map:
                queue.append(id_map[pid])
    ordered.reverse()
    return ordered
