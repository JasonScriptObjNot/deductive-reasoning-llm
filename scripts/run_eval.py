import os; os.environ.setdefault("PYTHONUTF8", "1")  # Unsloth chat_template.jinja contains non-ASCII on Windows
"""
python scripts/run_eval.py --mode retriever
python scripts/run_eval.py --mode reasoner
python scripts/run_eval.py --mode e2e
python scripts/run_eval.py --mode e2e --backend bm25
python scripts/run_eval.py --mode e2e --pair-search 12 --goal-proj 9   # full recovery stack
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dras.config import Config
from dras.utils import get_logger, set_seed

log = get_logger("run_eval")

def _load_benchmarks(cfg: Config) -> list[dict]:
    benchmark_path = os.path.join(cfg.data_dir, "benchmarks.json")
    if os.path.exists(benchmark_path):
        import json as _json
        with open(benchmark_path) as f:
            return _json.load(f)
    # fallback minimal set
    return [
        {"id": "T1", "tier": "T1", "pattern": "modus_ponens_chain", "label": "Socrates",
         "goal": "Socrates is mortal.", "seeds": ["All humans are mortal.", "Socrates is a human."], "provable": True},
        {"id": "T4", "tier": "T4", "pattern": "unprovable", "label": "Unprovable",
         "goal": "The stock market will rise tomorrow.",
         "seeds": ["The sky is blue.", "Penguins live in Antarctica."], "provable": False},
    ]


def mode_retriever(cfg: Config) -> None:
    from dras.evaluate import retrieval_metrics
    from dras.retriever import make_retriever
    from dras.utils import load_jsonl

    eval_path = os.path.join(cfg.data_dir, "eval.jsonl")
    if not os.path.exists(eval_path):
        log.error("eval.jsonl not found. Run run_preprocess.py first.")
        return

    retriever = make_retriever(cfg)
    rows = load_jsonl(eval_path)

    all_precision, all_recall, all_mrr = [], [], []

    for row in rows:
        text = row["text"]
        from dras.preprocess import GOAL_TOKEN, PREMISES_TOKEN, NEXT_STEP_TOKEN
        goal = text[text.index(GOAL_TOKEN) + len(GOAL_TOKEN): text.index("\n")].strip()
        prem_str = text[text.index(PREMISES_TOKEN) + len(PREMISES_TOKEN): text.index(NEXT_STEP_TOKEN)].strip()
        premises = [p.strip() for p in prem_str.split("  ") if p.strip()]
        if not premises:
            continue

        retriever.reset()
        for p in premises:
            retriever.add(p)

        # treat first premise as "ground truth" for a minimal sanity check
        retrieved = retriever.query(goal, cfg.retrieval_k)
        m = retrieval_metrics(retrieved, [premises[0]])
        all_precision.append(m["precision"])
        all_recall.append(m["recall"])
        all_mrr.append(m["mrr"])

    metrics = {
        "precision": round(sum(all_precision) / len(all_precision), 4) if all_precision else 0,
        "recall": round(sum(all_recall) / len(all_recall), 4) if all_recall else 0,
        "mrr": round(sum(all_mrr) / len(all_mrr), 4) if all_mrr else 0,
        "n": len(all_precision),
        "backend": cfg.retriever_backend,
    }
    log.info(f"Retriever eval: {metrics}")
    print(json.dumps(metrics, indent=2))


def mode_reasoner(cfg: Config) -> None:
    from dras.reasoner import evaluate_reasoner, load_trained_model

    model, tokenizer = load_trained_model(cfg)
    metrics = evaluate_reasoner(model, tokenizer, cfg)
    print(json.dumps(metrics, indent=2))


W = 70  # trace line width


def _label_proof(proof_steps_detail: list[dict]) -> dict[str, str]:
    """Assign [P1],[P2]... labels to seeds and [D1],[D2]... to derived steps."""
    label_map: dict[str, str] = {}
    p_idx = d_idx = 0
    for d in proof_steps_detail:
        if d["is_seed"]:
            p_idx += 1
            label_map[d["text"]] = f"P{p_idx}"
        else:
            d_idx += 1
            label_map[d["text"]] = f"D{d_idx}"
    return label_map


def _write_problem_trace(prob: dict, result: dict, path: str) -> None:
    """Write a human-readable reasoning trace for one benchmark problem."""
    events = result.get("events", [])
    proof_detail = result.get("proof_steps_detail", [])
    status = result["status"]
    goal = prob["goal"]

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "═" * W,
        f"{prob.get('id','?')} · {prob.get('pattern','?')} · {prob.get('label','?')}",
        f"Goal : {goal}",
        f"Tier : {prob.get('tier','?')}   Provable: {prob.get('provable', True)}",
        "═" * W, "",
    ]

    # ── Seed premises ─────────────────────────────────────────────────────────
    lines.append("GIVEN PREMISES")
    for i, seed in enumerate(prob["seeds"], 1):
        lines.append(f"  [{i}] {seed}")
    lines.append("")

    # ── Reasoning trace (one block per iteration) ─────────────────────────────
    lines += ["─" * W, "REASONING TRACE", "─" * W]

    iter_open = False
    for ev in events:
        t = ev["type"]

        if t == "seeds":
            continue

        elif t == "iteration":
            if iter_open:
                lines.append("")
            iter_open = True
            sub = f"  [sub-goal: {ev['effective_goal']}]" if ev["effective_goal"] != goal else ""
            lines.append(f"\n  ┌─ Iter {ev['iteration']}  (store: {ev['store_size']} items){sub}")
            lines.append(f"  │  Retrieved:")
            for r in ev["retrieved"]:
                lines.append(f"  │    • {r}")

        elif t == "generated":
            tag = " [sampled]" if ev.get("sampled") else ""
            lines.append(f"  │  Generated{tag}:")
            lines.append(f"  │    {ev['text']}")

        elif t == "rejected":
            reason = "; ".join(ev.get("issues", ["unknown"]))
            lines.append(f"  │  ✗ REJECTED — {reason}")

        elif t == "backchain_trigger":
            lines.append(f"  │  ↩ BACKCHAIN  (consecutive_rejects={ev['consecutive_rejects']})")
            if ev.get("anchor"):
                lines.append(f"  │    anchor: {ev['anchor']}")

        elif t == "backchain_result":
            mark = "✓" if ev["valid"] else "✗"
            lines.append(f"  │    → backchain {mark}: {ev['text']}")

        elif t == "goal_proj_trigger":
            lines.append(f"  │  ⟵ GOAL PROJECTION  (consecutive_rejects={ev['consecutive_rejects']})")

        elif t == "goal_proj_subgoals":
            for sg in ev["subgoals"]:
                lines.append(f"  │    → sub-goal projected: {sg}")

        elif t == "goal_proj_direct":
            lines.append(f"  │    → direct rule application: {ev['text']}")

        elif t == "pairwise_trigger":
            lines.append(f"  │  ⊗ PAIRWISE SEARCH  (consecutive_rejects={ev['consecutive_rejects']})")

        elif t == "pairwise_found":
            lines.append(f"  │    → pairwise found: {ev['text']}")
            for p in ev.get("premises", []):
                lines.append(f"  │      from: {p}")

        elif t == "dedup":
            lines.append(f"  │  ≈ NEAR-DUPLICATE (sim={ev['sim']:.3f}) — skipped")
            lines.append(f"  │    {ev['text']}")

        elif t == "backchain_dedup_result":
            mark = "✓" if ev["valid"] else "✗"
            lines.append(f"  │    → backchain-dedup {mark}: {ev['text']}")

        elif t == "accepted":
            lines.append(f"  └─ ✓ ACCEPTED  goal_sim={ev['goal_sim']:.3f}")
            lines.append(f"       {ev['text']}")

        elif t == "subgoal_achieved":
            lines.append(f"     ★ Sub-goal achieved (sim={ev['sim']:.3f}): {ev['subgoal']}")

        elif t == "proof_found":
            lines.append(f"  ✓✓ PROOF FOUND — iter {ev['iteration']}  goal_sim={ev['goal_sim']:.3f}")

        elif t == "max_iter":
            lines.append(f"\n  MAX ITERATIONS ({ev['iterations']}) — no proof found")

    lines.append("")

    # ── Attributed proof (expert verification section) ────────────────────────
    lines += ["", "═" * W]
    if status == "PROOF_FOUND" and proof_detail:
        n_derived = sum(1 for d in proof_detail if not d["is_seed"])
        lines += [f"PROOF  ({n_derived} derived step{'s' if n_derived != 1 else ''}, derivation order)", "═" * W, ""]
        label_map = _label_proof(proof_detail)

        # Single linear chain — seeds first (iter 0), then derived steps in order
        for d in proof_detail:
            lbl = label_map[d["text"]]
            lines.append(f"  [{lbl}]  {d['text']}")
            if not d["is_seed"]:
                parent_lbls = [label_map.get(p, "?") for p in d["parent_texts"]]
                if parent_lbls:
                    lines.append(f"         ↑ {', '.join('['+l+']' for l in parent_lbls)}")

        lines += ["", f"  ∴  {goal}", ""]
    else:
        lines += [
            f"STATUS: {status}",
            f"  The goal could not be derived from the given premises within",
            f"  {result['iterations']} iterations.",
            "",
        ]

    lines.append("═" * W)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_summary(benchmarks: list[dict], results: list[dict], path: str) -> None:
    """Write a one-page summary with all final proofs."""
    provable = [r for r in results if r["provable"]]
    unprovable = [r for r in results if not r["provable"]]
    pcr = (sum(1 for r in provable if r["status"] == "PROOF_FOUND") / len(provable)
           if provable else 0.0)
    fpr = (sum(1 for r in unprovable if r["status"] == "PROOF_FOUND") / len(unprovable)
           if unprovable else 0.0)

    lines: list[str] = [
        "═" * W,
        "DRAS Evaluation Summary",
        "═" * W, "",
        f"  proof_completion_rate : {pcr:.3f}  "
        f"({sum(1 for r in provable if r['status']=='PROOF_FOUND')}/{len(provable)} provable solved)",
        f"  false_proof_rate      : {fpr:.3f}  "
        f"({sum(1 for r in unprovable if r['status']=='PROOF_FOUND')}/{len(unprovable)} unprovable falsely proved)",
        "",
        f"  {'':1}{'ID':<5} {'Pattern':<30} {'Status':<14} {'Iters':<6} Steps",
        f"  {'─'*4} {'─'*29} {'─'*13} {'─'*5} {'─'*5}",
    ]
    for r in results:
        correct = (r["provable"] and r["status"] == "PROOF_FOUND") or \
                  (not r["provable"] and r["status"] != "PROOF_FOUND")
        tick = "✓" if correct else "✗"
        pid = r.get("id", "?")
        pat = (r.get("pattern") or "?")[:28]
        nsteps = str(len(r.get("proof_tree", [])))
        lines.append(f"  {tick} {pid:<4} {pat:<29} {r['status']:<14} {r['iterations']:<6} {nsteps}")

    lines += ["", "─" * W, "PROOF TRACES", "─" * W]

    for prob, r in zip(benchmarks, results):
        lines.append("")
        detail = r.get("proof_steps_detail", [])
        lines.append(f"{'─'*W}")
        lines.append(f"{r.get('id','?')} · {prob.get('label','?')}")
        lines.append(f"Goal: {prob['goal']}")

        if r["status"] != "PROOF_FOUND" or not detail:
            lines.append(f"  {r['status']} — no proof")
            continue

        label_map = _label_proof(detail)
        for d in detail:
            lbl = label_map[d["text"]]
            lines.append(f"  [{lbl}]  {d['text']}")
            if not d["is_seed"]:
                parent_lbls = [label_map.get(p, "?") for p in d["parent_texts"]]
                if parent_lbls:
                    lines.append(f"         ↑ {', '.join('['+l+']' for l in parent_lbls)}")
        lines.append(f"  ∴  {prob['goal']}")

    lines += ["", "═" * W]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def mode_e2e(cfg: Config, trace_dir: str | None = None) -> None:
    from dras.evaluate import proof_completion_rate, stall_rate
    from dras.loop import run_loop
    from dras.reasoner import load_trained_model
    from dras.retriever import make_retriever

    model, tokenizer = load_trained_model(cfg)
    retriever = make_retriever(cfg)
    benchmarks = _load_benchmarks(cfg)
    results = []

    for prob in benchmarks:
        label = prob.get("label", prob.get("id", "?"))
        log.info(f"\n{'='*55}")
        log.info(f"  {prob.get('id','?')} [{prob.get('tier','?')}] {label}")
        log.info(f"  goal: {prob['goal']}")
        result = run_loop(prob["goal"], prob["seeds"], retriever, model, tokenizer, cfg, verbose=True)
        result.update({
            "id": prob.get("id"), "tier": prob.get("tier"), "pattern": prob.get("pattern"),
            "provable": prob.get("provable", True), "goal": prob["goal"],
            "expected_steps": prob.get("expected_steps", 0),
        })
        results.append(result)

        log.info(f"  → status={result['status']}  iterations={result['iterations']}")
        if result["proof_tree"]:
            for i, step in enumerate(result["proof_tree"], 1):
                log.info(f"    {i}. {step}")

        if trace_dir:
            pid = result.get("id") or "unknown"
            trace_path = os.path.join(trace_dir, f"{pid}_{prob.get('pattern','?')}.txt")
            _write_problem_trace(prob, result, trace_path)
            log.info(f"  trace → {trace_path}")

    # ── summary ──────────────────────────────────────────────────────────────
    provable = [r for r in results if r["provable"]]
    unprovable = [r for r in results if not r["provable"]]

    by_pattern: dict = {}
    for r in results:
        p = r.get("pattern", "unknown")
        by_pattern.setdefault(p, []).append(r["status"])

    summary = {
        "backend": cfg.retriever_backend,
        "proof_completion_rate": proof_completion_rate(provable) if provable else None,
        "false_proof_rate": round(
            sum(1 for r in unprovable if r["status"] == "PROOF_FOUND") / len(unprovable), 4
        ) if unprovable else None,
        "by_problem": [
            {
                "id": r["id"], "tier": r["tier"], "pattern": r["pattern"],
                "status": r["status"], "iterations": r["iterations"],
                "proof_steps": len(r["proof_tree"]),
            }
            for r in results
        ],
        "by_pattern": {p: {"statuses": v, "completion_rate": round(v.count("PROOF_FOUND") / len(v), 3)}
                       for p, v in by_pattern.items()},
    }
    print(json.dumps(summary, indent=2))

    if trace_dir:
        summary_path = os.path.join(trace_dir, "summary.txt")
        _write_summary(benchmarks, results, summary_path)
        log.info(f"\nTrace directory: {trace_dir}")
        log.info(f"Summary       : {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["retriever", "reasoner", "e2e"], default="e2e")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["dense", "bm25"], default=None)
    parser.add_argument("--lookahead", type=int, default=0,
                        help="Number of sampled alternatives to evaluate per step (0=disabled)")
    parser.add_argument("--pair-search", type=int, default=0,
                        help="Consecutive rejects before exhaustive pairwise search (0=disabled, recommended: 12)")
    parser.add_argument("--goal-proj", type=int, default=0,
                        help="Consecutive rejects before goal projection (0=disabled, recommended: 9)")
    parser.add_argument("--trace-dir", type=str, default=None,
                        help="Directory to write per-problem trace files and summary.txt. "
                             "Defaults to outputs/eval_traces/<timestamp> when --mode e2e is used.")
    args = parser.parse_args()

    cfg = Config()
    if args.backend:
        cfg.retriever_backend = args.backend
    if args.lookahead:
        cfg.lookahead_k = args.lookahead
    if args.pair_search:
        cfg.pair_search_k = args.pair_search
    if args.goal_proj:
        cfg.goal_proj_k = args.goal_proj

    set_seed(args.seed)

    if args.mode == "e2e":
        import datetime
        trace_dir = args.trace_dir
        if trace_dir is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_dir = os.path.join("outputs", "eval_traces", ts)
        mode_e2e(cfg, trace_dir=trace_dir)
    else:
        {"retriever": mode_retriever, "reasoner": mode_reasoner}[args.mode](cfg)


if __name__ == "__main__":
    main()
