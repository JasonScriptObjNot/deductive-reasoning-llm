import os; os.environ.setdefault("PYTHONUTF8", "1")
"""
DRAS interactive demo.

Run a deductive proof attempt on your own premises and goal, or replay one of
the provided examples.  The reasoning loop runs verbosely so you can watch
each retrieval, inference, validation, and recovery step in real time.

Usage
-----
Interactive console (enter goal + premises at prompts):
    python scripts/demo.py

Load from a file:
    python scripts/demo.py --file examples/climate_treaty.txt

Run a built-in benchmark problem by ID:
    python scripts/demo.py --benchmark B6

File format
-----------
Lines starting with "goal:" set the goal (one per file).
Lines starting with "premise:" add a premise.
Blank lines and lines starting with "#" are ignored.

Example file:
    goal: Socrates is mortal.
    premise: All humans are mortal.
    premise: Socrates is a human.
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_from_file(path: str) -> tuple[str, list[str]]:
    goal = None
    premises = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("goal:"):
                goal = line[5:].strip()
            elif line.lower().startswith("premise:"):
                premises.append(line[8:].strip())
    if not goal:
        sys.exit("File must contain a 'goal: ...' line.")
    if not premises:
        sys.exit("File must contain at least one 'premise: ...' line.")
    return goal, premises


def _load_from_console() -> tuple[str, list[str]]:
    print()
    print("  DRAS — Interactive Demo")
    print("  ─────────────────────────────────────────────────────────")
    print("  Enter your goal (the statement you want to prove), then")
    print("  your premises one per line.  Press Enter twice when done.")
    print()

    goal = input("  Goal: ").strip()
    if not goal:
        sys.exit("No goal entered.")

    print()
    print("  Premises (one per line, blank line to finish):")
    premises = []
    while True:
        p = input(f"  [{len(premises)+1}] ").strip()
        if not p:
            break
        premises.append(p)

    if not premises:
        sys.exit("No premises entered.")
    return goal, premises


def _load_benchmark(bid: str) -> tuple[str, list[str]]:
    bench_path = os.path.join(os.path.dirname(__file__), "..", "data", "benchmarks.json")
    if not os.path.exists(bench_path):
        sys.exit("data/benchmarks.json not found.")
    with open(bench_path, encoding="utf-8") as f:
        benchmarks = json.load(f)
    for b in benchmarks:
        if b.get("id", "").upper() == bid.upper():
            return b["goal"], b["seeds"]
    ids = [b.get("id") for b in benchmarks]
    sys.exit(f"Benchmark '{bid}' not found. Available: {ids}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DRAS interactive demo")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--file", metavar="PATH",
                     help="Load goal and premises from a .txt file")
    src.add_argument("--benchmark", metavar="ID",
                     help="Run a built-in benchmark problem (e.g. B1, B6)")
    parser.add_argument("--pair-search", type=int, default=12,
                        help="Consecutive rejects before exhaustive pairwise search (default: 12)")
    parser.add_argument("--goal-proj", type=int, default=9,
                        help="Consecutive rejects before goal projection (default: 9)")
    parser.add_argument("--max-iter", type=int, default=20,
                        help="Maximum loop iterations (default: 20)")
    parser.add_argument("--save-trace", metavar="PATH", default=None,
                        help="Save the reasoning trace to a .txt file after the run")
    args = parser.parse_args()

    # ── Load problem ──────────────────────────────────────────────────────────
    if args.file:
        goal, premises = _load_from_file(args.file)
    elif args.benchmark:
        goal, premises = _load_benchmark(args.benchmark)
    else:
        goal, premises = _load_from_console()

    print()
    print("  ═" * 35)
    print(f"  Goal     : {goal}")
    print(f"  Premises : {len(premises)}")
    for i, p in enumerate(premises, 1):
        print(f"    [{i}] {p}")
    print("  ═" * 35)
    print()
    print("  Loading model … (this takes ~30s on first run)")
    print()

    # ── Load model and config ─────────────────────────────────────────────────
    from dras.config import Config
    from dras.loop import run_loop
    from dras.reasoner import load_trained_model
    from dras.retriever import make_retriever
    from dras.utils import set_seed

    set_seed(42)
    cfg = Config()
    cfg.max_iterations = args.max_iter
    cfg.pair_search_k = args.pair_search
    cfg.goal_proj_k = args.goal_proj

    model, tokenizer = load_trained_model(cfg)
    retriever = make_retriever(cfg)

    print("  Model loaded. Starting reasoning loop …")
    print()

    # ── Run ───────────────────────────────────────────────────────────────────
    result = run_loop(goal, premises, retriever, model, tokenizer, cfg, verbose=True)

    # ── Print summary ─────────────────────────────────────────────────────────
    print()
    print("  ═" * 35)
    print(f"  Status     : {result['status']}")
    print(f"  Iterations : {result['iterations']}")

    if result["status"] == "PROOF_FOUND":
        print()
        print("  PROOF TRACE")
        print("  ─" * 35)
        for i, step in enumerate(result["proof_tree"], 1):
            print(f"  {i:>2}. {step}")
    print("  ═" * 35)

    # ── Optionally save trace ─────────────────────────────────────────────────
    if args.save_trace:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from run_eval import _write_problem_trace
        prob = {
            "id": args.benchmark or "custom",
            "pattern": "custom",
            "label": goal[:60],
            "tier": "—",
            "provable": True,
            "goal": goal,
            "seeds": premises,
        }
        _write_problem_trace(prob, result, args.save_trace)
        print(f"\n  Trace saved → {args.save_trace}")


if __name__ == "__main__":
    main()
