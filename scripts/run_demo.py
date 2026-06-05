"""
Interactive demo — enter your own goal and premises, watch the loop run step by step.

    python scripts/run_demo.py
    python scripts/run_demo.py --backend bm25

Type 'quit' at any prompt to exit.
"""

import os; os.environ.setdefault("PYTHONUTF8", "1")
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dras.config import Config
from dras.loop import run_loop
from dras.reasoner import load_trained_model
from dras.retriever import make_retriever
from dras.utils import get_logger, set_seed

import argparse

log = get_logger("demo")

DIVIDER = "-" * 60

EXAMPLE_PROBLEMS = [
    {
        "label": "T1 — Direct syllogism",
        "goal": "Socrates is mortal.",
        "seeds": ["All humans are mortal.", "Socrates is a human."],
    },
    {
        "label": "T2 — 3-step chain",
        "goal": "Rex is a vertebrate.",
        "seeds": ["All mammals are vertebrates.", "All dogs are mammals.", "Rex is a dog."],
    },
    {
        "label": "T2 — Car wash reasoning",
        "goal": "Driving to the car wash is faster than walking.",
        "seeds": [
            "All cars travel faster than walking on roads.",
            "The car wash is reachable by road.",
            "We have a car available.",
        ],
    },
    {
        "label": "T3 — 6-premise chain (forces multi-step RAG accumulation)",
        "goal": "Alice can enter the building.",
        "seeds": [
            "Alice is an employee.",
            "All employees have a keycard.",
            "Anyone with a keycard can access the lobby.",
            "Anyone who can access the lobby can use the elevator.",
            "Anyone who can use the elevator can reach floor 3.",
            "Anyone who can reach floor 3 can enter the building.",
        ],
    },
    {
        "label": "T4 — Unprovable (should stall)",
        "goal": "The stock market will rise tomorrow.",
        "seeds": ["The sky is blue.", "Penguins live in Antarctica.", "Water boils at 100 degrees Celsius."],
    },
]


def print_result(result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  STATUS: {result['status']}  |  iterations: {result['iterations']}")
    print(f"{'='*60}")
    if result["proof_tree"]:
        print("\nProof tree:")
        for i, step in enumerate(result["proof_tree"], 1):
            prefix = "  GOAL →" if i == len(result["proof_tree"]) else f"  {i}."
            print(f"{prefix} {step}")
    else:
        print("\nAll generated steps:")
        for s in result["steps"]:
            if not s.is_seed:
                print(f"  iter {s.iteration}: {s.text}")
    print()


def run_preset_from_benchmarks(model, tokenizer, retriever, cfg: Config) -> None:
    import json as _json
    bpath = os.path.join("data", "benchmarks.json")
    if not os.path.exists(bpath):
        print("benchmarks.json not found — using built-in presets.")
        run_preset(model, tokenizer, retriever, cfg)
        return

    with open(bpath) as f:
        benchmarks = _json.load(f)

    print("\nBenchmark problems:\n")
    for i, b in enumerate(benchmarks, 1):
        print(f"  {i}. [{b['tier']}] {b['label']}")
    choice = input(f"\nChoose (1-{len(benchmarks)}) or press Enter to run all: ").strip()
    problems = benchmarks if not choice else [benchmarks[int(choice) - 1]]

    for prob in problems:
        print(f"\n{DIVIDER}")
        print(f"  [{prob['tier']}] {prob['label']}")
        print(f"  Pattern: {prob.get('pattern','?')}")
        print(f"  Goal:  {prob['goal']}")
        print(f"  Seeds ({len(prob['seeds'])}, retrieval_k={cfg.retrieval_k}):")
        for s in prob["seeds"]:
            print(f"    • {s}")
        print(DIVIDER)
        result = run_loop(prob["goal"], prob["seeds"], retriever, model, tokenizer, cfg, verbose=True)
        print_result(result)


def run_preset(model, tokenizer, retriever, cfg: Config) -> None:
    print("\nPreset problems:\n")
    for i, p in enumerate(EXAMPLE_PROBLEMS, 1):
        print(f"  {i}. {p['label']}")
    choice = input(f"\nChoose (1-{len(EXAMPLE_PROBLEMS)}) or press Enter to run all: ").strip()

    problems = EXAMPLE_PROBLEMS if not choice else [EXAMPLE_PROBLEMS[int(choice) - 1]]

    for prob in problems:
        print(f"\n{DIVIDER}")
        print(f"  {prob['label']}")
        print(f"  Goal:  {prob['goal']}")
        print(f"  Seeds ({len(prob['seeds'])}, retrieval_k={cfg.retrieval_k}):")
        for s in prob["seeds"]:
            print(f"    • {s}")
        print(DIVIDER)
        result = run_loop(prob["goal"], prob["seeds"], retriever, model, tokenizer, cfg, verbose=True)
        print_result(result)


def _validate_seed(premise: str, existing: list[str], retriever, model, tokenizer, cfg: Config) -> bool:
    """Validate a manually entered premise. Returns True if user confirms or no issues."""
    if not cfg.validate_manual:
        return True
    from dras.validator import validate_manual_premise
    from sentence_transformers import SentenceTransformer

    embed_model = cfg.retriever_model_path if cfg.retriever_model_path else cfg.embed_model
    _st = SentenceTransformer(embed_model)
    embed_fn = lambda t: _st.encode(t, normalize_embeddings=True)

    from dras.reasoner import _raw_generate as _raw
    infer_fn = lambda prompt: _raw(model, tokenizer, prompt)

    valid, issues = validate_manual_premise(premise, existing, embed_fn, infer_fn, cfg)

    if not valid:
        print(f"  REJECTED: {issues[0]}")
        return False

    if issues:
        print(f"  WARNING: {issues[0]}")
        override = input("  Add anyway? (y/N): ").strip().lower()
        return override == "y"

    return True


def run_custom(model, tokenizer, retriever, cfg: Config) -> None:
    print("\nEnter your own problem.")
    goal = input("Goal statement: ").strip()
    if goal.lower() == "quit":
        return

    print("Enter seed premises one per line. Empty line when done.")
    print("Each premise is validated before being accepted.")
    seeds = []
    while True:
        line = input(f"  Premise {len(seeds)+1}: ").strip()
        if not line:
            break
        if line.lower() == "quit":
            return
        if _validate_seed(line, seeds, retriever, model, tokenizer, cfg):
            seeds.append(line)
            print(f"  ✓ accepted")

    if not seeds:
        print("No premises entered.")
        return

    print(f"\n{DIVIDER}")
    print(f"  Goal:  {goal}")
    print(f"  Seeds ({len(seeds)}, retrieval_k={cfg.retrieval_k}):")
    for s in seeds:
        print(f"    • {s}")
    print(f"  Backend: {cfg.retriever_backend}")
    print(DIVIDER)
    result = run_loop(goal, seeds, retriever, model, tokenizer, cfg, verbose=True)
    print_result(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["dense", "bm25"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config()
    if args.backend:
        cfg.retriever_backend = args.backend
    set_seed(args.seed)

    print(f"\nLoading model (retriever backend: {cfg.retriever_backend})…")
    model, tokenizer = load_trained_model(cfg)
    retriever = make_retriever(cfg)
    print("Ready.\n")

    while True:
        print(f"{DIVIDER}")
        print("  1. Run preset benchmark problems")
        print("  2. Enter your own goal + premises")
        print("  3. Quit")
        choice = input("Choice: ").strip()

        if choice == "1":
            run_preset_from_benchmarks(model, tokenizer, retriever, cfg)
        elif choice == "2":
            run_custom(model, tokenizer, retriever, cfg)
        elif choice in ("3", "quit", "q"):
            break
        else:
            print("Enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
