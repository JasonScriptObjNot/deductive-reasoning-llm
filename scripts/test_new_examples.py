"""Quick test: load model once, run all 5 new examples, print a results table."""
import os, sys
os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EXAMPLES = [
    ("inheritance",        "examples/inheritance.txt"),
    ("building_inspection","examples/building_inspection.txt"),
    ("scholarship",        "examples/scholarship.txt"),
    ("river_ecosystem",    "examples/river_ecosystem.txt"),
    ("advisory_board",     "examples/advisory_board.txt"),
]

def load_file(path):
    goal, premises = None, []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"): continue
            if line.lower().startswith("goal:"): goal = line[5:].strip()
            elif line.lower().startswith("premise:"): premises.append(line[8:].strip())
    return goal, premises

from dras.config import Config
from dras.loop import run_loop
from dras.reasoner import load_trained_model
from dras.retriever import make_retriever
from dras.utils import set_seed

set_seed(42)
cfg = Config()
cfg.max_iterations = 25
cfg.pair_search_k = 12
cfg.goal_proj_k = 9

print("Loading model …")
model, tokenizer = load_trained_model(cfg)
retriever = make_retriever(cfg)
print("Model loaded.\n")

results = []
for name, path in EXAMPLES:
    goal, premises = load_file(path)
    print(f"{'─'*60}")
    print(f"  {name}")
    print(f"  Goal: {goal}")
    print()
    result = run_loop(goal, premises, retriever, model, tokenizer, cfg, verbose=True)
    results.append((name, result["status"], result["iterations"], result.get("proof_tree", [])))
    print()

print(f"\n{'═'*60}")
print(f"  {'Example':<25} {'Status':<15} {'Iters'}")
print(f"  {'─'*25} {'─'*15} {'─'*5}")
for name, status, iters, proof in results:
    print(f"  {name:<25} {status:<15} {iters}")
print(f"{'═'*60}\n")

for name, status, iters, proof in results:
    if status == "PROOF_FOUND":
        print(f"  PROOF — {name}")
        for i, step in enumerate(proof, 1):
            print(f"    {i}. {step}")
        print()
