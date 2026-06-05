# Deductive Reasoning LLM

Given a set of facts and rules in plain English, derive what logically follows — step by step, with a fully attributed proof trace anyone can verify.

Each inference is validated before it enters the knowledge store. Hallucinations cannot silently compound across a chain. The system knows when it cannot reach the goal and says so.

---

## What it solves

Standard LLMs produce fluent text but treat multi-step logical reasoning as a generation problem — they pattern-match toward a plausible conclusion, frequently skipping or fabricating intermediate steps. Formal logic engines (Prolog, Z3, Lean) require premises to be transcribed into a formal syntax before they can reason.

DRAS sits between these: it reasons over premises written in ordinary language, produces proofs that a domain expert can read and verify, and handles the full range of natural-language complexity that formal systems reject — hedged rules, passive constructions, implicit negations, referential phrasing.

**Target use cases:**
- Legal / policy reasoning: does this set of regulations require action X?
- Medical triage protocols: given these clinical observations, what is indicated?
- Contract analysis: do these clauses entail a specific obligation?
- Educational tools: construct verifiable proofs from student-entered premises
- Agent planning: deductive precondition checking before action selection

---

## How it works

The system builds a proof one validated inference step at a time.

```
Seed premises + goal
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│                         Reasoning Loop                            │
│                                                                   │
│  ┌──────────────┐  retrieve    ┌────────────────────────────┐    │
│  │  Knowledge   │ ──k=5 most──▶│  Fine-tuned 7B LLM         │    │
│  │  Store       │  relevant    │                            │    │
│  │  (grows each │  premises    │  "Given these premises     │    │
│  │   iteration) │              │   and this goal, what      │    │
│  └──────────────┘              │   single step follows?"    │    │
│         ▲                      └──────────────┬─────────────┘    │
│         │                                     │                  │
│         │  add if valid                        ▼                  │
│         │                      ┌──────────────────────────────┐  │
│         └──────────────────────│  Validator                   │  │
│                                │  · well-formed sentence?     │  │
│                                │  · grounded in premises?     │  │
│                                │  · logically follows?        │  │
│                                └──────────────┬───────────────┘  │
│                                               │ valid            │
│                                               ▼                  │
│                                ┌──────────────────────────────┐  │
│                                │  sim(step, goal) ≥ 0.85?     │  │
│                                │  yes → PROOF FOUND           │  │
│                                └──────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

The retriever selects which premises are most relevant to the current goal at each iteration, so long chains do not require the model to hold all premises in context simultaneously. The knowledge store feeds back into itself: derived facts become retrievable premises for subsequent steps.

When the model gets stuck (consecutive rejected steps), four escalating recovery mechanisms fire:

| Trigger | Mechanism | What it does |
|---|---|---|
| 3 rejects | **Sampling fallback** | Switches to temperature sampling to escape repeated failures |
| 6 rejects | **Backward chaining** | Isolates the most applicable rule, runs focused 2-premise inference to fill the missing link |
| 9 rejects | **Goal projection** | Works backward from the goal to identify a closer sub-goal; falls back to deterministic modus ponens via rule parsing |
| 12 rejects | **Exhaustive pairwise search** | BFS over all ordered pairs in the current store — provably complete over single-step derivations |

---

## Results

Evaluated across nine benchmark problems covering the standard deductive inference patterns:

| ID | Pattern | Chain length | Result |
|---|---|---|---|
| B1 | Modus tollens | 2 | PROOF_FOUND |
| B2 | Hypothetical syllogism | 3 | PROOF_FOUND |
| B3 | Disjunctive syllogism + chain | 3 | PROOF_FOUND |
| B4 | Contrapositive chain | 6 | PROOF_FOUND |
| B5 | Mixed negation | 5 | PROOF_FOUND |
| B6 | Long forward chain | 6 | PROOF_FOUND |
| B7 | Unprovable (should stall) | — | MAX_ITER (correct) |
| B8 | Disjunction identification + forward | 5 | PROOF_FOUND |
| B9 | Contrapositive + forward chain | 6 | PROOF_FOUND |

**proof_completion_rate: 1.000 &nbsp;·&nbsp; false_proof_rate: 0.000**

Pre-run traces for all nine problems are in [`outputs/eval_traces/`](outputs/eval_traces/) — readable without running the model.

---

## Quick start

### Read the pre-run traces (no GPU required)

The [`outputs/eval_traces/`](outputs/eval_traces/) directory contains a detailed reasoning log for each benchmark problem: every retrieval, every generated step, every validator decision, every recovery mechanism that fired, and the final attributed proof.

```
outputs/eval_traces/20260605_175948/
  summary.txt                       ← score table + all proofs
  B6_long_forward_chain.txt         ← most interesting: goal projection fires at iter 12
  ...
```

### Try your own premises (GPU required)

Enter a goal and premises at the console:

```bash
python scripts/demo.py
```

Or load from a file:

```bash
python scripts/demo.py --file examples/climate_treaty.txt
python scripts/demo.py --file examples/contrapositive.txt
python scripts/demo.py --file examples/medical_triage.txt
python scripts/demo.py --file examples/disjunctive_syllogism.txt
```

**File format:**
```
# Comments and blank lines are ignored
goal: Your conclusion here.
premise: First fact or rule, in plain English.
premise: Second fact or rule.
```

See the [`examples/`](examples/) directory for templates.

### Run the benchmark suite

```bash
python scripts/run_eval.py --mode e2e --pair-search 12 --goal-proj 9
```

Runs all nine benchmarks and writes a fresh trace set to `outputs/eval_traces/<timestamp>/`.

---

## Training

> Requires: NVIDIA GPU with ≥16GB VRAM (tested on RTX 3090 Ti, CUDA 12.4).

### Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install unsloth trl peft bitsandbytes transformers datasets
pip install chromadb sentence-transformers rank-bm25
```

On Windows:
```powershell
$env:PYTHONUTF8="1"
```

### Build training data

```bash
python scripts/run_preprocess.py
```

Downloads FOLIO (`tasksource/folio`) from HuggingFace, generates positive examples and refusal examples, combines with hand-authored synthetic chains covering all five inference patterns. Writes ~820 training rows and ~145 eval rows to `data/`.

### Train

```bash
python scripts/run_train.py --stage reasoner
```

Fine-tunes DeepSeek-R1-Distill-Qwen-7B (4-bit quantized) with LoRA (r=16) for 3 epochs. Takes ~15 minutes on an RTX 3090 Ti. Saves adapter to `outputs/reasoner_adapter/`.

---

## Project layout

```
dras/
  config.py          All hyperparameters
  loop.py            Reasoning loop + recovery mechanisms
  validator.py       Step validation
  retriever.py       Dense (ChromaDB) and BM25 backends
  reasoner.py        Model loading, training, inference
  preprocess.py      FOLIO → training data
  synth.py           Synthetic training chains
  evaluate.py        Metric functions

scripts/
  demo.py            Interactive demo — console or file input
  run_eval.py        Benchmark evaluation with trace writing
  run_train.py       LoRA fine-tuning
  run_preprocess.py  Dataset construction

examples/            Ready-to-run input files
outputs/eval_traces/ Pre-run reasoning traces for all benchmarks
data/benchmarks.json Nine evaluation problems
```

Full technical write-up — motivation, architecture rationale, training details, and benchmark analysis — in [`DRAS_System_Report.md`](DRAS_System_Report.md).

---

## Stack

- **Base model**: DeepSeek-R1-Distill-Qwen-7B, 4-bit quantized via Unsloth
- **Fine-tuning**: LoRA r=16 via TRL SFTTrainer
- **Retriever**: ChromaDB + `BAAI/bge-small-en-v1.5` (dense) or BM25 (sparse)
- **Training data**: FOLIO + hand-authored synthetic chains (~820 examples)
- **Hardware tested**: RTX 3090 Ti, CUDA 12.4, Windows 11
