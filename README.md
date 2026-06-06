# Deductive Reasoning LLM

**A proposed architecture for reliable, auditable reasoning over natural language — and a working proof of concept.**

---

## The problem with current LLM reasoning

Every existing approach to LLM-based reasoning shares a structural flaw: **generation and validation are fused**. When a model generates a chain of thought, each step flows directly into the next with no mechanism to catch errors. A hallucination at step 3 of a 6-step chain contaminates everything that follows — silently, with no attribution, no way for a reader to find where the reasoning went wrong. Scaling the model makes errors less frequent; it does not make them detectable or the reasoning auditable.

Formal logic engines (Prolog, Z3, Lean) solve the validation problem but require premises to be transcribed into rigid formal syntax before they can reason at all. That transcription step is where errors are introduced, and it makes these tools inaccessible to the people who actually need to reason over ordinary documents.

The gap — **a system that reasons over natural language with the validation discipline of a formal system** — is what this architecture is designed to fill.

---

## The architectural approach

Decompose reasoning into single atomic steps. Validate each step independently before it enters the reasoning chain. Accumulate only verified conclusions.

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
│         │  add ONLY if valid                   ▼                  │
│         │                      ┌──────────────────────────────┐  │
│         └──────────────────────│  Independent validator       │  │
│                                │  · well-formed?              │  │
│                                │  · grounded in premises?     │  │
│                                │  · logically follows?        │  │
│                                └──────────────┬───────────────┘  │
│                                               │ pass             │
│                                               ▼                  │
│                                ┌──────────────────────────────┐  │
│                                │  sim(step, goal) ≥ 0.85?     │  │
│                                │  yes → PROOF FOUND           │  │
│                                └──────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

Three properties follow directly from this structure:

**Hallucinations cannot silently compound.** Every step must survive independent validation before it enters the store. A step that does not follow from its premises is discarded — it does not become a premise for the next step.

**Every conclusion is attributed.** The proof trace records exactly which premises produced each derived step. A domain expert can verify the chain without any AI expertise, following each link back to the original facts.

**The system knows when it cannot reach the goal.** When no valid proof exists, the loop exhausts its recovery mechanisms and returns MAX_ITER — not a confabulated answer.

---

## Why this scales

The architecture is a proposed foundation, not a finished system. The current prototype uses a 7B model and simple dense retrieval. The path to a production-grade system is a sequence of composable additions, not a redesign:

**More training data, not more architecture.** The model was not programmed with logic rules — it learned atomic inference from ~820 examples. Broader coverage of defeasible reasoning, exception handling, quantifier scope, and domain-specific language is a data problem. More diverse training examples improve reliability without any structural change.

**Known NLP tools integrate at clean boundaries.** Coreference resolution (resolving "he," "it," "the company" to their referents) slots in before premises enter the store. Named entity recognition normalizes entity names across the chain. Scope disambiguation runs on incoming premises. These are preprocessing steps with no coupling to the reasoning loop. The components that handle them are mature and off-the-shelf.

**The validator is a pluggable interface.** For inference patterns where correctness is formally checkable — modus ponens, modus tollens, hypothetical syllogism — a symbolic verifier can replace the LLM judge for those patterns, providing formal correctness guarantees while the LLM handles the rest. This is a swap in one function.

**The architecture addresses the use cases where LLM reliability falls shortest.** Legal and policy reasoning, clinical protocol verification, contract analysis, agent precondition checking — these are domains where an auditable, attributed reasoning trace has concrete value and where the cost of silent hallucination is highest. The current architecture already produces the right output format for these use cases.

---

## Recovery mechanisms

When the model gets stuck, four escalating mechanisms search the logical space:

| Trigger | Mechanism | What it does |
|---|---|---|
| 3 rejects | **Sampling fallback** | Switches to temperature sampling to escape repeated failures |
| 6 rejects | **Backward chaining** | Isolates the most applicable rule; runs focused 2-premise inference |
| 9 rejects | **Goal projection** | Works backward from the goal; falls back to deterministic modus ponens via rule parsing |
| 12 rejects | **Exhaustive pairwise search** | BFS over all ordered premise pairs — provably complete over single-step derivations |

---

## Prototype results

Evaluated on nine benchmark problems covering the standard deductive inference patterns:

| ID | Pattern | Chain length | Result |
|---|---|---|---|
| B1 | Modus tollens | 2 | PROOF_FOUND |
| B2 | Hypothetical syllogism | 3 | PROOF_FOUND |
| B3 | Disjunctive syllogism + chain | 3 | PROOF_FOUND |
| B4 | Contrapositive chain | 6 | PROOF_FOUND |
| B5 | Mixed negation | 5 | PROOF_FOUND |
| B6 | Long forward chain | 6 | PROOF_FOUND |
| B7 | Unprovable — should stall | — | MAX_ITER ✓ |
| B8 | Disjunction identification + forward | 5 | PROOF_FOUND |
| B9 | Contrapositive + forward chain | 6 | PROOF_FOUND |

**proof_completion_rate: 1.000 &nbsp;·&nbsp; false_proof_rate: 0.000**

These are controlled benchmarks on a small model. They demonstrate that the architectural properties hold — validated atomic steps, correct refusal on unprovable problems, attributed proof traces — not that the system is production-ready. The full reasoning traces for all nine problems are in [`outputs/eval_traces/`](outputs/eval_traces/) and are readable without running the model.

---

## Quick start

### Read the pre-run traces (no GPU, no setup)

The [`outputs/eval_traces/`](outputs/eval_traces/) directory contains the full reasoning log for each benchmark problem: every retrieval, generated step, validator decision, recovery mechanism trigger, and the final attributed proof chain. Open any file directly on GitHub or clone and read locally.

```
outputs/eval_traces/20260605_175948/
  summary.txt                       ← score table + all proofs
  B6_long_forward_chain.txt         ← most interesting: goal projection fires at iter 12
  ...
```

### Try your own premises (GPU required)

**Step 1 — Install**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install unsloth trl peft bitsandbytes transformers datasets
pip install chromadb sentence-transformers rank-bm25
```

On Windows:
```powershell
$env:PYTHONUTF8="1"
```

**Step 2 — Download the trained adapter**

```bash
python scripts/download_weights.py
```

Downloads the LoRA adapter (~145 MB) from the [v1.0 release](https://github.com/JasonScriptObjNot/deductive-reasoning-llm/releases/tag/v1.0). The base model is pulled automatically from HuggingFace on first run.

**Step 3 — Run**

```bash
python scripts/demo.py --file examples/climate_treaty.txt      # 6-step forward chain
python scripts/demo.py --file examples/contrapositive.txt      # modus tollens
python scripts/demo.py --file examples/inheritance.txt         # legal chain
python scripts/demo.py --file examples/river_ecosystem.txt     # causal cascade
python scripts/demo.py --benchmark B6                          # replay a benchmark
python scripts/demo.py                                         # interactive console
```

**File format:**
```
goal: The conclusion you want to prove.
premise: First fact or rule, in plain English.
premise: Second fact or rule.
```

**Run the full benchmark suite:**

```bash
python scripts/run_eval.py --mode e2e --pair-search 12 --goal-proj 9
```

---

## Training from scratch

> Requires: NVIDIA GPU with ≥16GB VRAM (tested on RTX 3090 Ti, CUDA 12.4).

```bash
python scripts/run_preprocess.py   # build training data from FOLIO + synthetic chains
python scripts/run_train.py --stage reasoner  # fine-tune LoRA adapter (~15 min)
```

---

## Project layout

```
dras/
  config.py          All hyperparameters
  loop.py            Reasoning loop + recovery mechanisms
  validator.py       Per-step validation
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
  download_weights.py  Fetch trained adapter from GitHub release

examples/            Ready-to-run input files (9 problems across 5 domains)
outputs/eval_traces/ Pre-run reasoning traces for all benchmarks
data/benchmarks.json Nine evaluation problems
```

Full technical write-up in [`DRAS_System_Report.md`](DRAS_System_Report.md).

---

## Stack

- **Base model**: DeepSeek-R1-Distill-Qwen-7B, 4-bit quantized via Unsloth
- **Fine-tuning**: LoRA r=16, TRL SFTTrainer, ~820 training examples
- **Retriever**: ChromaDB + `BAAI/bge-small-en-v1.5` (dense) or BM25 (sparse)
- **Hardware tested**: RTX 3090 Ti, CUDA 12.4, Windows 11
