import os; os.environ.setdefault("PYTHONUTF8", "1")
"""
python scripts/run_train.py --stage reasoner
python scripts/run_train.py --stage retriever   (no-op in MVP)
python scripts/run_train.py --stage assembly    (requires reasoner checkpoint)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dras.config import Config
from dras.utils import get_logger, set_seed

log = get_logger("run_train")

STAGE_TARGETS = {
    "retriever": {"recall": 0.60},   # achievable with 511 pairs; 0.70 needs contrastive hard-neg mining
    "reasoner": {"step_accuracy": 0.85},
    "assembly": {"proof_completion_rate": 0.60},
}


def _check_prior_stage(stage: str, cfg: Config) -> None:
    """Fail fast if the previous stage hasn't met its targets."""
    prior = {"reasoner": "retriever", "assembly": "reasoner"}.get(stage)
    if prior is None:
        return
    metrics_path = os.path.join(cfg.output_dir, prior, "metrics.json")
    if not os.path.exists(metrics_path):
        raise RuntimeError(
            f"Stage '{prior}' metrics not found at {metrics_path}. "
            f"Run --stage {prior} first."
        )
    with open(metrics_path) as f:
        metrics = json.load(f)
    targets = STAGE_TARGETS.get(prior, {})
    for key, threshold in targets.items():
        actual = metrics.get(key, 0.0)
        if actual < threshold:
            raise RuntimeError(
                f"Stage '{prior}' did not meet target {key} >= {threshold} (got {actual}). "
                "Fix the prior stage before running assembly."
            )


def stage_retriever(cfg: Config) -> None:
    """
    Contrastive fine-tuning of the bi-encoder (bge-small) using
    MultipleNegativesRankingLoss over synthetic chain support pairs.

    Query  = goal statement
    Positive = a premise that directly supports a derivation step toward that goal

    In-batch negatives automatically form hard negatives: positives from other
    goals in the same batch are plausible but wrong.
    """
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    from dras.synth import build_retriever_pairs
    from dras.evaluate import retrieval_metrics

    pairs = build_retriever_pairs()
    log.info(f"Retriever training: {len(pairs)} (query, positive) pairs")

    examples = [InputExample(texts=[q, p]) for q, p in pairs]

    model_name = cfg.retriever_model_path if cfg.retriever_model_path else cfg.embed_model
    model = SentenceTransformer(model_name)

    loader = DataLoader(examples, shuffle=True, batch_size=32)
    loss = losses.MultipleNegativesRankingLoss(model=model)

    out_path = os.path.join(cfg.output_dir, "retriever_model")
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=10,
        warmup_steps=max(1, len(examples) // 32 // 5),
        show_progress_bar=True,
        output_path=out_path,
    )
    log.info(f"Fine-tuned retriever saved to {out_path}")

    # Quick sanity eval: for each chain goal, rank all chain premises,
    # check that support premises appear in top-k
    from dras.synth import ANNOTATED_CHAINS
    from sentence_transformers import SentenceTransformer as ST
    ft_model = ST(out_path)
    precisions, recalls = [], []
    for chain in ANNOTATED_CHAINS:
        goal = chain["goal"]
        all_premises = chain["seeds"] + [s["text"] if isinstance(s, dict) else s for s in chain["steps"]]
        goal_emb = ft_model.encode(goal, normalize_embeddings=True)
        prem_embs = ft_model.encode(all_premises, normalize_embeddings=True)
        import numpy as np
        scores = prem_embs @ goal_emb
        top_k_idx = set(np.argsort(scores)[::-1][:cfg.retrieval_k])
        # For annotated steps, check support premises are in top-k
        for step in chain["steps"]:
            if not isinstance(step, dict) or "support" not in step:
                continue
            support_set = set(step["support"])
            retrieved = {all_premises[i] for i in top_k_idx}
            m = retrieval_metrics(list(retrieved), list(support_set))
            precisions.append(m["precision"])
            recalls.append(m["recall"])

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    metrics = {"precision": round(avg_precision, 4), "recall": round(avg_recall, 4), "n_evals": len(recalls)}
    log.info(f"Retriever eval (support recall@{cfg.retrieval_k}): {metrics}")

    out_dir = os.path.join(cfg.output_dir, "retriever")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Tell the config where the fine-tuned model lives (for subsequent stages)
    log.info(f"Set cfg.retriever_model_path = '{out_path}' to use the fine-tuned retriever.")


def stage_reasoner(cfg: Config) -> None:
    from dras.reasoner import load_model, train, evaluate_reasoner

    model, tokenizer = load_model(cfg)
    train(model, tokenizer, cfg)
    metrics = evaluate_reasoner(model, tokenizer, cfg)

    out_dir = os.path.join(cfg.output_dir, "reasoner")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Reasoner metrics: {metrics}")


def stage_assembly(cfg: Config) -> None:
    """
    Short end-to-end fine-tuning pass (1–2 epochs) using the fine-tuned retriever
    to supply premises rather than oracle support sets.

    This bridges the distribution gap: at inference time the reasoner receives
    retrieved premises (possibly imperfect), but was trained only on oracle premises.
    Assembly teaches it to be robust to retrieval noise.
    """
    # load_model triggers Unsloth import — must happen before trl is imported so SFTTrainer is patched
    from dras.reasoner import load_model, _tokenize_dataset, evaluate_reasoner
    from dras.preprocess import _format, NEXT_STEP_TOKEN
    from dras.retriever import make_retriever
    from dras.synth import ANNOTATED_CHAINS, CHAINS, _step_text

    # Use fine-tuned retriever if available
    ft_retriever_path = os.path.join(cfg.output_dir, "retriever_model")
    if os.path.exists(ft_retriever_path):
        cfg.retriever_model_path = ft_retriever_path
        log.info(f"Assembly using fine-tuned retriever from {ft_retriever_path}")

    # Load model first — this imports Unsloth and patches SFTTrainer
    model, tokenizer = load_model(cfg)

    # Now safe to import trl (Unsloth has already patched it)
    from transformers import TrainingArguments
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    retriever = make_retriever(cfg)

    # Generate assembly examples: for each annotated chain step, use live retrieval
    # instead of oracle support to populate the premises field

    rows = []
    all_chains = list(CHAINS) + list(ANNOTATED_CHAINS)
    for chain in all_chains:
        seeds = chain["seeds"]
        goal = chain["goal"]
        steps = chain["steps"]

        retriever.reset()
        for s in seeds:
            retriever.add(s)

        accumulated = list(seeds)
        for step in steps:
            text = _step_text(step)
            # retrieve using live retriever (introduces realistic noise)
            retrieved = retriever.query_multi(goal, accumulated[-1] if len(accumulated) > len(seeds) else None,
                                              cfg.retrieval_k)
            rows.append({"text": _format(goal, retrieved, text)})
            retriever.add(text)
            accumulated.append(text)

    log.info(f"Assembly training: {len(rows)} examples with live-retrieved premises")

    # Very short fine-tuning pass (1 epoch, low LR) — adapt, don't re-learn
    assembly_cfg = Config(
        **{k: v for k, v in cfg.__dict__.items()},
    )
    assembly_cfg.num_epochs = 1
    assembly_cfg.learning_rate = 5e-5

    train_ds = _tokenize_dataset(rows, tokenizer, assembly_cfg.max_seq_length)
    collator = DataCollatorForCompletionOnlyLM(response_template=NEXT_STEP_TOKEN, tokenizer=tokenizer)

    out_dir_adapter = os.path.join(cfg.output_dir, "assembly_adapter")
    training_args = TrainingArguments(
        output_dir=out_dir_adapter,
        per_device_train_batch_size=assembly_cfg.batch_size,
        gradient_accumulation_steps=assembly_cfg.grad_accum,
        learning_rate=assembly_cfg.learning_rate,
        num_train_epochs=assembly_cfg.num_epochs,
        bf16=assembly_cfg.bf16,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_ds, data_collator=collator,
        max_seq_length=assembly_cfg.max_seq_length, packing=False,
        args=training_args,
    )
    trainer.train()
    trainer.save_model(out_dir_adapter)
    tokenizer.save_pretrained(out_dir_adapter)
    log.info(f"Assembly adapter saved to {out_dir_adapter}")

    # Point reasoner_adapter to the assembly output for eval
    cfg.output_dir = os.path.join(cfg.output_dir)
    metrics = evaluate_reasoner(model, tokenizer, cfg)
    out_metrics = os.path.join(cfg.output_dir, "assembly")
    os.makedirs(out_metrics, exist_ok=True)
    with open(os.path.join(out_metrics, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Assembly metrics: {metrics}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["retriever", "reasoner", "assembly"],
                        default="reasoner")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["dense", "bm25"], default=None,
                        help="Override cfg.retriever_backend")
    args = parser.parse_args()

    cfg = Config()
    if args.backend:
        cfg.retriever_backend = args.backend

    set_seed(args.seed)
    _check_prior_stage(args.stage, cfg)

    {"retriever": stage_retriever, "reasoner": stage_reasoner, "assembly": stage_assembly}[
        args.stage
    ](cfg)


if __name__ == "__main__":
    main()
