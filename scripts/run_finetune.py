"""
Micro-training pass from an existing adapter checkpoint.

Loads the Stage 2 reasoner adapter and runs 1 epoch at lr=1e-5 on:
  - all supplement examples (upweighted x3)
  - a random sample of existing training rows (prevents catastrophic forgetting)

python scripts/run_finetune.py
python scripts/run_finetune.py --out outputs/reasoner_adapter_v2   # save separately
python scripts/run_finetune.py --existing-sample 150               # how many existing rows to mix in
"""
import os; os.environ.setdefault("PYTHONUTF8", "1")
import argparse
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dras.config import Config
from dras.utils import get_logger, load_jsonl, save_jsonl, set_seed

log = get_logger("run_finetune")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="", help="Output adapter path (default: overwrite reasoner_adapter)")
    parser.add_argument("--existing-sample", type=int, default=150,
                        help="How many rows from train.jsonl to mix in (prevents forgetting)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = Config()

    # ── Load supplement ──────────────────────────────────────────────────────
    from dras.supplement import build_supplement
    supplement = build_supplement()
    log.info(f"Supplement: {len(supplement)} targeted examples "
             f"(B4 mixed-context + B3 DS direction)")

    # ── Mix with existing training sample ────────────────────────────────────
    train_path = os.path.join(cfg.data_dir, "train.jsonl")
    existing = load_jsonl(train_path)
    random.shuffle(existing)
    sample = existing[:args.existing_sample]
    log.info(f"Existing sample: {len(sample)} rows from train.jsonl")

    # Upweight supplement x3 — these are the targeted patterns we need to learn
    rows = supplement * 3 + sample
    random.shuffle(rows)
    log.info(f"Total micro-training rows: {len(rows)} "
             f"({len(supplement)*3} supplement, {len(sample)} existing)")

    # ── Load Stage 2 adapter ─────────────────────────────────────────────────
    from dras.reasoner import load_trained_model, _tokenize_dataset
    from dras.preprocess import NEXT_STEP_TOKEN

    log.info("Loading Stage 2 adapter (outputs/reasoner_adapter)...")
    model, tokenizer = load_trained_model(cfg)

    # ── Fine-tune ────────────────────────────────────────────────────────────
    from transformers import TrainingArguments
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    train_ds = _tokenize_dataset(rows, tokenizer, cfg.max_seq_length)
    collator = DataCollatorForCompletionOnlyLM(response_template=NEXT_STEP_TOKEN, tokenizer=tokenizer)

    out_path = args.out if args.out else os.path.join(cfg.output_dir, "reasoner_adapter")
    training_args = TrainingArguments(
        output_dir=out_path,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=1,
        bf16=cfg.bf16,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_ds, data_collator=collator,
        max_seq_length=cfg.max_seq_length, packing=False,
        args=training_args,
    )

    log.info(f"Starting micro-training: 1 epoch, lr={args.lr}")
    trainer.train()
    trainer.save_model(out_path)
    tokenizer.save_pretrained(out_path)
    log.info(f"Saved to {out_path}")

    # ── Quick eval ───────────────────────────────────────────────────────────
    from dras.reasoner import evaluate_reasoner
    metrics = evaluate_reasoner(model, tokenizer, cfg)
    log.info(f"Post-finetune eval: {metrics}")


if __name__ == "__main__":
    main()
