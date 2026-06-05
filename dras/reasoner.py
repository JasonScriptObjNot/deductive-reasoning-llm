"""
Reasoner: owns the Unsloth model.

Public API:
    load_model(cfg)                          → (model, tokenizer)
    train(model, tokenizer, cfg)             → None   (saves adapter)
    infer(model, tokenizer, goal, premises, cfg) → str
    evaluate_reasoner(model, tokenizer, eval_path, cfg) → dict
"""

from __future__ import annotations

import os

from dras.config import Config
from dras.preprocess import END_TOKEN, GOAL_TOKEN, NEXT_STEP_TOKEN, PREMISES_TOKEN
from dras.utils import get_logger, load_jsonl

log = get_logger(__name__)

# Response template for DataCollatorForCompletionOnlyLM —
# loss is computed only on tokens AFTER this string.
RESPONSE_TEMPLATE = NEXT_STEP_TOKEN


def _build_prompt(goal: str, premises: list[str]) -> str:
    premises_str = "  ".join(p.strip() for p in premises)
    return (
        f"{GOAL_TOKEN}{goal}\n"
        f"{PREMISES_TOKEN}{premises_str}\n"
        f"{NEXT_STEP_TOKEN}"
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg: Config):
    """Load base model with a fresh LoRA head — use for training only."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_id,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


def load_trained_model(cfg: Config):
    """Load base model + saved LoRA adapter — use for inference and eval."""
    import os
    from unsloth import FastLanguageModel

    adapter_path = os.path.join(cfg.output_dir, "reasoner_adapter")
    if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        log.info(f"Loading trained adapter from {adapter_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=cfg.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
    else:
        log.warning("No trained adapter found — loading base model without LoRA.")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_id,
            max_seq_length=cfg.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training (Stage 2 / reasoner stage)
# ---------------------------------------------------------------------------

def _tokenize_dataset(rows: list[dict], tokenizer, max_seq_length: int):
    """Pre-tokenize in the main process to avoid Unsloth compiled-cache subprocess issues on Windows."""
    from datasets import Dataset

    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    ds = Dataset.from_list(rows)
    return ds.map(tokenize, batched=True, remove_columns=["text"], num_proc=1)


def train(model, tokenizer, cfg: Config) -> None:
    from transformers import TrainingArguments
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    train_path = os.path.join(cfg.data_dir, "train.jsonl")
    eval_path = os.path.join(cfg.data_dir, "eval.jsonl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}. Run run_preprocess.py first.")

    train_ds = _tokenize_dataset(load_jsonl(train_path), tokenizer, cfg.max_seq_length)
    eval_ds = _tokenize_dataset(load_jsonl(eval_path), tokenizer, cfg.max_seq_length) if os.path.exists(eval_path) else None

    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    output_dir = os.path.join(cfg.output_dir, "reasoner_adapter")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        bf16=cfg.bf16,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds else "no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        max_seq_length=cfg.max_seq_length,
        packing=cfg.packing,
        args=training_args,
    )

    log.info(f"Starting reasoner training — {len(train_ds)} examples, {cfg.num_epochs} epochs")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info(f"Adapter saved to {output_dir}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _raw_generate(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> str:
    """Generate from a raw prompt without DRAS special-token formatting.
    Used for judge calls and other free-form completions."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def infer(
    model,
    tokenizer,
    goal: str,
    premises: list[str],
    cfg: Config,
    do_sample: bool = False,
    temperature: float = 0.7,
) -> str:
    prompt = _build_prompt(goal, premises)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # encode <|end|> as an additional stop token so generation halts there
    end_token_ids = tokenizer.encode(END_TOKEN, add_special_tokens=False)

    generate_kwargs: dict = dict(
        max_new_tokens=cfg.max_new_tokens,
        repetition_penalty=cfg.repetition_penalty,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=[tokenizer.eos_token_id] + end_token_ids,
    )
    if do_sample:
        generate_kwargs["temperature"] = temperature

    outputs = model.generate(**inputs, **generate_kwargs)

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=False)

    # strip our end marker and DeepSeek's EOS token, then truncate at any
    # prompt-format token that indicates the model ran past the answer
    for stop in (END_TOKEN, "<｜end▁of▁sentence｜>", GOAL_TOKEN, PREMISES_TOKEN, NEXT_STEP_TOKEN):
        if stop in result:
            result = result[:result.index(stop)]

    return result.strip()


# ---------------------------------------------------------------------------
# Isolated reasoner evaluation (oracle premises from eval.jsonl)
# ---------------------------------------------------------------------------

def evaluate_reasoner(model, tokenizer, cfg: Config) -> dict:
    from dras.evaluate import reasoner_metrics

    eval_path = os.path.join(cfg.data_dir, "eval.jsonl")
    rows = load_jsonl(eval_path)

    generated, premises_lists = [], []

    for row in rows:
        text = row["text"]
        # parse goal and premises from the text field
        goal_start = text.index(GOAL_TOKEN) + len(GOAL_TOKEN)
        prem_start = text.index(PREMISES_TOKEN) + len(PREMISES_TOKEN)
        step_start = text.index(NEXT_STEP_TOKEN) + len(NEXT_STEP_TOKEN)

        goal = text[goal_start:text.index("\n", goal_start)].strip()
        premises_str = text[prem_start:text.index("\n", prem_start)].strip()
        premises = [p.strip() for p in premises_str.split("  ") if p.strip()]

        pred = infer(model, tokenizer, goal, premises, cfg)
        generated.append(pred)
        premises_lists.append(premises)

    def _infer_fn(prompt: str) -> str:
        return _raw_generate(model, tokenizer, prompt)

    metrics = reasoner_metrics(generated, premises_lists, _infer_fn)
    log.info(f"Reasoner eval: {metrics}")
    return metrics
