# train_coder.py
import os, argparse, math, unsloth
from unsloth import FastLanguageModel
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from data_utils import build_datasets

parser = argparse.ArgumentParser()
parser.add_argument("--model",   default="codellama/CodeLlama-7b-Instruct-hf")
parser.add_argument("--data",    default="datasets/bug_report.jsonl")
parser.add_argument("--out",     default="coder-lora")
parser.add_argument("--bsz",     type=int, default=4)   # per-device
parser.add_argument("--accum",   type=int, default=8)
parser.add_argument("--steps",   type=int, default=3_000)
args = parser.parse_args()

# --- 2-bit + flashattention + RoPE scaling automatically enabled ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model,
    max_seq_length=2048,
    dtype="auto",
    load_in_4bit=True,
)
tokenizer.pad_token = tokenizer.eos_token
model.gradient_checkpointing_enable()

# LoRA config
lora = LoraConfig(
    r=32,             # rank
    lora_alpha=64,
    target_modules="all-linear",   # Unsloth handles names automatically
    lora_dropout=0.05,
    bias="none",
)
model = prepare_model_for_kbit_training(model)
model.add_adapter(lora)

# --- data ---
split = build_datasets(args.data)        # ONE DatasetDict
train_ds = split["train"]
val_ds   = split["test"]

def tokenize(ex):
    return tokenizer(
        ex["inst_prompt"] + ex["inst_target"],
        truncation=True,
        max_length=2048,
    )

train_ds = train_ds.map(tokenize, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(tokenize,   remove_columns=val_ds.column_names)

# --- trainer ---
args_hf = TrainingArguments(
    output_dir=args.out,
    per_device_train_batch_size=args.bsz,
    gradient_accumulation_steps=args.accum,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=1,                 # steps override below
    max_steps=args.steps,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=args_hf,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()

# --- save & merge ---
model.save_pretrained(args.out)
FastLanguageModel.merge_lora(
    model,
    tokenizer,
    save_pretrained_dir=f"{args.out}-merged",
    dtype="fp16",
)
