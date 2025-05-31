#!/usr/bin/env python3
# train_coder_hf.py
# Fine-tune CodeLlama-13B-Python with LoRA in pure HF Trainer
import os, argparse, torch
from datasets import disable_progress_bar
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from data_utils import build_datasets

torch.cuda.empty_cache()

# --------------- CLI -----------------
parser = argparse.ArgumentParser()
# IF CODER LLM USE THIS
parser.add_argument("--model", default="codellama/CodeLlama-13b-Python-hf")
parser.add_argument("--out",   default="coder-lora")
# IF ISNTRUCTOR LLM USE THIS
# parser.add_argument("--model",   default="codellama/CodeLlama-13b-Instruct-hf")
# parser.add_argument("--out",     default="instruct-lora")
# GENERAL ARGUMENTS
parser.add_argument("--data",  default="datasets/bug_report.jsonl")
parser.add_argument("--bsz",   type=int, default=1)
parser.add_argument("--accum", type=int, default=8)
parser.add_argument("--steps", type=int, default=3_000)
args = parser.parse_args()

# --------------- 4-bit base model -----------------
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    quantization_config=bnb_cfg,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# --------------- attach LoRA -----------------
lora_cfg = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# --------------- dataset -----------------
split = build_datasets(args.data)
train_ds, val_ds = split["train"], split["test"]

def tokenize_fn(ex):
    return tokenizer(
        ex["coder_prompt"] + ex["coder_target"],
        truncation=True,
        max_length=2048,
        padding="max_length",
    )

train_ds = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(tokenize_fn,   remove_columns=val_ds.column_names)
disable_progress_bar()               # cleaner logs

print(model.hf_device_map)
print(model.dtype)
print(model.hf_device_map)
print(model.config.quantization_config)

# --------------- trainer -----------------
training_args = TrainingArguments(
    output_dir=args.out,
    per_device_train_batch_size=args.bsz,
    gradient_accumulation_steps=args.accum,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    max_steps=args.steps,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()

# --------------- Save LoRA adapter ---------------
model.save_pretrained(args.out)
tokenizer.save_pretrained(args.out)

# --------------- Create proper merged model -----------------
print("Creating merged model without quantization...")
# Clear GPU memory
del model
torch.cuda.empty_cache()
# Load base model WITHOUT quantization for merging
base_model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# Load and merge the trained LoRA adapter
model_with_lora = PeftModel.from_pretrained(base_model, args.out)
merged_model = model_with_lora.merge_and_unload()
# Save the clean merged model
merged_model.save_pretrained(
    f"{args.out}-merged",
    safe_serialization=True,
    max_shard_size="5GB"
)
tokenizer.save_pretrained(merged_dir)
print(f"Clean merged model saved to {merged_dir}")
# Clean up
del base_model, model_with_lora, merged_model
torch.cuda.empty_cache()