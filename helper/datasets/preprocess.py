#!/usr/bin/env python3
"""
preprocess_dataset.py  ── create Arrow dataset with train & validation splits
---------------------------------------------------------------------------
Input : JSONL file where each record has
        { "api": ..., "bug description": ..., "Instructions": ... }

Output: Arrow directory containing DatasetDict({"train": ..., "validation": ...})
        Each example is {"prompt": ..., "completion": ...}, ready for CLM fine-tuning.
"""
import argparse, json, re, random
from pathlib import Path
from datasets import Dataset, DatasetDict


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--in_jsonl", required=True,  help="Original JSONL file")
parser.add_argument("--out_dir", required=True, help="Destination directory")

split_group = parser.add_mutually_exclusive_group()
split_group.add_argument("--val_ratio", type=float, default=0.05,
                         help="Fraction of examples to reserve for validation")
split_group.add_argument("--val_size",  type=int,
                         help="Exact number of validation examples (overrides --val_ratio)")
parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible split")
args = parser.parse_args()

OUT_INSTRUCT = Path(args.out_dir, "instruct")
OUT_CODER = Path(args.out_dir, "coder")

# --------------- Train / validation split ---------------------------------
def train_val_split(examples):
    random.shuffle(examples)
    val_n = (args.val_size if args.val_size is not None
             else round(len(examples) * args.val_ratio))
    return examples[val_n:], examples[:val_n]      # train, val

# -------------------- Save Arrow datasets ----------------------------------
def save_arrow(train, val, out_path):
    out_path.mkdir(parents=True, exist_ok=True)
    ds = DatasetDict({
        "train": Dataset.from_list(train),
        "val": Dataset.from_list(val),
    })
    ds.save_to_disk(out_path)
    return ds
    

#  Build examples
instruct_sep = re.compile(r"^\s*\d+\.\s*", re.M)
instruct_examples, coder_examples = [], []

with open(args.in_jsonl, "r", encoding="utf-8") as fh:
    for row in map(json.loads, fh):
        api   = row["API"]
        bug   = row["Bug Description"]
        instructions = [s.strip() for s in instruct_sep.split(row["Instructions"]) if s.strip()]
        code = row["Full Code"]
        # --- INSTRUCTION DATASET ------------------------------------
        for t in range(len(instructions) - 1):                # up to penultimate step
            prefix = "\n".join(
                f"{i+1}. {step}" for i, step in enumerate(instructions[: t + 1])
            )
            instruct_examples.append({
                "prompt": (
                    f"API: {api}\nBug: {bug}\n"
                    f"Instructions so far:\n{prefix}"
                    "\n===\nNext Instruction:"
                ),
                "completion": f"{t+2}. " + instructions[t + 1]
            })
        # --- CODER DATASET ------------------------------------
        coder_prompt = (
            f"API: {api}\nBug: {bug}\n"
            f"Instructions:\n" + " ".join(instructions) +
            "\n===\nCode:"
        )
        coder_examples.append({
            "prompt": coder_prompt,
            "completion": code
        })


random.seed(args.seed)
instruct_train, instruct_val = train_val_split(instruct_examples)
coder_train, coder_val = train_val_split(coder_examples)

print("Saving step-by-step dataset …")
ds_steps  = save_arrow(instruct_train, instruct_val, OUT_INSTRUCT)
print("Saving coder dataset …")
ds_coder  = save_arrow(coder_train, coder_val, OUT_CODER)

print("\nFinished:")
print(f"  steps/train:      {len(ds_steps['train']):>6}")
print(f"  steps/validation: {len(ds_steps['val']):>6}")
print(f"  coder/train:      {len(ds_coder['train']):>6}")
print(f"  coder/validation: {len(ds_coder['val']):>6}")
print(f"\nDatasets saved under {args.out_dir}")