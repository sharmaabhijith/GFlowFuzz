#!/usr/bin/env python3
"""
show_arrow_examples.py
-------------------------------------------------
Quickly inspect the contents of an Arrow dataset
created with Hugging-Face `Dataset.save_to_disk`.

Usage examples
--------------
# show 5 rows from the default "train" split
python show_arrow_examples.py --data_dir out_dir/instruct

# show 10 rows from the validation split
python show_arrow_examples.py --data_dir out_dir/coder --split val -n 10
"""
from tabulate import tabulate
import argparse, textwrap, pandas as pd
from datasets import load_from_disk

def main(data_dir: str, split: str, n: int) -> None:
    ds_dict = load_from_disk(data_dir)          # DatasetDict({"train":…, "val":…})
    if split not in ds_dict:
        raise ValueError(f"Split ‘{split}’ not found; "
                         f"available = {list(ds_dict.keys())}")

    ds = ds_dict[split]
    n  = min(n, len(ds))
    print(f"Loaded {len(ds):,} rows from {data_dir!r} split «{split}»")
    print(f"Displaying first {n} rows:\n")

    # Convert to DataFrame for pretty printing
    df = pd.DataFrame(ds[:n])
    with pd.option_context("display.max_colwidth", None):
        print(df.to_markdown(index=False, tablefmt="grid"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True,
                   help="Path to the dataset directory (e.g. out_dir/instruct)")
    p.add_argument("--split_name", default="train", choices=["train", "val", ],
                   help="Which split to preview")
    p.add_argument("-n", "--num_rows", type=int, default=2,
                   help="How many examples to show")
    args = p.parse_args()

    # Allow both 'val' and 'validation'
    main(args.data_dir, args.split_name, args.num_rows)
