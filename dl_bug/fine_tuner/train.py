import os
import argparse
import multiprocess

from dl_bug.fine_tuner.llm_engine import LLMEngine

multiprocess.set_start_method("spawn", force=True)

parser = argparse.ArgumentParser("LLMEngine CLI")
parser.add_argument("--mode", choices=["train", "inference"], required=True)
parser.add_argument("--model_name", required=True)
parser.add_argument("--model_type", choices=["coder", "instruct"], default="instruct")
parser.add_argument("--model_path", help="Directory to save / load adapter or merged model")
parser.add_argument("--data_path", default="datasets/arrow/instruct")
parser.add_argument("-M", "--use_merged", action="store_true")
parser.add_argument("--prompt_file", default="test_prompt.txt", help="Path to file containing raw prompt")

args = parser.parse_args()

manager = LLMEngine(
    mode=args.mode,
    model_name=args.model_name,
    model_type=args.model_type,
    model_path=args.model_path,
    use_merged=args.use_merged,
    additional_special_tokens=["<｜System｜>", "<｜User｜>", "<｜Assistant｜>"]
)


if args.mode == "train":
    # ───────── Training mode ─────────
    manager.train(
        data_path=args.data_path,
        batch_size=4,
        accumulation_steps=8,
        epochs=5,
    )
    manager._create_merged_model()
else:
    # ───────── Inference mode ─────────
    if not os.path.isfile(args.prompt_file):
            parser.error(f"Prompt file not found: {args.prompt_file}")
    with open(args.prompt_file, "r", encoding="utf-8") as fp:
        raw_prompt = fp.read()
        
    print("[Generating…]")
    result = manager.infer(raw_prompt)
    print("—" * 40)
    print(result)
    print("—" * 40)