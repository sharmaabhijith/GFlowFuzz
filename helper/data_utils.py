# data_utils.py
from datasets import load_dataset, DatasetDict
import json, re, textwrap

def stringify_instructions(instr):
    "Convert list -> clean numbered markdown bullet list"
    if isinstance(instr, list):
        return "\n".join(f"{i+1}. {s.strip()}" for i, s in enumerate(instr))
    return instr


def instructor_prompt(api: str, bug: str) -> str:
    SYSTEM_INSTRUCTOR = (
        "You are a instruction generation tool. Given an API and bug summary, "
        "write a precise numbered instruction list to guide another coder model "
        "for generating code."
    )
    return (
        "<s>[INST] <<SYS>>\n"
        f"{SYSTEM_INSTRUCTOR}\n"
        "<</SYS>>\n\n"
        f"API: {api}\n"
        f"Bug Description: {bug}\n"
        "[/INST]\n"          # the model’s answer (our target) starts after this
    )


def coder_prompt(api, bug, instr):
    SYSTEM_CODER = (
        "You are an expert PyTorch developer who writes the smallest, fully "
        "self-contained reproduction scripts for deep learning library bugs."
    )
    return (
        "<s>[INST] <<SYS>>\n"
        f"{SYSTEM_CODER}\n"
        "<</SYS>>\n\n"
        f"API: {api}\n"
        f"Bug Description: {bug}\n"
        f"Instructions:\n{instr}\n"
        "[/INST]\n"          # model’s answer starts after this
    )


# ---------- one function that returns ONE DatasetDict ----------
def build_datasets(json_path, val_ratio=0.03, seed=42):
    def _map(ex):
        api, bug, code = ex["API"], ex["Bug Description"], ex["Full Code"]
        instr_txt = stringify_instructions(ex["Instructions"])

        ex["coder_prompt"] = coder_prompt(api, bug, instr_txt)
        ex["coder_target"] = code + "</s>"

        ex["inst_prompt"]  = instructor_prompt(api, bug)
        ex["inst_target"]  = instr_txt + "</s>"
        return ex

    ds = load_dataset("json", data_files=json_path)["train"]
    ds = ds.map(_map, remove_columns=ds.column_names)
    ds = ds.shuffle(seed)
    return ds.train_test_split(test_size=val_ratio, seed=seed)
