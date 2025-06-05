def prompt_wrapper(
    model_type: str,
    model_name: str,
    prompt: str, 
    completion: Optional[str] = None,
    eos_token: Optional[str] = None,
):
        
    task = model_type.lower()
    family = model_name.lower()
    # System messaged based on the model type/task
    if task == "instruct":
        sys_prompt = (
            "You are a helpful assistant that writes the next instruction.\n"
            "Given an API, Bug Description and Instructions generated so far, "
            "generate next natural-language instruction."
        )
    elif task == "coder":
        sys_prompt = (
            "You are an expert PyTorch and Python coder. \n"
            "Given an API, Bug Description and a set of all Instructions, "
            "generate self-contained code for triggering deep learning library bugs."
        )
    else:
        raise ValueError(f"Unknown model type: {task}")

    # Wrap text cases
    if any(m in family for m in {"codellama"}):
        base_prompt = (
            f"<｜System｜>\n{sys_prompt}\n"
            f"<｜User｜>\n{prompt}\n"
            f"<｜Assistant｜>\n"
        )
        if completion:
            return base_prompt + f"{completion}\n{tokenizer.eos_token}"
        return {"text": base_prompt}

    elif any(m in family for m in {"llama3", "llama-3", "llama_3"}):
        base_prompt =  (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{sys_prompt}\n"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        if completion:
            return base_prompt + f"{completion}<|eot_id|>"
        return base_prompt

    elif any(m in family for m in {"mistral"}):
        base_prompt = (
            f"<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n"
            f"{prompt} [/INST]\n{completion}</s>"
        )
        if completion:
            return base_prompt + f"{completion}</s>"
        return base_prompt

    elif any(m in family for m in {"gemma", "gemma-7b", "google/gemma"}):
        base_prompt = (
            f"<start_of_turn>user\n{sys_prompt}\n{prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        if completion:
            return base_prompt + f"{completion}<end_of_turn>"
        return base_prompt
    else:
        raise ValueError(f"Unknown model family: {family}")