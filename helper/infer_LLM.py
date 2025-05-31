# Inference script for the fine-tuned CodeLlama model
# Example usage:
"""
# Using merged model:
python infer_coder.py --model_path ./coder-lora-merged --merged --interactive

# Using LoRA adapter:
python infer_coder.py --model_path ./coder-lora --interactive

# Single prompt inference:
python infer_coder.py --model_path ./coder-lora-merged --merged --prompt "Write a Python function to reverse a string"

# With 4-bit quantization:
python infer_coder.py --model_path ./coder-lora --use_4bit --interactive
"""
import os
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


def load_model_and_tokenizer(model_path, use_merged=False, use_4bit=False):
    """
    Load the model and tokenizer for inference.
    
    Args:
        model_path: Path to the model (adapter or merged)
        use_merged: If True, load merged model directly. If False, load base + adapter
        use_4bit: Whether to use 4-bit quantization for inference
    """
    if use_merged:
        # Load the merged model directly
        print(f"Loading merged model from {model_path}")
        
        if use_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_cfg,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    else:
        # Load base model + LoRA adapter
        print(f"Loading base model with LoRA adapter from {model_path}")
        base_model_name = "codellama/CodeLlama-13b-Python-hf"
        
        if use_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_cfg,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9):
    """
    Generate a response from the model given a prompt.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens (excluding input prompt)
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response

def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned CodeLlama model")
    # IF CODER LLM USE THIS
    parser.add_argument("--model_path", default = "coder-lora-merged", help="Path to model (adapter or merged)")
    # IF ISNTRUCTOR LLM USE THIS
    # parser.add_argument("--model_path", default = "instruct-lora-merged", help="Path to model (adapter or merged)")
    parser.add_argument("--merged", action="store_false", help="Use merged model (default: use base + adapter)")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, 
        use_merged=args.merged,
        use_4bit=args.use_4bit
    )
    print("Model loaded successfully!")
    
    
    # Prompt
    prompt = """
    ### API: torch.nn
    ### Bug Description: Fails on large input tensors with sparse gradients.
    ### Instructions: Generate minimal PyTorch code to trigger this issue using torch.nn.Linear with sparse gradients.
    ### Code:
    """
    response = generate_response(
        model, tokenizer, prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    print(response)

if __name__ == "__main__":
    main()
