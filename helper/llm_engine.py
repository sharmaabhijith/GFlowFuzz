#!/usr/bin/env python3
"""
CodeLlamaManager - A unified class for training and inference with CodeLlama models
Usage examples:

python3 CodeLlama_manager.py --mode inference --model_type instruct --model_name meta-llama/Meta-Llama-3-8B-Instruct --model_path instruct-lora-llama
"""

import os
import json
import torch
from typing import Optional, Dict, Any

from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)


class LLMEngine:
    """
    Unified class for training and inference with LLM models using LoRA.
    """
    
    def __init__(
        self,
        model_name: str,
        model_type: str,
        model_path: Optional[str] = None,
        use_merged: bool = False,
        use_4bit: bool = True
    ):
        """
        Initialize the LLM Manager.
        
        Args:
            model_name: Base model name from HuggingFace
            model_type: Type of model for training
            model_path: Path to trained model/adapter (for inference)
            use_merged: Whether to use merged model for inference
            use_4bit: Whether to use 4-bit quantization
        """
        self.model_name = model_name
        self.model_type = model_type
        self.model_path = model_path
        self.use_merged = use_merged
        self.use_4bit = use_4bit
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    
    def _get_bnb_config(self) -> BitsAndBytesConfig:
        """Get BitsAndBytes configuration for 4-bit quantization."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    
    def _load_base_model(self, for_training: bool = False) -> None:
        """Load the base model with appropriate configuration."""
        print(f"Loading base model: {self.model_name}")
        
        if self.use_4bit and (for_training or not self.use_merged):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self._get_bnb_config(),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            # Disable caching to optimize for fine-tuning
            self.model.config.use_cache = False
            self.model.config.pretraining_tp = 1
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _wrap_prompt(self, prompt: str, completion: str):
        
        task = self.model_type.lower()
        family = self.model_name.lower()
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
        if any(m in family for m in {"llama", "llama2", "mistral"}):
            return (
                f"<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n"
                f"{prompt} [/INST]\n{completion}</s>"
            )
        elif any(m in family for m in {"llama3", "llama-3", "llama_3"}):
            return (
                "<|begin_of_text|>"
                "<|start_header_id|>system<|end_header_id|>\n"
                f"{sys_prompt}\n"
                "<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n"
                f"{prompt}\n"
                "<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n"
                f"{completion}"
                "<|eot_id|>"
            )
        elif any(m in family for m in {"openai", "gpt", "chatgpt"}):
            return (
                f"<|system|>\n{sys_prompt}\n"
                f"<|user|>\n{prompt}\n"
                f"<|assistant|>\n{completion}"
            )
        elif any(m in family for m in {"deepseek", "deepseek-llm", "deepseek-coder"}):
            return (
                f"{sys_prompt}\n\n"
                f"User: {prompt}\n\n"
                f"Assistant: {completion}"
            )
        else:
            raise ValueError(f"Unknown model family: {family}")
    
    
    def _tokenize_function(self, examples):
        """Tokenization function for the dataset."""

        text = self._wrap_prompt(examples['prompt'], examples['completion'])
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    
    def train(
        self,
        data_path: str,
        batch_size: int = 1,
        accumulation_steps: int = 8,
        max_steps: int = 300,
        learning_rate: float = 2e-4,
        eval_steps: int = 250,
        save_steps: int = 250
    ) -> None:
        """
        Train the model with LoRA.
        
        Args:
            data_path: Path to training data (JSONL format)
            batch_size: Training batch size
            accumulation_steps: Gradient accumulation steps
            max_steps: Maximum training steps
            learning_rate: Learning rate
            eval_steps: Steps between evaluations
            save_steps: Steps between saves
        """
        print("Starting training process...")
        torch.cuda.empty_cache()
        
        # Load base model
        self._load_base_model(for_training=True)

        # 2. Prepare it for training (adds fp32 inputs, gradient checkpointing, etc.)
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Attach LoRA
        print("Attaching LoRA adapter...")
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
        # Prepare datasets
        split_dataset = load_from_disk(data_path)
        train_ds = split_dataset["train"].map(
            self._tokenize_function, 
            remove_columns=split_dataset["train"].column_names,
            batched=True
        )
        val_ds = split_dataset["val"].map(
            self._tokenize_function,
            remove_columns=split_dataset["val"].column_names,
            batched=True
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.model_path,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            max_steps=max_steps,
            bf16=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            report_to="tensorboard",
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        
        # Train
        print("Starting training...")
        self.trainer.train()
        
        # Save LoRA adapter
        print(f"Saving LoRA adapter to {self.model_path}")
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        
        # Create merged model
        self._create_merged_model()
    
    def _create_merged_model(self) -> None:
        """Create and save merged model without quantization."""
        print("Creating merged model without quantization...")
        merged_dir = f"{self.model_path}-merged"
        
        # Clear GPU memory
        del self.model
        torch.cuda.empty_cache()
        
        # Load base model WITHOUT quantization for merging
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # Load and merge the trained LoRA adapter
        model_with_lora = PeftModel.from_pretrained(base_model, self.model_path)
        merged_model = model_with_lora.merge_and_unload()
        
        # Save the clean merged model
        merged_model.save_pretrained(
            merged_dir,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        self.tokenizer.save_pretrained(merged_dir)
        print(f"Clean merged model saved to {merged_dir}")
        
        # Clean up
        del base_model, model_with_lora, merged_model
        torch.cuda.empty_cache()
    
    def load_for_inference(self) -> None:
        """Load model for inference."""
        if self.use_merged:
            self._load_merged_model()
        else:
            self._load_model_with_adapter()
        
        print("Model loaded successfully for inference!")
    
    def _load_merged_model(self) -> None:
        """Load merged model for inference."""
        if not self.model_path:
            raise ValueError("model_path must be specified for merged model inference")
        
        print(f"Loading merged model from {self.model_path}")
        
        if self.use_4bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=self._get_bnb_config(),
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _load_model_with_adapter(self) -> None:
        """Load base model with LoRA adapter for inference."""
        if not self.model_path:
            raise ValueError("model_path must be specified for adapter inference")
        
        print(f"Loading base model with LoRA adapter from {self.model_path}")
        
        # Load base model
        if self.use_4bit:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self._get_bnb_config(),
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a response from the model given a prompt.
        
        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated response string
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_for_inference() first.")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens (excluding input prompt)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Engine")
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_type", choices=["coder", "instruct"], default="instruct")
    parser.add_argument("--model_path", help="Path to model to save/load")
    parser.add_argument("--data_path", default="datasets/arrow/instruct")
    parser.add_argument("--use_merged", action="store_true")
    
    args = parser.parse_args()

    manager = LLMEngine(
        model_name=args.model_name,
        model_type=args.model_type,
        model_path=args.model_path,
        use_merged=args.use_merged
    )
    
    if args.mode == "train":
        manager.train(
            data_path=args.data_path,
            batch_size=1,
            accumulation_steps=8,
            max_steps=1000
        )
    else:  
        manager.load_for_inference()
        # Instruct example
        prompt = (
            "[INST] <<SYS>>\n"
            "You are an instruction generation tool. Given an API and bug summary, "
            "you have to create one natural language instruction at a time.\n"
            "<</SYS>>\n\n"
            "API: torch.nn\n"
            "Bug Description: Fails on large input tensors with sparse gradients\n"
            "[/INST]"
        )
        # prompt = (
        #     "<s>[INST] <<SYS>>\n"
        #     "You are a instruction generation tool. Given an API and bug summary, "
        #     "you have to create one natural language instruction at a time.\n"
        #     "<</SYS>>\n\n"
        #     "API: torch.nn\n"
        #     "Bug Description: Fails on large input tensors with sparse gradients\n"
        #     "[/INST]\n"
        # )
        # # Coder example
        # prompt = (
        #     "You are an expert PyTorch developer who writes the smallest, fully "
        #     "self-contained reproduction scripts for deep learning library bugs."
        #     "API: torch.nn\n"
        #     "Bug Description: Fails on large input tensors with sparse gradients\n"
        #     "Instruction: \n"
        #     "Code: "
        # )
        response = manager.generate(prompt)
        print(response)