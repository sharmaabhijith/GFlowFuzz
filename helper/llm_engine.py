#!/usr/bin/env python3
"""
CodeLlamaManager - A unified class for training and inference with CodeLlama models
Usage examples:
python3 llm_engine.py --mode train --model_name deepseek-ai/deepseek-coder-6.7b-instruct --model_type coder --model_path DS_Coder --data_path datasets/arrow/coder
"""

import os
import json
import torch
import argparse
from typing import Optional, Dict, Any
import multiprocess

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
from trl import DataCollatorForCompletionOnlyLM as CollatorForCompletion

from utils import prompt_wrapper


class LLMEngine:
    """
    Unified class for training and inference with LLM models using LoRA.
    """
    
    def __init__(
        self,
        mode:str,
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
        self.mode = mode
        self.model_name = model_name
        self.model_type = model_type
        self.model_path = model_path
        self.use_merged = use_merged
        self.use_4bit = use_4bit

        self._load_model()

        # LoRA configuration
        self.lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    
    def _load_model(self) -> None:
        """Load the base model with appropriate configuration."""

        if self.mode == "train":
            print(f"Loading base model: {self.model_name}")
        elif self.mode == "inference":
            print(f"Loading fine tuned model from {self.model_path} | Merged: {self.use_merged}")
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation = "flash_attention_2"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation = "flash_attention_2"
            )
        if self.use_merged:
            self.model = base_model
        else:
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(tokenizer.special_tokens_map)
        print(tokenizer.additional_special_tokens)
    
    
    def _tokenize_function(self, examples):
        """Tokenization function for the dataset."""

        text = prompt_wrapper(
            model_type = self.model_type,
            model_name = self.model_name,
            prompt = examples['prompt'], 
            completion = examples['completion'],
            eos_token = self.tokenizer.eos_token
        )
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=1024,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    
    
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

    def train(
        self,
        data_path: str,
        batch_size: int = 1,
        accumulation_steps: int = 8,
        epochs: int = 2,
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
        )
        val_ds = split_dataset["val"].map(
            self._tokenize_function,
            remove_columns=split_dataset["val"].column_names,
        )
        if self.model_type=="coder":
            collator = DataCollatorForCompletionOnlyLM(
                tokenizer=self.tokenizer, response_template="<|assistant|>\n"
            )
        else:
            collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=self.model_path,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            num_train_epochs=epochs,
            bf16=True,
            logging_steps=50,
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
            data_collator=collator,
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
        generated_tokens = outputs[0]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response



if __name__ == "__main__":
    
    multiprocess.set_start_method('spawn', force=True)
    
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
            batch_size=2,
            accumulation_steps=8,
        )
    else:
        # Instruct example
        raw_prompt = (
            "API: torch.nn\n"
            "Bug Description: Fails on large input tensors with sparse gradients\n"
            "Instructions so far: \n"
            "1. Invoke `torch.nn.functional.conv2d` exactly as in the full script; this call is expected to surface the issue described: fails on large input tensors with sparse gradients\n"
            "2. If training is involved, compute loss and call `backward()` to trigger autograd logic where the fault occurs."
            "\n===\nNext Instruction:"
        )
        # # Coder example
        # prompt = (
        #     "API: torch.nn\n"
        #     "Bug Description: Fails on large input tensors with sparse gradients\n"
        #     "Instruction: \n"
        #     "Code: "
        # )
        prompt = prompt_wrapper(
            model_type = args.model_type, 
            model_name = args.model_name, 
            prompt = raw_prompt, 
            eos_token = manager.tokenizer.eos_token
        )
        response = manager.generate(prompt)
        print(response)