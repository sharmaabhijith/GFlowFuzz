from .trainer import TrainerConfig, Trainer

def build_instruct_LM_trainer(args):
    """
    Create Instructor LM Trainer from command line arguments
    
    Args:
        args: Command line arguments
        
    Returns:
        Instructor LM Trainer instance
    """
    # Convert args to FuzzTrainerConfig
    config = TrainerConfig(
        # General arguments
        exp_name=args.exp_name,
        save_dir=args.save_dir,
        wandb_project=args.wandb_project,
        prompt_file=args.prompt_file,
        few_shot_file=args.few_shot_file,
        
        # Model arguments
        model_name=args.model_name,
        sft_ckpt=args.sft_ckpt,
        victim_model=args.victim_model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        
        # Training arguments
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        train_steps=args.train_steps,
        grad_acc_steps=args.grad_acc_steps,
        lr=args.lr,
        max_norm=args.max_norm,
        num_warmup_steps=args.num_warmup_steps,
        eval_period=args.eval_period,
        
        # LoRA arguments
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        
        # Sampling arguments
        temp_low=args.temp_low,
        temp_high=args.temp_high,
        max_len=args.max_len,
        max_instructions=getattr(args, 'max_instructions', 5),
        victim_max_len=args.victim_max_len,
        min_len=args.min_len,
        victim_temp=args.victim_temp,
        victim_top_p=args.victim_top_p,
        num_r_samples=args.num_r_samples,
        
        # Buffer arguments
        metric=args.metric,
        buffer_size=args.buffer_size,
        prioritization=args.prioritization,
        compare=args.compare,
        
        # Reward arguments
        beta=args.beta,
        reward_sched_start=args.reward_sched_start,
        reward_sched_end=args.reward_sched_end,
        reward_sched_horizon=args.reward_sched_horizon,
        lm_sched_start=args.lm_sched_start,
        lm_sched_end=args.lm_sched_end,
        lm_sched_horizon=args.lm_sched_horizon,
        
        # Instruction template
        instruction_template=getattr(args, 'instruction_template', "Generate the next instruction:"),
        instruction_separator=getattr(args, 'instruction_separator', "\n\n")
    )
    
    return Trainer(config)



# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Instruction Fuzzing Trainer")
#     # Add your command line arguments here
#     # ...
    
#     args = parser.parse_args()
#     trainer = create_fuzz_trainer(args)
#     trainer.train()