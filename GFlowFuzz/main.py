import os
import sys
import click
import yaml

from GFlowFuzz.distiller_LM import DistillerConfig
from GFlowFuzz.instruct_LM import InstructorConfig
from GFlowFuzz.coder_LM import CoderConfig
from GFlowFuzz.trainer.trainer import TrainerConfig
from GFlowFuzz.fuzzer import Fuzzer


def load_yaml_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--config", default="config/main_config.yaml", help="Path to config YAML.")
def main(config: str):
    config_data = load_yaml_config(config)
    distiller_config = DistillerConfig(
        folder=config_data["distiller"]["folder"],
        logger=config_data["distiller"]["logger"],
        wrap_prompt_func=config_data["distiller"]["wrap_prompt_func"],
        validate_prompt_func=config_data["distiller"]["validate_prompt_func"],
        prompt_components=config_data["distiller"]["prompt_components"],
        openai_config=config_data["distiller"]["openai_config"],
        system_message=config_data["distiller"]["system_message"],
        instruction=config_data["distiller"]["instruction"],
    )
    instructor_config = InstructorConfig(
        model=config_data["instructor"]["model"],
        tokenizer=config_data["instructor"]["tokenizer"],
        instruction_template=config_data["instructor"]["instruction_template"],
        separator=config_data["instructor"]["separator"],
        max_instructions=config_data["instructor"]["max_instructions"],
        temperature=config_data["instructor"]["temperature"],
        max_len=config_data["instructor"]["max_len"],
    )
    coder_config = CoderConfig(
        batch_size=config_data["coder"]["batch_size"],
        temperature=config_data["coder"]["temperature"],
        device=config_data["coder"]["device"],
        coder_name=config_data["coder"]["coder_name"],
        max_length=config_data["coder"]["max_length"],
    )
    trainer_section = config_data["trainer"]
    trainer_config = TrainerConfig(
        model_name=trainer_section["model_name"],
        sft_ckpt=trainer_section["sft_ckpt"],
        buffer_size=trainer_section["buffer_size"],
        train_steps=trainer_section["train_steps"],
        grad_acc_steps=trainer_section["grad_acc_steps"],
        lr=trainer_section["lr"],
        lora_r=trainer_section["lora_r"],
        lora_alpha=trainer_section["lora_alpha"],
        lora_dropout=trainer_section["lora_dropout"],
        max_norm=trainer_section["max_norm"],
        num_warmup_steps=trainer_section["num_warmup_steps"],
        wandb_project=trainer_section["wandb_project"],
        exp_name=trainer_section["exp_name"],
        save_dir=trainer_section["save_dir"],
        prioritization=trainer_section["prioritization"],
    )

    fuzzer = Fuzzer(
        SUT=None,  # Replace with actual SUT
        number_of_iterations=10,
        distiller_config=distiller_config,
        instructor_config=instructor_config,
        coder_config=coder_config,
        trainer_config=trainer_config,
        total_time=1,
        output_folder=config_data["fuzzing"]["output_folder"],
        resume=False,
        otf=False
    )
    fuzzer.run()

if __name__ == "__main__":
    main()
