import os
import sys
import click
import yaml

from distiller_LM import DistillerConfig
from instruct_LM import InstructorConfig
from coder_LM import CoderConfig
from trainer import Fuzzer, TrainerConfig, FuzzerConfig


def load_yaml_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--config", default="config/main_config.yaml", help="Path to config YAML.")
def main(main_config: str):
    main_config = load_yaml_config(main_config)
    
    distiller_config = DistillerConfig(
        folder=main_config["distiller"]["folder"],
        logger=main_config["distiller"]["logger"],
        wrap_prompt_func=main_config["distiller"]["wrap_prompt_func"],
        validate_prompt_func=main_config["distiller"]["validate_prompt_func"],
        prompt_components=main_config["distiller"]["prompt_components"],
        openai_config=main_config["distiller"]["openai_config"],
        system_message=main_config["distiller"]["system_message"],
        instruction=main_config["distiller"]["instruction"],
    )
    instructor_config = InstructorConfig(
        model=main_config["instructor"]["model"],
        tokenizer=main_config["instructor"]["tokenizer"],
        instruction_template=main_config["instructor"]["instruction_template"],
        separator=main_config["instructor"]["separator"],
        max_instructions=main_config["instructor"]["max_instructions"],
        temperature=main_config["instructor"]["temperature"],
        max_len=main_config["instructor"]["max_len"],
    )
    coder_config = CoderConfig(
        batch_size=main_config["coder"]["batch_size"],
        temperature=main_config["coder"]["temperature"],
        device=main_config["coder"]["device"],
        coder_name=main_config["coder"]["coder_name"],
        max_length=main_config["coder"]["max_length"],
    )
    trainer_config = TrainerConfig(
        model_name=main_config["trainer"]["model_name"],
        sft_ckpt=main_config["trainer"]["sft_ckpt"],
        train_steps=main_config["trainer"]["train_steps"],
        grad_acc_steps=main_config["trainer"]["grad_acc_steps"],
        lr=main_config["trainer"]["lr"],
        max_norm=main_config["trainer"]["max_norm"],
        num_warmup_steps=main_config["trainer"]["num_warmup_steps"],
        lora_r=main_config["trainer"]["lora_r"],
        lora_alpha=main_config["trainer"]["lora_alpha"],
        lora_dropout=main_config["trainer"]["lora_dropout"],
        buffer_size=main_config["trainer"]["buffer_size"],
        prioritization=main_config["trainer"]["prioritization"],
    )
    fuzzer_config = FuzzerConfig(
        number_of_iterations=main_config["fuzzer"]["number_of_iterations"],
        total_time=main_config["fuzzer"]["total_time"],
        output_folder=main_config["fuzzer"]["output_folder"],
        resume=main_config["fuzzer"].get("resume", False),
        otf=main_config["fuzzer"].get("otf", False),
    )

    fuzzer = Fuzzer(
        SUT=None,  # Replace with actual SUT
        fuzzer_config=fuzzer_config,
        distiller_config=distiller_config,
        instructor_config=instructor_config,
        coder_config=coder_config,
        trainer_config=trainer_config,
    )

    fuzzer.train()

if __name__ == "__main__":
    main()
