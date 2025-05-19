import re
import yaml

from distiller_LM import DistillerConfig
from instruct_LM import InstructorConfig
from coder_LM import CoderConfig
from trainer import TrainerConfig, FuzzerConfig
from SUT import SUTConfig
from client_LLM import LLMConfig
from logger import LEVEL


def natural_sort_key(s):
    _nsre = re.compile("([0-9]+)")
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)
    ]


def load_configurations(main_config_path: str):
    """Load all configurations from the main YAML file."""
    with open(main_config_path, "r") as f:
        main_config = yaml.load(f, Loader=yaml.FullLoader)

    configs = {}
    configs["exp_name"] = main_config.get("exp_name", "exp")
    configs["distiller_config"] = DistillerConfig(
        folder=main_config["distiller"]["folder"],
        api_name=main_config["distiller"]["api_name"],
        llm_config=LLMConfig(**main_config["distiller"]["llm_config"]),
        system_message=main_config["distiller"]["system_message"],
        instruction=main_config["distiller"]["instruction"],      
    )
    configs["instructor_config"] = InstructorConfig(
        engine_name=main_config["instructor"]["engine_name"],
        tokenizer=main_config["instructor"]["tokenizer"],
        template=main_config["instructor"]["template"],
        separator=main_config["instructor"]["separator"],
        max_instructions=main_config["instructor"]["max_instructions"],
        temperature=main_config["instructor"]["temperature"],
        max_len=main_config["instructor"]["max_len"],
        device=main_config["instructor"]["device"],
    )
    configs["coder_config"] = CoderConfig(
        system_message=main_config["coder"]["system_message"],
        instruction=main_config["coder"]["instruction"],
        api_name=main_config["coder"]["api_name"],
        llm_config=LLMConfig(**main_config["coder"]["llm_config"]),
        device=main_config["coder"]["device"],
    )
    configs["trainer_config"] = TrainerConfig(
        device=main_config["trainer"]["device"],
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
        batch_size=main_config["trainer"]["batch_size"],
    )
    configs["fuzzer_config"] = FuzzerConfig(
        number_of_iterations=main_config["fuzzer"]["number_of_iterations"],
        total_time=main_config["fuzzer"]["total_time"],
        output_folder=main_config["fuzzer"]["output_folder"],
        resume=main_config["fuzzer"].get("resume", False),
        otf=main_config["fuzzer"].get("otf", False),
        log_level=main_config["fuzzer"].get("log_level", LEVEL.INFO),
    )
    configs["SUT_config"] = SUTConfig(
        language=main_config["SUT"]["language"],
        path_documentation=main_config["SUT"]["path_documentation"],
        path_example_code=main_config["SUT"]["path_example_code"],
        trigger_to_generate_input=main_config["SUT"]["trigger_to_generate_input"],
        input_hint=main_config["SUT"]["input_hint"],
        SUT_string=main_config["SUT"]["SUT_string"],
        device=main_config["SUT"]["device"],
        timeout=main_config["SUT"]["timeout"],
        folder=main_config["SUT"]["folder"],
        batch_size=main_config["SUT"]["batch_size"],
        temperature=main_config["SUT"]["temperature"],
        max_length=main_config["SUT"]["max_length"],
        log_level=main_config["SUT"]["log_level"],
        path_hand_written_prompt=main_config["SUT"].get("path_hand_written_prompt", None),
        template=main_config["SUT"].get("template", None),
        lambda_hyper=main_config["SUT"]["lambda_hyper"],
        beta1_hyper=main_config["SUT"]["beta1_hyper"],
        special_eos=main_config["SUT"]["special_eos"],
        oracle_type=main_config["SUT"]["oracle_type"],
    )

    return configs