import re
import yaml

from distiller_LM import DistillerConfig, OpenAIConfig
from instruct_LM import InstructorConfig
from coder_LM import CoderConfig
from trainer import TrainerConfig, FuzzerConfig, SUTConfig


def comment_remover(text, lang="cpp"):
    if lang == "cpp" or lang == "go" or lang == "java":

        def replacer(match):
            s = match.group(0)
            if s.startswith("/"):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE,
        )
        return re.sub(pattern, replacer, text)
    elif lang == "smt2":
        return re.sub(r";.*", "", text)
    else:
        # TODO (Add other lang support): temp, only facilitate basic c/cpp syntax
        # raise NotImplementedError("Only cpp supported for now")
        return text

# most fuzzing targets should be some variation of source code
# so this function is likely fine, but we can experiment with
# other more clever variations
def simple_parse(gen_body: str):
    # first check if its a code block
    if "```" in gen_body:
        func = gen_body.split("```")[1]
        func = "\n".join(func.split("\n")[1:])
    else:
        func = ""
    return func


def create_chatgpt_docstring_template(
    system_message: str, user_message: str, docstring: str, example: str, first: str
):
    messages = [{"role": "system", "content": system_message}]
    messages.append({"role": "user", "content": docstring})
    messages.append({"role": "user", "content": example})
    if first != "":
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": "```\n{}\n```".format(first)})
    messages.append({"role": "user", "content": user_message})
    return messages


def natural_sort_key(s):
    _nsre = re.compile("([0-9]+)")
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)
    ]


def load_config_file(filepath: str):
    """Load the config file."""
    with open(filepath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load_configurations(main_config_path: str):
    """Load all configurations from the main YAML file."""
    with open(main_config_path, "r") as f:
        main_config = yaml.load(f, Loader=yaml.FullLoader)

    configs = {}
    configs["openai_config"] = OpenAIConfig(
        engine_name=main_config["openai_config"]["engine_name"],
        max_tokens=main_config["openai_config"]["max_tokens"],
        temperature=main_config["openai_config"]["temperature"],
        stop=main_config["openai_config"]["stop"],
        top_p=main_config["openai_config"]["top_p"],
    )
    configs["distiller_config"] = DistillerConfig(
        folder=main_config["distiller"]["folder"],
        prompt_components=main_config["distiller"]["prompt_components"],
        openai_config=configs["openai_config"],
        system_message=main_config["distiller"]["system_message"],
        instruction=main_config["distiller"]["instruction"],
    )
    configs["instructor_config"] = InstructorConfig(
        engine_name=main_config["instructor"]["engine_name"],
        tokenizer=main_config["instructor"]["tokenizer"],
        instruction_template=main_config["instructor"]["instruction_template"],
        separator=main_config["instructor"]["separator"],
        max_instructions=main_config["instructor"]["max_instructions"],
        temperature=main_config["instructor"]["temperature"],
        max_len=main_config["instructor"]["max_len"],
    )
    configs["coder_config"] = CoderConfig(
        temperature=main_config["coder"]["temperature"],
        engine_name=main_config["coder"]["engine_name"],
        max_length=main_config["coder"]["max_length"],
        eos=main_config["coder"]["eos"],
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