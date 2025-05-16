import os
import click
from GFlowFuzz.logger import set_global_log_dir
import datetime
from trainer import Fuzzer
from utils import load_configurations

openai.api_key = os.environ.get("OPENAI_API_KEY", "dummy")
client = openai.OpenAI()


@click.command()
@click.option("--main_config", default="config/main.yaml", help="Path to config YAML.")
def main(main_config: str):
    exp_name = main_config.get("exp_name", "exp")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"{exp_name}_{timestamp}")
    set_global_log_dir(log_dir)
    configs = load_configurations(main_config)

    fuzzer = Fuzzer(
        SUT_config=configs["SUT_config"],
        fuzzer_config=configs["fuzzer_config"],
        distiller_config=configs["distiller_config"],
        instructor_config=configs["instructor_config"],
        coder_config=configs["coder_config"],
        trainer_config=configs["trainer_config"]
    )

    fuzzer.train()

if __name__ == "__main__":
    main()
