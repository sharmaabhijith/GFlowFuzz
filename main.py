
import os
import click
from GFlowFuzz import base_SUT, make_SUT_with_config
from GFlowFuzz.utils import load_config_file
from GFlowFuzz import Fuzzer



@click.group()
@click.option(
    "config_file",
    "--config",
    type=str,
    default=None,
    help="Path to the configuration file.",
)
@click.pass_context
def cli(ctx, config_file):
    """Run the main using a configuration file."""
    if config_file is not None:
        config_dict = load_config_file(config_file)
        ctx.ensure_object(dict)
        ctx.obj["CONFIG_DICT"] = config_dict


@cli.command("main_with_config")
@click.pass_context
@click.option(
    "folder",
    "--folder",
    type=str,
    default="Results/test",
    help="folder to store results",
)
@click.option(
    "cpu",
    "--cpu",
    is_flag=True,
    help="to use cpu",  # this is for GPU resource low situations where only cpu is available
)
@click.option(
    "batch_size",
    "--batch_size",
    type=int,
    default=30,
    help="batch size for the model",
)
@click.option(
    "coder_name",
    "--coder_name",
    type=str,
    default="bigcode/starcoderbase",
    help="model to use",
)
@click.option(
    "SUT",
    "--SUT",
    type=str,
    default="",
    help="specific SUT to run",
)

def main_with_config(ctx, folder, cpu, batch_size, SUT, coder_name):
    """Run the main using a configuration file."""
    config_dict = ctx.obj["CONFIG_DICT"]
    fuzzing = config_dict["fuzzing"]
    config_dict["fuzzing"]["output_folder"] = folder
    if cpu:
        config_dict["coder"]["device"] = "cpu"
    if batch_size:
        config_dict["coder"]["batch_size"] = batch_size
    if coder_name != "":
        config_dict["coder"]["coder_name"] = coder_name
    if SUT != "":
        config_dict["fuzzing"]["SUT_name"] = SUT
    print(config_dict)

    SUT_obj = make_SUT_with_config(config_dict)
    if not fuzzing["evaluate"]:
        assert (
            not os.path.exists(folder) or fuzzing["resume"]
        ), f"{folder} already exists!"
        os.makedirs(fuzzing["output_folder"], exist_ok=True)
        
        # Use the Fuzzer class instead of the fuzz function
        fuzzer = Fuzzer(
            SUT=SUT_obj,
            number_of_iterations=fuzzing["num"],
            total_time=fuzzing["total_time"],
            output_folder=folder,
            resume=fuzzing["resume"],
            otf=fuzzing["otf"],
        )
        fuzzer.run()
    else:
        fuzzer = Fuzzer(SUT_obj, 0, 0, folder, False, False)
        fuzzer.evaluate_all()

def fuzz(
    SUT: base_SUT,
    number_of_iterations: int,
    total_time: int,
    output_folder: str,
    resume: bool,
    otf: bool,
):
    """
    Legacy function for backward compatibility.
    
    Use Fuzzer class instead for new code.
    """
    fuzzer = Fuzzer(SUT, number_of_iterations, total_time, output_folder, resume, otf)
    fuzzer.run()


# evaluate against the oracle to discover any potential bugs
# used after the generation
def evaluate_all(SUT: base_SUT):
    """
    Legacy function for backward compatibility.
    
    Use Fuzzer.evaluate_all() instead for new code.
    """
    fuzzer = Fuzzer(SUT, 0, 0, "", False, False)
    fuzzer.evaluate_all()


if __name__ == "__main__":
    cli()