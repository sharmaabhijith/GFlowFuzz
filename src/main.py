import os
import click
from logger import set_global_log_dir
import datetime
from trainer import Fuzzer
from utils import make_ouput_dirs, load_configurations

@click.command()
@click.option(
    "--target_name", 
    default="../gcc-13/bin/gcc", 
    required=True, 
    help="Full path to the compiler/interpreter."
)
def main(target_name: str):
    # Derive target from basename
    binary = os.path.basename(target_name)

    # Select config file based on binary name
    if binary == "gcc":
        main_config = "config/c_std.yaml"
    elif binary == "g++":
        main_config = "config/cpp_23.yaml"
    elif binary == "go":
        main_config = "config/go_std.yaml"
    elif binary == "javac":
        main_config = "config/java_std.yaml"
    elif binary == "cvc5":
        main_config = "config/smt_general.yaml"
    elif binary == "python":
        main_config = "config/qiskit_opt_and_qasm.yaml"
    else:
        raise ValueError(f"Invalid target_name: {target_name}")

    # Load configurations
    configs = load_configurations(main_config)
    configs["target_name"] = target_name  # make target available to your Fuzzer

    # Set up logs
    exp_name = configs["exp_name"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    expdump_dir = os.path.join("ExpDump", binary, f"{exp_name}")
    folder_path_dict = make_ouput_dirs(expdump_dir)
    set_global_log_dir(os.path.join(expdump_dir, timestamp, "logs"))

    # Instantiate and run fuzzer
    fuzzer = Fuzzer(
        SUT_config=configs["SUT_config"],
        fuzzer_config=configs["fuzzer_config"],
        distiller_config=configs["distiller_config"],
        instructor_config=configs["instructor_config"],
        coder_config=configs["coder_config"],
        trainer_config=configs["trainer_config"],
        target_name=target_name,
        output_folders=folder_path_dict
    )
    fuzzer.train()

if __name__ == "__main__":
    main()
