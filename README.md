# Fuzz Everything, Everywhere, All At Once

### Developer Setup

1. Create a virtual environment (Python 3.10) and install the requirements.
    ```shell
    conda create -n fuzz-everything python=3.10
    ```

2. Activate the environment.
    ```shell
    conda activate fuzz-everything
    ```

3. Install the requirements.
    ```shell
    pip install -r requirements.txt
    pre-commit install
    pip install -e .
    ```

Note: to save the conda environment, run `conda env export > environment.yml` and to load it, run `conda env create -f environment.yml`.

### Usage

To download the models for the local language model, you need a way to retireve them, either from HuggingFace directly or from a local repository.
To point the program to the right location, you need to set either the environment variable `HUGGING_FACE_HUB_TOKEN` or `HF_HOME`.
See the two examples of `.env` files in the `envs` folder.
Copy one of them in the root directory and rename it `.env`.

You can run the approach either passing the arguments directly to the script or by using a configuration file.

Note that you need to be logged into HuggingFace and have access to `bigcode/starcoderbase` to use the language model
(for both cached locally or from HuggingFace directly)

#### Arguments

```shell
    python FuzzAll/fuzz.py main \
    --language=cpp --num=10 --otf --level=1 \
    --template=cpp_expected \
    --bs=1 --temperature=1.0 --prompt_strategy=0 --use_hw
```

#### Configuration file

```shell
    python FuzzAll/fuzz.py \
        --config=<config_file_path> main_with_config \
        --folder output_folder
    e.g. python FuzzAll/fuzz.py \
        --config=config/v01_cpp_expected.yaml main_with_config \
        --folder /tmp/fuzzing_output
```
The path to the configuration file is relative to the root directory of the project.

