# GFlowFuzz: Diversity-Promoting Fuzzing with Generative Flow Networks 

(Repo Implementation in Progress)

**GFlowFuzz** is a novel LLM-driven fuzzing framework that integrates **Generative Flow Networks (GFlowNets)** with large language models to generate **diverse, high-reward fuzz inputs**. It systematically improves code coverage and bug discovery in software testing through a two-phase training procedure that fine-tunes an instruction-generation policy.

---

## 🔍 Motivation
LLM-based fuzzers often suffer from **mode collapse**, repeatedly producing similar inputs that miss critical edge cases. GFlowFuzz tackles this by optimizing a latent instruction-generation policy with the GFlowNet objective, enabling exploratory, high-reward sampling of fuzzing code conditioned on problem specifications.

---

## 🧠 Key Components

| Component | Role | Implementation |
|-----------|------|----------------|
| **Distiller (Φ<sub>d</sub>)** | AutoPrompting ⇒ distils specs/docs to a concise prompt **X** | GPT-4o |
| **Instructor (Φ<sub>g</sub>)** | Generates instruction sequence **Z** given **X** | LoRA-adapted LLM trained with **Trajectory Balance** |
| **Coder (Φ<sub>c</sub>)** | Converts *(X, Z)* into executable fuzz code **C** | Code LLM (e.g., StarCoder) |
| **Oracle (𝒪)** | Executes **C** on SUT, returns coverage Δ and crash info | Coverage+bug instrumentation |

---

## 🚀 Training Workflow

### Phase I — Exploration
1. **Sample** instruction sequences with the current Instructor policy.  
2. **Generate** fuzz input via the Coder and **execute** on the SUT.  
3. **Compute reward** `R = Δcoverage + λ·coverage + β·1[bug]`.  
4. **Store** trajectories in a replay buffer.  
5. **Update** Instructor by minimizing **SubTB** loss on mixed on-policy + high-reward batches.

### Phase II — Exploitation
- **Freeze** the trained policy.  
- **Generate & execute** fuzz inputs for the remaining budget to harvest bugs without additional optimisation overhead.

---

## 🧮 Reward Function

```text
R(C) = Δcoverage(C) + λ · coverage(C) + β · 1[bug(C)]
```

## 📂 Repository Structure

GFlowFuzz/
├── distiller/         # Prompt distillation (Φd)
├── instructor/        # GFlowNet training for instruction gen (Φg)
├── coder/             # Code generation using LLMs (Φc)
├── oracle/            # Coverage + bug reward computation
├── replay_buffer/     # Trajectory / reward storage
├── configs/           # YAML/JSON hyper-params
├── scripts/           # Train & evaluate pipelines
├── utils/             # Common helpers
└── README.md          # ← you are here!

## Usage

To download the models for the local language model, you need a way to retireve them, either from HuggingFace directly or from a local repository.
To point the program to the right location, you need to set either the environment variable `HUGGING_FACE_HUB_TOKEN` or `HF_HOME`.
See the two examples of `.env` files in the `envs` folder.
Copy one of them in the root directory and rename it `.env`.

You can run the approach either passing the arguments directly to the script or by using a configuration file.

Note that you need to be logged into HuggingFace and have access to `bigcode/starcoderbase` to use the language model
(for both cached locally or from HuggingFace directly)


## 📖 References
- Bengio et al., 2023 – GFlowNet Foundations
- Hu et al., 2024 – Amortizing Intractable Inference in LLMs
- Lee et al., 2024 – Learning Diverse Attacks for Red-Teaming
- Xia et al., 2024 – Fuzz4All: Universal Fuzzing with LLMs


## Developer Setup

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

## 📌 TODO
- Detailed setup and dependency list
- Pretrained Distiller / Instructor checkpoints
- Automatic coverage instrumentation scripts
- Example fuzzing campaign with sample SUT

## 🤝 Contributing
Contributions are welcome! Please open an issue or pull request to propose features, improvements, or bug fixes.






