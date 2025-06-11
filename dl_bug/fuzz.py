import re
import argparse
import multiprocess
from oracle import RewardModel, Outcome
from fine_tuner.llm_engine import LLMEngine

multiprocess.set_start_method("spawn", force=True)

parser = argparse.ArgumentParser("Fuzzing CLI")
parser.add_argument("--instruct_name", required=True)
parser.add_argument("--instruct_path", help="Directory to save / load adapter or merged model")
parser.add_argument("--coder_name", required=True)
parser.add_argument("--coder_path", help="Directory to save / load adapter or merged model")
parser.add_argument("-M", "--use_merged", action="store_true")
parser.add_argument("--prompt_file", default="test_prompt.txt", help="Path to file containing raw prompt")

args = parser.parse_args()

print("Loading models ...")
instruct = LLMEngine(
    mode="inference",
    model_name=args.instruct_name,
    model_type="instruct",
    model_path=args.instruct_path,
    use_merged=args.use_merged,
    additional_special_tokens=["<｜System｜>", "<｜User｜>", "<｜Assistant｜>"]
)
print("Instruct LLM loaded.")

coder = LLMEngine(
    mode="inference",
    model_name=args.coder_name,
    model_type="coder",
    model_path=args.coder_path,
    use_merged=args.use_merged,
    additional_special_tokens=["<｜System｜>", "<｜User｜>", "<｜Assistant｜>"]
)
print("Coder LLM loaded.")

# try:
#     with open(args.prompt_file, "r", encoding="utf-8") as fp:
#         raw_prompt = fp.read()
# except:
raw_prompt = (
    "API: torch.view_as_complex"
    "Bug Description: Segmentation Fault: torch.view_as_complex fails with segfault for a zero dimensional tensor"
)


instructions_so_far = raw_prompt

print("[Generating instructions .................]")
print(instructions_so_far)
for i in range(10):
    next_instruction = instruct.infer(
        raw_prompt = instructions_so_far,
        max_new_tokens=512,
        temperature=0.9,
    )
    next_instruction = re.split(r"[\n|===]", next_instruction)[0]
    print(next_instruction)
    instructions_so_far = f"{instructions_so_far}\n{next_instruction}"
full_instructions = instructions_so_far
print("—" * 40)

print("[Generating code .................]")
code = coder.infer(
    raw_prompt=full_instructions,
    max_new_tokens=1024,
    temperature=0.1,
)
print("—" * 40)
print(code)
print("—" * 40)

print("[Evaluating reward .................]")
reward_model = RewardModel()
reward = reward_model.evaluate_fuzzing_code(code, timeout = 100)

# Map to original format
if reward.outcome == Outcome.SUCCESS:
    status = "ok"
elif reward.outcome == Outcome.BUG:
    status = "library_bug"  
else:
    status = "user_code_error"

full_reward =  {
    "status": status,
    "returncode": reward.returncode,
    "stderr": reward.stderr,
    "reward_score": reward.reward_score,
    "confidence": reward.confidence,
    "bug_likelihood": reward.bug_likelihood
}


print(full_reward)