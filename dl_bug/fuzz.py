import argparse
import multiprocess
from oracle import RewardModel, Outcome
from dl_bug.fine_tuner.llm_engine import LLMEngine

multiprocess.set_start_method("spawn", force=True)

parser = argparse.ArgumentParser("Fuzzing CLI")
parser.add_argument("--coder_name", required=True)
parser.add_argument("--instruct_name", required=True)
parser.add_argument("--coder_path", help="Directory to save / load adapter or merged model")
parser.add_argument("--instruct_path", help="Directory to save / load adapter or merged model")
parser.add_argument("--data_path", default="datasets/arrow/")
parser.add_argument("-M", "--use_merged", action="store_true")
parser.add_argument("--prompt_file", default="test_prompt.txt", help="Path to file containing raw prompt")

args = parser.parse_args()

instruct = LLMEngine(
    mode="inference",
    model_name=args.instruct_name,
    model_type="instruct",
    model_path=args.model_path,
    use_merged=args.use_merged,
    additional_special_tokens=["<｜System｜>", "<｜User｜>", "<｜Assistant｜>"]
)

coder = LLMEngine(
    mode="inference",
    model_name=args.coder_name,
    model_type="coder",
    model_path=args.model_path,
    use_merged=args.use_merged,
    additional_special_tokens=["<｜System｜>", "<｜User｜>", "<｜Assistant｜>"]
)


with open(args.prompt_file, "r", encoding="utf-8") as fp:
    raw_prompt = fp.read()


instructions_so_far = raw_prompt

print("[Generating instructions .................]")
for i in range(10):
    next_instruction = instruct.infer(instructions_so_far)
    instructions_so_far = f"{instructions_so_far}\n{next_instruction}"
full_instructions = instructions_so_far
print("—" * 40)
print(full_instructions)
print("—" * 40)

print("[Generating code .................]")
code = coder.infer(full_instructions)
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