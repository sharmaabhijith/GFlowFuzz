from peft import PeftModel
base = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter_path = "instruct-lora"

model = PeftModel.from_pretrained(base, adapter_path, device_map="auto")
print(model.peft_config)      # Should list *only* that one adapter
prompt = (
    "<s>[INST] <<SYS>>\n"
    "You are a instruction generation tool. Given an API and bug summary, "
    "write a precise numbered instruction list.\n"
    "<</SYS>>\n\n"
    "API: torch.nn\n"
    "Bug Description: Fails on large input tensors with sparse gradients\n"
    "[/INST]\n"
)
# 2️⃣  Run a temperature-0 generation to see the raw bias
out = generate(prompt, max_new_tokens=200, temperature=0.0, top_p=1.0)
print(out[:500])