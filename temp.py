# system messages for prompting
self.SYSTEM_MESSAGE = None
self.AP_SYSTEM_MESSAGE = "You are an auto-prompting tool"
self.AP_INSTRUCTION = (
    "Please summarize the above documentation in a concise manner to describe the usage and "
    "functionality of the target "
)
# prompt based variables
self.hw = kwargs["use_hw"]
self.no_input_prompt = kwargs["no_input_prompt"]

self.coder_name = kwargs["coder_name"]
self.coder = None

self.prompt = None
self.initial_prompt = None
self.prev_example = None
# prompt strategies
self.se_prompt = self.wrap_in_comment(
    "Please create a semantically equivalent program to the previous "
    "generation"
)
self.m_prompt = self.wrap_in_comment(
    "Please create a mutated program that modifies the previous generation"
)
self.c_prompt = self.wrap_in_comment(
    "Please combine the two previous programs into a single program"
)
self.p_strategy = kwargs["prompt_strategy"]
# eos based
self.special_eos = None
if "coder_name" in kwargs:
    self.coder_name = kwargs["coder_name"]
if "target_name" in kwargs:
    self.target_name = kwargs["target_name"]



# initialize through either some templates or auto-prompting to determine prompts
def initialize(self):
    self.m_logger.logo(
        "Initializing ... this may take a while ...", level=LEVEL.INFO
    )
    self.m_logger.logo("Loading model ...", level=LEVEL.INFO)
    eos = [
        self.prompt_used["separator"],
        "<eom>",  # for codegen2
        self.se_prompt,
        self.m_prompt,
        self.c_prompt,
    ]
    # if the config_dict is an attribute, add additional eos from config_dict
    # which might be model specific
    if hasattr(self, "config_dict"):
        coder = self.config_dict["coder"]
        coder_name = coder["coder_name"]
        additional_eos = coder.get("additional_eos", [])
        if additional_eos:
            eos = eos + additional_eos
    else:
        coder_name = self.coder_name

    if self.special_eos is not None:
        eos = eos + [self.special_eos]

    self.coder = build_coder_LM(
        eos=eos,
        coder_name=coder_name,
        device=self.device,
        max_length=self.max_length,
    )
    self.m_logger.logo("Model Loaded", level=LEVEL.INFO)
    self.initial_prompt = self.auto_prompt(
        message=self.prompt_used["docstring"],
        hw_prompt=self.prompt_used["hw_prompt"] if self.hw else None,
        hw=self.hw,
        no_input_prompt=self.no_input_prompt,
    )
    self.prompt = self.initial_prompt
    self.m_logger.logo("Done", level=LEVEL.INFO)


def generate_instructions(self, temperature=0.8, num_instructions=3):
    """
    Generate additional instructions based on the current prompt
    
    Args:
        temperature: Temperature to use for instruction generation
        num_instructions: Number of instructions to generate
        
    Returns:
        Updated prompt with additional instructions
    """
    self.m_logger.logo("Generating additional instructions with instruct_LM...", level=LEVEL.INFO)
    
    self.instruction_trainer = build_instruct_LM_trainer(self.instruct_args)
    
    # Generate additional instructions based on the prompt
    additional_instructions = self.instruction_trainer.generate_instructions(
        prompt=self.prompt,
        num_instructions=num_instructions,
        temperature=temperature
    )
    
    # Format and append the additional instructions
    formatted_instructions = self.instruct_args.instruction_separator.join(additional_instructions)
    updated_prompt = self.prompt + f"\n\n{formatted_instructions}"
    
    # Log the instructions for debugging
    self.g_logger.logo(f"Generated additional instructions:\n{formatted_instructions}", level=LEVEL.VERBOSE)
    self.m_logger.logo("Done adding instructions with instruct_LM.", level=LEVEL.INFO)
    
    return updated_prompt

def generate_code(self) -> List[str]:
    self.g_logger.logo(self.prompt, level=LEVEL.VERBOSE)
    
    return self.coder.generate(
        self.prompt,
        batch_size=self.batch_size,
        temperature=self.temperature,
        max_length=1024,
    )

# generation
def generate(self, **kwargs) -> Union[List[str], bool]:
    try:
        fos = self.generate_code()
    except RuntimeError:
        # catch cuda out of memory error.
        self.m_logger.logo("cuda out of memory...", level=LEVEL.INFO)
        del self.coder
        torch.cuda.empty_cache()
        return False
    new_fos = []
    for fo in fos:
        self.g_logger.logo("========== sample =========", level=LEVEL.VERBOSE)
        new_fos.append(self.clean(self.prompt_used["begin"] + "\n" + fo))
        self.g_logger.logo(
            self.clean(self.prompt_used["begin"] + "\n" + fo), level=LEVEL.VERBOSE
        )
        self.g_logger.logo("========== sample =========", level=LEVEL.VERBOSE)
    return new_fos