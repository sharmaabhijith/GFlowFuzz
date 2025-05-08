from typing import Any, Dict, List, Optional, Tuple, Union

from .C.C import C_SUT
from .CPP.CPP import CPP_SUT
from .GO.GO import GO_SUT
from .JAVA.JAVA import JAVA_SUT
from .QISKIT.QISKIT import Qiskit_SUT
from .SMT.SMT import SMT_SUT
from .base_sut import base_SUT


def make_SUT(kwargs: Dict[str, Any]) -> base_SUT:
    """Make a SUT from the given command line arguments."""
    language = kwargs["language"]
    if language == "cpp":  # G++
        return CPP_SUT(**kwargs)
    elif language == "c":  # GCC
        return C_SUT(**kwargs)
    elif language == "qiskit":  # Qiskit
        return Qiskit_SUT(**kwargs)
    elif language == "smt2":  # SMT solvers
        return SMT_SUT(**kwargs)
    elif language == "go":  # GO
        return GO_SUT(**kwargs)
    elif language == "java":  # Java
        return JAVA_SUT(**kwargs)
    else:
        raise ValueError(f"Invalid SUT {language}")


def make_SUT_with_config(config_dict: Dict[str, Any]) -> base_SUT:
    """Create a SUT from a configuration dictionary."""

    coder = config_dict["coder"]
    fuzzing = config_dict["fuzzing"]
    SUT = config_dict["target"]

    SUT_compat_dict = {
        # simple mapping
        "language": SUT["language"],
        "folder": fuzzing["output_folder"],
        "bs": coder.get("batch_size", 1),
        "temperature": coder.get("temperature", 1.0),
        "device": coder.get("device", "cuda"),
        "coder_name": coder.get("coder_name", "bigcode/starcoder"),
        "max_length": coder.get("max_length", 1024),
        "use_hw": fuzzing.get("use_hand_written_prompt", False),
        "no_input_prompt": fuzzing.get("no_input_prompt", False),
        "prompt_strategy": fuzzing.get("prompt_strategy", 0),
        "level": fuzzing.get("log_level", 0),
        # compatibility conversion
        # signalling the target to use the config file
        "template": "fuzzing_with_config_file",
        "config_dict": config_dict,
        "SUT_name": fuzzing.get("SUT_name", "SUT"),
    }

    # print the SUT config
    print("=== SUT Config ===")
    for k, v in SUT_compat_dict.items():
        print(f"{k}: {v}")
    print("====================")

    if SUT["language"] == "cpp":
        return CPP_SUT(**SUT_compat_dict)
    elif SUT["language"] == "c":
        return C_SUT(**SUT_compat_dict)
    elif SUT["language"] == "qiskit":
        return Qiskit_SUT(**SUT_compat_dict)
    elif SUT["language"] == "smt2":
        return SMT_SUT(**SUT_compat_dict)
    elif SUT["language"] == "go":
        return GO_SUT(**SUT_compat_dict)
    elif SUT["language"] == "java":
        return JAVA_SUT(**SUT_compat_dict)
    else:
        raise ValueError(f"Invalid SUT {SUT['language']}")
