from typing import Any

from .C.C import C_SUT
from .CPP.CPP import CPP_SUT
from .GO.GO import GO_SUT
from .JAVA.JAVA import JAVA_SUT
from .QISKIT.QISKIT import Qiskit_SUT
from .SMT.SMT import SMT_SUT
from .base_sut import BaseSUT
from .utils import SUTConfig, FResult

# Dictionary to map language to SUT class
__SUT_CLASS_MAP = {
    "cpp": CPP_SUT,
    "c": C_SUT,
    "qiskit": Qiskit_SUT,
    "smt2": SMT_SUT,
    "smt": SMT_SUT,  # Alias for smt2
    "go": GO_SUT,
    "java": JAVA_SUT,
}

def make_SUT(sut_config: SUTConfig) -> BaseSUT:
    """Create a SUT from a SUTConfig object and an optional coder instance."""
    # Optional: Print SUT config for verification (can be controlled by a flag in sut_config if needed)
    print("=== SUT Config ===")
    # Simple way to print dataclass fields and values
    for field in sut_config.__dataclass_fields__:
        print(f"{field}: {getattr(sut_config, field)}")
    print("====================")

    sut_class = __SUT_CLASS_MAP.get(sut_config.language)
    if sut_class:
        return sut_class(sut_config)
    else:
        raise ValueError(f"Invalid SUT language in SUTConfig: {sut_config.language}")
