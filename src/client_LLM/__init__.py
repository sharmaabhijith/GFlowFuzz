from .factory import LLMClient
from .utils import LLMConfig

def get_LLM_client(api_name: str, engine_name: str = None) -> LLMClient:
    """
    Factory function to get the appropriate LLM client with advanced features.
    """
    return LLMClient(api_name=api_name, engine_name=engine_name) 
