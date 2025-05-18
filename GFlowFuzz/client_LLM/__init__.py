from .factory import LLMClient
from .utils import LLMConfig

def get_LLM_client(api_name: str, api_key: str = None, engine_name: str = None):
    """
    Factory function to get the appropriate LLM client with advanced features.
    """
    return LLMClient(api_name=api_name, api_key=api_key, engine_name=engine_name) 
