from .factory import FactoryLLMClient
from .utils import LLMConfig

def get_client(provider: str, api_key: str = None, model: str = None):
    """
    Factory function to get the appropriate LLM client with advanced features.
    """
    return FactoryLLMClient(provider=provider, api_key=api_key, model=model) 
