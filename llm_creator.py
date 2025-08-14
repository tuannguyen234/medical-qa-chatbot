from typing import Dict, Optional, Union, Any
import os

from src.llms.base_llm import BaseLLM
from src.llms.openai_llm import OpenAILLM
from src.llms.local_llm import LocalLLM
from config.settings import settings as core_settings

class LLMCreator:
    """Factory class for creating LLM instances based on configuration."""
    
    @staticmethod
    def create_llm(
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """Create an LLM based on provider type and configuration.
        
        Args:
            provider: The LLM provider ('openai', 'local', etc.)
            model_name: Name of the model to use
            api_key: API key for the provider (if needed)
            base_url: Custom endpoint URL (if needed)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            An instance of a BaseLLM implementation
            
        Raises:
            ValueError: If an unsupported provider is specified
        """
        provider = provider.lower()
        
        # Create OpenAI LLM
        if provider == "openai":
            return OpenAILLM(
                model_name=model_name,
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                base_url=base_url,
                **kwargs
            )
        
        # Create local LLM (Ollama, etc.)
        elif provider == "local":
            return LocalLLM(
                model_name=model_name,
                base_url=base_url or "http://localhost:11434/v1",
                api_key=api_key,
                **kwargs
            )
        
        # Other providers could be added here
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseLLM:
        """Create an LLM from a configuration dictionary.
        
        Args:
            config: Dictionary containing LLM configuration
                Required keys:
                - provider: LLM provider name
                - model_name: Name of the model
                Optional keys:
                - api_key: API key
                - base_url: Custom endpoint
                - Any other provider-specific parameters
                
        Returns:
            An instance of a BaseLLM implementation
        """
        provider = config.pop("provider")
        model_name = config.pop("model_name")
        
        # Extract common parameters
        api_key = config.pop("api_key", None)
        base_url = config.pop("base_url", None)
        
        # Pass remaining config as kwargs
        return cls.create_llm(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            **config
        )
    
    @classmethod
    def from_core_settings(cls) -> BaseLLM:
        """Create an LLM based on the core settings configuration.
        
        This method uses the application's core settings to create an LLM instance.
        
        Returns:
            An instance of a BaseLLM implementation
        """
        llm_config = core_settings.LLM_MODEL_CONFIG
        
        # Determine provider based on settings
        provider = "local" if llm_config.get("type") == "local" else "openai"
        
        # Get model name from settings
        model_name = llm_config.get("model_name") or core_settings.MODEL_NAME or core_settings.OPENAI_MODEL
        
        # Create config dictionary for LLM creation
        config = {
            "provider": provider,
            "model_name": model_name,
            "api_key": llm_config.get("api_key"),
            "base_url": llm_config.get("base_url") or core_settings.BASE_URL,
            "temperature": llm_config.get("temperature", 0.7),
        }
        
        # Add max tokens if specified
        if "max_tokens" in llm_config:
            config["max_tokens"] = llm_config["max_tokens"]
            
        # Add other parameters that might be in the config
        for param in ["top_p", "frequency_penalty", "presence_penalty"]:
            if param in llm_config:
                config[param] = llm_config[param]
        
        return cls.from_config(config)
