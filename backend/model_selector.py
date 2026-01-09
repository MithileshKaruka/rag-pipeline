"""
Model Selection and Configuration Utility

Maps frontend model selections to appropriate provider (Ollama/Bedrock)
and model IDs.
"""

from typing import Dict, Tuple, Optional
from constants import (
    OLLAMA_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    BEDROCK_MODEL,
    BEDROCK_EMBEDDING_MODEL
)


# Model mapping: frontend_id -> (provider, model_id, embedding_model)
# Uses constants from constants.py to avoid hardcoding
MODEL_MAPPING = {
    # Ollama models
    "ollama-qwen2.5": ("ollama", OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL),

    # AWS Bedrock models - Llama 3.2 uses cross-region inference profile
    "bedrock-llama3-2-3b": ("bedrock", BEDROCK_MODEL, BEDROCK_EMBEDDING_MODEL),
}


def parse_model_selection(model_id: Optional[str]) -> Tuple[str, str, str]:
    """
    Parse frontend model selection and return provider, model ID, and embedding model.

    Args:
        model_id: Frontend model identifier (e.g., "ollama-llama2", "bedrock-llama3-2-3b")
                 If None, returns default from environment

    Returns:
        Tuple of (provider, model_id, embedding_model)

    Examples:
        >>> parse_model_selection("ollama-llama2")
        ("ollama", "llama2:7b-chat-q4_0", "nomic-embed-text")

        >>> parse_model_selection("bedrock-llama3-2-3b")
        ("bedrock", "meta.llama3-2-3b-instruct-v1:0", "amazon.titan-embed-text-v2:0")

        >>> parse_model_selection(None)
        # Returns default from environment variables
    """
    # If no model specified, use defaults from environment
    if not model_id:
        from constants import (
            LLM_PROVIDER,
            OLLAMA_MODEL,
            BEDROCK_MODEL,
            OLLAMA_EMBEDDING_MODEL,
            BEDROCK_EMBEDDING_MODEL
        )

        if LLM_PROVIDER.lower() == "bedrock":
            return ("bedrock", BEDROCK_MODEL, BEDROCK_EMBEDDING_MODEL)
        else:
            return ("ollama", OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL)

    # Look up model in mapping
    if model_id in MODEL_MAPPING:
        return MODEL_MAPPING[model_id]

    # If not found, default to Ollama Qwen2.5
    return MODEL_MAPPING["ollama-qwen2.5"]


def get_model_info(model_id: Optional[str]) -> Dict[str, str]:
    """
    Get detailed information about a model selection.

    Args:
        model_id: Frontend model identifier

    Returns:
        Dictionary with provider, model, embedding_model, and display_name
    """
    provider, model, embedding = parse_model_selection(model_id)

    # Get display name from mapping or construct it
    if model_id and model_id in MODEL_MAPPING:
        display_name = model_id.replace("-", " ").title()
    else:
        display_name = f"{provider.title()} - {model}"

    return {
        "provider": provider,
        "model": model,
        "embedding_model": embedding,
        "display_name": display_name
    }


def is_bedrock_model(model_id: Optional[str]) -> bool:
    """Check if the selected model uses AWS Bedrock."""
    provider, _, _ = parse_model_selection(model_id)
    return provider == "bedrock"


def is_ollama_model(model_id: Optional[str]) -> bool:
    """Check if the selected model uses Ollama."""
    provider, _, _ = parse_model_selection(model_id)
    return provider == "ollama"
