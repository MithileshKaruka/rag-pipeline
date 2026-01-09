"""
Models API Router

Provides endpoints for retrieving available model configurations.
"""

from fastapi import APIRouter
from typing import List, Dict
from model_selector import MODEL_MAPPING, get_model_info


def create_models_router() -> APIRouter:
    """
    Create and configure the models router.

    Returns:
        Configured APIRouter instance
    """
    router = APIRouter()

    @router.get("/models", response_model=List[Dict])
    async def get_available_models():
        """
        Get list of available models with their configurations.

        Returns:
            List of model configurations
        """
        models = []

        for model_id in MODEL_MAPPING.keys():
            provider, model_name, embedding_model = MODEL_MAPPING[model_id]

            # Determine display information
            if provider == "ollama":
                if "qwen2.5:0.5b" in model_name:
                    display_name = "Qwen2.5 0.5B"
                    description = "Ollama - Local, ~400MB"
                    cost = "Free (Local)"
                else:
                    display_name = model_name
                    description = "Ollama - Local"
                    cost = "Free (Local)"
            elif provider == "bedrock":
                if "llama3-2-3b" in model_name:
                    display_name = "Llama 3.2 3B"
                    description = "AWS Bedrock"
                    cost = "$0.15/$0.15 per M tokens"
                else:
                    display_name = model_name
                    description = "AWS Bedrock"
                    cost = "Pay-per-use"
            else:
                display_name = model_name
                description = provider
                cost = "Unknown"

            models.append({
                "id": model_id,
                "name": display_name,
                "description": description,
                "provider": provider,
                "model": model_name,
                "embedding_model": embedding_model,
                "cost": cost
            })

        return models

    return router
