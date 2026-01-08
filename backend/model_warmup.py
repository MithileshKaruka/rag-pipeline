"""
Model Warmup Module

Provides utilities for warming up the Ollama LLM model on backend startup
to reduce first query latency.
"""

import logging
from typing import Optional

from constants import (
    OLLAMA_MODEL,
    OLLAMA_DEFAULT_KEEP_ALIVE,
    WARMUP_PROMPT,
    COLD_START_TIME,
    WARM_START_TIME,
    FIRST_TOKEN_TIME_COLD,
    FIRST_TOKEN_TIME_WARM,
    SUBSEQUENT_TOKEN_TIME,
    MODEL_MEMORY_USAGE,
    MODEL_DISK_SIZE
)

logger = logging.getLogger(__name__)


async def warmup_ollama_model(llm) -> bool:
    """
    Warm up Ollama model by sending a simple inference request.

    This loads the model into memory, significantly reducing latency
    for the first user query. Without warmup, first query can take
    30-60 seconds for model loading + 10-15 seconds for inference.
    With warmup, first query takes only 5-10 seconds.

    Args:
        llm: Ollama LLM instance from config.get_ollama_llm()

    Returns:
        bool: True if warmup succeeded, False otherwise

    Example:
        >>> from config import get_ollama_llm
        >>> from model_warmup import warmup_ollama_model
        >>> llm = get_ollama_llm()
        >>> success = await warmup_ollama_model(llm)
        >>> if success:
        ...     print("Model ready for fast inference")
    """
    if not llm:
        logger.warning("Ollama LLM not initialized - skipping warmup")
        return False

    try:
        logger.info("=" * 80)
        logger.info("WARMING UP OLLAMA MODEL")
        logger.info("=" * 80)

        logger.info(f"Model: {OLLAMA_MODEL}")
        logger.info("Loading model into memory...")

        # Send a minimal warmup prompt to load the model
        # Using a very short prompt to minimize warmup time
        # Invoke the model (this loads it into memory if not already loaded)
        response = llm.invoke(WARMUP_PROMPT)

        # Log success with truncated response
        response_preview = response[:50] + "..." if len(response) > 50 else response
        logger.info("✓ Ollama model warmed up successfully")
        logger.info(f"Warmup response: {response_preview}")
        logger.info(f"Model loaded and ready for inference")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.warning("=" * 80)
        logger.warning(f"Failed to warm up Ollama model: {e}")
        logger.warning("First query may take 30-60 seconds longer than usual")
        logger.warning("Subsequent queries will be fast once model is loaded")
        logger.warning("=" * 80)
        return False


def configure_model_keep_alive(keep_alive: str = "24h") -> dict:
    """
    Get configuration for Ollama model keep-alive setting.

    Keep-alive determines how long Ollama keeps the model in memory
    after the last inference request. Longer durations reduce cold
    starts but use more memory.

    Args:
        keep_alive: Duration to keep model in memory
                   Options: "5m", "1h", "12h", "24h", "-1" (indefinite)
                   Default: "24h" (recommended for production)

    Returns:
        dict: Configuration parameters for Ollama LLM initialization

    Memory Impact (for llama2:7b-chat-q4_0):
        - Model size: ~3.8GB on disk
        - RAM usage: ~4.5GB when loaded
        - Safe for t3.large (8GB RAM)

    Keep-Alive Recommendations:
        - Development: "5m" (default)
        - Low traffic: "1h"
        - Medium traffic: "12h"
        - High traffic: "24h" (recommended)
        - Always-on: "-1" (only if 16GB+ RAM)

    Example:
        >>> from langchain_community.llms import Ollama
        >>> config = configure_model_keep_alive("24h")
        >>> llm = Ollama(
        ...     base_url="http://localhost:11434",
        ...     model="llama2:7b-chat-q4_0",
        ...     **config
        ... )
    """
    logger.info(f"Configuring Ollama with keep_alive={keep_alive}")

    return {
        "keep_alive": keep_alive
    }


def get_warmup_stats() -> dict:
    """
    Get statistics about model warmup performance.

    Returns:
        dict: Statistics including model name, expected times, memory usage

    Example:
        >>> stats = get_warmup_stats()
        >>> print(f"Model: {stats['model']}")
        >>> print(f"Cold start time: {stats['cold_start_time']}")
    """
    return {
        "model": OLLAMA_MODEL,
        "cold_start_time": COLD_START_TIME,
        "warm_start_time": WARM_START_TIME,
        "first_token_time_cold": FIRST_TOKEN_TIME_COLD,
        "first_token_time_warm": FIRST_TOKEN_TIME_WARM,
        "subsequent_token_time": SUBSEQUENT_TOKEN_TIME,
        "memory_usage": MODEL_MEMORY_USAGE,
        "disk_size": MODEL_DISK_SIZE,
        "recommended_keep_alive": OLLAMA_DEFAULT_KEEP_ALIVE,
        "warmup_prompt": WARMUP_PROMPT,
        "optimization_status": "enabled"
    }
