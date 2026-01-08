"""
Constants and Configuration Defaults

Central location for all default values, model configurations, and system constants.
All modules should import from this file rather than hardcoding values.
"""

import os

# ============================================================================
# OLLAMA CONFIGURATION
# ============================================================================

OLLAMA_DEFAULT_HOST = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llama2:7b-chat-q4_0"
OLLAMA_DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_DEFAULT_KEEP_ALIVE = "24h"

# Get from environment or use defaults
OLLAMA_HOST = os.getenv("OLLAMA_HOST", OLLAMA_DEFAULT_HOST)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", OLLAMA_DEFAULT_MODEL)
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", OLLAMA_DEFAULT_EMBEDDING_MODEL)

# ============================================================================
# CHROMADB CONFIGURATION
# ============================================================================

CHROMA_DEFAULT_HOST = "localhost"
CHROMA_DEFAULT_PORT = 8001

# Get from environment or use defaults
CHROMA_HOST = os.getenv("CHROMA_HOST", CHROMA_DEFAULT_HOST)
CHROMA_PORT = int(os.getenv("CHROMA_PORT", str(CHROMA_DEFAULT_PORT)))

# ============================================================================
# BACKEND CONFIGURATION
# ============================================================================

BACKEND_DEFAULT_HOST = "localhost"
BACKEND_DEFAULT_PORT = 8000

# Get from environment or use defaults
BACKEND_HOST = os.getenv("BACKEND_HOST", BACKEND_DEFAULT_HOST)
BACKEND_PORT = int(os.getenv("BACKEND_PORT", str(BACKEND_DEFAULT_PORT)))

# ============================================================================
# CORS CONFIGURATION
# ============================================================================

ALLOWED_ORIGINS_DEFAULT = "*"

# Get from environment or use defaults
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", ALLOWED_ORIGINS_DEFAULT)

# ============================================================================
# ENVIRONMENT
# ============================================================================

ENVIRONMENT_DEFAULT = "development"

# Get from environment or use defaults
ENVIRONMENT = os.getenv("ENVIRONMENT", ENVIRONMENT_DEFAULT)

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

# Cache settings for query responses
CACHE_TTL_SECONDS = 3600  # 1 hour
CACHE_MAX_SIZE = 100  # Maximum number of cached queries

# ============================================================================
# QUERY CONFIGURATION
# ============================================================================

# Default number of documents to retrieve for RAG
DEFAULT_N_RESULTS = 3

# Timeout settings for external services
CHROMA_TIMEOUT_SECONDS = 120.0
OLLAMA_CONNECT_TIMEOUT_SECONDS = 10.0
OLLAMA_READ_TIMEOUT_SECONDS = 600.0  # 10 minutes for long responses

# ============================================================================
# INGESTION CONFIGURATION
# ============================================================================

# Default chunking parameters for document ingestion
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# ============================================================================
# MODEL WARMUP CONFIGURATION
# ============================================================================

# Warmup prompt to load model into memory on startup
WARMUP_PROMPT = "Hi"

# Expected performance metrics (for monitoring/documentation)
COLD_START_TIME = "30-60 seconds"
WARM_START_TIME = "5-10 seconds"
FIRST_TOKEN_TIME_COLD = "10-15 seconds"
FIRST_TOKEN_TIME_WARM = "5-10 seconds"
SUBSEQUENT_TOKEN_TIME = "50-200ms"

# Model memory usage (for llama2:7b-chat-q4_0)
MODEL_MEMORY_USAGE = "~4.5GB RAM"
MODEL_DISK_SIZE = "~3.8GB"

# ============================================================================
# API INFORMATION
# ============================================================================

API_TITLE = "RAG API"
API_DESCRIPTION = "Retrieval Augmented Generation API using Ollama and ChromaDB"
API_VERSION = "1.0.0"

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL_DEFAULT = "INFO"
LOG_LEVEL = os.getenv("LOG_LEVEL", LOG_LEVEL_DEFAULT)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_ollama_config() -> dict:
    """
    Get Ollama configuration as a dictionary.

    Returns:
        dict: Ollama configuration parameters
    """
    return {
        "host": OLLAMA_HOST,
        "model": OLLAMA_MODEL,
        "embedding_model": OLLAMA_EMBEDDING_MODEL,
        "keep_alive": OLLAMA_DEFAULT_KEEP_ALIVE
    }


def get_chroma_config() -> dict:
    """
    Get ChromaDB configuration as a dictionary.

    Returns:
        dict: ChromaDB configuration parameters
    """
    return {
        "host": CHROMA_HOST,
        "port": CHROMA_PORT,
        "timeout": CHROMA_TIMEOUT_SECONDS
    }


def get_backend_config() -> dict:
    """
    Get backend configuration as a dictionary.

    Returns:
        dict: Backend configuration parameters
    """
    return {
        "host": BACKEND_HOST,
        "port": BACKEND_PORT,
        "environment": ENVIRONMENT,
        "allowed_origins": ALLOWED_ORIGINS
    }


def get_cache_config() -> dict:
    """
    Get cache configuration as a dictionary.

    Returns:
        dict: Cache configuration parameters
    """
    return {
        "ttl_seconds": CACHE_TTL_SECONDS,
        "max_size": CACHE_MAX_SIZE
    }


def print_configuration():
    """Print current configuration (useful for debugging/startup logs)."""
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Environment: {ENVIRONMENT}")
    print(f"\nOllama:")
    print(f"  Host: {OLLAMA_HOST}")
    print(f"  Model: {OLLAMA_MODEL}")
    print(f"  Embedding Model: {OLLAMA_EMBEDDING_MODEL}")
    print(f"\nChromaDB:")
    print(f"  Host: {CHROMA_HOST}")
    print(f"  Port: {CHROMA_PORT}")
    print(f"\nBackend:")
    print(f"  Host: {BACKEND_HOST}")
    print(f"  Port: {BACKEND_PORT}")
    print(f"  CORS Origins: {ALLOWED_ORIGINS}")
    print(f"\nCache:")
    print(f"  TTL: {CACHE_TTL_SECONDS}s")
    print(f"  Max Size: {CACHE_MAX_SIZE}")
    print("=" * 80)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_configuration():
    """
    Validate configuration values.

    Raises:
        ValueError: If any configuration value is invalid
    """
    if CHROMA_PORT < 1 or CHROMA_PORT > 65535:
        raise ValueError(f"Invalid CHROMA_PORT: {CHROMA_PORT}. Must be between 1 and 65535.")

    if BACKEND_PORT < 1 or BACKEND_PORT > 65535:
        raise ValueError(f"Invalid BACKEND_PORT: {BACKEND_PORT}. Must be between 1 and 65535.")

    if CACHE_TTL_SECONDS < 0:
        raise ValueError(f"Invalid CACHE_TTL_SECONDS: {CACHE_TTL_SECONDS}. Must be >= 0.")

    if CACHE_MAX_SIZE < 1:
        raise ValueError(f"Invalid CACHE_MAX_SIZE: {CACHE_MAX_SIZE}. Must be >= 1.")

    if ENVIRONMENT not in ["development", "production", "test"]:
        raise ValueError(f"Invalid ENVIRONMENT: {ENVIRONMENT}. Must be 'development', 'production', or 'test'.")
