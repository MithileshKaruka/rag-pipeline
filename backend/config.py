"""
Configuration and initialization for ChromaDB and Ollama clients
"""

import logging
from dotenv import load_dotenv
import chromadb
from langchain_community.llms import Ollama

from constants import (
    CHROMA_HOST,
    CHROMA_PORT,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_DEFAULT_KEEP_ALIVE,
    ALLOWED_ORIGINS
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_chroma_client():
    """
    Initialize and return ChromaDB client

    Returns:
        ChromaDB HttpClient or None if initialization fails
    """
    try:
        # ChromaDB HttpClient for version 0.5+
        # We bypass buggy client methods (list_collections, get_collection) with direct API calls
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        logger.info(f"ChromaDB client initialized at {CHROMA_HOST}:{CHROMA_PORT}")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB client: {e}")
        return None


def get_ollama_llm():
    """
    Initialize and return Ollama LLM client

    Returns:
        Ollama LLM instance or None if initialization fails
    """
    try:
        llm = Ollama(
            base_url=OLLAMA_HOST,
            model=OLLAMA_MODEL,
            keep_alive=OLLAMA_DEFAULT_KEEP_ALIVE
        )
        logger.info(f"Ollama LLM initialized with model: {OLLAMA_MODEL}")
        logger.info(f"Keep-alive: {OLLAMA_DEFAULT_KEEP_ALIVE}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Ollama LLM: {e}")
        return None


def get_allowed_origins():
    """
    Get CORS allowed origins from environment

    Returns:
        List of allowed origins
    """
    return ALLOWED_ORIGINS.split(",")
