"""
Configuration and initialization for ChromaDB, Ollama, and AWS Bedrock clients
"""

import logging
from dotenv import load_dotenv
import chromadb
from langchain_community.llms import Ollama

from constants import (
    LLM_PROVIDER,
    CHROMA_HOST,
    CHROMA_PORT,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_DEFAULT_KEEP_ALIVE,
    BEDROCK_REGION,
    BEDROCK_MODEL,
    ALLOWED_ORIGINS
)
from bedrock_config import get_bedrock_client, test_bedrock_connection

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


def get_llm_client():
    """
    Initialize and return LLM client based on LLM_PROVIDER setting.

    Returns Ollama or Bedrock client depending on configuration.

    Returns:
        LLM client instance or None if initialization fails
    """
    if LLM_PROVIDER.lower() == "bedrock":
        logger.info("=" * 80)
        logger.info(f"Initializing AWS Bedrock LLM (Region: {BEDROCK_REGION}, Model: {BEDROCK_MODEL})")
        logger.info("=" * 80)

        client = get_bedrock_client()

        if client:
            # Test connection
            if test_bedrock_connection(client):
                return client
            else:
                logger.error("Bedrock connection test failed")
                return None
        else:
            return None
    else:
        # Default to Ollama
        logger.info("=" * 80)
        logger.info(f"Initializing Ollama LLM (Host: {OLLAMA_HOST}, Model: {OLLAMA_MODEL})")
        logger.info("=" * 80)
        return get_ollama_llm()


def get_allowed_origins():
    """
    Get CORS allowed origins from environment

    Returns:
        List of allowed origins
    """
    return ALLOWED_ORIGINS.split(",")
