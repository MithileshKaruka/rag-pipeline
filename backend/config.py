"""
Configuration and initialization for ChromaDB and Ollama clients
"""

import os
import logging
from dotenv import load_dotenv
import chromadb
from langchain_community.llms import Ollama

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
        # Simple HttpClient initialization for ChromaDB 0.4.24
        # Test connection with a simple heartbeat call
        import httpx
        host = os.getenv("CHROMA_HOST", "localhost")
        port = int(os.getenv("CHROMA_PORT", "8001"))

        # Test if ChromaDB server is reachable
        test_url = f"http://{host}:{port}/api/v1/heartbeat"
        response = httpx.get(test_url, timeout=5.0)
        response.raise_for_status()

        # If reachable, create client
        client = chromadb.HttpClient(host=host, port=port)
        logger.info(f"ChromaDB client initialized successfully at {host}:{port}")
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
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama2")
        )
        logger.info(f"Ollama LLM initialized with model: {os.getenv('OLLAMA_MODEL', 'llama2')}")
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
    return os.getenv("ALLOWED_ORIGINS", "*").split(",")
