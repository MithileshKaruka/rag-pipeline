"""
Health check API endpoints
"""

import logging
from fastapi import APIRouter

from models import HealthResponse
from constants import OLLAMA_MODEL, LLM_PROVIDER

logger = logging.getLogger(__name__)
router = APIRouter()


def create_health_router(chroma_client, llm):
    """
    Create health router with dependencies

    Args:
        chroma_client: ChromaDB client instance
        llm: Ollama LLM instance or Bedrock client

    Returns:
        APIRouter with health check endpoint
    """

    @router.get("/health", response_model=HealthResponse)
    async def health_check():
        """
        Health check endpoint
        Returns the status of all connected services
        """
        # Check ChromaDB connection
        chromadb_connected = False
        if chroma_client:
            try:
                chroma_client.heartbeat()
                chromadb_connected = True
            except Exception as e:
                logger.error(f"ChromaDB health check failed: {e}")

        # Check Ollama connection
        ollama_connected = False
        if llm and LLM_PROVIDER.lower() == "ollama":
            try:
                # Simple test to verify Ollama is responding
                test_response = llm("test")
                ollama_connected = True
            except Exception as e:
                logger.error(f"Ollama health check failed: {e}")

        # Check Bedrock connection
        bedrock_connected = False
        try:
            from bedrock_config import get_bedrock_client
            bedrock_client = get_bedrock_client()
            if bedrock_client:
                # Simple test to verify Bedrock is accessible
                # Just checking if we can create the client successfully
                bedrock_connected = True
        except Exception as e:
            logger.error(f"Bedrock health check failed: {e}")

        # Determine overall health status
        if LLM_PROVIDER.lower() == "ollama":
            overall_healthy = chromadb_connected and ollama_connected
        else:  # bedrock
            overall_healthy = chromadb_connected and bedrock_connected

        return HealthResponse(
            status="healthy" if overall_healthy else "degraded",
            chromadb_connected=chromadb_connected,
            ollama_connected=ollama_connected,
            bedrock_connected=bedrock_connected,
            model=OLLAMA_MODEL
        )

    return router
