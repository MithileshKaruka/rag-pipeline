"""
Health check API endpoints
"""

import logging
from fastapi import APIRouter

from models import HealthResponse
from constants import OLLAMA_MODEL

logger = logging.getLogger(__name__)
router = APIRouter()


def create_health_router(chroma_client, llm):
    """
    Create health router with dependencies

    Args:
        chroma_client: ChromaDB client instance
        llm: Ollama LLM instance

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
        if llm:
            try:
                # Simple test to verify Ollama is responding
                test_response = llm("test")
                ollama_connected = True
            except Exception as e:
                logger.error(f"Ollama health check failed: {e}")

        return HealthResponse(
            status="healthy" if (chromadb_connected and ollama_connected) else "degraded",
            chromadb_connected=chromadb_connected,
            ollama_connected=ollama_connected,
            model=OLLAMA_MODEL
        )

    return router
