"""
RAG Backend API
FastAPI application that provides RAG (Retrieval Augmented Generation) functionality
using ChromaDB for vector storage and Ollama for LLM inference.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from constants import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    ENVIRONMENT,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    CHROMA_HOST,
    CHROMA_PORT,
    BACKEND_HOST,
    BACKEND_PORT,
    CACHE_TTL_SECONDS,
    CACHE_MAX_SIZE
)
from config import get_chroma_client, get_ollama_llm, get_allowed_origins
from model_warmup import warmup_ollama_model
from api.health import create_health_router
# from api.query import create_query_router  # Replaced by enhanced version
from api.query_enhanced import create_enhanced_query_router
from api.collections import create_collections_router
from api.ingest import create_ingest_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
chroma_client = get_chroma_client()
llm = get_ollama_llm()

# Create and include routers
health_router = create_health_router(chroma_client, llm)
query_router = create_enhanced_query_router(chroma_client, llm)  # Using enhanced version with caching & streaming
collections_router = create_collections_router(chroma_client)
ingest_router = create_ingest_router(chroma_client)

# Include routers in the app
app.include_router(health_router, tags=["Health"])
app.include_router(query_router, prefix="/api", tags=["Query"])
app.include_router(collections_router, prefix="/api", tags=["Collections"])
app.include_router(ingest_router, prefix="/api", tags=["Ingestion"])


# Startup event to warm up Ollama model
@app.on_event("startup")
async def warmup_ollama():
    """
    Warm up Ollama model on backend startup.
    This loads the model into memory to reduce first query latency.
    See model_warmup.py for implementation details.
    """
    await warmup_ollama_model(llm)


# Root endpoint
@app.get("/", response_model=dict)
async def root():
    """
    Root endpoint - API information
    """
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "query": "/api/query",
            "query_stream": "/api/query/stream",
            "cache_stats": "/api/cache/stats",
            "cache_clear": "/api/cache",
            "collections": "/api/collections",
            "ingest_text": "/api/ingest/text",
            "ingest_file": "/api/ingest/file",
            "create_collection": "/api/collections/create"
        },
        "features": {
            "streaming": "Use /api/query/stream for real-time token-by-token responses",
            "caching": f"Responses cached for {CACHE_TTL_SECONDS // 3600} hour (max {CACHE_MAX_SIZE} entries)",
            "quantized_model": f"Using {OLLAMA_MODEL} for 2-3x faster inference"
        }
    }


# Error Handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return {
        "error": "Not Found",
        "message": str(exc.detail) if hasattr(exc, 'detail') else "Resource not found"
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    return {
        "error": "Internal Server Error",
        "message": str(exc.detail) if hasattr(exc, 'detail') else "An unexpected error occurred"
    }


# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """
    Run on application startup
    """
    logger.info("=" * 50)
    logger.info("RAG API Starting Up")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(f"Ollama Host: {OLLAMA_HOST}")
    logger.info(f"Ollama Model: {OLLAMA_MODEL}")
    logger.info(f"ChromaDB: {CHROMA_HOST}:{CHROMA_PORT}")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Run on application shutdown
    """
    logger.info("RAG API Shutting Down")


# Main entry point for direct execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        reload=True if ENVIRONMENT == "development" else False,
        log_level="info"
    )
