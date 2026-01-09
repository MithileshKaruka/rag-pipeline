"""
Enhanced Query API endpoints with streaming and caching
"""

import logging
import httpx
import hashlib
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from cachetools import TTLCache
from typing import AsyncGenerator

from models import QueryRequest, QueryResponse, SourceDocument
from constants import (
    CHROMA_HOST,
    CHROMA_PORT,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    BEDROCK_EMBEDDING_MODEL,
    CACHE_TTL_SECONDS,
    CACHE_MAX_SIZE,
    CHROMA_TIMEOUT_SECONDS,
    OLLAMA_CONNECT_TIMEOUT_SECONDS,
    OLLAMA_READ_TIMEOUT_SECONDS
)
from model_selector import parse_model_selection, is_bedrock_model
from bedrock_config import get_bedrock_client, stream_bedrock_llm, generate_bedrock_embedding

logger = logging.getLogger(__name__)
router = APIRouter()

# Cache for storing query results
query_cache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS)


def create_enhanced_query_router(chroma_client, llm):
    """
    Create enhanced query router with streaming and caching

    Args:
        chroma_client: ChromaDB client instance
        llm: Ollama LLM instance

    Returns:
        APIRouter with enhanced query endpoints
    """

    def get_cache_key(question: str, collection_name: str, model: str = "") -> str:
        """Generate cache key from question, collection, and model."""
        cache_string = f"{collection_name}:{model}:{question.lower().strip()}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    async def get_collection_id(collection_name: str) -> str:
        """Get collection UUID by name via direct API call."""
        try:
            url = f"http://{CHROMA_HOST}:{CHROMA_PORT}/api/v2/tenants/default_tenant/databases/default_database/collections"

            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(url)
                response.raise_for_status()
                collections_data = response.json()

            for col in collections_data:
                if col.get("name") == collection_name:
                    return col.get("id")

            return None
        except Exception as e:
            logger.error(f"Error getting collection ID: {e}")
            return None

    def generate_query_embedding(query_text: str, provider: str = "ollama", embedding_model: str = None) -> list:
        """Generate embedding for query using specified provider and model."""
        import requests

        if provider == "bedrock":
            # Use Bedrock embedding
            bedrock_client = get_bedrock_client()
            if not bedrock_client:
                raise Exception("Failed to initialize Bedrock client for embeddings")

            # Use the embedding model from model selection or default
            return generate_bedrock_embedding(bedrock_client, query_text)

        # Default to Ollama embedding
        embedding_model = embedding_model or OLLAMA_EMBEDDING_MODEL

        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={
                    "model": embedding_model,
                    "prompt": query_text
                },
                timeout=30
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            logger.info(f"Generated query embedding using {OLLAMA_EMBEDDING_MODEL}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

    async def stream_llm_response(prompt: str, provider: str = "ollama", model_id: str = None) -> AsyncGenerator[str, None]:
        """Stream LLM response token by token using specified provider."""
        try:
            if provider == "bedrock":
                # Use Bedrock streaming
                bedrock_client = get_bedrock_client()
                if not bedrock_client:
                    yield "[Error: Failed to initialize Bedrock client]"
                    return

                logger.info(f"Streaming with Bedrock model: {model_id}")
                async for token in stream_bedrock_llm(bedrock_client, prompt):
                    yield token
            else:
                # Use Ollama streaming
                model_id = model_id or OLLAMA_MODEL
                logger.info(f"Streaming with Ollama model: {model_id}")

                # Use httpx for true async streaming
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(OLLAMA_CONNECT_TIMEOUT_SECONDS, read=OLLAMA_READ_TIMEOUT_SECONDS)
                ) as client:
                    async with client.stream(
                        'POST',
                        f"{OLLAMA_HOST}/api/generate",
                        json={
                            "model": model_id,
                            "prompt": prompt,
                            "stream": True
                        }
                    ) as response:
                        response.raise_for_status()

                        # Stream line by line
                        async for line in response.aiter_lines():
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    if 'response' in data and data['response']:
                                        yield data['response']
                                    if data.get('done', False):
                                        break
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to parse JSON: {line}")
                                    continue
        except Exception as e:
            logger.error(f"Error streaming LLM response: {e}")
            yield f"[Error: {str(e)}]"

    @router.post("/query/stream")
    async def query_rag_stream(request: QueryRequest):
        """
        Unified RAG Query Endpoint with Streaming and Caching

        Streams the LLM response token by token for immediate user feedback.
        If the query is cached, indicates cache hit and streams cached response quickly.
        After streaming completes, caches the response for future requests.

        Args:
            request: QueryRequest containing question, collection name, and optional model selection

        Returns:
            StreamingResponse with answer tokens and cache status
        """
        # Parse model selection
        provider, model_id, embedding_model = parse_model_selection(request.model)

        logger.info("=" * 80)
        logger.info("QUERY REQUEST (STREAMING)")
        logger.info("=" * 80)
        logger.info(f"Request payload: {json.dumps(request.model_dump(), indent=2)}")
        logger.info(f"Selected provider: {provider}")
        logger.info(f"Selected model: {model_id}")
        logger.info(f"Embedding model: {embedding_model}")

        cache_key = get_cache_key(request.question, request.collection_name, request.model or "")

        if not chroma_client:
            raise HTTPException(status_code=503, detail="ChromaDB client not available")

        try:
            # Check cache first
            if cache_key in query_cache:
                logger.info(f"Cache hit for streaming query: {request.question[:50]}...")
                cached_response = query_cache[cache_key]

                async def cached_response_generator():
                    """Stream cached response with cache hit indication."""
                    # Indicate cache hit
                    yield f"data: {json.dumps({'type': 'cache_hit', 'data': True})}\n\n"

                    # Send sources
                    sources_data = {
                        "sources": [
                            {
                                "content": source.content,
                                "metadata": source.metadata
                            }
                            for source in cached_response.sources
                        ],
                        "model_used": cached_response.model_used
                    }
                    yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"

                    # Stream cached answer token by token (fast)
                    # Split into words for natural streaming feel
                    words = cached_response.answer.split()
                    for i, word in enumerate(words):
                        token = word + (" " if i < len(words) - 1 else "")
                        yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"
                        # Small delay to make streaming visible but fast
                        import asyncio
                        await asyncio.sleep(0.02)

                    # Signal completion
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"

                return StreamingResponse(
                    cached_response_generator(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    }
                )

            # Cache miss - generate fresh response
            logger.info(f"Cache miss for streaming query: {request.question[:50]}...")

            # Step 1: Get collection ID
            collection_id = await get_collection_id(request.collection_name)
            if not collection_id:
                raise HTTPException(
                    status_code=404,
                    detail=f"Collection '{request.collection_name}' not found"
                )

            # Step 2: Generate embedding for query with EMBEDDING_PROVIDER
            # IMPORTANT: Always use EMBEDDING_PROVIDER for query embeddings to match ingestion embeddings
            # The LLM provider (provider variable) is used for text generation only
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            # Determine embedding provider and model
            embedding_provider = EMBEDDING_PROVIDER.lower()
            query_embedding_model = BEDROCK_EMBEDDING_MODEL if embedding_provider == "bedrock" else OLLAMA_EMBEDDING_MODEL

            logger.info(f"Using embedding provider for query: {embedding_provider} (model: {query_embedding_model})")

            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                query_embedding = await loop.run_in_executor(
                    pool,
                    generate_query_embedding,
                    request.question,
                    embedding_provider,
                    query_embedding_model
                )

            # Step 3: Perform similarity search
            url = f"http://{CHROMA_HOST}:{CHROMA_PORT}/api/v2/tenants/default_tenant/databases/default_database/collections/{collection_id}/query"

            query_body = {
                "query_embeddings": [query_embedding],
                "n_results": request.n_results,
                "include": ["documents", "metadatas"]
            }

            async with httpx.AsyncClient(timeout=120.0) as http_client:
                response = await http_client.post(url, json=query_body)
                response.raise_for_status()
                results = response.json()

            # Step 4: Extract documents
            if not results.get('documents') or not results['documents'][0]:
                raise HTTPException(
                    status_code=404,
                    detail=f"No documents found in collection '{request.collection_name}'"
                )

            documents = results['documents'][0]
            metadatas = results['metadatas'][0] if results.get('metadatas') else [{}] * len(documents)

            # Step 5: Build context and construct prompt
            context = "\n\n".join([
                f"Document {i+1}:\n{doc}"
                for i, doc in enumerate(documents)
            ])

            prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {request.question}

Answer:"""

            # Step 6: Stream response and cache after completion
            logger.info("Generating streaming response with LLM")

            async def response_generator():
                """Generate streaming response, accumulate tokens, and cache result."""
                # Indicate cache miss (fresh generation)
                yield f"data: {json.dumps({'type': 'cache_hit', 'data': False})}\n\n"

                # Send sources as JSON
                sources_data = {
                    "sources": [
                        {
                            "content": doc,
                            "metadata": meta
                        }
                        for doc, meta in zip(documents, metadatas)
                    ],
                    "model_used": model_id
                }
                yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"

                # Stream the answer tokens and accumulate for caching
                full_answer = ""
                async for token in stream_llm_response(prompt, provider, model_id):
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

                # After streaming completes, cache the result
                sources = [
                    SourceDocument(content=doc, metadata=meta)
                    for doc, meta in zip(documents, metadatas)
                ]

                cached_result = QueryResponse(
                    answer=full_answer.strip(),
                    sources=sources,
                    model_used=model_id
                )
                query_cache[cache_key] = cached_result
                logger.info(f"Cached streaming response for: {request.question[:50]}...")

                # Signal completion
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            return StreamingResponse(
                response_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during streaming query: {e}")
            raise HTTPException(status_code=500, detail=f"Streaming query failed: {str(e)}")

    @router.delete("/cache")
    async def clear_cache():
        """
        Clear the query cache

        Returns:
            dict with status message
        """
        cache_size = len(query_cache)
        query_cache.clear()
        logger.info(f"Cache cleared ({cache_size} entries removed)")
        return {
            "status": "success",
            "message": f"Cache cleared ({cache_size} entries removed)"
        }

    @router.get("/cache/stats")
    async def get_cache_stats():
        """
        Get cache statistics

        Returns:
            dict with cache stats
        """
        return {
            "size": len(query_cache),
            "maxsize": query_cache.maxsize,
            "ttl": query_cache.ttl,
            "currsize": query_cache.currsize
        }

    return router
