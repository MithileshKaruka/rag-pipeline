"""
Enhanced Query API endpoints with streaming and caching
"""

import logging
import httpx
import os
import hashlib
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from cachetools import TTLCache
from typing import AsyncGenerator

from models import QueryRequest, QueryResponse, SourceDocument

logger = logging.getLogger(__name__)
router = APIRouter()

# Cache for storing query results (TTL: 1 hour, max 100 entries)
query_cache = TTLCache(maxsize=100, ttl=3600)


def create_enhanced_query_router(chroma_client, llm):
    """
    Create enhanced query router with streaming and caching

    Args:
        chroma_client: ChromaDB client instance
        llm: Ollama LLM instance

    Returns:
        APIRouter with enhanced query endpoints
    """

    def get_cache_key(question: str, collection_name: str) -> str:
        """Generate cache key from question and collection."""
        cache_string = f"{collection_name}:{question.lower().strip()}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    async def get_collection_id(collection_name: str) -> str:
        """Get collection UUID by name via direct API call."""
        try:
            chroma_host = os.getenv("CHROMA_HOST", "localhost")
            chroma_port = os.getenv("CHROMA_PORT", "8001")
            url = f"http://{chroma_host}:{chroma_port}/api/v2/tenants/default_tenant/databases/default_database/collections"

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

    def generate_query_embedding(query_text: str) -> list:
        """Generate embedding for query using Ollama nomic-embed-text."""
        import requests

        try:
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            response = requests.post(
                f"{ollama_host}/api/embeddings",
                json={
                    "model": "nomic-embed-text",
                    "prompt": query_text
                },
                timeout=30
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            logger.info(f"Generated query embedding using nomic-embed-text")
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

    async def stream_llm_response(prompt: str) -> AsyncGenerator[str, None]:
        """Stream LLM response token by token."""
        import requests

        try:
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            ollama_model = os.getenv("OLLAMA_MODEL", "llama2:7b-chat-q4_0")

            response = requests.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": prompt,
                    "stream": True
                },
                stream=True,
                timeout=180
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'response' in data:
                        yield data['response']
                    if data.get('done', False):
                        break
        except Exception as e:
            logger.error(f"Error streaming LLM response: {e}")
            yield f"[Error: {str(e)}]"

    @router.post("/query", response_model=QueryResponse)
    async def query_rag(request: QueryRequest):
        """
        RAG Query Endpoint with Caching (Non-streaming)

        Performs retrieval-augmented generation:
        1. Checks cache for previously answered questions
        2. Retrieves relevant documents from ChromaDB
        3. Constructs prompt with context
        4. Generates response using Ollama LLM
        5. Caches result for future requests

        Args:
            request: QueryRequest containing question and collection name

        Returns:
            QueryResponse with answer and source documents
        """
        # Check cache first
        cache_key = get_cache_key(request.question, request.collection_name)
        if cache_key in query_cache:
            logger.info(f"Cache hit for question: {request.question}")
            return query_cache[cache_key]

        logger.info(f"Cache miss for question: {request.question}")

        # Log incoming request
        logger.info("=" * 80)
        logger.info("QUERY REQUEST")
        logger.info("=" * 80)
        logger.info(f"Request payload: {json.dumps(request.model_dump(), indent=2)}")

        if not chroma_client:
            raise HTTPException(status_code=503, detail="ChromaDB client not available")

        if not llm:
            raise HTTPException(status_code=503, detail="Ollama LLM not available")

        try:
            # Step 1: Get collection ID
            logger.info(f"Querying collection: {request.collection_name}")
            collection_id = await get_collection_id(request.collection_name)
            if not collection_id:
                raise HTTPException(
                    status_code=404,
                    detail=f"Collection '{request.collection_name}' not found"
                )

            # Step 2: Generate embedding for query
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                query_embedding = await loop.run_in_executor(
                    pool,
                    generate_query_embedding,
                    request.question
                )

            # Step 3: Perform similarity search via direct API call
            logger.info(f"Searching for: {request.question}")
            chroma_host = os.getenv("CHROMA_HOST", "localhost")
            chroma_port = os.getenv("CHROMA_PORT", "8001")
            url = f"http://{chroma_host}:{chroma_port}/api/v2/tenants/default_tenant/databases/default_database/collections/{collection_id}/query"

            query_body = {
                "query_embeddings": [query_embedding],
                "n_results": request.n_results,
                "include": ["documents", "metadatas"]
            }

            # Log ChromaDB query request
            log_query_body = {
                "query_embeddings": f"[[{len(query_embedding)} dimensional embedding vector]]",
                "n_results": request.n_results,
                "include": ["documents", "metadatas"]
            }
            logger.info("=" * 80)
            logger.info("CHROMADB QUERY REQUEST")
            logger.info("=" * 80)
            logger.info(f"URL: POST {url}")
            logger.info(f"Request body (abbreviated): {json.dumps(log_query_body, indent=2)}")

            async with httpx.AsyncClient(timeout=120.0) as http_client:
                response = await http_client.post(url, json=query_body)
                response.raise_for_status()
                results = response.json()

            # Log ChromaDB query response
            logger.info("=" * 80)
            logger.info("CHROMADB QUERY RESPONSE")
            logger.info("=" * 80)
            logger.info(f"Status: {response.status_code}")
            logger.info(f"Found {len(results.get('documents', [[]])[0])} matching documents")
            logger.info(f"Response body: {json.dumps(results, indent=2)}")
            logger.info("=" * 80)

            # Step 4: Extract documents and metadata
            if not results.get('documents') or not results['documents'][0]:
                raise HTTPException(
                    status_code=404,
                    detail=f"No documents found in collection '{request.collection_name}'"
                )

            documents = results['documents'][0]
            metadatas = results['metadatas'][0] if results.get('metadatas') else [{}] * len(documents)

            # Step 5: Build context from retrieved documents
            context = "\n\n".join([
                f"Document {i+1}:\n{doc}"
                for i, doc in enumerate(documents)
            ])

            # Step 6: Construct prompt
            prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {request.question}

Answer:"""

            # Step 7: Generate response using LLM (non-streaming for cache)
            logger.info("Generating response with LLM (non-streaming)")
            response_text = llm(prompt)

            # Step 8: Prepare source documents
            sources = [
                SourceDocument(
                    content=doc,
                    metadata=meta
                )
                for doc, meta in zip(documents, metadatas)
            ]

            query_response = QueryResponse(
                answer=response_text.strip(),
                sources=sources,
                model_used=os.getenv("OLLAMA_MODEL", "llama2:7b-chat-q4_0")
            )

            # Cache the response
            query_cache[cache_key] = query_response
            logger.info(f"Cached response for question: {request.question}")

            # Log final response
            logger.info("=" * 80)
            logger.info("QUERY RESPONSE")
            logger.info("=" * 80)
            logger.info(f"Response payload: {json.dumps(query_response.model_dump(), indent=2)}")
            logger.info("=" * 80)

            return query_response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during RAG query: {e}")
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    @router.post("/query/stream")
    async def query_rag_stream(request: QueryRequest):
        """
        RAG Query Endpoint with Streaming Response

        Streams the LLM response token by token for immediate user feedback.
        Note: Streaming responses are not cached.

        Args:
            request: QueryRequest containing question and collection name

        Returns:
            StreamingResponse with answer tokens
        """
        logger.info("=" * 80)
        logger.info("QUERY REQUEST (STREAMING)")
        logger.info("=" * 80)
        logger.info(f"Request payload: {json.dumps(request.model_dump(), indent=2)}")

        if not chroma_client:
            raise HTTPException(status_code=503, detail="ChromaDB client not available")

        if not llm:
            raise HTTPException(status_code=503, detail="Ollama LLM not available")

        try:
            # Step 1: Get collection ID
            collection_id = await get_collection_id(request.collection_name)
            if not collection_id:
                raise HTTPException(
                    status_code=404,
                    detail=f"Collection '{request.collection_name}' not found"
                )

            # Step 2: Generate embedding for query
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                query_embedding = await loop.run_in_executor(
                    pool,
                    generate_query_embedding,
                    request.question
                )

            # Step 3: Perform similarity search
            chroma_host = os.getenv("CHROMA_HOST", "localhost")
            chroma_port = os.getenv("CHROMA_PORT", "8001")
            url = f"http://{chroma_host}:{chroma_port}/api/v2/tenants/default_tenant/databases/default_database/collections/{collection_id}/query"

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

            # Step 6: Stream response
            logger.info("Generating streaming response with LLM")

            async def response_generator():
                """Generate streaming response with metadata."""
                # First, send sources as JSON
                sources_data = {
                    "sources": [
                        {
                            "content": doc,
                            "metadata": meta
                        }
                        for doc, meta in zip(documents, metadatas)
                    ],
                    "model_used": os.getenv("OLLAMA_MODEL", "llama2:7b-chat-q4_0")
                }
                yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"

                # Then stream the answer tokens
                async for token in stream_llm_response(prompt):
                    yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

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
