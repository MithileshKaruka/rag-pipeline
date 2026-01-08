"""
Query API endpoints for RAG retrieval
"""

import logging
import httpx
from fastapi import APIRouter, HTTPException

from models import QueryRequest, QueryResponse, SourceDocument
from constants import CHROMA_HOST, CHROMA_PORT, OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL

logger = logging.getLogger(__name__)
router = APIRouter()


def create_query_router(chroma_client, llm):
    """
    Create query router with dependencies

    Args:
        chroma_client: ChromaDB client instance
        llm: Ollama LLM instance

    Returns:
        APIRouter with query endpoints
    """

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

    def generate_query_embedding(query_text: str) -> list:
        """Generate embedding for query using Ollama nomic-embed-text."""
        import requests

        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={
                    "model": OLLAMA_EMBEDDING_MODEL,
                    "prompt": query_text
                },
                timeout=30  # Add timeout to prevent hanging
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            logger.info(f"Generated query embedding using nomic-embed-text")
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

    @router.post("/query", response_model=QueryResponse)
    async def query_rag(request: QueryRequest):
        """
        RAG Query Endpoint

        Performs retrieval-augmented generation:
        1. Retrieves relevant documents from ChromaDB
        2. Constructs prompt with context
        3. Generates response using Ollama LLM

        Args:
            request: QueryRequest containing question and collection name

        Returns:
            QueryResponse with answer and source documents
        """
        import json

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
            url = f"http://{CHROMA_HOST}:{CHROMA_PORT}/api/v2/tenants/default_tenant/databases/default_database/collections/{collection_id}/query"

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

            # Step 7: Generate response using LLM (llama2:7b-chat-q4_0 quantized model)
            logger.info("Generating response with LLM")
            response = llm(prompt)

            # Step 8: Prepare source documents
            sources = [
                SourceDocument(
                    content=doc,
                    metadata=meta
                )
                for doc, meta in zip(documents, metadatas)
            ]

            query_response = QueryResponse(
                answer=response.strip(),
                sources=sources,
                model_used=OLLAMA_MODEL
            )

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

    return router
