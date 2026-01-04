"""
Query API endpoints for RAG retrieval
"""

import logging
from fastapi import APIRouter, HTTPException

from models import QueryRequest, QueryResponse, SourceDocument

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
        if not chroma_client:
            raise HTTPException(status_code=503, detail="ChromaDB client not available")

        if not llm:
            raise HTTPException(status_code=503, detail="Ollama LLM not available")

        try:
            # Step 1: Get collection
            logger.info(f"Querying collection: {request.collection_name}")
            collection = chroma_client.get_collection(request.collection_name)

            # Step 2: Perform similarity search
            logger.info(f"Searching for: {request.question}")
            results = collection.query(
                query_texts=[request.question],
                n_results=request.n_results
            )

            # Step 3: Extract documents and metadata
            if not results['documents'] or not results['documents'][0]:
                raise HTTPException(
                    status_code=404,
                    detail=f"No documents found in collection '{request.collection_name}'"
                )

            documents = results['documents'][0]
            metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)

            # Step 4: Build context from retrieved documents
            context = "\n\n".join([
                f"Document {i+1}:\n{doc}"
                for i, doc in enumerate(documents)
            ])

            # Step 5: Construct prompt
            prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {request.question}

Answer:"""

            # Step 6: Generate response using LLM
            logger.info("Generating response with LLM")
            response = llm(prompt)

            # Step 7: Prepare source documents
            sources = [
                SourceDocument(
                    content=doc,
                    metadata=meta
                )
                for doc, meta in zip(documents, metadatas)
            ]

            import os
            return QueryResponse(
                answer=response.strip(),
                sources=sources,
                model_used=os.getenv("OLLAMA_MODEL", "llama2")
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during RAG query: {e}")
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return router
