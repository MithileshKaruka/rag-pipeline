"""
Ingestion API endpoints for adding documents to ChromaDB

Chunking Strategy:
=================
This module uses RecursiveCharacterTextSplitter for intelligent text chunking.

How Recursive Splitting Works:
1. Attempts to split on the first separator in the list
2. If resulting chunks are still too large, moves to next separator
3. Continues recursively until chunks meet size requirements
4. Preserves semantic meaning by splitting on natural boundaries

Text Files (.txt):
  Separators: paragraphs (\\n\\n) → sentences (. ! ?) → words ( ) → characters

Markdown Files (.md):
  Separators: headers (## ###) → paragraphs → code blocks → sentences → words

Benefits:
- Preserves document structure and hierarchy
- Keeps related content together
- Maintains readability and context
- Better retrieval quality in RAG applications
"""

import logging
import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File

from models import IngestTextRequest, IngestResponse
from utils import chunk_text, chunk_markdown_text, generate_document_ids, create_chunk_metadata, validate_file_extension
from constants import CHROMA_HOST, CHROMA_PORT, OLLAMA_HOST, OLLAMA_EMBEDDING_MODEL

logger = logging.getLogger(__name__)
router = APIRouter()


def create_ingest_router(chroma_client):
    """
    Create ingestion router with dependencies

    Args:
        chroma_client: ChromaDB client instance

    Returns:
        APIRouter with ingestion endpoints
    """

    async def get_collection_id(collection_name: str) -> str:
        """
        Get collection UUID by name via direct API call.
        Returns collection ID if exists, None otherwise.
        """
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

    async def collection_exists(collection_name: str) -> bool:
        """
        Check if a collection exists by calling ChromaDB API directly.
        Bypasses the buggy client deserialization.
        """
        collection_id = await get_collection_id(collection_name)
        return collection_id is not None

    async def create_collection_direct(collection_name: str, metadata: dict = None) -> dict:
        """
        Create a collection via direct API call with embedding function configured.
        The server will use this embedding function to auto-generate embeddings.
        """
        try:
            url = f"http://{CHROMA_HOST}:{CHROMA_PORT}/api/v2/tenants/default_tenant/databases/default_database/collections"

            # Configure collection to use default embedding function
            # ChromaDB will then auto-generate embeddings server-side
            payload = {
                "name": collection_name,
                "metadata": metadata or {},
                "configuration": {
                    "hnsw_configuration": {
                        "space": "cosine"
                    }
                }
            }

            async with httpx.AsyncClient() as http_client:
                response = await http_client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error creating collection via API: {e}")
            raise

    def generate_embeddings_with_ollama(documents: list) -> list:
        """
        Generate embeddings using Ollama with nomic-embed-text model.
        This model is optimized for embeddings (faster and better quality than llama2).
        Uses concurrent requests to speed up batch processing.
        """
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def generate_single_embedding(doc: str) -> list:
            """Generate embedding for a single document."""
            response = requests.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={
                    "model": OLLAMA_EMBEDDING_MODEL,
                    "prompt": doc
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]

        try:
            embeddings = []

            # Use ThreadPoolExecutor to parallelize embedding generation
            # Limit to 4 concurrent requests to avoid overwhelming Ollama
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all documents for embedding generation
                future_to_index = {
                    executor.submit(generate_single_embedding, doc): i
                    for i, doc in enumerate(documents)
                }

                # Create a list to store embeddings in order
                embeddings = [None] * len(documents)

                # Collect results as they complete
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        embedding = future.result()
                        embeddings[index] = embedding
                    except Exception as e:
                        logger.error(f"Error generating embedding for document {index}: {e}")
                        raise

            logger.info(f"Generated embeddings for {len(documents)} documents using Ollama (nomic-embed-text) with parallel processing")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings with Ollama: {e}")
            raise

    async def add_documents_with_client_async(collection_name: str, documents: list, metadatas: list, ids: list):
        """
        Add documents via direct API call with Ollama-generated embeddings.
        Uses Ollama for embedding generation, then calls ChromaDB API directly.
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        try:
            # Get collection UUID
            collection_id = await get_collection_id(collection_name)
            if not collection_id:
                raise ValueError(f"Collection '{collection_name}' not found")

            # Generate embeddings using Ollama in thread pool (blocking operation)
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                embeddings = await loop.run_in_executor(
                    pool,
                    generate_embeddings_with_ollama,
                    documents
                )

            # Direct API call to add documents
            chroma_host = os.getenv("CHROMA_HOST", "localhost")
            chroma_port = os.getenv("CHROMA_PORT", "8001")
            url = f"http://{chroma_host}:{chroma_port}/api/v2/tenants/default_tenant/databases/default_database/collections/{collection_id}/add"

            # Prepare add request with embeddings
            add_body = {
                "ids": ids,
                "embeddings": embeddings,
                "documents": documents,
                "metadatas": metadatas
            }

            # Log ChromaDB add request (without full embeddings for brevity)
            import json
            log_body = {
                "ids": ids,
                "embeddings": f"[{len(embeddings)} embeddings of dimension {len(embeddings[0]) if embeddings else 0}]",
                "documents": documents[:2] + ["..."] if len(documents) > 2 else documents,  # Show first 2 docs
                "metadatas": metadatas[:2] + ["..."] if len(metadatas) > 2 else metadatas
            }
            logger.info("=" * 80)
            logger.info("CHROMADB ADD REQUEST")
            logger.info("=" * 80)
            logger.info(f"URL: POST {url}")
            logger.info(f"Request body (abbreviated): {json.dumps(log_body, indent=2)}")

            async with httpx.AsyncClient(timeout=120.0) as http_client:
                response = await http_client.post(url, json=add_body)
                response.raise_for_status()
                response_data = response.json()

            # Log ChromaDB add response
            logger.info("=" * 80)
            logger.info("CHROMADB ADD RESPONSE")
            logger.info("=" * 80)
            logger.info(f"Status: {response.status_code}")
            logger.info(f"Response body: {json.dumps(response_data, indent=2)}")
            logger.info("=" * 80)

            logger.info(f"Successfully added {len(documents)} documents to '{collection_name}'")
            return {"status": "success", "count": len(documents)}
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error adding documents: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    @router.post("/ingest/text", response_model=IngestResponse)
    async def ingest_text(request: IngestTextRequest):
        """
        Ingest text directly into ChromaDB using recursive chunking strategy

        Uses RecursiveCharacterTextSplitter which intelligently splits text by:
        1. Paragraphs (double newlines) - preserves document structure
        2. Sentences (periods, exclamation, question marks) - keeps ideas together
        3. Words (spaces) - maintains readability
        4. Characters - only as last resort

        This recursive approach ensures chunks are semantically meaningful
        and preserves context better than simple character splitting.

        Args:
            request: IngestTextRequest with text content and parameters
                - text: Content to ingest
                - chunk_size: Max characters per chunk (default: 1000)
                - chunk_overlap: Characters overlap between chunks (default: 200)
                - collection_name: Target ChromaDB collection
                - metadata: Optional custom metadata

        Returns:
            IngestResponse with ingestion details including chunk count and IDs
        """
        import json

        # Log incoming request
        logger.info("=" * 80)
        logger.info("INGEST TEXT REQUEST")
        logger.info("=" * 80)
        logger.info(f"Request payload: {json.dumps(request.model_dump(), indent=2)}")

        if not chroma_client:
            raise HTTPException(status_code=503, detail="ChromaDB client not available")

        try:
            # Check if collection exists, create if needed
            # We bypass client methods completely due to deserialization bugs
            if not await collection_exists(request.collection_name):
                await create_collection_direct(request.collection_name)
                logger.info(f"Created new collection: {request.collection_name}")
            else:
                logger.info(f"Using existing collection: {request.collection_name}")

            # Split text into chunks using recursive strategy
            # This intelligently splits on paragraphs, then sentences, then words
            chunks = chunk_text(request.text, request.chunk_size, request.chunk_overlap)

            logger.info(f"Split text into {len(chunks)} chunks using recursive strategy")

            # Generate unique IDs for each chunk
            document_ids = generate_document_ids(len(chunks))

            # Prepare metadata for each chunk
            metadatas = create_chunk_metadata(
                num_chunks=len(chunks),
                source="text_input",
                additional_metadata=request.metadata
            )

            # Add documents using client's internal HTTP method (handles embedding generation)
            await add_documents_with_client_async(
                collection_name=request.collection_name,
                documents=chunks,
                metadatas=metadatas,
                ids=document_ids
            )

            logger.info(f"Ingested {len(chunks)} chunks into collection '{request.collection_name}'")

            response = IngestResponse(
                message=f"Successfully ingested {len(chunks)} chunks",
                collection_name=request.collection_name,
                num_chunks=len(chunks),
                document_ids=document_ids
            )

            # Log response
            logger.info("=" * 80)
            logger.info("INGEST TEXT RESPONSE")
            logger.info("=" * 80)
            logger.info(f"Response payload: {json.dumps(response.model_dump(), indent=2)}")
            logger.info("=" * 80)

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to ingest text: {e}")
            raise HTTPException(status_code=500, detail=f"Text ingestion failed: {str(e)}")

    @router.post("/ingest/file", response_model=IngestResponse)
    async def ingest_file(
        file: UploadFile = File(...),
        collection_name: str = "knowledge_base",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Ingest a file into ChromaDB using recursive chunking strategy

        Supports: .txt, .md files with intelligent splitting

        Chunking Strategy:
        - .txt files: Splits by paragraphs → sentences → words → characters
        - .md files: Splits by markdown headers (##, ###) → paragraphs → sentences → words

        This recursive approach preserves semantic structure:
        - Keeps related paragraphs together
        - Maintains sentence boundaries when possible
        - Preserves markdown hierarchy for .md files
        - Uses chunk_overlap to maintain context across boundaries

        Args:
            file: Uploaded file (.txt or .md)
            collection_name: Target ChromaDB collection (default: "knowledge_base")
            chunk_size: Max characters per chunk (default: 1000)
            chunk_overlap: Characters overlap between chunks (default: 200)
                         This helps maintain context across chunk boundaries

        Returns:
            IngestResponse with ingestion details including chunk count and IDs
        """
        # Log incoming request
        logger.info("=" * 80)
        logger.info("INGEST FILE REQUEST")
        logger.info("=" * 80)
        logger.info(f"File: {file.filename}")
        logger.info(f"Collection: {collection_name}")
        logger.info(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")

        if not chroma_client:
            raise HTTPException(status_code=503, detail="ChromaDB client not available")

        try:
            # Validate file type
            allowed_extensions = [".txt", ".md"]
            is_valid, file_extension = validate_file_extension(file.filename, allowed_extensions)

            if not is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_extension}. Supported: {', '.join(allowed_extensions)}"
                )

            # Read file content
            content = await file.read()
            text = content.decode("utf-8")

            # Check if collection exists, create if needed
            # We bypass client methods completely due to deserialization bugs
            if not await collection_exists(collection_name):
                await create_collection_direct(collection_name)
                logger.info(f"Created new collection: {collection_name}")
            else:
                logger.info(f"Using existing collection: {collection_name}")

            # Split text into chunks using appropriate strategy based on file type
            if file_extension == ".md":
                # Use markdown-optimized chunking for .md files
                # Splits by headers (##, ###) first to preserve document structure
                chunks = chunk_markdown_text(text, chunk_size, chunk_overlap)
                logger.info(f"Using markdown-optimized recursive chunking for {file.filename}")
            else:
                # Use general text chunking for .txt files
                # Splits by paragraphs first, then sentences, then words
                chunks = chunk_text(text, chunk_size, chunk_overlap)
                logger.info(f"Using standard recursive chunking for {file.filename}")

            logger.info(f"Split file into {len(chunks)} chunks")

            # Generate unique IDs for each chunk
            document_ids = generate_document_ids(len(chunks))

            # Prepare metadata for each chunk
            metadatas = create_chunk_metadata(
                num_chunks=len(chunks),
                source="file_upload",
                filename=file.filename,
                file_type=file_extension
            )

            # Add documents using client's internal HTTP method (handles embedding generation)
            await add_documents_with_client_async(
                collection_name=collection_name,
                documents=chunks,
                metadatas=metadatas,
                ids=document_ids
            )

            logger.info(f"Ingested file '{file.filename}' ({len(chunks)} chunks) into '{collection_name}'")

            response = IngestResponse(
                message=f"Successfully ingested file '{file.filename}' with {len(chunks)} chunks",
                collection_name=collection_name,
                num_chunks=len(chunks),
                document_ids=document_ids
            )

            # Log response
            logger.info("=" * 80)
            logger.info("INGEST FILE RESPONSE")
            logger.info("=" * 80)
            logger.info(f"Response payload: {json.dumps(response.model_dump(), indent=2)}")
            logger.info("=" * 80)

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to ingest file: {e}")
            raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")

    return router
