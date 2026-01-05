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
import os
from fastapi import APIRouter, HTTPException, UploadFile, File

from models import IngestTextRequest, IngestResponse
from utils import chunk_text, chunk_markdown_text, generate_document_ids, create_chunk_metadata, validate_file_extension

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

    async def collection_exists(collection_name: str) -> bool:
        """
        Check if a collection exists by calling ChromaDB API directly.
        Bypasses the buggy client deserialization.
        """
        collection_id = await get_collection_id(collection_name)
        return collection_id is not None

    async def create_collection_direct(collection_name: str, metadata: dict = None) -> dict:
        """
        Create a collection via direct API call.
        Bypasses the buggy client deserialization.
        """
        try:
            chroma_host = os.getenv("CHROMA_HOST", "localhost")
            chroma_port = os.getenv("CHROMA_PORT", "8001")
            url = f"http://{chroma_host}:{chroma_port}/api/v2/tenants/default_tenant/databases/default_database/collections"

            payload = {
                "name": collection_name,
                "metadata": metadata or {}
            }

            async with httpx.AsyncClient() as http_client:
                response = await http_client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error creating collection via API: {e}")
            raise

    async def add_documents_with_client(collection_name: str, documents: list, metadatas: list, ids: list):
        """
        Add documents using ChromaDB client's add method.
        The client handles embedding generation automatically.
        We use get_or_create_collection to avoid deserialization of get_collection.
        """
        try:
            # Use get_or_create_collection which has fewer deserialization issues
            collection = chroma_client.get_or_create_collection(name=collection_name)

            # Call add() - the client will auto-generate embeddings
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"Successfully added {len(documents)} documents to '{collection_name}'")
            return {"status": "success", "count": len(documents)}
        except Exception as e:
            logger.error(f"Error adding documents with client: {e}")
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

            # Add documents using client method (handles embedding generation)
            await add_documents_with_client(
                collection_name=request.collection_name,
                documents=chunks,
                metadatas=metadatas,
                ids=document_ids
            )

            logger.info(f"Ingested {len(chunks)} chunks into collection '{request.collection_name}'")

            return IngestResponse(
                message=f"Successfully ingested {len(chunks)} chunks",
                collection_name=request.collection_name,
                num_chunks=len(chunks),
                document_ids=document_ids
            )

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

            # Add documents using client method (handles embedding generation)
            await add_documents_with_client(
                collection_name=collection_name,
                documents=chunks,
                metadatas=metadatas,
                ids=document_ids
            )

            logger.info(f"Ingested file '{file.filename}' ({len(chunks)} chunks) into '{collection_name}'")

            return IngestResponse(
                message=f"Successfully ingested file '{file.filename}' with {len(chunks)} chunks",
                collection_name=collection_name,
                num_chunks=len(chunks),
                document_ids=document_ids
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to ingest file: {e}")
            raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")

    return router
