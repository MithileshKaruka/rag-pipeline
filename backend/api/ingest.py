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
            # Get or create collection with proper error handling for HTTP client
            try:
                collection = chroma_client.get_collection(name=request.collection_name)
                logger.info(f"Using existing collection: {request.collection_name}")
            except ValueError:
                # Collection doesn't exist, create it
                collection = chroma_client.create_collection(name=request.collection_name)
                logger.info(f"Created new collection: {request.collection_name}")
            except Exception as e:
                # Collection might exist but get failed, try get_or_create
                if "already exists" in str(e).lower():
                    collection = chroma_client.get_collection(name=request.collection_name)
                    logger.info(f"Collection exists, retrieved: {request.collection_name}")
                else:
                    raise

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

            # Add documents to collection
            collection.add(
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

            # Get or create collection with proper error handling for HTTP client
            try:
                collection = chroma_client.get_collection(name=collection_name)
                logger.info(f"Using existing collection: {collection_name}")
            except ValueError:
                # Collection doesn't exist, create it
                collection = chroma_client.create_collection(name=collection_name)
                logger.info(f"Created new collection: {collection_name}")
            except Exception as e:
                # Collection might exist but get failed, try get_or_create
                if "already exists" in str(e).lower():
                    collection = chroma_client.get_collection(name=collection_name)
                    logger.info(f"Collection exists, retrieved: {collection_name}")
                else:
                    raise

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

            # Add documents to collection
            collection.add(
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
