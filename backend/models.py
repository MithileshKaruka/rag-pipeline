"""
Pydantic models for request and response schemas
"""

from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    """Request model for RAG queries"""
    question: str
    collection_name: str = "knowledge_base"
    n_results: int = 3


class SourceDocument(BaseModel):
    """Model for source document metadata"""
    content: str
    metadata: Optional[dict] = None


class QueryResponse(BaseModel):
    """Response model for RAG queries"""
    answer: str
    sources: List[SourceDocument]
    model_used: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    chromadb_connected: bool
    ollama_connected: bool
    model: str


class IngestTextRequest(BaseModel):
    """Request model for ingesting text directly"""
    text: str
    collection_name: str = "knowledge_base"
    metadata: Optional[dict] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200


class IngestResponse(BaseModel):
    """Response model for document ingestion"""
    message: str
    collection_name: str
    num_chunks: int
    document_ids: List[str]


class CreateCollectionRequest(BaseModel):
    """Request model for creating a new collection"""
    collection_name: str
    metadata: Optional[dict] = None
