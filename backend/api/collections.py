"""
Collection management API endpoints
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException

from models import CreateCollectionRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def create_collections_router(chroma_client):
    """
    Create collections router with dependencies

    Args:
        chroma_client: ChromaDB client instance

    Returns:
        APIRouter with collection management endpoints
    """

    @router.get("/collections", response_model=List[str])
    async def list_collections():
        """
        List all available ChromaDB collections
        """
        if not chroma_client:
            raise HTTPException(status_code=503, detail="ChromaDB client not available")

        try:
            collections = chroma_client.list_collections()
            # Handle both dict and object responses from ChromaDB client
            collection_names = []
            for col in collections:
                if isinstance(col, dict):
                    collection_names.append(col.get('name', col.get('id', str(col))))
                else:
                    collection_names.append(col.name if hasattr(col, 'name') else str(col))
            return collection_names
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")

    @router.post("/collections/create")
    async def create_collection(request: CreateCollectionRequest):
        """
        Create a new ChromaDB collection

        Args:
            request: CreateCollectionRequest with collection name and metadata

        Returns:
            Success message and collection details
        """
        if not chroma_client:
            raise HTTPException(status_code=503, detail="ChromaDB client not available")

        try:
            # Check if collection already exists
            collections = chroma_client.list_collections()
            existing_collections = []
            for col in collections:
                if isinstance(col, dict):
                    existing_collections.append(col.get('name', col.get('id', str(col))))
                else:
                    existing_collections.append(col.name if hasattr(col, 'name') else str(col))

            if request.collection_name in existing_collections:
                raise HTTPException(
                    status_code=400,
                    detail=f"Collection '{request.collection_name}' already exists"
                )

            # Create new collection
            collection = chroma_client.create_collection(
                name=request.collection_name,
                metadata=request.metadata or {}
            )

            logger.info(f"Created collection: {request.collection_name}")

            return {
                "message": f"Collection '{request.collection_name}' created successfully",
                "collection_name": request.collection_name,
                "metadata": request.metadata
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create collection: {str(e)}")

    return router
