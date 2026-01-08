"""
Collection management API endpoints
"""

import logging
import httpx
from typing import List
from fastapi import APIRouter, HTTPException

from models import CreateCollectionRequest
from constants import CHROMA_HOST, CHROMA_PORT

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

        Note: We bypass the ChromaDB client's list_collections() method because it has
        a deserialization bug with the HTTP client. Instead, we call the API directly.
        """
        if not chroma_client:
            raise HTTPException(status_code=503, detail="ChromaDB client not available")

        try:
            # Get ChromaDB connection details from client

            # Call ChromaDB v2 API directly (v1 returns 410 Gone)
            url = f"http://{CHROMA_HOST}:{CHROMA_PORT}/api/v2/tenants/default_tenant/databases/default_database/collections"

            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                collections_data = response.json()

            # Extract collection names from the response
            collection_names = [col.get("name") for col in collections_data if col.get("name")]

            logger.info(f"Found {len(collection_names)} collections")
            return collection_names

        except httpx.HTTPError as e:
            logger.error(f"HTTP error listing collections: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")
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
            # Get ChromaDB connection details

            # Check if collection already exists by calling API directly
            # ChromaDB 0.5.23 uses v2 API
            list_url = f"http://{chroma_host}:{chroma_port}/api/v2/tenants/default_tenant/databases/default_database/collections"

            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(list_url)
                response.raise_for_status()
                collections_data = response.json()

            existing_collections = [col.get("name") for col in collections_data if col.get("name")]

            if request.collection_name in existing_collections:
                raise HTTPException(
                    status_code=400,
                    detail=f"Collection '{request.collection_name}' already exists"
                )

            # Create new collection via direct API call to avoid deserialization bug
            create_url = f"http://{chroma_host}:{chroma_port}/api/v2/tenants/default_tenant/databases/default_database/collections"

            payload = {
                "name": request.collection_name,
                "metadata": request.metadata or {}
            }

            async with httpx.AsyncClient() as http_client:
                response = await http_client.post(create_url, json=payload)
                response.raise_for_status()

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
