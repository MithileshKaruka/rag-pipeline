"""
Rate Limit API Router

Provides endpoints for checking Bedrock rate limiter statistics and admin controls.
"""

from fastapi import APIRouter, HTTPException
from bedrock_config import rate_limiter
import logging

logger = logging.getLogger(__name__)


def create_rate_limit_router() -> APIRouter:
    """
    Create and configure the rate limit router.

    Returns:
        Configured APIRouter instance
    """
    router = APIRouter()

    @router.get("/rate-limit/stats")
    async def get_rate_limit_stats():
        """
        Get current rate limiter statistics.

        Returns:
            Rate limiter statistics including daily cost, remaining budget, etc.
        """
        try:
            stats = rate_limiter.get_stats()
            return {
                "status": "success",
                "data": stats
            }
        except Exception as e:
            logger.error(f"Error getting rate limit stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/rate-limit/reset")
    async def reset_rate_limit(admin_key: str = None):
        """
        Reset daily rate limit counters (admin only).

        Args:
            admin_key: Admin authentication key (optional for now)

        Returns:
            Success message
        """
        try:
            # TODO: Add actual admin authentication
            # For now, just log a warning
            logger.warning("Rate limit manually reset - implement admin auth!")

            rate_limiter.reset_daily_limit()

            return {
                "status": "success",
                "message": "Daily rate limit has been reset"
            }
        except Exception as e:
            logger.error(f"Error resetting rate limit: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return router
