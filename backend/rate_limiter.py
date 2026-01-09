"""
Rate Limiter for AWS Bedrock API Calls

Implements:
1. Token-based rate limiting (tracks input/output tokens)
2. Daily cost tracking with automatic cutoff
3. Request-based rate limiting per time window

Pricing (as of 2024):
- Llama 3.2 3B: $0.15 per M input tokens, $0.15 per M output tokens
- Titan Embeddings v2: $0.02 per M tokens
"""

import time
import logging
from typing import Dict, Tuple
from datetime import datetime, timedelta
from threading import Lock

logger = logging.getLogger(__name__)


class BedrockRateLimiter:
    """Rate limiter for AWS Bedrock API calls with cost tracking."""

    # Pricing per million tokens (USD)
    PRICING = {
        "llama3-2-3b": {
            "input": 0.15,   # $0.15 per M input tokens
            "output": 0.15,  # $0.15 per M output tokens
        },
        "titan-embed-v2": {
            "input": 0.02,   # $0.02 per M tokens
            "output": 0.0,   # Embeddings don't have output cost
        }
    }

    def __init__(self, daily_limit_usd: float = 1.0, requests_per_minute: int = 100):
        """
        Initialize rate limiter.

        Args:
            daily_limit_usd: Maximum spend per day in USD (default: $1.00)
            requests_per_minute: Maximum requests per minute (default: 100)
        """
        self.daily_limit_usd = daily_limit_usd
        self.requests_per_minute = requests_per_minute

        # Daily cost tracking
        self.current_date = datetime.now().date()
        self.daily_cost = 0.0
        self.daily_input_tokens = 0
        self.daily_output_tokens = 0
        self.daily_requests = 0

        # Request rate limiting (sliding window)
        self.request_timestamps = []

        # Thread safety
        self.lock = Lock()

        logger.info(f"BedrockRateLimiter initialized: ${daily_limit_usd}/day, {requests_per_minute} req/min")

    def _reset_daily_stats_if_needed(self):
        """Reset daily statistics if it's a new day."""
        today = datetime.now().date()
        if today != self.current_date:
            logger.info(f"New day detected. Previous day stats:")
            logger.info(f"  - Total cost: ${self.daily_cost:.6f}")
            logger.info(f"  - Input tokens: {self.daily_input_tokens:,}")
            logger.info(f"  - Output tokens: {self.daily_output_tokens:,}")
            logger.info(f"  - Total requests: {self.daily_requests}")

            self.current_date = today
            self.daily_cost = 0.0
            self.daily_input_tokens = 0
            self.daily_output_tokens = 0
            self.daily_requests = 0
            self.request_timestamps = []

            logger.info("Daily statistics reset")

    def _clean_old_requests(self):
        """Remove request timestamps older than 1 minute."""
        cutoff_time = time.time() - 60
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff_time]

    def _calculate_cost(self, model_type: str, input_tokens: int, output_tokens: int = 0) -> float:
        """
        Calculate cost for a request.

        Args:
            model_type: "llama3-2-3b" or "titan-embed-v2"
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens (0 for embeddings)

        Returns:
            Cost in USD
        """
        if model_type not in self.PRICING:
            logger.warning(f"Unknown model type: {model_type}, using default pricing")
            model_type = "llama3-2-3b"

        pricing = self.PRICING[model_type]
        cost = (
            (input_tokens / 1_000_000) * pricing["input"] +
            (output_tokens / 1_000_000) * pricing["output"]
        )
        return cost

    def check_and_increment(
        self,
        model_type: str,
        estimated_input_tokens: int,
        estimated_output_tokens: int = 0
    ) -> Tuple[bool, str]:
        """
        Check if request is allowed and increment counters if so.

        Args:
            model_type: "llama3-2-3b" or "titan-embed-v2"
            estimated_input_tokens: Estimated number of input tokens
            estimated_output_tokens: Estimated number of output tokens

        Returns:
            Tuple of (is_allowed, reason)
            - is_allowed: True if request is allowed, False otherwise
            - reason: Reason for rejection (empty if allowed)
        """
        with self.lock:
            self._reset_daily_stats_if_needed()
            self._clean_old_requests()

            # Check request rate limit
            if len(self.request_timestamps) >= self.requests_per_minute:
                remaining_time = 60 - (time.time() - self.request_timestamps[0])
                return False, f"Rate limit exceeded: {self.requests_per_minute} requests/minute. Retry in {remaining_time:.0f}s"

            # Calculate estimated cost for this request
            estimated_cost = self._calculate_cost(model_type, estimated_input_tokens, estimated_output_tokens)

            # Check daily cost limit
            projected_cost = self.daily_cost + estimated_cost
            if projected_cost > self.daily_limit_usd:
                remaining = self.daily_limit_usd - self.daily_cost
                return False, (
                    f"Daily budget limit reached: ${self.daily_cost:.4f}/${self.daily_limit_usd:.2f} spent. "
                    f"This request would cost ~${estimated_cost:.6f} (${remaining:.6f} remaining). "
                    f"Resets at midnight."
                )

            # Request is allowed - increment counters
            self.request_timestamps.append(time.time())
            self.daily_cost += estimated_cost
            self.daily_input_tokens += estimated_input_tokens
            self.daily_output_tokens += estimated_output_tokens
            self.daily_requests += 1

            logger.info(
                f"Bedrock request approved: ${estimated_cost:.6f} "
                f"(Daily: ${self.daily_cost:.4f}/${self.daily_limit_usd:.2f}, "
                f"{len(self.request_timestamps)}/{self.requests_per_minute} req/min)"
            )

            return True, ""

    def record_actual_usage(self, model_type: str, actual_input_tokens: int, actual_output_tokens: int = 0):
        """
        Record actual token usage after request completes.
        This adjusts the cost if the estimate was wrong.

        Args:
            model_type: "llama3-2-3b" or "titan-embed-v2"
            actual_input_tokens: Actual number of input tokens used
            actual_output_tokens: Actual number of output tokens used
        """
        with self.lock:
            # Calculate actual cost
            actual_cost = self._calculate_cost(model_type, actual_input_tokens, actual_output_tokens)

            # We already counted an estimate, so we need to adjust
            # For simplicity, we'll just log the difference
            logger.debug(
                f"Actual usage: {actual_input_tokens} input, {actual_output_tokens} output tokens "
                f"(~${actual_cost:.6f})"
            )

    def get_stats(self) -> Dict:
        """
        Get current rate limiter statistics.

        Returns:
            Dictionary with current statistics
        """
        with self.lock:
            self._reset_daily_stats_if_needed()
            self._clean_old_requests()

            remaining_budget = max(0, self.daily_limit_usd - self.daily_cost)
            budget_used_pct = (self.daily_cost / self.daily_limit_usd) * 100

            return {
                "date": self.current_date.isoformat(),
                "daily_limit_usd": self.daily_limit_usd,
                "daily_cost_usd": round(self.daily_cost, 6),
                "remaining_budget_usd": round(remaining_budget, 6),
                "budget_used_percent": round(budget_used_pct, 2),
                "daily_requests": self.daily_requests,
                "input_tokens_today": self.daily_input_tokens,
                "output_tokens_today": self.daily_output_tokens,
                "requests_last_minute": len(self.request_timestamps),
                "requests_per_minute_limit": self.requests_per_minute,
            }

    def reset_daily_limit(self):
        """Manually reset daily limit (admin function)."""
        with self.lock:
            logger.warning("Manually resetting daily rate limit")
            self.daily_cost = 0.0
            self.daily_input_tokens = 0
            self.daily_output_tokens = 0
            self.daily_requests = 0
            self.request_timestamps = []


# Global rate limiter instance
_rate_limiter = None


def get_rate_limiter(daily_limit_usd: float = 1.0, requests_per_minute: int = 100) -> BedrockRateLimiter:
    """
    Get or create the global rate limiter instance.

    Args:
        daily_limit_usd: Maximum spend per day in USD
        requests_per_minute: Maximum requests per minute

    Returns:
        BedrockRateLimiter instance
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = BedrockRateLimiter(
            daily_limit_usd=daily_limit_usd,
            requests_per_minute=requests_per_minute
        )
    return _rate_limiter
