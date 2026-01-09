"""
AWS Bedrock Configuration and Client Initialization

Provides utilities for initializing AWS Bedrock clients for:
- LLM inference (Claude, Llama, and other Bedrock models)
- Text embeddings (Titan Embeddings)
"""

import logging
import json
import boto3
from typing import Optional, Dict, Any, AsyncGenerator
from botocore.config import Config

from constants import (
    BEDROCK_REGION,
    BEDROCK_MODEL,
    BEDROCK_EMBEDDING_MODEL,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_SESSION_TOKEN,
    BEDROCK_DAILY_LIMIT_USD,
    BEDROCK_REQUESTS_PER_MINUTE
)
from rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)

# Initialize rate limiter
rate_limiter = get_rate_limiter(
    daily_limit_usd=BEDROCK_DAILY_LIMIT_USD,
    requests_per_minute=BEDROCK_REQUESTS_PER_MINUTE
)


def get_bedrock_client():
    """
    Initialize and return AWS Bedrock runtime client.

    Uses IAM role if on EC2, otherwise falls back to provided credentials.

    Returns:
        boto3 Bedrock runtime client or None if initialization fails
    """
    try:
        # Configure retry and timeout settings
        config = Config(
            region_name=BEDROCK_REGION,
            retries={'max_attempts': 3, 'mode': 'adaptive'},
            connect_timeout=10,
            read_timeout=300  # 5 minutes for long responses
        )

        # Create client with explicit credentials if provided
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            # Build client kwargs
            client_kwargs = {
                'region_name': BEDROCK_REGION,
                'aws_access_key_id': AWS_ACCESS_KEY_ID,
                'aws_secret_access_key': AWS_SECRET_ACCESS_KEY,
                'config': config
            }

            # Add session token if provided (for temporary credentials)
            if AWS_SESSION_TOKEN:
                client_kwargs['aws_session_token'] = AWS_SESSION_TOKEN
                logger.info(f"Bedrock client initialized with temporary credentials in {BEDROCK_REGION}")
            else:
                logger.info(f"Bedrock client initialized with long-term credentials in {BEDROCK_REGION}")

            client = boto3.client('bedrock-runtime', **client_kwargs)
        else:
            # Use IAM role (recommended for EC2)
            client = boto3.client(
                'bedrock-runtime',
                region_name=BEDROCK_REGION,
                config=config
            )
            logger.info(f"Bedrock client initialized with IAM role in {BEDROCK_REGION}")

        logger.info(f"Using Bedrock LLM model: {BEDROCK_MODEL}")
        logger.info(f"Using Bedrock embedding model: {BEDROCK_EMBEDDING_MODEL}")

        return client

    except Exception as e:
        logger.error(f"Failed to initialize Bedrock client: {e}")
        return None


def invoke_bedrock_llm(client, prompt: str, max_tokens: int = 2048) -> Optional[str]:
    """
    Invoke Bedrock LLM model synchronously (non-streaming) with rate limiting.

    Args:
        client: Bedrock runtime client
        prompt: The prompt to send to the model
        max_tokens: Maximum tokens in response

    Returns:
        Generated text or None if invocation fails
    """
    try:
        # Estimate input tokens (rough estimate: 1 token ≈ 4 characters)
        estimated_input_tokens = len(prompt) // 4

        # Check rate limit before making API call
        is_allowed, reason = rate_limiter.check_and_increment(
            model_type="llama3-2-3b",
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=max_tokens  # Worst case: full output
        )

        if not is_allowed:
            logger.error(f"Bedrock LLM request blocked by rate limiter: {reason}")
            raise Exception(f"Rate limit exceeded: {reason}")

        # Prepare request body based on model type
        if "claude" in BEDROCK_MODEL.lower():
            # Claude 3 models use Messages API
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "top_p": 0.9
            }
        elif "llama" in BEDROCK_MODEL.lower():
            # Llama models use max_gen_len instead of max_tokens
            body = {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            }
        else:
            # Generic format for other models
            body = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            }

        response = client.invoke_model(
            modelId=BEDROCK_MODEL,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(response['body'].read())

        # Extract text based on model type
        if "claude" in BEDROCK_MODEL.lower():
            # Claude 3 returns content in specific format
            return response_body.get("content", [{}])[0].get("text", "")
        elif "llama" in BEDROCK_MODEL.lower():
            # Llama models return generation in specific format
            return response_body.get("generation", "")
        elif "titan" in BEDROCK_MODEL.lower():
            return response_body.get("results", [{}])[0].get("outputText", "")
        else:
            # Try common response formats
            return response_body.get("completion") or response_body.get("generated_text") or ""

    except Exception as e:
        logger.error(f"Error invoking Bedrock LLM: {e}")
        return None


async def stream_bedrock_llm(client, prompt: str, max_tokens: int = 2048) -> AsyncGenerator[str, None]:
    """
    Stream Bedrock LLM response token by token with rate limiting.

    Args:
        client: Bedrock runtime client
        prompt: The prompt to send to the model
        max_tokens: Maximum tokens in response

    Yields:
        Generated tokens as they arrive
    """
    try:
        # Estimate input tokens (rough estimate: 1 token ≈ 4 characters)
        estimated_input_tokens = len(prompt) // 4

        # Check rate limit before making API call
        is_allowed, reason = rate_limiter.check_and_increment(
            model_type="llama3-2-3b",
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=max_tokens  # Worst case: full output
        )

        if not is_allowed:
            logger.error(f"Bedrock streaming request blocked by rate limiter: {reason}")
            yield f"[Error: Rate limit exceeded - {reason}]"
            return

        # Prepare request body for streaming
        if "claude" in BEDROCK_MODEL.lower():
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "top_p": 0.9
            }
        elif "llama" in BEDROCK_MODEL.lower():
            # Llama models use max_gen_len instead of max_tokens
            body = {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            }
        else:
            body = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            }

        response = client.invoke_model_with_response_stream(
            modelId=BEDROCK_MODEL,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        # Process streaming response
        for event in response['body']:
            chunk = json.loads(event['chunk']['bytes'].decode())

            # Extract token based on model type
            if "claude" in BEDROCK_MODEL.lower():
                # Claude 3 streaming format
                if chunk.get('type') == 'content_block_delta':
                    delta = chunk.get('delta', {})
                    if delta.get('type') == 'text_delta':
                        text = delta.get('text', '')
                        if text:
                            yield text
            elif "llama" in BEDROCK_MODEL.lower():
                # Llama streaming format
                text = chunk.get('generation', '')
                if text:
                    yield text
            elif "titan" in BEDROCK_MODEL.lower():
                # Titan streaming format
                text = chunk.get('outputText', '')
                if text:
                    yield text
            else:
                # Generic streaming format
                text = chunk.get('completion') or chunk.get('generated_text') or ''
                if text:
                    yield text

    except Exception as e:
        logger.error(f"Error streaming Bedrock LLM: {e}")
        yield f"[Error: {str(e)}]"


def generate_bedrock_embedding(client, text: str) -> Optional[list]:
    """
    Generate embeddings using Bedrock embedding model with rate limiting.

    Args:
        client: Bedrock runtime client
        text: Text to embed

    Returns:
        Embedding vector as list of floats, or None if generation fails
    """
    try:
        # Estimate tokens (rough estimate: 1 token ≈ 4 characters for English)
        estimated_tokens = len(text) // 4

        # Check rate limit before making API call
        is_allowed, reason = rate_limiter.check_and_increment(
            model_type="titan-embed-v2",
            estimated_input_tokens=estimated_tokens,
            estimated_output_tokens=0  # Embeddings don't have output tokens
        )

        if not is_allowed:
            logger.error(f"Bedrock embedding request blocked by rate limiter: {reason}")
            raise Exception(f"Rate limit exceeded: {reason}")

        # Prepare request body for embedding
        body = {
            "inputText": text
        }

        response = client.invoke_model(
            modelId=BEDROCK_EMBEDDING_MODEL,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(response['body'].read())

        # Extract embedding vector
        embedding = response_body.get("embedding")

        if embedding:
            logger.info(f"Generated embedding using {BEDROCK_EMBEDDING_MODEL} ({len(embedding)} dimensions)")

            # Record actual usage (embeddings use input tokens only)
            rate_limiter.record_actual_usage(
                model_type="titan-embed-v2",
                actual_input_tokens=estimated_tokens,
                actual_output_tokens=0
            )

            return embedding
        else:
            logger.error("No embedding found in response")
            return None

    except Exception as e:
        logger.error(f"Error generating Bedrock embedding: {e}")
        return None


def test_bedrock_connection(client) -> bool:
    """
    Test Bedrock connection with a simple prompt.

    Args:
        client: Bedrock runtime client

    Returns:
        True if connection successful, False otherwise
    """
    try:
        logger.info("Testing Bedrock connection...")
        response = invoke_bedrock_llm(client, "Say 'Hello' in one word.")

        if response and len(response) > 0:
            logger.info(f"✓ Bedrock connection successful. Response: {response[:50]}...")
            return True
        else:
            logger.error("✗ Bedrock connection failed: Empty response")
            return False

    except Exception as e:
        logger.error(f"✗ Bedrock connection test failed: {e}")
        return False
