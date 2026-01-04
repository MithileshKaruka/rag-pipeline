"""
Helper functions for document processing and text chunking
"""

import uuid
from datetime import datetime
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks using RecursiveCharacterTextSplitter

    This uses a recursive chunking strategy that tries to split on:
    1. Double newlines (paragraphs)
    2. Single newlines (lines)
    3. Spaces (words)
    4. Characters (as last resort)

    This preserves semantic meaning by keeping related content together.

    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk (default: 1000 characters)
        chunk_overlap: Overlap between consecutive chunks (default: 200 characters)
                      This helps maintain context across chunks

    Returns:
        List of text chunks
    """
    # Recursive splitting strategy with custom separators
    # Tries these separators in order: paragraphs -> sentences -> words -> characters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Separators in order of preference (tries first to last)
        separators=[
            "\n\n",  # Split by double newlines (paragraphs) first
            "\n",    # Then single newlines (lines)
            ". ",    # Then sentences
            "! ",    # Exclamation sentences
            "? ",    # Question sentences
            "; ",    # Semi-colons
            ", ",    # Commas
            " ",     # Words
            ""       # Characters (last resort)
        ],
        # Keep separators to maintain formatting
        keep_separator=True,
    )
    return text_splitter.split_text(text)


def chunk_markdown_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split markdown/code text into chunks using specialized separators

    Optimized for markdown and code files with hierarchical structure.

    Args:
        text: Markdown or code text to split
        chunk_size: Maximum size of each chunk (default: 1000 characters)
        chunk_overlap: Overlap between consecutive chunks (default: 200 characters)

    Returns:
        List of text chunks
    """
    # Markdown/code-specific recursive splitting strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Separators optimized for markdown and code
        separators=[
            "\n## ",      # Markdown H2 headers
            "\n### ",     # Markdown H3 headers
            "\n#### ",    # Markdown H4 headers
            "\n\n",       # Paragraphs
            "\n```\n",    # Code blocks
            "\n",         # Lines
            ". ",         # Sentences
            " ",          # Words
            ""            # Characters
        ],
        keep_separator=True,
    )
    return text_splitter.split_text(text)


def generate_document_ids(count: int) -> List[str]:
    """
    Generate unique document IDs using UUID

    Args:
        count: Number of IDs to generate

    Returns:
        List of unique UUIDs as strings
    """
    return [str(uuid.uuid4()) for _ in range(count)]


def create_chunk_metadata(
    num_chunks: int,
    source: str = "text_input",
    additional_metadata: dict = None,
    filename: str = None,
    file_type: str = None
) -> List[dict]:
    """
    Create metadata dictionaries for document chunks

    Args:
        num_chunks: Total number of chunks
        source: Source type (text_input, file_upload, etc.)
        additional_metadata: Additional metadata to include
        filename: Name of the source file (for file uploads)
        file_type: Type of the source file

    Returns:
        List of metadata dictionaries for each chunk
    """
    metadatas = []
    ingestion_time = datetime.now().isoformat()

    for i in range(num_chunks):
        chunk_metadata = {
            "chunk_index": i,
            "total_chunks": num_chunks,
            "ingestion_time": ingestion_time,
            "source": source
        }

        # Add filename and file type if provided
        if filename:
            chunk_metadata["filename"] = filename
        if file_type:
            chunk_metadata["file_type"] = file_type

        # Add any additional metadata
        if additional_metadata:
            chunk_metadata.update(additional_metadata)

        metadatas.append(chunk_metadata)

    return metadatas


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> Tuple[bool, str]:
    """
    Validate if file extension is allowed

    Args:
        filename: Name of the file
        allowed_extensions: List of allowed extensions (e.g., ['.txt', '.md'])

    Returns:
        Tuple of (is_valid, file_extension)
    """
    import os
    file_extension = os.path.splitext(filename)[1].lower()
    is_valid = file_extension in allowed_extensions
    return is_valid, file_extension
