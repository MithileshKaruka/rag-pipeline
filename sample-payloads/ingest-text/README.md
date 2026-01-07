# Text Ingestion Sample Payloads

This directory contains sample request and response payloads for the `/api/ingest/text` endpoint.

## Files

### `ingest-request.json`
The API request sent by the client to ingest text into ChromaDB.

**Endpoint**: `POST /api/ingest/text`

**Fields**:
- `text`: The text content to ingest (Tesla Roadster Wikipedia excerpt)
- `chunk_size`: Maximum characters per chunk (1000)
- `chunk_overlap`: Character overlap between chunks (200)
- `collection_name`: Target ChromaDB collection ("knowledge_base")
- `metadata`: Optional custom metadata (null in this example)

### `ingest-response.json`
The API response returned to the client after successful ingestion.

**Fields**:
- `message`: Success message with chunk count
- `collection_name`: The collection where documents were stored
- `num_chunks`: Total number of chunks created (3)
- `document_ids`: Array of UUIDs for each ingested chunk

### `chromadb-add-request.json`
The internal request sent from our backend to ChromaDB's v2 API to store the documents.

**Endpoint**: `POST /api/v2/tenants/default_tenant/databases/default_database/collections/{collection_id}/add`

**Key Details**:
- Uses collection UUID (not name) in the endpoint
- Embeddings generated using Ollama `nomic-embed-text` model (768 dimensions)
- Documents are split into 3 chunks with metadata
- Parallel embedding generation (up to 4 concurrent requests)

### `chromadb-add-response.json`
ChromaDB's response after successfully storing the documents.

**Status**: `201 Created`

**Body**: Empty object `{}`

## Workflow

1. **Client → Backend**: POST `/api/ingest/text` with `ingest-request.json`
2. **Backend Processing**:
   - Text is split into 3 chunks using recursive strategy
   - Embeddings generated for each chunk using Ollama
   - Metadata created with timestamps and chunk indices
3. **Backend → ChromaDB**: POST to `/add` endpoint with `chromadb-add-request.json`
4. **ChromaDB → Backend**: Returns `201 Created` (`chromadb-add-response.json`)
5. **Backend → Client**: Returns success response (`ingest-response.json`)

## Chunking Strategy

The text was split using `RecursiveCharacterTextSplitter`:
- First attempts: Paragraph splits (`\n\n`)
- Then: Sentence splits (`. `, `! `, `? `)
- Then: Word splits (` `)
- Finally: Character splits (last resort)

This resulted in 3 semantically meaningful chunks that preserve context.

## Performance

- **Total time**: ~3 seconds
- **Embedding generation**: ~3 seconds (parallel processing with 4 workers)
- **ChromaDB storage**: <1 second

## Testing

Use these payloads to test the ingestion endpoint:

```bash
curl -X POST http://your-server/api/ingest/text \
  -H "Content-Type: application/json" \
  -d @ingest-request.json
```
