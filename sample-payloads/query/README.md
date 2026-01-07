# Query Sample Payloads

This directory contains sample request and response payloads for the `/api/query` endpoint.

## Files

### `query-request.json`
The API request sent by the client to query the RAG system.

**Endpoint**: `POST /api/query`

**Fields**:
- `question`: The user's question ("when was musk asked to step down as CEO ?")
- `collection_name`: Target ChromaDB collection ("knowledge_base")
- `n_results`: Number of similar documents to retrieve (3)

### `query-response.json`
The API response returned to the client with the LLM-generated answer.

**Fields**:
- `answer`: The generated answer from the LLM (llama2:7b-chat-q4_0)
- `sources`: Array of 3 source documents with content and metadata
- `model_used`: The Ollama model used for generation ("llama2:7b-chat-q4_0")

### `chromadb-query-request.json`
The internal request sent from our backend to ChromaDB's v2 API to perform similarity search.

**Endpoint**: `POST /api/v2/tenants/default_tenant/databases/default_database/collections/{collection_id}/query`

**Key Details**:
- Uses collection UUID (not name) in the endpoint
- Query embedding generated using Ollama `nomic-embed-text` model (768 dimensions)
- Requests top 3 most similar documents
- Includes documents and metadata in response

### `chromadb-query-response.json`
ChromaDB's response with the 3 most relevant documents.

**Status**: `200 OK`

**Body**:
- `ids`: Document IDs for the retrieved chunks
- `documents`: Full text content of the 3 most similar chunks
- `metadatas`: Metadata for each chunk (source, chunk index, ingestion time)
- `distances`: Not included in response (null)
- `embeddings`: Not included in response (null)

## Workflow

1. **Client → Backend**: POST `/api/query` with `query-request.json`
2. **Backend Processing**:
   - Query embedding generated using Ollama nomic-embed-text
   - Collection UUID lookup via ChromaDB API
3. **Backend → ChromaDB**: POST to `/query` endpoint with `chromadb-query-request.json`
4. **ChromaDB → Backend**: Returns `200 OK` with 3 most similar documents (`chromadb-query-response.json`)
5. **Backend Processing**:
   - Context built from retrieved documents
   - Prompt constructed with context and question
   - LLM (llama2:7b-chat-q4_0 - 4-bit quantized) generates answer
6. **Backend → Client**: Returns answer with sources (`query-response.json`)

## Query Processing

**Similarity Search**:
- ChromaDB uses cosine similarity between query embedding and stored document embeddings
- Returns top 3 most relevant chunks
- In this example, chunk 1 (about Eberhard stepping down) was most relevant to the question

**Context Window**:
- All 3 retrieved documents are concatenated as context
- Total context: ~700 words from 3 chunks
- LLM generates answer based only on provided context

## Performance

- **Query embedding generation**: ~500ms (Ollama nomic-embed-text)
- **ChromaDB similarity search**: <100ms
- **LLM response generation**: ~1-2 minutes (llama2:7b-chat-q4_0 on CPU - 4-bit quantized for 2-3x speedup)
- **Total time**: ~1-2 minutes (bottleneck is LLM generation)

**Note**: Performance significantly improved by switching from full `llama2` to quantized `llama2:7b-chat-q4_0` model, reducing inference time by ~50-70% with minimal quality loss.

## Testing

Use these payloads to test the query endpoint:

```bash
curl -X POST http://your-server/api/query \
  -H "Content-Type: application/json" \
  -d @query-request.json
```

## Notes

- The LLM's answer in this example is technically incorrect - it states "Musk was not asked to step down as CEO" when the question was actually about Eberhard, not Musk
- The first source document clearly states: "In August 2007, Eberhard was asked by the board, led by Musk, to step down as CEO"
- This demonstrates the importance of prompt engineering and potentially using more advanced models for better reading comprehension
