# Streaming and Caching Features

This document describes the enhanced query capabilities with streaming responses and intelligent caching.

## Table of Contents
1. [Overview](#overview)
2. [Response Caching](#response-caching)
3. [Streaming Responses](#streaming-responses)
4. [API Endpoints](#api-endpoints)
5. [Performance Benefits](#performance-benefits)
6. [Usage Examples](#usage-examples)

---

## Overview

The RAG API now provides two query modes:

1. **Standard Query** (`/api/query`) - With response caching
2. **Streaming Query** (`/api/query/stream`) - Real-time token-by-token responses

Both modes benefit from the quantized `llama2:7b-chat-q4_0` model for 2-3x faster inference.

---

## Response Caching

### How It Works

- Responses are cached in memory using an LRU (Least Recently Used) cache
- Cache key is generated from the question + collection name combination
- Identical questions return instantly from cache without LLM inference

### Cache Configuration

```python
TTLCache(maxsize=100, ttl=3600)
```

- **Max Entries**: 100 cached responses
- **TTL (Time To Live)**: 1 hour (3600 seconds)
- **Eviction Policy**: LRU - oldest entries removed when cache is full

### Benefits

- **Instant responses** for repeated questions
- **Reduced LLM costs** - no API calls for cached queries
- **Lower server load** - no embedding generation or ChromaDB queries for cached results

### Cache Management

**Get Cache Statistics:**
```bash
curl http://localhost:8000/api/cache/stats
```

Response:
```json
{
  "size": 15,
  "maxsize": 100,
  "ttl": 3600,
  "currsize": 15
}
```

**Clear Cache:**
```bash
curl -X DELETE http://localhost:8000/api/cache
```

Response:
```json
{
  "status": "success",
  "message": "Cache cleared (15 entries removed)"
}
```

---

## Streaming Responses

### How It Works

Instead of waiting for the complete response, the LLM generates tokens in real-time:

1. Client sends query to `/api/query/stream`
2. Server retrieves relevant documents from ChromaDB
3. LLM generates response tokens as they're produced
4. Tokens are streamed to client via Server-Sent Events (SSE)
5. Client displays tokens progressively

### Benefits

- **Immediate feedback** - users see response forming in real-time
- **Perceived faster** - tokens appear within seconds, not minutes
- **Better UX** - users know the system is working, not frozen

### Stream Format

The streaming endpoint uses **Server-Sent Events (SSE)** format:

```
data: {"type": "sources", "data": {...}}

data: {"type": "token", "data": "Based"}

data: {"type": "token", "data": " on"}

data: {"type": "token", "data": " the"}

...

data: {"type": "done"}
```

**Event Types:**
1. `sources` - Source documents and metadata (sent first)
2. `token` - Individual token from LLM response
3. `done` - Signals end of stream

---

## API Endpoints

### 1. Standard Query with Caching

**Endpoint:** `POST /api/query`

**Request:**
```json
{
  "question": "What is Tesla Roadster?",
  "collection_name": "knowledge_base",
  "n_results": 3
}
```

**Response:**
```json
{
  "answer": "The Tesla Roadster is an electric sports car...",
  "sources": [
    {
      "content": "Tesla Roadster (2008-2012)...",
      "metadata": {"source": "text_input", "chunk_index": 0}
    }
  ],
  "model_used": "llama2:7b-chat-q4_0"
}
```

**Timing:**
- First request: ~1-2 minutes (full LLM inference)
- Cached requests: <100ms (instant)

### 2. Streaming Query

**Endpoint:** `POST /api/query/stream`

**Request:**
```json
{
  "question": "What is Tesla Roadster?",
  "collection_name": "knowledge_base",
  "n_results": 3
}
```

**Response:** Server-Sent Events stream

**Timing:**
- First token: ~5-10 seconds
- Subsequent tokens: Real-time as generated
- Total time: ~1-2 minutes (same as standard, but progressive)

**Note:** Streaming responses are NOT cached (each request generates fresh response).

---

## Performance Benefits

### Before Enhancements

| Operation | Time |
|-----------|------|
| Query (first time) | ~4 minutes |
| Query (repeated) | ~4 minutes |
| User sees response | After 4 minutes |

### After Enhancements

| Operation | Standard (Cached) | Streaming |
|-----------|-------------------|-----------|
| Query (first time) | ~1-2 minutes | ~1-2 minutes |
| Query (repeated) | <100ms | ~1-2 minutes |
| User sees response | After 1-2 min (or instant) | Within 5-10 seconds |

### Combined Benefits

1. **Quantized Model**: 2-3x faster LLM inference (4 min → 1-2 min)
2. **Caching**: Instant responses for repeated questions (<100ms)
3. **Streaming**: Progressive display eliminates perceived wait time

**Result:**
- First-time queries: 50-70% faster
- Repeated queries: 99.9% faster (instant)
- User experience: Much better due to streaming

---

## Usage Examples

### Example 1: Standard Query (Python)

```python
import requests

response = requests.post(
    "http://localhost:8000/api/query",
    json={
        "question": "When was the Tesla Roadster released?",
        "collection_name": "knowledge_base",
        "n_results": 3
    }
)

data = response.json()
print(f"Answer: {data['answer']}")
print(f"Sources: {len(data['sources'])} documents")
```

### Example 2: Streaming Query (Python)

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/api/query/stream",
    json={
        "question": "When was the Tesla Roadster released?",
        "collection_name": "knowledge_base",
        "n_results": 3
    },
    stream=True
)

print("Streaming response:")
for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = json.loads(line[6:])  # Remove 'data: ' prefix

            if data['type'] == 'sources':
                print(f"Found {len(data['data']['sources'])} source documents")
            elif data['type'] == 'token':
                print(data['data'], end='', flush=True)
            elif data['type'] == 'done':
                print("\n\nStream complete!")
```

### Example 3: Streaming Query (JavaScript/Browser)

```javascript
async function streamQuery(question) {
    const response = await fetch('http://localhost:8000/api/query/stream', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            question: question,
            collection_name: 'knowledge_base',
            n_results: 3
        })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));

                if (data.type === 'token') {
                    // Append token to display
                    document.getElementById('answer').innerText += data.data;
                } else if (data.type === 'sources') {
                    console.log('Sources:', data.data);
                } else if (data.type === 'done') {
                    console.log('Stream complete');
                }
            }
        }
    }
}

// Usage
streamQuery("What is the Tesla Roadster?");
```

### Example 4: Cache Management

```bash
# Check cache status
curl http://localhost:8000/api/cache/stats

# Clear cache (useful after updating documents)
curl -X DELETE http://localhost:8000/api/cache

# Regular query (will be cached)
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Tesla?",
    "collection_name": "knowledge_base",
    "n_results": 3
  }'

# Same query again (instant from cache)
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Tesla?",
    "collection_name": "knowledge_base",
    "n_results": 3
  }'
```

---

## Best Practices

### When to Use Standard Query

- **Repeated questions** - Takes advantage of caching
- **Batch processing** - When you need the complete response
- **API integrations** - Simpler to handle complete responses
- **Cost optimization** - Cached responses save LLM inference costs

### When to Use Streaming Query

- **Interactive chat** - Real-time user experience
- **Long responses** - Users see progress immediately
- **First-time questions** - Better UX even if not cached
- **User interfaces** - Progressive display keeps users engaged

### Cache Considerations

**Clear cache when:**
- Documents are updated/reingested
- You want fresh responses for all queries
- Cache grows too large (monitor with `/api/cache/stats`)

**Don't clear cache when:**
- Documents haven't changed
- You want fast responses for common questions
- System is under high load

---

## Monitoring

### Cache Hit Rate

Monitor cache effectiveness:

```python
import requests
import time

# Make request
start = time.time()
response = requests.post("http://localhost:8000/api/query", ...)
elapsed = time.time() - start

# Check if cached (cached requests < 1 second)
if elapsed < 1.0:
    print("Cache HIT - instant response")
else:
    print(f"Cache MISS - generated in {elapsed:.1f}s")
```

### Cache Statistics

```bash
watch -n 5 'curl -s http://localhost:8000/api/cache/stats | jq'
```

Output:
```json
{
  "size": 42,
  "maxsize": 100,
  "ttl": 3600,
  "currsize": 42
}
```

---

## Deployment Notes

### Environment Variables

No additional environment variables needed. The system uses existing configuration:

```bash
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2:7b-chat-q4_0
CHROMA_HOST=localhost
CHROMA_PORT=8001
```

### Memory Usage

**Cache memory usage:**
- Average cached response: ~2-5 KB
- 100 cached entries: ~200-500 KB
- Negligible compared to model memory (~2GB for quantized llama2)

### Production Considerations

1. **Redis for distributed caching**: Current implementation uses in-memory cache (single instance)
2. **Cache size tuning**: Adjust `maxsize` and `ttl` based on usage patterns
3. **Streaming timeouts**: Set appropriate nginx/load balancer timeouts for SSE
4. **CORS for streaming**: Ensure CORS headers allow streaming from your frontend

---

## Troubleshooting

### Issue: Cache not working

**Symptoms:** All queries take full time
**Solution:** Check cache stats - if `size` is 0, caching is working but no repeated queries yet

### Issue: Streaming disconnects

**Symptoms:** Stream cuts off mid-response
**Solution:**
- Check nginx timeout settings
- Increase `timeout` in streaming code (currently 180s)
- Check network stability

### Issue: Cache grows too large

**Symptoms:** High memory usage
**Solution:**
- Clear cache: `curl -X DELETE http://localhost:8000/api/cache`
- Reduce `maxsize` in `query_enhanced.py`
- Reduce `ttl` to expire entries sooner

---

## Summary

| Feature | Benefit | Performance Gain |
|---------|---------|------------------|
| Quantized Model | Faster inference | 2-3x (4min → 1-2min) |
| Response Caching | Instant repeated queries | 99.9% (1-2min → <100ms) |
| Streaming | Real-time feedback | Perceived instant response |

**Combined Result:** First-time queries 2-3x faster, repeated queries 1000x faster, streaming provides instant user feedback.
