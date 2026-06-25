# Memory in FastAgent

`MemoryStore` is the only memory primitive you need. It bundles two halves:

| Half | What it stores | How it's searched |
|---|---|---|
| **Short-term** (`store.short_term`) | Chat history. Bounded deque (default cap: 200). Thread-safe with `RLock`. | Linear scan, returns last N |
| **Long-term** (`store._records`) | Text chunks + metadata + vectors. Optional LRU eviction (`max_records=10_000`). | Cosine similarity (vector index) + BM25 rerank |

## Basic usage

```python
from fastagent import MemoryStore

store = MemoryStore()

# Add text with optional metadata.
record_id = await store.add("The CEO is Priya.", {"topic": "people"})

# Search by meaning, not exact words.
hits = await store.search("who runs the company?", limit=3)
for hit in hits:
    print(f"  score={hit.score:.2f}  text={hit.text}")
```

## Pluggable embedder

By default, FastAgent uses a deterministic hash-based embedder (bag-of-words)
that works offline. For production, plug in a real embedder:

```python
async def my_embed(text: str) -> list[float]:
    # Call OpenAI, sentence-transformers, Cohere, anything that returns a list of floats.
    return [0.1] * 384   # dim must match across calls

store = MemoryStore(embed_fn=my_embed, dim=384)
```

The framework calls `embed_fn(text)` once per `add()` and once per `search()`.
Make sure `dim` matches the embedder's output size, otherwise similarity
scores will be wrong.

## LRU eviction cap

```python
store = MemoryStore(max_records=10_000)
```

When `add()` would push the count over `max_records`, the record with the
lowest "recency" score is dropped. Recency is bumped every time a record
shows up in a `search()` result, so popular items survive.

## Persistence

```python
await store.save_jsonl("memory.jsonl")
await store.load_jsonl("memory.jsonl")
```

`save_jsonl` writes one JSON object per line:
`{"id": "...", "text": "...", "metadata": {...}, "vector": [...]}`.

`load_jsonl` is **incremental** — safe to call repeatedly. Existing records
(by id) are preserved; new records are added. Re-loading the same file
twice does NOT duplicate.

## Hybrid search (default)

By default, `search()` uses:

- cosine similarity over the vector store
- reranked with a small BM25 term-match score
- final score = `0.5 * cosine + 0.5 * bm25` (each normalized to [0, 1])

To turn off reranking (pure vector search):

```python
store = MemoryStore(rerank=False)
```

## Common gotchas

1. **Empty store search returns `[]`.** If your agent says "I don't know",
   the first thing to check is whether `add()` ever ran.
2. **Dim mismatch with embed_fn crashes silently.** If `my_embed` returns
   768-dim vectors but `MemoryStore(dim=384)`, the index produces wrong
   scores. Match the dims.
3. **Tokens don't overlap = low score.** The offline embedder is bag-of-words.
   If your query is "who runs the company?" and your memory is "The CEO is
   Priya.", the score is LOW because the tokens don't overlap. Rephrase
   queries to use the SAME words as the source ("CEO" instead of "runs the
   company"), OR plug in a real embedder.
4. **Short-term cap is per-store.** If two agents share a store via
   `ctx.memory = shared`, they share the chat history too.
5. **JSONL files are not encrypted.** Don't store secrets there.

## When to upgrade to a real vector DB

Switch to Chroma / Qdrant / Weaviate when:

- corpus exceeds ~10K docs
- you need sub-100ms search at scale
- you need metadata filters, hybrid search, or persistence across deploys

Switch to a real embedder when:

- the offline hash-embedder misses obvious synonyms ("CEO" vs "head")
- you need multilingual support
- you need semantic understanding beyond bag-of-words

## Diagnostic recipe

When the agent answers wrong:

```bash
# 1. Dump what's in memory.
python skill/scripts/memory_dump.py memory.jsonl

# 2. Read it; check the question tokens appear in stored text.
# 3. If not, rephrase queries or rephrase stored text.
# 4. Or plug in a real embedder.
```
