# Memory in FastAgent

How the `MemoryStore` works under the hood, and how to tune it for your use case.

## The two stores in one

Every `MemoryStore` has:

1. **Short-term** — `store.short_term`. A bounded deque of `Message` records
   (role + content + timestamp). Thread-safe via `RLock`. Capped at
   `max_messages` (default 200). When full, the OLDEST message is dropped.

2. **Long-term** — `store._records` (a dict by id) + `_vectors` (a dict by id).
   Each record is `{id, text, metadata, created_at}`. Search uses
   cosine similarity over the vectors.

Both halves work together. The framework, by default, pulls top-`memory_k`
long-term hits per agent turn and concatenates them with the short-term
history when building the chat messages list.

## Vector indexing — how

FastAgent ships a **deterministic, offline embedder** (a hash-based one) so
the framework works with NO API key. The math:

1. Tokenize text on `[A-Za-z0-9]+` boundaries.
2. For each token, hash to a bucket in `[0, dim)`.
3. L2-normalize the resulting vector.
4. Cosine similarity = dot product of normalized vectors.

This is intentionally simple. It works "well enough" for demos and small
knowledge bases. For production, plug in a real embedder.

## Pluggable embedder

```python
import asyncio
from fastagent import MemoryStore

async def my_embed(text: str) -> list[float]:
    # Call OpenAI, sentence-transformers, Cohere, anything that returns a list of floats.
    # The framework calls this with one text at a time.
    return [0.1] * 384   # dim must match across calls

store = MemoryStore(embed_fn=my_embed, dim=384)
```

When `embed_fn` is supplied, FastAgent uses it for both `add()` and
`search()`. Make sure `dim` matches the embedder's output size, otherwise
similarity scores will be wrong.

## Tuning the search

By default, `search()` uses:

- cosine similarity over the vector store
- reranked with a small BM25 term-match score
- final score = `0.5 * cosine + 0.5 * bm25` (each normalized to [0, 1])

To turn off reranking (pure vector search):

```python
store = MemoryStore(rerank=False)
```

## LRU eviction cap

```python
store = MemoryStore(max_records=10_000)
```

When `add()` would push the count over `max_records`, the record with the
lowest "recency" score is dropped. Recency is bumped every time a record
shows up in a `search()` result, so popular items survive.

## Persistence

Two methods, both async:

```python
await store.save_jsonl("memory.jsonl")
await store.load_jsonl("memory.jsonl")
```

`save_jsonl` writes one JSON object per line:
`{"id": "...", "text": "...", "metadata": {...}, "vector": [...]}`.

`load_jsonl` is **incremental** — safe to call repeatedly. Existing records
(by id) are preserved; new records are added. Re-loading the same file
twice does NOT duplicate.

Both use stdlib only (no third-party deps).

## Common gotchas

1. **Empty store search returns `[]`.** If your agent says "I don't know",
   the first thing to check is whether `add()` ever ran.
2. **Dim mismatch with embed_fn crashes.** If `my_embed` returns 768-dim
   vectors but `MemoryStore(dim=384)`, the index silently produces wrong
   scores. Match the dims.
3. **Tokens don't overlap = low score.** The offline embedder is bag-of-words.
   If your query is "who runs the company?" and your memory is
   "The CEO is Priya.", the score is LOW because tokens "who/runs/company"
   don't appear in the memory. Rephrase queries to use the SAME words as
   the source ("CEO" instead of "runs the company").
4. **Short-term cap is per-store.** If two agents share a store via
   `ctx.memory = shared`, they share the chat history too.
5. **JSONL files are not encrypted.** Don't store secrets there.

## When to upgrade

Switch to a real vector DB (Chroma, Weaviate, Qdrant) when:

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
# 1. Dump what's in memory
python scripts/memory_dump.py memory.jsonl

# 2. Read it; check the question tokens appear in stored text
# 3. If not, rephrase queries or rephrase stored text
# 4. Or plug in a real embedder
```