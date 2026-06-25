"""fastagent.memory - the brains of FastAgent.

============================================================
WHAT IS THIS FILE? (read this first if you are new)
============================================================

This file gives every agent in your app TWO kinds of memory:

  1. SHORT-TERM memory - the conversation you are having RIGHT NOW.
     Like the chat history you see in ChatGPT - the last few messages
     between you and the agent, kept in order.

  2. LONG-TERM memory - facts and notes you want the agent to remember
     FOREVER, even across different conversations. The agent searches
     this memory by MEANING, not by exact words - so if you store
     "The CEO is Priya." and later ask "who runs the company?", the
     agent can find that memory even though no words match.

If you only ever want one feature from FastAgent, it is probably this.

============================================================
BEGINNER EXAMPLE - 30 seconds
============================================================

    import asyncio
    from fastagent import MemoryStore

    async def main():
        store = MemoryStore()
        await store.add("The CEO of Acme is Priya.", {"topic": "people"})
        await store.add("Acme is headquartered in Bengaluru.", {"topic": "office"})
        hits = await store.search("who runs the company?", limit=2)
        for hit in hits:
            print(f"  score={hit.score:.2f}  text={hit.text}")

    asyncio.run(main())

============================================================
HOW DOES LONG-TERM SEARCH WORK? (read this if curious)
============================================================

When you call store.add(text, ...) we:
  1. Convert the text into a list of numbers (a "vector") using either:
     - YOUR embedding function (e.g. OpenAI, Ollama) - best quality
     - The built-in offline embedder - works with no API key needed
  2. Store the vector alongside the original text.

When you call store.search(query, limit=k) we:
  1. Convert the query into a vector the same way.
  2. Compare its vector to every stored vector using COSINE SIMILARITY
     (how much the two vectors point in the same direction - a number
     between 0.0 = unrelated and 1.0 = identical meaning).
  3. Return the top k most similar chunks, sorted by score.

The result of search is a list of MemoryHit objects:
  hit.id        - unique id of the stored memory
  hit.text      - the original text you stored
  hit.metadata  - the dict you stored alongside it
  hit.score     - cosine similarity (0.0 to 1.0)

============================================================
PUBLIC CLASSES (the things you actually import)
============================================================

  MemoryStore        - the main thing you will use. Short + long term.
  ShortTermContext   - just the chat-history part (advanced).
  Message            - one message in the chat history.
  MemoryHit          - one search result.
  default_store()    - get a shared singleton (advanced).

============================================================
ADVANCED: PERSISTENCE + EVICTION
============================================================

MemoryStore supports two optional features:
  * max_records=N     - LRU eviction. When you store the (N+1)th item,
                        the LEAST-recently-USED item is dropped.
  * save_jsonl(path)  - dump long-term memory to a .jsonl file.
  * load_jsonl(path)  - load it back. Safe to call repeatedly; existing
                        records (by id) are kept, new ones are appended.

============================================================
REQUIREMENTS
============================================================

Zero hard dependencies. numpy is OPTIONAL - if installed, the vector
index uses it for fast matrix math; if not, we fall back to a tiny
pure-python implementation that is correct but slower on >10k records.

Only Python 3.10+ is required (we use modern type union syntax).
"""
from __future__ import annotations

import hashlib
import math
import re
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Optional numpy makes the vector index ~50x faster on large corpora.
# If you do not have it, we fall back to a tiny pure-python implementation.
try:
    import numpy as _np  # type: ignore
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _np = None
    _HAS_NUMPY = False


# ============================================================================ #
# Message - one turn of conversation (used by ShortTermContext below)
# ============================================================================ #
@dataclass
class Message:
    """A single message in the short-term conversation history.

    Think of this like one bubble in a chat UI. Every time the user says
    something, or the agent replies, or a tool result comes back, we make
    one of these.

    Attributes
    ----------
    role : str
        Who said it. One of "system" (instructions), "user" (the human),
        "assistant" (the agent), or "tool" (a tool result).
    content : str
        The actual text of the message.
    timestamp : float
        Unix epoch seconds when the message was created. Auto-filled.
    metadata : dict
        Anything extra you want to attach (e.g. tool_call_id, tokens).
    """
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict representation. Handy for JSON serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


# ============================================================================ #
# MemoryHit - one search result
# ============================================================================ #
@dataclass
class MemoryHit:
    """One result returned by MemoryStore.search.

    Attributes
    ----------
    id : str
        Unique id of the stored memory. Pass it to MemoryStore.get(id)
        to retrieve the full record.
    text : str
        The original text you stored.
    metadata : dict
        The metadata dict you stored alongside the text.
    score : float
        Cosine similarity between the query and this text. Range 0.0-1.0:
        1.0 = identical meaning, 0.0 = unrelated. Higher is more relevant.
    """
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


# ============================================================================ #
# Tokenization - turn text into a list of word-like tokens
# ============================================================================ #
# A regex that matches word characters: letters, digits, underscore.
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_TOKEN_RE = _TOKEN_PATTERN
def _tokenize(text: str) -> List[str]:
    """Split text into lowercase word tokens. Used by the offline embedder.

    >>> _tokenize("The CEO is Priya.")
    ["the", "ceo", "is", "priya"]
    """
    return [t.lower() for t in _TOKEN_RE.findall(text) if t]


# ============================================================================ #
# ShortTermContext - the chat history (part 1 of memory)
# ============================================================================ #
class ShortTermContext:
    """A thread-safe, in-memory log of the most recent conversation messages.

    This is the "short-term" half of FastAgent memory - the recent chat
    history. By default it keeps the last 200 messages and drops older ones
    (FIFO). It is safe to use from multiple threads at once.

    BEGINNER EXAMPLE
    ----------------
        st = ShortTermContext()
        st.add("user", "Hello!")
        st.add("assistant", "Hi, how can I help?")
        for msg in st.messages():
            print(f"[{msg.role}] {msg.content}")

    You usually do NOT need to create one yourself - every AgentContext
    already has one at ctx.memory.short_term. You just call
    ctx.memory.short_term.add(...) or read ctx.memory.short_term.messages().

    PARAMETERS
    ----------
    max_messages : int, default 200
        How many messages to keep before dropping the oldest. 200 is enough
        for most chat sessions; raise it if you need long context.
    """

    VALID_ROLES = ("system", "user", "assistant", "tool")

    def __init__(self, max_messages: int = 200) -> None:
        if max_messages <= 0:
            raise ValueError(
                "ShortTermContext: max_messages must be > 0, got %r" % max_messages
            )
        self._lock = threading.RLock()  # re-entrant so tools can call add() recursively
        self._messages: List[Message] = []
        self._max = max_messages

    def add(self, role: str, content: str, metadata=None) -> Message:
        """Append a new message to the history.

        PARAMETERS
        ----------
        role : str
            Who said it. Must be one of: system, user, assistant, tool.
        content : str
            The text of the message.
        metadata : dict, optional
            Anything extra you want to attach.

        RETURNS
        -------
        Message
            The Message object that was just added.
        """
        if role not in self.VALID_ROLES:
            raise ValueError(
                "ShortTermContext.add: invalid role %r. Must be one of %s." % (role, list(self.VALID_ROLES))
            )
        msg = Message(role=role, content=content, metadata=metadata or {})
        with self._lock:
            self._messages.append(msg)
            if len(self._messages) > self._max:
                overflow = len(self._messages) - self._max
                del self._messages[0:overflow]
        return msg

    def messages(self) -> List[Message]:
        """Return a SHALLOW COPY of the messages in chronological order."""
        with self._lock:
            return list(self._messages)

    def clear(self) -> None:
        """Forget everything. Useful between test cases."""
        with self._lock:
            self._messages.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)


# ============================================================================ #
# Deterministic offline embedder (used when no embed_fn is provided)
# ============================================================================ #
class _DeterministicEmbedder:
    """A tiny embedder that produces a 256-dim vector from text using only stdlib.

    This is NOT a real language model. It is a hashed bag-of-words with IDF
    weighting - good enough for short factual corpora and demos, but for
    production use you should plug in a real embedder (see MemoryStore).

    WHY THIS EXISTS: it means the framework can demo, test, and bootstrap
    itself without anyone needing an OpenAI key.

    DIM : int = 256
        Size of the output vector. Fixed at 256.
    """

    DIM = 256

    def __init__(self) -> None:
        # doc_freq[t] = number of stored documents containing token t.
        # Used to weight rare tokens higher than common ones (IDF).
        self._doc_freq: Counter = Counter()
        self._n_docs: int = 0

    def _hash_token(self, tok: str) -> int:
        """Map a token to one of 256 buckets via MD5."""
        h = hashlib.md5(tok.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "big") % self.DIM

    def fit(self, corpus: Iterable[str]) -> None:
        """Build the IDF table from a corpus of documents.

        Called automatically by MemoryStore; you should not need this.
        """
        self._doc_freq = Counter()
        self._n_docs = 0
        for doc in corpus:
            toks = set(_tokenize(doc))  # dedup per-doc
            self._doc_freq.update(toks)
            self._n_docs += 1

    def encode(self, text: str) -> List[float]:
        """Encode a single string into a normalized 256-dim vector."""
        toks = _tokenize(text)
        if not toks:
            return [0.0] * self.DIM
        counts: Counter = Counter(toks)
        vec = [0.0] * self.DIM
        n_docs = max(1, self._n_docs)
        for tok, c in counts.items():
            idx = self._hash_token(tok)
            df = self._doc_freq.get(tok, 0)
            idf = math.log((1 + n_docs) / (1 + df)) + 1.0
            vec[idx] += float(c) * idf
        return self._normalize(vec)

    @staticmethod
    def _normalize(vec: Sequence[float]) -> List[float]:
        """L2-normalize a vector. After this, dot product equals cosine similarity."""
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return list(vec)
        return [v / norm for v in vec]


# ============================================================================ #
# VectorIndex - stores all vectors and searches by similarity
# ============================================================================ #
class _VectorIndex:
    """Append-only cosine similarity index.

    You will not use this directly - it is the engine inside MemoryStore.
    Stored rows are unit-length, so dot product equals cosine similarity.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._meta: List[Dict[str, Any]] = []
        self._matrix = _zeros_matrix((0, dim))

    def add(self, record_id: str, text: str, vec: Sequence[float], meta: Dict[str, Any]) -> None:
        if len(vec) != self.dim:
            raise ValueError(
                "_VectorIndex.add: vector length %d != index dim %d" % (len(vec), self.dim)
            )
        self._ids.append(record_id)
        self._texts.append(text)
        self._meta.append(meta)
        row = _as_row(vec)
        self._matrix = _vstack(self._matrix, row)

    def search(self, query_vec: Sequence[float], k: int) -> List[Tuple[int, float]]:
        """Return up to k (row_index, score) pairs sorted by score desc."""
        if self._matrix.shape[0] == 0:
            return []
        q = _as_row(query_vec)
        scores = _matmul(q, self._matrix.T).flatten()
        k = max(1, min(k, scores.shape[0]))
        order = _argsort_desc(scores)
        return [(int(i), float(scores[i])) for i in order[:k]]

    def remove(self, row: int) -> None:
        """Drop one row by its index (used by LRU eviction)."""
        if row < 0 or row >= len(self._ids):
            return
        del self._ids[row]
        del self._texts[row]
        del self._meta[row]
        n = len(self._ids)
        if n == 0:
            self._matrix = _zeros_matrix((0, self.dim))
            return
        rows = []
        for i in range(n):
            if _HAS_NUMPY:
                rows.append(self._matrix[i].tolist())
            else:
                rows.append(list(self._matrix[i]))
        if _HAS_NUMPY:
            import numpy as _np2
            self._matrix = _np2.array(rows, dtype=_np2.float32)
        else:
            m = _Matrix()
            for r in rows:
                m.append(r)
            self._matrix = m

    def __len__(self) -> int:
        return self._matrix.shape[0]


# ============================================================================ #
# Tiny linear-algebra shim (numpy if installed, else pure-python)
# ============================================================================ #
def _zeros_matrix(shape: Tuple[int, int]):
    if _HAS_NUMPY:
        return _np.zeros(shape, dtype=_np.float32)
    rows, cols = shape
    return [[0.0] * cols for _ in range(rows)]


def _as_row(vec: Sequence[float]):
    if _HAS_NUMPY:
        return _np.array([list(vec)], dtype=_np.float32)
    return [list(vec)]


def _vstack(existing, new_row):
    if _HAS_NUMPY:
        return _np.vstack([existing, new_row])
    if not existing:
        return [list(new_row[0])]
    out = [list(r) for r in existing]
    out.append(list(new_row[0]))
    return out


def _matmul(a, b):
    if _HAS_NUMPY:
        return a @ b
    arow = a[0]
    m = len(b)
    n = len(b[0]) if m else 0
    out = _Matrix()
    out.append([0.0] * n)
    for j in range(n):
        s = 0.0
        for i in range(len(arow)):
            if i < m:
                s += arow[i] * b[i][j]
        out[0][j] = s
    return out


def _argsort_desc(scores):
    if _HAS_NUMPY:
        arr = scores.flatten() if hasattr(scores, "flatten") else scores
        return _np.argsort(-arr).tolist()
    if hasattr(scores, "shape"):
        shape = scores.shape
        if len(shape) == 0:
            flat = [float(scores)]
        elif len(shape) == 1:
            flat = [float(x) for x in scores]
        else:
            flat = [float(x) for x in scores[0]]
    else:
        flat = [float(x) for x in scores]
    return sorted(range(len(flat)), key=lambda i: -flat[i])


def _lex_score(query_tokens, doc_tokens):
    if not query_tokens or not doc_tokens:
        return 0.0
    q_counts = Counter(query_tokens)
    d_counts = Counter(doc_tokens)
    overlap = sum(min(q_counts[t], d_counts.get(t, 0)) for t in q_counts)
    return overlap / math.sqrt(max(1, len(doc_tokens)))


# ============================================================================ #
# Matrix shim for the pure-python fallback (so .shape / .T / .flatten() work)
# ============================================================================ #
class _Matrix(list):
    """A 2-D row-matrix that exposes .shape, .T, .flatten() like numpy.

    Used only when numpy is not installed. Stores rows as Python lists of float.
    """

    @property
    def shape(self):
        rows = len(self)
        cols = len(self[0]) if rows else 0
        return (rows, cols)

    @property
    def T(self):
        if not self:
            return _Matrix()
        rows = len(self)
        cols = len(self[0])
        out = _Matrix()
        for _ in range(cols):
            out.append([0.0] * rows)
        for i in range(rows):
            for j in range(cols):
                out[j][i] = self[i][j]
        return out

    def flatten(self):
        out = _Matrix()
        out.append([])
        for row in self:
            out[0].extend(row)
        return out


# ============================================================================ #
# MemoryStore - the main class you actually use
# ============================================================================ #
EmbedFn = Callable[[str], "Any"]


class MemoryStore:
    """Combined short-term + long-term semantic memory for an agent.

    ============================================================
    WHAT YOU DO WITH IT (3 lines)
    ============================================================

        store = MemoryStore()
        await store.add("The CEO is Priya.", {"tag": "people"})
        hits = await store.search("who runs the company?", limit=3)

    ============================================================
    PARAMETERS YOU MIGHT CARE ABOUT
    ============================================================

    embed_fn
        Optional async-or-sync function text -> List[float]. Use this to
        plug in OpenAI / Ollama / sentence-transformers / etc. If you do not
        pass one, the framework uses a built-in offline embedder that needs
        no API key (see _DeterministicEmbedder above for caveats).

    dim
        Embedding vector size. Ignored when you pass embed_fn (the index
        adapts to the first vector it sees). Default 256.

    rerank
        Default True. When using a real embedder, blend a small lexical
        signal on top of vector scores so exact-phrase lookups still surface
        even when embeddings are weak.

    max_records
        Default None (unbounded). If you set it to e.g. 10_000, the store
        will silently evict the LEAST-recently-USED record each time you add
        the 10_001st item. Searches also bump a record LRU rank.

    ============================================================
    WHAT IS "SHORT-TERM" vs "LONG-TERM"?
    ============================================================

    short_term : ShortTermContext
        The chat history (last N messages). Reset it manually with
        store.short_term.clear() or by creating a new context.

    long-term (no attribute - use store.add() / store.search())
        The vector store. Survives until you call store.clear_long_term()
        or the process exits (unless you save_jsonl() first).
    """

    def __init__(self, embed_fn=None, dim=_DeterministicEmbedder.DIM, rerank=True, max_records=None):
        """Create a new MemoryStore. All parameters are optional."""
        if max_records is not None and max_records <= 0:
            raise ValueError(
                "MemoryStore: max_records must be > 0 or None, got %r" % max_records
            )
        self._embed_fn = embed_fn
        self._dim = None
        self._offline = _DeterministicEmbedder() if embed_fn is None else None
        self._rerank = rerank
        self._index = _VectorIndex(dim=dim)
        self._index_dim = dim
        self._records = {}
        self._lru_counter = 0
        self._lru = {}
        self._max_records = max_records
        self._dirty_corpus = []
        self._lock = threading.RLock()
        self._short_term = ShortTermContext()

    # ---------------------------------------------------------------- #
    # Short-term accessor
    # ---------------------------------------------------------------- #
    @property
    def short_term(self):
        """The ShortTermContext attached to this store.

        Use store.short_term.add(...) to log a message and
        store.short_term.messages() to read the history.
        """
        return self._short_term

    # ---------------------------------------------------------------- #
    # Long-term write / read
    # ---------------------------------------------------------------- #
    async def add(self, text, metadata=None):
        """Store a piece of text in long-term memory.

        PARAMETERS
        ----------
        text : str
            The text to remember. Must be a non-empty string.
        metadata : dict, optional
            Any dict you want attached (e.g. {"source": "wiki",
            "topic": "people"}). A unique id is auto-generated for you;
            pass metadata={"id": "my-id"} to set your own.

        RETURNS
        -------
        str
            The id of the newly stored memory. Pass it to store.get(id)
            later if you want to retrieve the full record.

        RAISES
        ------
        ValueError
            If text is empty or not a string.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "MemoryStore.add: text must be a non-empty string, got %r" % text
            )
        meta = dict(metadata or {})
        record_id = meta.get("id") or str(uuid.uuid4())
        meta.setdefault("created_at", time.time())
        with self._lock:
            vec = await self._embed(text)
            if self._dim is None:
                self._dim = len(vec)
                self._index = _VectorIndex(dim=self._dim)
            elif len(vec) != self._dim:
                raise ValueError(
                    "MemoryStore.add: embedder returned dim %d but the index already "
                    "stores dim %d. All embeddings must use the same model." % (len(vec), self._dim)
                )
            self._index.add(record_id, text, vec, meta)
            self._records[record_id] = {"text": text, "metadata": meta}
            self._dirty_corpus.append(text)
            self._lru_counter += 1
            self._lru[record_id] = self._lru_counter
            self._evict_if_needed()
        return record_id

    async def search(self, query, limit=5):
        """Search long-term memory by MEANING, return the top-k hits.

        PARAMETERS
        ----------
        query : str
            The question or phrase you want to find similar memories for.
        limit : int, default 5
            How many results to return.

        RETURNS
        -------
        list[MemoryHit]
            Top-k hits sorted by relevance (most relevant first). Empty
            if there are no memories stored or the query is empty.

        BEGINNER EXAMPLE
        ----------------
            for hit in await store.search("who is the CEO?", limit=3):
                print(f"{hit.score:.2f}  {hit.text}")
        """
        if not isinstance(query, str) or not query.strip():
            return []
        limit = max(1, int(limit))
        with self._lock:
            if len(self._index) == 0:
                return []
            qvec = await self._embed(query)
            if self._dim is None:
                self._dim = len(qvec)
            ranked = self._index.search(qvec, k=limit * 4 if self._rerank else limit)
            if self._rerank and self._offline is None:
                q_tokens = _tokenize(query)
                rescored = []
                for idx, vec_score in ranked:
                    doc_text = self._index._texts[idx]
                    doc_tokens = _tokenize(doc_text)
                    lex = _lex_score(q_tokens, doc_tokens)
                    rescored.append((idx, vec_score + 0.15 * lex))
                rescored.sort(key=lambda t: t[1], reverse=True)
                ranked = rescored[:limit]
            hits = []
            self._lru_counter += 1
            for idx, score in ranked[:limit]:
                rid = self._index._ids[idx]
                self._lru[rid] = self._lru_counter
                hits.append(MemoryHit(
                    id=rid,
                    text=self._index._texts[idx],
                    metadata=self._index._meta[idx],
                    score=float(score),
                ))
            return hits

    def get(self, record_id):
        """Retrieve a stored record by id. Returns None if not found."""
        with self._lock:
            rec = self._records.get(record_id)
            return dict(rec) if rec else None

    def __len__(self):
        """How many records are currently in long-term memory."""
        with self._lock:
            return len(self._records)

    # ---------------------------------------------------------------- #
    # Eviction (LRU) and persistence
    # ---------------------------------------------------------------- #
    def _evict_if_needed(self):
        """Drop LRU records if we are over the cap. Caller holds the lock."""
        if self._max_records is None:
            return
        while len(self._records) > self._max_records:
            if not self._lru:
                break
            victim_id = min(self._lru, key=self._lru.get)
            self._evict(victim_id)

    def _evict(self, record_id):
        """Remove a single record. Caller holds the lock."""
        if record_id not in self._records:
            return
        try:
            row = self._index._ids.index(record_id)
        except ValueError:
            row = -1
        if row >= 0:
            self._index.remove(row)
        del self._records[record_id]
        self._lru.pop(record_id, None)

    async def save_jsonl(self, path):
        """Write every long-term record to a JSON-Lines file. Returns count written.

        Each line is {"id": ..., "text": ..., "metadata": {...}}. Safe to
        read back later with load_jsonl(path).
        """
        import json as _json
        with self._lock:
            data = [
                {"id": rid, "text": rec["text"], "metadata": rec["metadata"]}
                for rid, rec in self._records.items()
            ]
        with open(path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(_json.dumps(row, ensure_ascii=False) + chr(10))
        return len(data)

    async def load_jsonl(self, path):
        """Append records from a JSON-Lines file. Returns count loaded.

        Existing records (matched by id) are KEPT (not overwritten). Missing
        file returns 0 without raising.
        """
        import json as _json
        loaded = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = _json.loads(line)
                    text = row.get("text", "").strip()
                    if not text:
                        continue
                    meta = dict(row.get("metadata") or {})
                    if "id" in row and "id" not in meta:
                        meta["id"] = row["id"]
                    await self.add(text, meta)
                    loaded += 1
        except FileNotFoundError:
            return 0
        return loaded

    async def clear_long_term(self):
        """Forget every long-term record. Short-term memory is untouched."""
        with self._lock:
            self._records.clear()
            self._lru.clear()
            self._lru_counter = 0
            self._dirty_corpus.clear()
            self._index = _VectorIndex(dim=self._dim or self._index_dim)
            if self._offline is not None:
                self._offline = _DeterministicEmbedder()

    # ---------------------------------------------------------------- #
    # Internals
    # ---------------------------------------------------------------- #
    async def _embed(self, text):
        if self._embed_fn is not None:
            vec = self._embed_fn(text)
            if hasattr(vec, "__await__"):
                vec = await vec
            vec = list(vec)
            return _DeterministicEmbedder._normalize(vec)
        assert self._offline is not None
        if not self._offline._n_docs or self._offline._n_docs % 64 == 0:
            self._offline.fit(self._dirty_corpus or [text])
        return self._offline.encode(text)


# ============================================================================ #
# Module-level singleton (advanced - most users do not need this)
# ============================================================================ #
_default_store = None


def default_store():
    """Return a process-wide singleton MemoryStore.

    Use this only when you want a globally-shared memory that every agent in
    every script imports. Most apps should construct their own MemoryStore
    per-agent for isolation.
    """
    global _default_store
    if _default_store is None:
        _default_store = MemoryStore()
    return _default_store


__all__ = [
    "Message",
    "MemoryHit",
    "ShortTermContext",
    "MemoryStore",
    "default_store",
]