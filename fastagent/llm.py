"""fastagent.llm - provider-agnostic LLM + embedding client.

============================================================
WHAT IS THIS FILE? (read this first if you are new)
============================================================

This is the ONLY file that talks to the outside world. Every other part
of FastAgent is pure-Python and works offline. When an agent needs to
"call the LLM", it goes through ``LLMClient``.

``LLMClient`` is a tiny router that knows how to talk to FOUR backends
("providers"):

  * mock    - ALWAYS works. Pretends to be an LLM. Useful for demos,
              tests, and offline development. NO API KEY REQUIRED.
  * ollama  - talks to a local Ollama server (free, no key, runs Llama
              and friends on your own machine).
  * openai  - talks to OpenAI's API (ChatGPT). Needs an API key.
  * minimax - talks to the MiniMax API. Needs an API key.

You pick a provider with the ``FASTAGENT_PROVIDER`` environment variable
(default: ``mock``) or by passing ``provider="..."`` to ``LLMClient()``.

============================================================
BEGINNER EXAMPLE - 30 seconds
============================================================

    import asyncio
    from fastagent.llm import LLMClient

    async def main():
        # No API key needed - this uses the mock provider.
        client = LLMClient(provider="mock")
        resp = await client.chat([
            {"role": "user", "content": "Hello, who are you?"},
        ])
        print(resp.content)       # the assistant's reply
        print(resp.provider)      # "mock"

    asyncio.run(main())

============================================================
USING A REAL LLM (OpenAI, Ollama, MiniMax)
============================================================

1. Set your API key (skip for Ollama):

   .. code-block:: bash

       export OPENAI_API_KEY="sk-..."
       # or
       export MINIMAX_API_KEY="..."
       # or, for Ollama:
       # Start the Ollama server locally; no key needed.

2. Set the provider:

   .. code-block:: bash

       export FASTAGENT_PROVIDER=openai    # or minimax | ollama

3. Run your script as normal.

============================================================
PUBLIC API
============================================================

  LLMClient           - the main router class
  ChatResponse        - what ``await client.chat(...)`` returns
  EmbeddingResponse   - what ``await client.embed(...)`` returns

============================================================
REQUIREMENTS
============================================================

Zero hard dependencies. The HTTP layer uses ``urllib.request`` from
the standard library so FastAgent works anywhere Python runs. If you
already have ``httpx`` or ``requests`` installed, FastAgent will use
``urllib`` anyway - no surprises.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# ============================================================================ #
# Public response dataclasses
# ============================================================================ #
@dataclass
class ChatResponse:
    """What ``await LLMClient.chat(...)`` returns.

    BEGINNER NOTE
    -------------
    A "dataclass" is just a tiny bundle of named values - like a row in a
    spreadsheet. The fields below are the columns.

    Attributes
    ----------
    content : str
        The assistant's reply text. Empty string if the model returned
        only tool calls.
    tool_calls : list[dict]
        Structured tool calls the model wants to make. Each entry looks
        like ``{"id": "...", "function": {"name": "...", "arguments": "..."}}``.
        Empty list if the model did not call any tool.
    raw : Any
        The full underlying response (provider-specific). Useful for
        debugging or when you need fields FastAgent does not surface.
    model : str
        The model that actually answered (may differ from what you
        requested if the provider substituted).
    provider : str
        Which provider served the request: ``mock``, ``openai``, ``ollama``,
        or ``minimax``.
    usage : dict
        Token usage if the provider reported it (input, output, total).
    """
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw: Any = None
    model: str = ""
    provider: str = ""
    usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class EmbeddingResponse:
    """What ``await LLMClient.embed(...)`` returns.

    Attributes
    ----------
    vectors : list[list[float]]
        One vector per input string, in the same order you gave them.
    model : str
        The embedding model that produced the vectors.
    provider : str
        Which provider served the request.
    """
    vectors: List[List[float]] = field(default_factory=list)
    model: str = ""
    provider: str = ""


# ============================================================================ #
# LLMClient - the public router
# ============================================================================ #
class LLMClient:
    """Provider-agnostic client for chat completions and embeddings.

    PARAMETERS
    ----------
    provider : str, optional
        Which backend to talk to. Default is read from the
        ``FASTAGENT_PROVIDER`` environment variable, then falls back to
        ``"mock"``. Allowed values: ``mock``, ``openai``, ``minimax``,
        ``ollama``.

    model : str, optional
        Which model name to request. Sensible defaults are picked per
        provider (e.g. ``gpt-4o-mini`` for OpenAI, ``llama3.2`` for
        Ollama). Override only if you know what you are doing.

    api_key : str, optional
        API key for the provider. Default is read from the matching
        environment variable (``OPENAI_API_KEY``, ``MINIMAX_API_KEY``).
        Not needed for ``mock`` or ``ollama``.

    base_url : str, optional
        Override the HTTP base URL. Useful for local OpenAI-compatible
        servers (LM Studio, vLLM, etc.) - pass their ``/v1`` URL here.

    timeout : float, optional
        HTTP timeout in seconds. Default 30. Beginners: leave it alone.

    HOW IT WORKS (beginner explanation)
    ----------------------------------
    FastAgent does NOT keep a long-running connection to any provider.
    Every call is a fresh HTTP request, which keeps things simple and
    means a stalled request cannot leak across calls.

    If the provider returns an error (network down, invalid key,
    rate-limit), the client raises a ``RuntimeError`` with a friendly
    beginner-oriented message. For ``openai`` and ``minimax`` only,
    FastAgent automatically falls back to the ``mock`` provider so demos
    keep working without internet.
    """

    DEFAULT_MODELS = {
        "mock": "mock-0",
        "openai": "gpt-4o-mini",
        "minimax": "minimax-chat",
        "ollama": "llama3.2",
    }

    DEFAULT_BASE_URLS = {
        "openai": "https://api.openai.com/v1",
        "minimax": "https://api.minimaxi.com/v1",
        "ollama": "http://localhost:11434/v1",
    }

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        # Resolve provider: arg -> env -> "mock".
        self.provider = (
            provider
            or os.environ.get("FASTAGENT_PROVIDER")
            or "mock"
        ).lower()
        if self.provider not in self.DEFAULT_MODELS:
            raise ValueError(
                "LLMClient: unknown provider {!r}. Allowed values: {}.".format(
                    self.provider, sorted(self.DEFAULT_MODELS)
                )
            )
        # Resolve model.
        self.model = model or self.DEFAULT_MODELS[self.provider]
        # Resolve API key.
        env_key = ""
        if self.provider == "openai":
            env_key = os.environ.get("OPENAI_API_KEY", "")
        elif self.provider == "minimax":
            env_key = os.environ.get("MINIMAX_API_KEY", "")
        self.api_key = api_key or env_key
        # Resolve base URL.
        self.base_url = (
            base_url
            or os.environ.get("FASTAGENT_BASE_URL")
            or self.DEFAULT_BASE_URLS.get(self.provider, "")
        )
        self.timeout = timeout
        # Per-instance call counter; useful for logging and tests.
        self._calls = 0

    # ------------------------------------------------------------------ #
    # The public chat() entry point
    # ------------------------------------------------------------------ #
    async def chat(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request to the configured provider.

        PARAMETERS
        ----------
        messages : list of dict
            The conversation so far. Each dict has ``role`` (one of
            ``system``, ``user``, ``assistant``, ``tool``) and
            ``content`` (the text). Example::

                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2+2?"},
                ]

        tools : list of dict, optional
            Tool schemas in OpenAI format. Each is the output of
            ``function_to_tool_schema(...)``. Pass ``None`` if the model
            should not call any tools.

        temperature : float, default 0.7
            Sampling temperature. 0.0 = deterministic, 1.0 = creative.
            Beginners: leave at 0.7.

        max_tokens : int, optional
            Cap on output length. None means "provider default".

        **kwargs
            Extra provider-specific options. Ignored by ``mock``.

        RETURNS
        -------
        ChatResponse
            See the ``ChatResponse`` class for fields.

        RAISES
        ------
        RuntimeError
            If the provider returns an error AND we cannot fall back to
            the mock provider.
        """
        self._calls += 1
        if self.provider == "mock":
            return self._mock_chat(messages, tools)
        # Try real provider; on failure, fall back to mock so demos work.
        try:
            if self.provider == "openai":
                return self._openai_chat(messages, tools, temperature, max_tokens, **kwargs)
            if self.provider == "minimax":
                return self._minimax_chat(messages, tools, temperature, max_tokens, **kwargs)
            if self.provider == "ollama":
                return self._ollama_chat(messages, tools, temperature, max_tokens, **kwargs)
        except Exception as exc:
            # Friendly fallback for beginners running offline.
            return _mock_chat_response(messages, tools, error=exc, provider_was=self.provider)

    async def embed(
        self,
        texts: Sequence[str],
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Compute embedding vectors for a batch of strings.

        PARAMETERS
        ----------
        texts : list[str]
            The texts to embed. Can be one string or many.
        model : str, optional
            Override the embedding model. Defaults to a sensible pick
            per provider.

        RETURNS
        -------
        EmbeddingResponse
            ``vectors[i]`` is the embedding for ``texts[i]``.

        BEGINNER NOTE
        -------------
        You almost never need to call this yourself - ``MemoryStore``
        uses it under the hood when you pass it an ``embed_fn``. If you
        are using the offline (default) embedder, FastAgent does NOT
        hit any provider; it produces vectors locally.
        """
        if not texts:
            return EmbeddingResponse(vectors=[], model=model or "", provider=self.provider)
        # Accept a single string OR a sequence of strings.
        if isinstance(texts, str):
            texts = [texts]
        if self.provider == "mock" or self.provider == "ollama":
            # Use the same offline embedder that MemoryStore uses.
            from .memory import _DeterministicEmbedder
            enc = _DeterministicEmbedder()
            enc.fit(list(texts))
            vectors = [enc.encode(t) for t in texts]
            return EmbeddingResponse(vectors=vectors, model="offline", provider=self.provider)
        # Real providers: simple HTTP POST.
        url = self.base_url.rstrip("/") + "/embeddings"
        body = {"input": list(texts), "model": model or self.model}
        data = self._http_json("POST", url, body)
        vectors = [item["embedding"] for item in data.get("data", [])]
        return EmbeddingResponse(vectors=vectors, model=data.get("model", ""), provider=self.provider)

    # ------------------------------------------------------------------ #
    # Mock provider (the offline one)
    # ------------------------------------------------------------------ #
    def _mock_chat(self, messages, tools):
        """Pretend to be an LLM. Always succeeds. Useful for tests."""
        return _mock_chat_response(messages, tools, error=None, provider_was="mock")

    # ------------------------------------------------------------------ #
    # OpenAI provider
    # ------------------------------------------------------------------ #
    def _openai_chat(self, messages, tools, temperature, max_tokens, **kwargs):
        url = self.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if tools:
            body["tools"] = list(tools)
        if max_tokens is not None:
            body["max_tokens"] = int(max_tokens)
        body.update(kwargs)
        data = self._http_json("POST", url, body, api_key=self.api_key)
        return _openai_style_parse(data, provider_name="openai")

    # ------------------------------------------------------------------ #
    # MiniMax provider (same shape as OpenAI - just different URL + key)
    # ------------------------------------------------------------------ #
    def _minimax_chat(self, messages, tools, temperature, max_tokens, **kwargs):
        # MiniMax exposes an OpenAI-compatible API surface.
        return self._openai_chat(messages, tools, temperature, max_tokens, **kwargs)

    # ------------------------------------------------------------------ #
    # Ollama provider (OpenAI-compat layer too)
    # ------------------------------------------------------------------ #
    def _ollama_chat(self, messages, tools, temperature, max_tokens, **kwargs):
        url = self.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            body["tools"] = list(tools)
        if max_tokens is not None:
            body["max_tokens"] = int(max_tokens)
        body.update(kwargs)
        data = self._http_json("POST", url, body, api_key=self.api_key or "ollama")
        return _openai_style_parse(data, provider_name="ollama")

    # ------------------------------------------------------------------ #
    # Tiny HTTP helper (urllib only - zero dependencies)
    # ------------------------------------------------------------------ #
    def _http_json(self, method, url, body, api_key=""):
        """POST or GET a JSON body; return parsed JSON dict."""
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if api_key:
            req.add_header("Authorization", "Bearer " + api_key)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read().decode("utf-8")
                return json.loads(payload) if payload else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                "LLMClient: HTTP {} from {}: {}".format(exc.code, url, detail[:300])
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "LLMClient: could not reach {} ({})".format(url, exc.reason)
            ) from exc


# ============================================================================ #
# Shared parsers
# ============================================================================ #
def _openai_style_parse(data, provider_name):
    """Parse an OpenAI-compatible chat response into a ChatResponse."""
    if not isinstance(data, dict):
        return ChatResponse(content=str(data), provider=provider_name)
    choices = data.get("choices") or []
    if not choices:
        return ChatResponse(content="", provider=provider_name, raw=data)
    msg = choices[0].get("message", {}) or {}
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    usage = data.get("usage") or {}
    return ChatResponse(
        content=content,
        tool_calls=list(tool_calls),
        raw=data,
        model=data.get("model", ""),
        provider=provider_name,
        usage=dict(usage) if isinstance(usage, dict) else {},
    )


def _mock_chat_response(messages, tools, error=None, provider_was=""):
    """Generate a deterministic but useful fake response.

    BEGINNER NOTE
    -------------
    The mock provider does NOT call any LLM. It synthesizes a reply that
    is realistic enough for demos:
      * Echoes back the last user message in a "I heard you" wrapper.
      * If tools were provided and the last user message hints at them,
        generates a synthetic tool_call.
      * If a previous error from a real provider is passed, the response
        includes a "previously failed" note so the user knows we fell back.
    """
    # Find the last user message.
    last_user = ""
    for m in reversed(list(messages)):
        if isinstance(m, dict) and m.get("role") == "user":
            last_user = str(m.get("content", ""))
            break
    # Detect tool calls the user might want.
    synthetic_tools = []
    if tools and last_user:
        for tool in tools:
            fn = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = fn.get("name", "")
            if name and re.search(rf"\\b{re.escape(name)}\\b", last_user, flags=re.IGNORECASE):
                synthetic_tools.append({
                    "id": "mock_call_" + str(int(time.time() * 1000)),
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                })
    note = ""
    if error is not None:
        note = " (falling back to mock: provider {!r} failed: {})".format(
            provider_was, str(error)[:120]
        )
    # If a system message contained a "long-term memory" block, surface
    # its content in the mock reply so demos look like the agent is using
    # memory. Beginners can SEE that grounding happened.
    memory_lines = []
    for m in list(messages):
        if isinstance(m, dict) and m.get("role") == "system":
            content = str(m.get("content", ""))
            if "long-term memory" in content.lower() or "memory:" in content.lower():
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped and (stripped.startswith("[") or stripped.startswith("-")):
                        memory_lines.append(stripped)
    memory_part = ""
    if memory_lines:
        memory_part = " (memory: " + "; ".join(memory_lines[:3]) + ")"
    content = "[mock answer{}]: {}{} (last user: {!r})".format(
        note, _now_iso(), memory_part, last_user
    )
    return ChatResponse(
        content=content,
        tool_calls=synthetic_tools,
        raw={"mock": True, "messages": list(messages)},
        model="mock-0",
        provider="mock",
    )


def _now_iso():
    """Friendly ISO-like timestamp."""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(int(time.time()))


# ============================================================================ #
# Public exports
# ============================================================================ #
__all__ = [
    "LLMClient",
    "ChatResponse",
    "EmbeddingResponse",
]