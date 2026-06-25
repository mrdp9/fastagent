# Changelog

All notable changes to FastAgent are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-06-26

### Added
- `@app.tool()` decorator — register plain Python functions as LLM tools; JSON schema auto-generated from type hints + docstrings.
- `AgentResult` envelope — every `@app.agent` call returns `AgentResult(ok, value, error, agent, tool_calls, iterations)`. Errors no longer crash the caller.
- `@app.structured_agent(name, output_schema=MyModel)` — validates LLM output against a Pydantic v2 `BaseModel`. Returns the parsed instance in `AgentResult.value`.
- `MemoryStore(max_records=N)` — LRU eviction cap on long-term memory.
- `await store.save_jsonl(path)` / `await store.load_jsonl(path)` — JSONL persistence; load is incremental and safe to call repeatedly.
- `_RegisteredAgent.execute_tool_calls(...)` — when the LLM returns `tool_calls`, dispatch to the matching tool, capture results, and feed them back to the model.
- Per-tool resilience — `on_error` policy (`"raise" | "return_error" | "retry"`) on `@app.agent`.
- 87 pytest tests across memory, utils, llm, core, integration.

### Changed
- All 5 modules rewritten with beginner-friendly documentation: top-level "WHAT IS THIS FILE?" sections, BEGINNER EXAMPLES, COMMON PITFALLS, structured PARAMETERS/RETURNS/RAISES blocks.
- README rewritten for three audiences (noobs, devs, generalists) — 30-second quickstart, copy-paste patterns, full API reference, FAQ.
- `format_prompt` now has two clear modes — chat-message builder vs. template substitution. No more guessing.
- `LLMClient.embed("string")` and `LLMClient.embed(["s1","s2"])` both work.
- Mock provider's reply surfaces memory hits when present, so beginners can SEE grounding happen.
- MemorySearch defaults to cosine + BM25 hybrid rerank; pure cosine is opt-in via `rerank=False`.

### Fixed
- `function_to_tool_schema` now resolves string annotations from `from __future__ import annotations` via `typing.get_type_hints`.
- `MemoryStore.search` and `_argsort_desc` are now numpy-safe across 0-D / 1-D / 2-D arrays.
- Workflow `yield` semantics: each step's output is now a separate list entry, not a single nested list.

## [0.1.0] - 2026-06-25

### Added
- Initial release. 5 modules (`memory.py`, `utils.py`, `llm.py`, `core.py`, `__init__.py`).
- `@app.agent`, `@app.workflow`, `@app.loop` decorators.
- Offline mock LLM provider.
- Pluggable embedder (`embed_fn=...`).
- Dual-store `MemoryStore` (short-term chat + long-term vector).
- 51 pytest tests.

[0.2.0]: https://github.com/mrdp9/fastagent/releases/tag/v0.2.0
[0.1.0]: https://github.com/mrdp9/fastagent/releases/tag/v0.1.0
