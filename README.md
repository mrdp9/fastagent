<div align="center">

# ⚡ FastAgent

**Decorator-driven AI & memory framework for Python.**

Zero boilerplate. Works offline. Ship a working agent in 30 seconds.

[![Tests](https://github.com/mrdp9/fastagent/actions/workflows/tests.yml/badge.svg)](https://github.com/mrdp9/fastagent/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zero deps](https://img.shields.io/badge/runtime%20deps-1%20(pydantic)-success)](#install)
[![Offline mode](https://img.shields.io/badge/works%20offline-yes-success)](#-30-second-quickstart)

</div>

---

## Why FastAgent?

You want to build an AI agent. Today that means choosing between:

- **LangChain / LlamaIndex** — 200 MB of transitive deps, three competing abstractions, YAML chains
- **AutoGen / CrewAI** — opinionated multi-agent runtime, locked into one LLM-provider flow
- **smolagents** — code-execution sandbox you probably don't want in prod
- **DIY with `openai.OpenAI()`** — five files of glue before you write the actual prompt

**FastAgent** is none of those. It's the smallest possible framework that does the four things every agent app needs:

1. **Decorate a function → get an agent** (with auto memory retrieval + tool-call execution + error envelope)
2. **Chain agents into a workflow** (`yield` each step's output)
3. **Make any of them self-improving** (`@app.loop` + custom evaluator)
4. **Built-in memory** (short-term chat history + long-term semantic search, no DB required)

Total: **1 required dependency** (`pydantic`). Total LOC: **~3,500**. Time-to-first-agent: **30 seconds**, no API key.

---

## 🚀 30-second quickstart

```bash
pip install pydantic
git clone https://github.com/mrdp9/fastagent.git
cd fastagent
python app.py
```

You should see:

```
======================================================================
FastAgent demo
======================================================================
[1] agent: query 'Who is the CEO?'
   → The CEO of Acme Corp is Priya Sharma.  (grounded in memory, score 0.62)
[2] workflow: support-agent → summarizer
   → 2 step outputs returned
[3] loop: refine-draft ran 3 iterations, stopped at '3 sentences reached'
======================================================================
All steps passed.
```

No API key, no DB, no `pip install openai` ceremony. The mock LLM provider ships built-in.

---

## The 5-line FastAgent app

```python
import asyncio
from fastagent import FastAgent

app = FastAgent(name="hello")

@app.agent(name="greeter", system_prompt="Greet warmly.")
async def greeter(ctx, user_input, messages=None):
    return "Hello, " + user_input + "!"

print(asyncio.run(app.run_agent("greeter", "world")).value)
# -> Hello, world!
```

That's it. `app.run_agent(...)` returns an `AgentResult` envelope:

```python
result = await app.run_agent("greeter", "world")
result.ok          # True
result.value       # 'Hello, world!'
result.agent       # 'greeter'
result.tool_calls  # []
result.iterations  # 0
```

---

## 🎯 The five decorators (the entire framework surface)

| Decorator | What it does | Returns |
|---|---|---|
| `@app.tool()` | Register a Python function as an LLM-callable tool. Schema is auto-built from your type hints + docstring. | (used by agents) |
| `@app.agent(name, system_prompt=...)` | One agent run. Auto-retrieves top-k memories, calls the LLM, returns an `AgentResult` envelope. | `AgentResult` |
| `@app.workflow(name)` | Chain steps via `yield`. Each `yield` is one step's output. | `list[Any]` |
| `@app.loop(name, max_iterations, evaluator)` | Self-evaluating loop. Body runs until evaluator says "stop" or `max_iterations` reached. | `LoopResult` |
| `@app.structured_agent(name, output_schema=MyModel)` | Agent that validates the LLM's reply against a Pydantic `BaseModel`. | `AgentResult` |

---

## 📚 Examples

### 1. Agent with long-term memory

```python
@app.agent(name="qa", system_prompt="Answer using only the context provided.")
async def qa(ctx, user_input, messages=None):
    # Seed once (or persist to disk via ctx.memory.save_jsonl(...)).
    if len(ctx.memory) == 0:
        await ctx.memory.add("The CEO of Acme Corp is Priya Sharma.", {"topic": "people"})
        await ctx.memory.add("Acme is headquartered in Bengaluru.", {"topic": "office"})

    # Semantic search returns ranked hits.
    hits = await ctx.memory.search(user_input, limit=3)
    if not hits:
        return "I don't know yet."
    best = hits[0]
    return f"[{best.metadata['topic']}] (score={best.score:.2f}) {best.text}"
```

→ See [`examples/01_memory_agent.py`](examples/01_memory_agent.py)

### 2. Multi-step workflow

```python
@app.workflow(name="intake-and-summarize")
async def intake(ctx):
    answer = await app.run_agent("support-agent", "How long do refunds take?", ctx=ctx)
    yield answer.value                       # step 1

    summary = await app.run_agent("summarizer", answer.value, ctx=ctx)
    yield summary.value                      # step 2
```

→ See [`examples/02_workflow.py`](examples/02_workflow.py)

### 3. Self-improving loop

```python
async def draft_evaluator(ctx):
    sentences = ctx.state.get("draft", "").count(".")
    return ("stop", None) if sentences >= 3 else ("continue", None)

@app.loop(name="refine-draft", max_iterations=5, evaluator=draft_evaluator)
async def refine(ctx, i):
    sentences = [
        "Acme Corp is a great place to work.",
        "Engineering teams own their services end to end.",
        "We invest in on-call ergonomics and incident review.",
    ]
    ctx.state["draft"] = (ctx.state.get("draft", "") + " " + sentences[i]).strip()
    return ctx.state["draft"]
```

→ See [`examples/03_loop.py`](examples/03_loop.py)

### 4. Tool with auto-generated JSON schema

```python
@app.tool()
def get_weather(city: str, units: str = "metric") -> dict:
    """Look up current weather.

    Args:
        city: City name, e.g. "Mumbai".
        units: "metric" or "imperial".
    """
    return {"city": city, "temp": 28, "units": units}
```

No JSON written by hand. `function_to_tool_schema(get_weather)` introspects the signature and emits an OpenAI-compatible JSON schema. The LLM calls your function directly.

→ See [`examples/04_tool.py`](examples/04_tool.py)

### 5. Structured output (typed Pydantic)

```python
from pydantic import BaseModel, Field

class Answer(BaseModel):
    name: str = Field(description="The person's name")
    role: str = Field(description="Their role, e.g. 'CEO', 'CTO'")

@app.structured_agent(name="extract-person", output_schema=Answer)
async def extract(ctx, user_input, messages=None):
    pass  # body unused; framework calls the LLM directly
```

`result.value` is a validated `Answer` instance, not a string.

→ See [`examples/05_structured.py`](examples/05_structured.py)

---

## 🧠 Memory in detail

`MemoryStore` combines two halves in one object:

| Half | What it stores | How it's searched |
|---|---|---|
| **Short-term** (`store.short_term`) | Chat history. Bounded deque (default cap: 200). Thread-safe with `RLock`. | Linear scan, returns last N |
| **Long-term** (`store._records`) | Text chunks + metadata + vectors. Optional LRU eviction (`max_records=10_000`). | Cosine similarity (vector index) + BM25 rerank |

```python
from fastagent import MemoryStore

store = MemoryStore(max_records=10_000)        # LRU cap
await store.add("The CEO is Priya.", {"topic": "people"})

# Search returns ranked hits.
hits = await store.search("who runs the company?", limit=3)
for h in hits:
    print(f"  {h.score:.2f}  {h.text}")

# Persist to disk and reload.
await store.save_jsonl("memory.jsonl")
await store.load_jsonl("memory.jsonl")        # incremental, safe to re-call

# Plug in your own embedder (OpenAI, sentence-transformers, anything).
async def my_embed(text):
    return [0.1] * 384
store = MemoryStore(embed_fn=my_embed, dim=384)
```

The default offline embedder is a deterministic hash-based bag-of-words. It works "well enough" for demos and small corpora. For production, plug in a real embedder.

→ Full memory deep-dive: [`docs/memory.md`](docs/memory.md)

---

## 🔌 LLM providers

FastAgent supports four providers out of the box. The **mock provider is the default** so demos work with zero setup.

| Provider | Needs API key? | Default model | Setup |
|---|---|---|---|
| `mock` | no | `mock-0` | Default. Always works. |
| `openai` | yes | `gpt-4o-mini` | `export FASTAGENT_PROVIDER=openai && export OPENAI_API_KEY=sk-...` |
| `minimax` | yes | `minimax-chat` | `export FASTAGENT_PROVIDER=minimax && export MINIMAX_API_KEY=...` |
| `ollama` | no | `llama3.2` | Install Ollama locally, then `export FASTAGENT_PROVIDER=ollama` |

```bash
# Real LLM
export FASTAGENT_PROVIDER=openai
export OPENAI_API_KEY=sk-...
python my_app.py

# If the real call fails, FastAgent falls back to mock automatically.
```

→ Full provider setup: [`docs/llm-providers.md`](docs/llm-providers.md)

---

## ⚡ Use it as a Claude Code / OpenCode / Hermes skill

This repo also ships a **slash-command skill** (`/fastagent`) so you can use FastAgent from inside any of those CLIs:

```bash
# Hermes
/fastagent build me an agent that answers questions about Acme Corp

# Claude Code
cp -r skill/ ~/.claude/skills/fastagent/
# Then in any session:  /fastagent build me ...

# OpenCode
cp -r skill/ ~/.opencode/skills/fastagent/
```

The skill bundles **5 ready-to-run templates** (hello / memory / workflow / loop / structured) and **4 helper scripts** (`scaffold.py`, `verify.py`, `memory_dump.py`, `install.py`).

→ See [`skill/SKILL.md`](skill/SKILL.md) for the full skill spec.

---

## 📦 Install

```bash
# Minimum
pip install pydantic

# Recommended (faster vector index)
pip install numpy

# From source (this repo)
pip install -e .

# Or just copy the framework/ directory into your project
# (it's pure stdlib + pydantic, no build step needed)
```

---

## 🆚 Comparison vs other frameworks

| | **FastAgent** | LangChain | AutoGen | CrewAI | smolagents |
|---|---|---|---|---|---|
| LOC of core | **~3,500** | 100,000+ | 30,000+ | 15,000+ | 8,000+ |
| Required deps | **pydantic** | 30+ | 10+ | 15+ | 5+ |
| API surface | **5 decorators** | 100+ classes | 20+ classes | 12+ classes | Code-only |
| Works offline | **✓** | ✗ | ✗ | ✗ | partial |
| Pydantic-native | **✓** | partial | ✗ | ✗ | ✗ |
| Tool schema auto-gen | **✓** | manual | manual | manual | manual |
| Built-in memory | **✓** | ✗ (separate pkg) | ✗ | ✗ | ✗ |
| Self-evaluating loops | **✓** | partial | ✓ | ✗ | ✗ |
| Time-to-first-agent | **30s** | 30min | 1h | 1h | 15min |
| Slash-command skill | **✓** | ✗ | ✗ | ✗ | ✗ |

> FastAgent is not trying to replace LangChain. If you need 200 integrations, use LangChain. If you want a 200-line decorator-driven app that you can hold in your head, use FastAgent.

---

## 🧪 Tests

```bash
pip install pytest pytest-asyncio numpy
python -m pytest tests/ -v
```

You should see **87 tests pass** in ~3 seconds.

CI runs the same suite across Python 3.10, 3.11, 3.12 on every push.

---

## 🗺️ Project structure

```
fastagent/
├── fastagent/              # the framework (the actual code)
│   ├── __init__.py         # 17 public exports, version 0.2.0
│   ├── core.py             # FastAgent + 5 decorators + AgentContext + AgentResult + LoopResult
│   ├── memory.py           # MemoryStore + ShortTermContext + vector index
│   ├── llm.py              # LLMClient (mock/openai/minimax/ollama)
│   ├── utils.py            # function_to_tool_schema + format_prompt + safe_run
│   └── py.typed            # PEP 561 marker
├── tests/                  # 87 tests, all passing
├── app.py                  # end-to-end demo (agent + workflow + loop)
├── examples/               # 5 copy-paste-ready examples
├── docs/                   # deep-dive docs (memory, providers, decorators)
├── skill/                  # Claude Code / OpenCode / Hermes slash-command
│   ├── SKILL.md            # the /fastagent spec
│   ├── templates/          # 5 starter apps
│   ├── scripts/            # scaffold + verify + memory_dump + install
│   └── references/         # progressive-disclosure docs
├── pyproject.toml          # setuptools build, console_script: fastagent=fastagent.cli:main
├── Dockerfile              # Python 3.12 slim, runs app.py
├── README.md               # you are here
├── CHANGELOG.md            # release notes
├── LICENSE                 # MIT
├── CONTRIBUTING.md         # how to contribute
└── CODE_OF_CONDUCT.md      # community standards
```

---

## 🤝 Contributing

We welcome PRs! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, code style, and the PR process.

Good first contributions:
- Add a new LLM provider (Anthropic, Gemini, Mistral)
- Improve the offline embedder (real BM25, smarter tokenization)
- Add a vector DB backend (Chroma, Qdrant)
- Write more examples (RAG over a folder, code-review agent, etc.)

---

## 📄 License

MIT — see [`LICENSE`](LICENSE).

---

## 🙏 Acknowledgments

FastAgent is inspired by:

- **FastMCP** — for the decorator-driven developer experience (`@mcp.tool`, `@mcp.resource`, `@mcp.prompt`)
- **Pydantic AI** — for type-driven agents (`Agent[DepsT, ResultT]`)
- **Atomic Agents** — for "output schema renders the system prompt" UX
- **smolagents** — for keeping the core small

---

<div align="center">

**If this saved you an afternoon, [⭐ star the repo](https://github.com/mrdp9/fastagent).**

Made with ⚡ by [mrdp9](https://github.com/mrdp9) and the FastAgent community.

</div>
