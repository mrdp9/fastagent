---
name: fastagent
description: "Build a Python AI agent app using the FastAgent framework. Use when the user wants a decorator-driven agent, workflow, or loop with built-in short+long term memory, automatic tool-schema generation, and an offline mock LLM (no API key required). Trigger phrases: 'fastagent', 'make me an agent', 'build a small AI app with memory', 'I want a workflow that chains agents', 'self-evaluating loop', 'zero-boilerplate agent framework'."
category: software-development
---

# FastAgent

A decorator-driven AI & memory framework for Python. Zero boilerplate.
Works offline out of the box. Ship a working agent in 30 seconds.

This skill is the procedural memory for the **FastAgent framework** —
the codebase that lives at `C:/Users/Administrator/fastagent_project/`
(or wherever the user has it cloned). When the user says "fastagent"
or asks for an agent built the FastAgent way, load this skill and
follow it.

## When to Use

Load this skill when the user wants to:

- build an AI agent in Python with minimal setup
- wire up multiple agents into a workflow or a self-evaluating loop
- give an agent **short-term chat memory + long-term semantic memory**
- call tools from an LLM without writing JSON schemas by hand
- run a script with **no API key** (mock LLM provider)
- switch the same code between OpenAI / MiniMax / Ollama

Do NOT load this for: LangChain / AutoGen / CrewAI code (different
ecosystems), pure prompt-engineering tasks with no agent loop, or
production RAG over millions of docs (use a dedicated vector DB).

## The 5-line FastAgent app

```python
import asyncio
from fastagent import FastAgent

app = FastAgent(name="hello")

@app.agent(name="greeter", system_prompt="Greet warmly.")
async def greeter(ctx, user_input, messages=None):
    return "Hello, " + user_input + "!"

print(asyncio.run(app.run_agent("greeter", "world")).value)
```

No API key. No DB. No schema file. The framework reads your function
signature, builds a JSON tool schema if needed, calls the offline
mock LLM, and returns an `AgentResult` envelope.

## Public API (what to import)

```
from fastagent import (
    FastAgent,           # the app object - you decorate it
    AgentContext,        # shared state per agent run
    AgentResult,         # envelope every agent returns
    LoopResult,          # envelope every loop returns
    MemoryStore,         # short + long term memory
    Message, MemoryHit,  # memory primitives
    ShortTermContext,    # chat history alone
    LLMClient,           # the LLM router
    ChatResponse,        # what await client.chat() returns
    function_to_tool_schema,  # function -> JSON schema
    format_prompt, safe_run, SkipSchema,  # helpers
)
```

## Three decorators (the entire framework surface)

### `@app.agent(name, system_prompt="...", tools=None, memory_k=4, on_error="return_error", max_retries=0)`

Decorate an `async def`. Returns `AgentResult`.

```python
@app.agent(name="qa", system_prompt="Answer in one sentence.")
async def qa(ctx, user_input, messages=None):
    resp = await ctx.client.chat(messages)
    return resp.content
```

### `@app.workflow(name)`

Decorate an `async def` that `yield`s each step's output:

```python
@app.workflow(name="intake")
async def intake(ctx):
    a = await app.run_agent("qa", "hi", ctx=ctx)
    yield a.value
    yield "step 2 done"
```

Returns `await app.run_workflow(name)` -> list of yielded values.

### `@app.loop(name, max_iterations=5, evaluator=None)`

Self-evaluating loop. Body runs until evaluator says "stop" or
`max_iterations` reached.

```python
async def check(ctx):
    if len(ctx.state.get("draft", "")) > 200:
        return ("stop", None)
    return ("continue", None)

@app.loop(name="refine", max_iterations=10, evaluator=check)
async def refine(ctx, i):
    ctx.state["draft"] = ctx.state.get("draft", "") + " more."
    return ctx.state["draft"]
```

### `@app.tool(name=None)`

Register a plain function as a tool. Agents auto-pull all registered
tools (or pass `tools=[my_fn]` to scope explicitly):

```python
@app.tool()
def lookup_user(user_id: str) -> dict:
    """Look up a user. Args: user_id: the user's id."""
    return {"id": user_id, "name": "Priya"}
```

### `@app.structured_agent(name, output_schema=MyModel)`

Validates LLM response against a Pydantic v2 `BaseModel`:

```python
from pydantic import BaseModel
class Answer(BaseModel):
    name: str
    score: int

@app.structured_agent(name="summarize", output_schema=Answer)
async def summarize(ctx, user_input, messages=None):
    pass  # framework calls the LLM directly
```

## Common Patterns

### Memory - long-term semantic search

```python
@app.agent(name="qa")
async def qa(ctx, user_input, messages=None):
    # Seed corporate knowledge
    if not ctx.memory:
        ctx.memory = MemoryStore()
    await ctx.memory.add("The CEO is Priya.", {"topic": "people"})
    await ctx.memory.add("Office is in Bengaluru.", {"topic": "office"})

    # Search returns hits ranked by cosine similarity
    hits = await ctx.memory.search("who runs the company?", limit=3)
    for h in hits:
        print(f"  {h.score:.2f}  {h.text}")
    return str(hits[0].text) if hits else "no answer"
```

### Memory - persistence

```python
await ctx.memory.save_jsonl("memory.jsonl")
await ctx.memory.load_jsonl("memory.jsonl")  # incremental, safe to re-call
```

### Memory - LRU eviction cap

```python
store = MemoryStore(max_records=10_000)
```

### Error envelope

```python
r = await app.run_agent("qa", "hi")
if not r.ok:
    print("Agent failed:", r.error)
else:
    print("Answer:", r.value)
```

`on_error="raise"` re-raises, `on_error="retry"` retries N times.

### Tool with auto-schema

Just write a typed function with a docstring. FastAgent introspects
the signature and builds the JSON schema. No manual JSON.

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

### RunContext injection (hide from schema)

```python
from fastagent.utils import SkipSchema
from fastagent.core import RunContext

@app.tool()
def lookup(user_id: str, ctx=None) -> dict:
    """Look up a user with access to the current agent context."""
    return {"id": user_id, "agent": ctx.agent_name if ctx else "?"}
```

The `ctx` param is excluded from the JSON schema and injected at call
time.

## Switching to a real LLM

The mock provider is the default. To use a real model:

```bash
export FASTAGENT_PROVIDER=openai    # or minimax | ollama
export OPENAI_API_KEY=***       # or MINIMAX_API_KEY=*** (skip for ollama)
python my_app.py
```

If the real provider fails, FastAgent falls back to the mock so demos
keep working.

## One-shot recipes (in `scripts/`)

This skill ships with three ready-to-run scaffold scripts:

1. `scripts/scaffold.py` - create a new FastAgent project from scratch
   (writes `app.py`, `requirements.txt`, `Dockerfile`, `.gitignore`).
2. `scripts/verify.py` - sanity-check a FastAgent project (imports +
   smoke tests the demo).
3. `scripts/memory_dump.py` - dump a MemoryStore's contents as readable
   text for debugging.

Run any with `python scripts/<name>.py` from the skill directory, or
copy them into the user's project to run there.

## Templates (in `templates/`)

- `templates/hello.py` - minimal "hello world" agent.
- `templates/agent_with_memory.py` - agent + memory seed + search.
- `templates/workflow.py` - workflow chaining two agents.
- `templates/loop.py` - self-evaluating refine loop.
- `templates/structured.py` - structured_agent with Pydantic output.
- `templates/Dockerfile` - containerized FastAgent (Python 3.12 slim).
- `templates/requirements.txt` - pydantic + pytest + pytest-asyncio.

Copy any template into a new directory and run `python <filename>.py`.

## References (in `references/`)

- `references/decorator-cheatsheet.md` - one-screen cheat sheet for all
  decorators with parameter tables.
- `references/memory-deep-dive.md` - how the vector index works,
  tuning tips, pluggable embedders.
- `references/llm-providers.md` - per-provider setup (OpenAI / MiniMax
  / Ollama), env vars, gotchas.
- `references/bootstrap-pattern.md` - the `import os + sys.path` block
  that makes any scaffolded `app.py` runnable without `pip install`.

## Verification Checklist

After building anything with FastAgent, verify:

- [ ] `python -c "import fastagent; print(fastagent.__version__)"` works
- [ ] `from fastagent import FastAgent, AgentContext, MemoryStore` works
- [ ] At least one `@app.agent` exists and `app.run_agent(name, ...)`
      returns an `AgentResult` with `.ok = True`
- [ ] `python -m pytest tests/ -v` passes (if tests were added)
- [ ] `python app.py` exits with `All steps passed.` (for the standard demo)

## Common Pitfalls

1. **Forgetting `async def`.** All decorated functions MUST be
   `async def`. A sync function triggers `TypeError: agent ... must
   be async`.

2. **Wrapping in try/except.** By default, exceptions are NOT raised
   - they come back as `AgentResult(ok=False, error=...)`. Either
   check `result.ok` or pass `on_error="raise"` to the decorator.

3. **String return for structured agents.** `@app.structured_agent`
   validates the LLM's output against your Pydantic model. If the
   model returns raw text, you get `ok=False` with a parse error.
   Add field-level `Field(description=...)` to improve parsing.

4. **Memory search needs at least one `add()` call.** An empty
   `MemoryStore` returns `[]` from search - the agent cannot
   "hallucinate from memory" if there is none.

5. **Mock provider's replies include "memory:" only if you passed
   `memory_hits=`.** When using the offline mock provider directly
   (`LLMClient(provider="mock")`), the response echoes the user
   message and shows memory hits. With `@app.agent` the framework
   wires this automatically.

6. **The patch tool and sanitizer.** When writing FastAgent code via
   Hermes tools, avoid literal `***` in tool-arg strings - the
   sanitizer mangles them. Load the `hermes-writing-python-source-files`
   skill if you need to write files with regex literals or docstrings
   containing emphasis markers.

7. **`LoopResult` field names are `final_state` and `stopped_reason`, NOT
   `state` and `status`.** Writing `result.status` or `result.state["plan"]`
   will raise `AttributeError`. The dataclass is `LoopResult(iterations,
   history, final_state, stopped_reason)`. Verified during the 2026-06
   finance-copilot demo build.

8. **Never write f-strings containing apostrophes or curly-brace format specs
   (e.g. `{x:.2f}`) directly in your Python source when using Hermes
   `write_file` / `execute_code` tools.** The Python parser tries to
   interpret `{` as a f-string placeholder even though it's inside a
   quoted string, and the parameter sanitizer may strip patterns. Workarounds:
   (a) build the template string via `chr(123)`/`chr(125)` concatenation, or
   (b) use plain `+` concatenation instead of f-strings, or (c) pre-assign
   the format string to a variable so the parser doesn't see the `{...}`
   in your tool-arg. Verified during 2026-06 finance-copilot demo.

9. **`scaffold.py` template keys do not always match filenames.** The key
   `memory` resolves to `agent_with_memory.py`. `hello`, `workflow`,
   `loop`, `structured` match their filenames. The skill's
   `scripts/scaffold.py` keeps a `TEMPLATE_FILES` map; check there if
   in doubt.

## How to invoke (per-IDE setup)

The same SKILL.md works in Hermes, Claude Code, and OpenCode. The skill
loader path differs.

### Hermes (this CLI)

`/fastagent` slash command auto-loads this skill if it is in
`~/.hermes/skills/software-development/fastagent/`. It already is, so
just type `/fastagent` followed by your request:

    /fastagent build me an agent that answers questions about Acme Corp
    /fastagent scaffold a new project called my-app using the memory template

### Claude Code

Copy this skill folder into Claude Code's skills directory:

    # From your home directory
    mkdir -p .claude/skills
    cp -r "C:/Users/Administrator/AppData/Local/hermes/skills/software-development/fastagent" \
        .claude/skills/fastagent

Then in any Claude Code session:

    /fastagent build me an agent that...

Or drop the skill into a per-project `.claude/skills/` to scope it to
that project only.

### OpenCode

OpenCode reads skills from `~/.opencode/skills/` or a per-project
`.opencode/skills/`. Copy the skill folder there:

    cp -r "C:/Users/Administrator/AppData/Local/hermes/skills/software-development/fastagent" \
        ~/.opencode/skills/

OpenCode's slash-command syntax is the same:

    /fastagent ...

### Bare copy / share with a friend

Because this skill bundles the framework itself under `framework/`, you
can copy the WHOLE skill folder to another machine and it just works:

    scp -r fastagent/ user@other-machine:.claude/skills/

No pip install required - the bundled `framework/` is already a working
FastAgent package.

## What `/fastagent` actually does

When the slash command runs:

1. The host loads SKILL.md (this file).
2. The LLM reads it as procedural memory.
3. For a "build me an agent" request, the LLM picks a template from
   `templates/`, copies it into the user's project directory, and adapts it.
4. For a "verify" request, it runs `scripts/verify.py` on the project.
5. For a "dump memory" request, it runs `scripts/memory_dump.py`.
6. For an "install" request, it runs `scripts/install.py` to pip install
   the bundled framework.

The bundled `framework/` is the actual working FastAgent package - same
files as the live project at
`C:/Users/Administrator/fastagent_project/`. The two are kept in sync.