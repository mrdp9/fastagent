# FastAgent Decorator Cheat Sheet

One-screen reference for all five FastAgent decorators. Print this and tape it to your monitor.

## At a glance

| Decorator | Purpose | Your function shape | Returns from `run_X` |
|---|---|---|---|
| `@app.tool(name=None)` | Register a tool the LLM can call | `def fn(...) -> Any` | (used by agents) |
| `@app.agent(name, ...)` | Single agent run | `async def fn(ctx, user_input, messages=None)` | `AgentResult` |
| `@app.structured_agent(name, output_schema=Model)` | Agent with typed Pydantic output | `async def fn(ctx, user_input, messages=None)` (body ignored) | `AgentResult` |
| `@app.workflow(name)` | Multi-step pipeline | `async def fn(ctx)` that `yield`s each step's output | `list[Any]` (all yields) |
| `@app.loop(name, max_iterations, evaluator)` | Self-evaluating loop | `async def fn(ctx, iteration)` returning a value | `LoopResult(iterations, history, final_state, stopped_reason)` |

## `@app.tool(name=None)`

```python
@app.tool()
def lookup_user(user_id: str) -> dict:
    """Look up a user.

    Args:
        user_id: The user's id, e.g. "u_42".
    """
    return {"id": user_id, "name": "Priya"}
```

**Auto-schema:** FastAgent reads your type hints + docstring and builds an
OpenAI-format JSON tool schema. No manual JSON required.

**Optional RunContext injection:**

```python
from fastagent.utils import SkipSchema

@app.tool()
def lookup(user_id: str, ctx=None) -> dict:
    """Look up a user with access to the agent context."""
    return {"id": user_id, "agent": ctx.agent_name if ctx else "?"}
```

The `ctx=None` parameter is excluded from the JSON schema and injected at
call time. Useful for logging, cancel checks, memory reads.

## `@app.agent(name, system_prompt="...", tools=None, memory_k=4, on_error="return_error", max_retries=0, max_tool_iterations=4)`

**Required:** `name`. **Default system prompt:** "You are a helpful assistant."

```python
@app.agent(name="qa", system_prompt="Answer in one short sentence.")
async def qa(ctx, user_input, messages=None):
    resp = await ctx.client.chat(messages)
    return resp.content
```

**Parameters:**
- `name` (str, required) - unique key for `app.run_agent(name, ...)`.
- `system_prompt` (str) - shown to the model on every turn.
- `tools` (list[callable] or None) - if None, all `@app.tool`-registered tools are used.
- `memory_k` (int, default 4) - top-k long-term memories retrieved per turn.
- `on_error` - one of:
  - `"raise"` - re-raise the wrapped function's exception.
  - `"return_error"` (default) - return `AgentResult(ok=False, error=exc)`.
  - `"retry"` - retry up to `max_retries` times.
- `max_retries` (int, default 0) - retries on exception.
- `max_tool_iterations` (int) - max tool-call rounds per turn.

**Returns:** `AgentResult(value, ok, error, agent, tool_calls, iterations)`.

## `@app.structured_agent(name, output_schema=MyBaseModel)`

```python
from pydantic import BaseModel, Field

class Answer(BaseModel):
    name: str = Field(description="The person's name")
    role: str = Field(description="Their role")

@app.structured_agent(name="extract", output_schema=Answer)
async def extract(ctx, user_input, messages=None):
    pass  # body is unused - framework calls the LLM directly
```

The framework injects a "respond with this JSON shape..." instruction into
the system prompt and validates the response against `output_schema`.
Returns `AgentResult(value=<parsed_instance>)` on success.

**Tip:** add `Field(description=...)` to every field — those become
field-level guidance in the rendered system prompt.

## `@app.workflow(name)`

```python
@app.workflow(name="intake")
async def intake(ctx):
    a = await app.run_agent("qa", "hi", ctx=ctx)
    yield a.value                  # step 1
    yield "step 2 done"            # step 2
```

**Each `yield` is one step.** `await app.run_workflow(name)` returns the
list of all yielded values, in order.

**Sharing state between steps:** use `ctx.state`:

```python
@app.workflow(name="pipeline")
async def pipeline(ctx):
    ctx.state["count"] = 0
    a = await app.run_agent("step-a", "first", ctx=ctx)
    ctx.state["count"] += 1
    yield a.value
```

## `@app.loop(name, max_iterations=5, evaluator=None)`

```python
async def my_evaluator(ctx):
    if ctx.state.get("done"):
        return ("stop", None)
    return ("continue", None)

@app.loop(name="refine", max_iterations=10, evaluator=my_evaluator)
async def refine(ctx, i):
    # Body runs each iteration. Mutate ctx.state.
    ctx.state.setdefault("draft", "")
    ctx.state["draft"] += " more."
    return ctx.state["draft"]
```

**Evaluator signature:** `async def eval(ctx) -> Tuple[str, Optional[dict]]`.
Returns `(status, suggestion)` where status is `"continue"` or `"stop"`
and suggestion is passed back to the body on the next iteration.

**Default evaluator** (when `evaluator=None`): stops when
`ctx.state["done"]` is True OR after `max_iterations` iterations.

**Returns:** `LoopResult(iterations, history, final_state, stopped_reason)`.

NOTE: The actual field names are `final_state` (NOT `state`) and `stopped_reason`
(NOT `status`). Earlier docs used the shorter names; check the dataclass if
in doubt - it's defined in `fastagent/core.py`.

## Method cheat sheet

| Method | Purpose | Returns |
|---|---|---|
| `await app.run_agent(name, user_input, ctx=None)` | Run an agent | `AgentResult` |
| `await app.run_workflow(name, ctx=None)` | Run a workflow | `list` |
| `await app.run_loop(name, ctx=None, max_iterations=None)` | Run a loop | `LoopResult` |
| `app.list_components()` | List registered names | `dict` |
| `app.tool(name=None)` | Tool decorator | wrapper |
| `app.agent(name, ...)` | Agent decorator | wrapper |
| `app.structured_agent(name, output_schema=...)` | Structured agent decorator | wrapper |
| `app.workflow(name)` | Workflow decorator | wrapper |
| `app.loop(name, max_iterations, evaluator)` | Loop decorator | wrapper |

## Common pitfalls

1. **Decorated function must be `async def`** - sync functions raise
   `TypeError: ... must be async`.
2. **Don't forget `messages=None`** - omitting it works but disables the
   auto-built chat history.
3. **Default error policy is `return_error`** - exceptions don't crash
   the caller; check `result.ok`.
4. **For structured_agent**, the body is ignored - the LLM is called directly.
5. **Workflow yields are the output** - `return value` in a workflow
   doesn't surface as an output (use `yield` instead).
6. **Loop `max_iterations` is a hard cap** - even if the evaluator never
   says stop, the loop exits at this number.