# FastAgent — decorator reference

The entire framework is five decorators. This page is the canonical
parameter table.

## `@app.tool(name=None)`

Register a plain Python function as a tool the LLM can call.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` or `None` | `fn.__name__` | Override the tool's name in the JSON schema |

**Your function shape:** any `def` (sync or async). Type hints + Google/Sphinx
docstring are introspected into a JSON schema automatically.

```python
@app.tool()
def lookup_user(user_id: str) -> dict:
    """Look up a user by id."""
    return {"id": user_id, "name": "Priya"}
```

## `@app.agent(name, system_prompt=..., tools=None, memory_k=4, on_error="return_error", max_retries=0, max_tool_iterations=4)`

Register a single agent run.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | (required) | Unique key for `app.run_agent(name, ...)` |
| `system_prompt` | `str` | `"You are a helpful assistant."` | Sent on every turn |
| `tools` | `list[callable]` or `None` | `None` (use all `@app.tool`-registered tools) | Scope tools to a specific agent |
| `memory_k` | `int` | `4` | Top-k long-term memory hits per turn |
| `on_error` | `str` | `"return_error"` | `"raise"` re-raises, `"retry"` retries up to `max_retries` |
| `max_retries` | `int` | `0` | Retries when `on_error="retry"` |
| `max_tool_iterations` | `int` | `4` | Tool-call rounds per turn |

**Your function shape:**

```python
async def my_agent(ctx: AgentContext, user_input: str, messages=None) -> str:
    ...
```

Returns `AgentResult(value, ok, error, agent, tool_calls, iterations)`.

## `@app.structured_agent(name, output_schema=MyModel)`

Validates the LLM's reply against a Pydantic v2 `BaseModel`.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | (required) | Unique key |
| `output_schema` | `type[BaseModel]` | (required) | Pydantic v2 model |

**Your function shape:** body is **ignored**. The framework calls the
LLM directly with a schema-rendered system prompt and validates the
response. Returns `AgentResult(value=<parsed_instance>)`.

```python
class Answer(BaseModel):
    name: str = Field(description="The person's name")
    role: str = Field(description="Their role")

@app.structured_agent(name="extract", output_schema=Answer)
async def extract(ctx, user_input, messages=None):
    pass  # unused
```

## `@app.workflow(name)`

Chain steps via `yield`. Each `yield` is one step's output.

**Your function shape:**

```python
@app.workflow(name="intake")
async def intake(ctx: AgentContext):
    a = await app.run_agent("qa", "hi", ctx=ctx)
    yield a.value           # step 1 output
    yield "step 2 done"     # step 2 output
```

Returns `await app.run_workflow(name)` → `list[Any]` (all yields, in order).

**Sharing state:** use `ctx.state`:

```python
ctx.state["count"] = 0
ctx.state["count"] += 1
```

## `@app.loop(name, max_iterations=5, evaluator=None)`

Self-evaluating loop.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | (required) | Unique key |
| `max_iterations` | `int` | `5` | Hard cap on iterations |
| `evaluator` | `async callable` or `None` | Default (stops when `ctx.state["done"]` is True) | Custom stop condition |

**Evaluator signature:**

```python
async def my_evaluator(ctx) -> Tuple[str, Optional[dict]]:
    return ("continue", None)   # or ("stop", {"reason": "..."})
```

**Your function shape:**

```python
@app.loop(name="refine", max_iterations=10, evaluator=my_evaluator)
async def refine(ctx, i):
    ctx.state.setdefault("draft", "")
    ctx.state["draft"] += " more."
    return ctx.state["draft"]
```

Returns `LoopResult(iterations, history, final_state, stopped_reason)`.

## What's NOT a decorator (but you might expect one)

- **`@app.system_prompt` (dynamic prompt)** — not in 0.2.0. Pass a static
  `system_prompt="..."` instead. Dynamic prompts are on the roadmap.
- **`@app.chain` (different from workflow)** — use `@app.workflow` with
  a `yield` per step.
- **`@app.rag` / `@app.retrieve`** — use `ctx.memory.search(...)` directly
  inside your agent. Keeps the decorator count to five.
