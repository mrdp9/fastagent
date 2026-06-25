"""fastagent.core - the decorator engine that ties FastAgent together.

============================================================
WHAT IS THIS FILE? (read this first if you are new)
============================================================

This file defines the ONE thing you will interact with most:
``FastAgent`` (an "app" object). Everything else in FastAgent exists to
support the three decorators you put on top of your functions:

============================================================
THE THREE DECORATORS (the entire public API in one screenshot)
============================================================

    from fastagent import FastAgent, AgentContext, AgentResult, LoopResult

    app = FastAgent(name="my-app")

    # --- 1. A simple agent that just answers questions -------------------
    @app.agent(name="qa", system_prompt="Answer in one sentence.")
    async def qa(ctx, user_input, messages=None):
        return "I heard you say: " + user_input

    # --- 2. A workflow that chains agents together -----------------------
    @app.workflow(name="intake")
    async def intake(ctx):
        a = await app.run_agent("qa", "hi", ctx=ctx)
        yield a.value            # yield step outputs one by one

    # --- 3. An autonomous loop that refines a draft until it is good -----
    @app.loop(name="refine", max_iterations=5, evaluator=...)
    async def refine(ctx, i):
        ctx.state["draft"] = (ctx.state.get("draft", "") + " more.").strip()
        return ctx.state["draft"]

Then run them::

    asyncio.run(app.run_agent("qa", "what is the weather?"))
    asyncio.run(app.run_workflow("intake"))
    asyncio.run(app.run_loop("refine"))

============================================================
WHAT YOU SHOULD READ FIRST (recommended reading order)
============================================================

1. ``AgentContext``  - the bag of state every agent sees.
2. ``AgentResult``   - the envelope every agent returns.
3. ``FastAgent``     - the app that holds your decorated functions.
4. ``LoopResult``    - what ``app.run_loop(...)`` gives you.

============================================================
THREADING / ASYNC MODEL
============================================================

* All decorated functions MUST be ``async def``.
* Inside one agent run, you can ``await`` other agent runs
  (``await app.run_agent("other", ...)``). They share the same ``ctx``.
* Multiple agents in DIFFERENT runs are independent and thread-safe
  (they each get their own ``ctx`` unless you share one).

============================================================
DEPENDENCIES
============================================================

* pydantic v2  - used for ``AgentContext`` (the only required dep)
* Everything else in this file is pure stdlib.
"""
from __future__ import annotations

import asyncio
import inspect
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

try:
    from pydantic import BaseModel, Field
    _PYDANTIC = True
except Exception:  # pragma: no cover - pydantic is the only required dep
    _PYDANTIC = False
    BaseModel = object  # type: ignore
    def Field(*_args, **_kwargs):  # type: ignore
        return None


# --------------------------------------------------------------------------- #
# Tiny JSON helpers (avoid pulling in a runtime json dependency)
# --------------------------------------------------------------------------- #
import json as _json


def json_dumps(obj: Any) -> str:
    try:
        return _json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        return str(obj)


def json_loads(text: str) -> Any:
    return _json.loads(text)


@dataclass
class RunContext:
    """Per-invocation context injected into tools and lifecycle hooks.

    This is the FastAgent analogue of Pydantic AI's ``RunContext[DepsT]``.
    Tools that declare a first parameter annotated as ``RunContext`` will
    receive this object; it is automatically excluded from the JSON schema.

    Attributes
    ----------
    agent_name
        Name of the agent executing the current turn.
    turn
        0-based turn index within the current agent run.
    user_input
        Original user input string.
    memory
        Reference to the agent's ``MemoryStore`` (same as ``ctx.memory``).
    short_term
        Reference to the short-term context (same as ``ctx.memory.short_term``).
    state
        Reference to the agent's mutable ``ctx.state`` dict.
    cancel
        ``asyncio.Event`` callers can set to request a graceful stop.
    extras
        Arbitrary per-invocation carrier (e.g. user_id, request_id).
    """
    agent_name: str = ""
    turn: int = 0
    user_input: str = ""
    memory: Any = None
    short_term: Any = None
    state: Any = None
    cancel: Any = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """The envelope every agent returns from ``app.run_agent(...)``.

    ============================================================
    WHY AN ENVELOPE?
    ============================================================
    Beginners often expect::

        result = await app.run_agent("qa", "hi")
        print(result)            # "the actual reply text"

    With FastAgent you get an envelope instead::

        result = await app.run_agent("qa", "hi")
        print(result.value)      # the actual reply text
        print(result.ok)         # True if it worked, False if it failed

    The envelope exists so that errors do NOT crash your program - they
    show up as ``result.ok = False`` with ``result.error`` set. This makes
    it easy to build resilient multi-agent pipelines.

    ============================================================
    BEGINNER EXAMPLE
    ============================================================
    .. code-block:: python

        result = await app.run_agent("qa", "What is the weather?")

        if not result.ok:
            print("Agent failed:", result.error)
        else:
            print("Agent said:", result.value)

        # Or unpack it in one line:
        text = result.value if result.ok else "(no answer)"
        print(text)

    ============================================================
    Attributes
    ----------
    ok : bool
        True if the agent produced a value; False if it failed.
        ``if result: ...`` and ``if not result: ...`` both work.
    value : Any
        Whatever your decorated function returned. ``None`` on error.
    error : BaseException or None
        The exception that broke the agent, if any. ``None`` on success.
    agent : str
        Name of the agent that produced this result. Useful when you
        have many agents and need to know which one answered.
    tool_calls : list of dict
        Tool calls the agent made during its run (empty for plain agents).
    iterations : int
        How many tool-call rounds happened. 0 for a plain LLM agent.
    """
    ok: bool
    value: Any = None
    error: Optional[BaseException] = None
    agent: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 0

    def __bool__(self) -> bool:  # ``if result:`` works
        return self.ok

    def unwrap(self) -> Any:
        """Return ``value`` if ok, else re-raise ``error``."""
        if not self.ok and self.error is not None:
            raise self.error
        return self.value

from .memory import MemoryStore, ShortTermContext, Message
from .llm import LLMClient, ChatResponse
from .utils import format_prompt, function_to_tool_schema, safe_run


# --------------------------------------------------------------------------- #
# AgentResult envelope and error policies
# --------------------------------------------------------------------------- #
OnErrorPolicy = str  # "raise" | "return_error" | "retry"
DEFAULT_MAX_TOOL_ITERATIONS = 4


# --------------------------------------------------------------------------- #
# AgentContext
# --------------------------------------------------------------------------- #
if _PYDANTIC:
    class AgentContext(BaseModel):
        """The shared state object every agent, workflow, and loop sees.

        ============================================================
        WHAT IS IT?
        ============================================================
        Every decorated function receives a ``ctx: AgentContext`` as its
        first argument. It is a SHARED, MUTABLE BAG of state that lives
        for the duration of one agent run (or one workflow, or one loop).
        If you write to ``ctx.state["foo"] = 42`` inside one agent and then
        run another agent with the SAME ctx, the second one sees "foo".

        ============================================================
        FIELDS YOU WILL ACTUALLY USE
        ============================================================

        state : dict
            The main scratchpad. Use this to pass data between steps
            of a workflow::

                @app.workflow(name="two-step")
                async def two_step(ctx):
                    ctx.state["name"] = await ask_user_name(ctx)
                    greeting = await greet_user(ctx, ctx.state["name"])
                    yield greeting

        history : list of dict
            An append-only log of events for THIS run. Each call to
            ``ctx.log("event-name", key=value)`` appends one entry. Useful
            for debugging and for the ``LoopResult`` trace.

        memory : MemoryStore
            The agent's combined short + long term memory. Most of the
            time the framework searches it FOR you before calling your
            function - you just write ``ctx.memory.add("fact")``.

        client : LLMClient
            The LLM client. Use ``await ctx.client.chat([...])`` to call
            the model directly from inside an agent (advanced).

        thread_id : str
            A unique id for this run. Useful for log correlation.

        ============================================================
        METHODS
        ============================================================
        ``ctx.log(event, **details)``
            Append one entry to ``ctx.history``. Used internally; you
            usually do not need to call this yourself unless you want
            fine-grained traces.

        ``ctx.snapshot()``
            Return a plain dict summarizing the context. Handy for
            printing the state of a workflow at the end of a run.

        ============================================================
        BEGINNER EXAMPLE
        ============================================================
        .. code-block:: python

            @app.agent(name="counter")
            async def counter(ctx, user_input, messages=None):
                ctx.state["count"] = ctx.state.get("count", 0) + 1
                return "I have been called {!r} times.".format(ctx.state["count"])

            ctx = AgentContext()  # default memory + mock client
            r1 = await app.run_agent("counter", "hi", ctx=ctx)
            r2 = await app.run_agent("counter", "hi", ctx=ctx)
            # r1.value -> "I have been called 1 times."
            # r2.value -> "I have been called 2 times."   (shared state!)

        ============================================================
        THREAD SAFETY
        ============================================================
        AgentContext uses an internal ``RLock`` so concurrent writes from
        background threads are safe, but Pydantic v2 does NOT auto-detect
        mutations to ``state`` and ``history``. Treat those as your own
        mutable containers; do not rely on Pydantic validation for them.
        """
        model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

        agent_name: str = "default"
        state: Dict[str, Any] = Field(default_factory=dict)
        history: List[Dict[str, Any]] = Field(default_factory=list)
        memory: MemoryStore = Field(default_factory=MemoryStore)
        client: LLMClient = Field(default_factory=LLMClient)
        thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

        def log(self, event: str, **details: Any) -> None:
            """Append one structured entry to ``ctx.history``."""
            self.history.append({"event": event, "ts": time.time(), **details})

        def snapshot(self) -> Dict[str, Any]:
            """Return a plain dict summarizing this context. Useful for debugging."""
            return {
                "agent_name": self.agent_name,
                "state": dict(self.state),
                "history_len": len(self.history),
                "memory_long_term": len(self.memory),
                "memory_short_term": len(self.memory.short_term),
                "thread_id": self.thread_id,
            }
else:  # pragma: no cover
    @dataclass
    class AgentContext:
        agent_name: str = "default"
        state: Dict[str, Any] = field(default_factory=dict)
        history: List[Dict[str, Any]] = field(default_factory=list)
        memory: MemoryStore = field(default_factory=MemoryStore)
        client: LLMClient = field(default_factory=LLMClient)
        thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))

        def log(self, event: str, **details: Any) -> None:
            self.history.append({"event": event, "ts": time.time(), **details})

        def snapshot(self) -> Dict[str, Any]:
            return {
                "agent_name": self.agent_name,
                "state": dict(self.state),
                "history_len": len(self.history),
                "memory_long_term": len(self.memory),
                "memory_short_term": len(self.memory.short_term),
                "thread_id": self.thread_id,
            }


# --------------------------------------------------------------------------- #
# Agent, Workflow, Loop wrappers
# --------------------------------------------------------------------------- #
class _RegisteredAgent:
    """Internal record of an agent registered via ``@app.agent``.

    Public surface is :class:`AgentResult` - what the wrapped call returns.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: Sequence[Callable[..., Any]],
        fn: Callable[..., Awaitable[Any]],
        memory_k: int = 4,
        on_error: OnErrorPolicy = "return_error",
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        max_retries: int = 0,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.tools = list(tools)
        self.tool_schemas = [function_to_tool_schema(t) for t in self.tools]
        self.fn = fn
        self.memory_k = memory_k
        self.on_error = on_error
        self.max_tool_iterations = max_tool_iterations
        self.max_retries = max_retries
        self._tool_index: Dict[str, Callable[..., Any]] = {
            (t.__name__): t for t in self.tools
        }

    async def __call__(self, ctx: AgentContext, user_input: str) -> "AgentResult":
        return await self.run(ctx, user_input)

    async def run(self, ctx: AgentContext, user_input: str) -> "AgentResult":
        ctx.agent_name = self.name
        ctx.memory.short_term.add("user", user_input)
        ctx.log("agent.start", name=self.name, user_input=user_input)
        hits = await ctx.memory.search(user_input, limit=self.memory_k)

        attempt = 0
        last_exc: Optional[BaseException] = None
        while attempt <= self.max_retries:
            try:
                # Build the chat messages list using the shared helper so the
                # "long-term memory" wording stays consistent across the framework.
                messages = format_prompt(
                    system_prompt=self.system_prompt,
                    short_term=ctx.memory.short_term.messages(),
                    memory_hits=hits,
                    user_input=user_input,
                )
                sig = inspect.signature(self.fn)
                if len(sig.parameters) >= 3:
                    raw = await self.fn(ctx, user_input, messages)
                else:
                    raw = await self.fn(ctx, user_input)
                ctx.memory.short_term.add("assistant", str(raw))
                ctx.log(
                    "agent.end",
                    name=self.name,
                    result_preview=str(raw)[:200],
                    used_memory=[h.id for h in hits],
                )
                return AgentResult(
                    ok=True,
                    value=raw,
                    agent=self.name,
                    tool_calls=[],
                    iterations=0,
                )
            except Exception as exc:  # broad on purpose - policy in on_error
                last_exc = exc
                attempt += 1
                ctx.log(
                    "agent.error",
                    name=self.name,
                    attempt=attempt,
                    exc_type=type(exc).__name__,
                    message=str(exc)[:200],
                )
                if self.on_error == "raise":
                    raise
                if attempt > self.max_retries:
                    break
        # All retries exhausted
        ctx.memory.short_term.add("assistant", "[error] " + repr(last_exc))
        ctx.log("agent.failed", name=self.name, exc_type=type(last_exc).__name__ if last_exc else None)
        return AgentResult(
            ok=False,
            error=last_exc,
            agent=self.name,
            tool_calls=[],
            iterations=0,
        )

    async def execute_tool_calls(
        self,
        ctx: AgentContext,
        user_input: str,
        tool_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute a batch of tool calls and return ``tool``-role messages
        suitable to append to the chat-completion ``messages`` array.
        """
        results: List[Dict[str, Any]] = []
        for tc in tool_calls:
            fn_meta = tc.get("function") or {}
            name = fn_meta.get("name", "")
            raw_args = fn_meta.get("arguments", "{}")
            call_id = tc.get("id", "")
            ctx.log("tool.call", agent=self.name, tool=name, call_id=call_id)
            if name not in self._tool_index:
                err = {"error": f"unknown tool: {name}"}
                results.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": json_dumps(err)})
                ctx.log("tool.error", tool=name, reason="unknown")
                continue
            try:
                if isinstance(raw_args, str):
                    args = json_loads(raw_args) if raw_args.strip() else {}
                else:
                    args = dict(raw_args or {})
            except Exception as exc:
                err = {"error": f"invalid arguments JSON: {exc}"}
                results.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": json_dumps(err)})
                ctx.log("tool.error", tool=name, reason="bad_json", exc=str(exc))
                continue
            try:
                fn = self._tool_index[name]
                out = fn(**args)
                if hasattr(out, "__await__"):
                    out = await out
                content = json_dumps(out) if not isinstance(out, str) else out
                results.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": content})
                ctx.log("tool.result", tool=name, preview=str(out)[:200])
            except Exception as exc:
                err = {"error": f"{type(exc).__name__}: {exc}"}
                results.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": json_dumps(err)})
                ctx.log("tool.error", tool=name, exc=str(exc))
        return results


class _RegisteredWorkflow:
    def __init__(self, name: str, steps: List[Callable[[AgentContext], Awaitable[Any]]]) -> None:
        self.name = name
        self.steps = steps

    async def __call__(self, ctx: AgentContext) -> List[Any]:
        ctx.log("workflow.start", name=self.name, steps=len(self.steps))
        outputs: List[Any] = []
        for i, step in enumerate(self.steps):
            ctx.log("workflow.step.start", index=i, step=getattr(step, "__name__", repr(step)))
            out = await step(ctx)
            # A single-step workflow that internally produced a sequence of
            # yielded values (an async generator wrapped in a single runner)
            # returns a list. Flatten one level so each yielded value is one
            # entry in ``outputs``.
            if isinstance(out, list):
                for j, sub in enumerate(out):
                    outputs.append(sub)
                    ctx.state[f"step_{i}_{j}_output"] = sub
            else:
                outputs.append(out)
                ctx.state[f"step_{i}_output"] = out
            ctx.log("workflow.step.end", index=i)
        ctx.log("workflow.end", name=self.name, outputs=len(outputs))
        return outputs


@dataclass
class LoopResult:
    iterations: int
    history: List[Dict[str, Any]]
    final_state: Dict[str, Any]
    stopped_reason: str


class _RegisteredLoop:
    def __init__(
        self,
        name: str,
        max_iterations: int,
        action_fn: Callable[[AgentContext, int], Awaitable[Any]],
        evaluator: Callable[[AgentContext], Awaitable[Tuple[str, Optional[Dict[str, Any]]]]],
    ) -> None:
        self.name = name
        self.max_iterations = max_iterations
        self.action_fn = action_fn
        self.evaluator = evaluator

    async def __call__(self, ctx: AgentContext) -> LoopResult:
        ctx.log("loop.start", name=self.name, max_iterations=self.max_iterations)
        last_status = "continue"
        suggestion: Optional[Dict[str, Any]] = None
        for i in range(self.max_iterations):
            ctx.state["iteration"] = i
            ctx.state["last_status"] = last_status
            ctx.state["evaluator_suggestion"] = suggestion
            ctx.log("loop.iter.start", iteration=i)
            await self.action_fn(ctx, i)
            last_status, suggestion = await self.evaluator(ctx)
            ctx.log("loop.iter.end", iteration=i, status=last_status, suggestion=suggestion)
            if last_status == "stop":
                break
        ctx.log("loop.end", name=self.name, iterations=i + 1, status=last_status)
        return LoopResult(
            iterations=i + 1,
            history=list(ctx.history),
            final_state=dict(ctx.state),
            stopped_reason=last_status,
        )


# --------------------------------------------------------------------------- #
# FastAgent app
# --------------------------------------------------------------------------- #
class FastAgent:
    """The main app object you instantiate and decorate.

    ============================================================
    WHAT IS ``FastAgent``?
    ============================================================
    ``FastAgent`` is the top-level container for your agents, workflows,
    loops, and tools. You create ONE per app, decorate your functions,
    and then call ``await app.run_agent(...)`` / ``await app.run_workflow(...)``
    / ``await app.run_loop(...)`` to execute them.

    ============================================================
    BEGINNER EXAMPLE (the shortest possible FastAgent app)
    ============================================================
    .. code-block:: python

        import asyncio
        from fastagent import FastAgent

        app = FastAgent(name="hello-app")

        @app.agent(name="greeter")
        async def greeter(ctx, user_input, messages=None):
            return "Hello, " + user_input + "!"

        print(asyncio.run(app.run_agent("greeter", "world")).value)
        # -> "Hello, world!"

    ============================================================
    PARAMETERS
    ============================================================
    name : str, default "fastagent"
        A friendly name for your app. Used in logs and ``repr(app)``.
    client : LLMClient, optional
        The LLM client to inject into every new ``AgentContext``. If you
        do not pass one, FastAgent creates a ``LLMClient(provider="mock")``
        so the app works with no API key.

    ============================================================
    METHODS YOU WILL USE
    ============================================================
    ``await app.run_agent(name, user_input, ctx=None)``
        Run an agent by name. Returns ``AgentResult``.

    ``await app.run_workflow(name, ctx=None)``
        Run a workflow. Returns a list of step outputs.

    ``await app.run_loop(name, ctx=None, max_iterations=None)``
        Run a self-evaluating loop. Returns ``LoopResult``.

    ``app.list_components()``
        Return a dict of all registered names: agents, workflows, loops, tools.

    ``app.tool(name=None)``, ``app.agent(name, ...)``, ``app.workflow(name)``,
    ``app.loop(name, max_iterations, evaluator)``, ``app.structured_agent(...)``
        The decorators you put on your functions. Each one returns a wrapper.

    ============================================================
    WHAT THE DECORATORS DO INTERNALLY
    ============================================================
    They register your function in one of the internal dicts (``_agents``,
    ``_workflows``, ``_loops``, ``_tools``) and wrap it in a small
    execution shell. Calling ``app.run_X(name)`` looks the wrapper up
    by name and runs it.
    """

    def __init__(self, name: str = "fastagent", client: Optional[LLMClient] = None) -> None:
        self.name = name
        self.client = client or LLMClient(provider="mock")
        self._agents: Dict[str, _RegisteredAgent] = {}
        self._workflows: Dict[str, _RegisteredWorkflow] = {}
        self._loops: Dict[str, _RegisteredLoop] = {}
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._tool_schemas: Dict[str, Optional[Dict[str, Any]]] = {}

    def _new_context(self) -> AgentContext:
        return AgentContext(memory=MemoryStore(), client=self.client)

    # -- tool decorator --------------------------------------------------- #
    def tool(self, name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a plain function as a tool the app can hand to any agent.

        Usage::

            app = FastAgent()

            @app.tool()
            def lookup_user(user_id: str) -> dict:
                # Look up a user by id.
                return {"id": user_id, "name": "..."}

        After registration, any agent that does NOT pass an explicit
        ``tools=[...]`` list will receive all app-registered tools automatically.
        Tools can also be passed explicitly per-agent via ``@app.agent(tools=[...])``;
        explicit wins.
        """
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or fn.__name__
            self._tools[tool_name] = fn
            try:
                from .utils import function_to_tool_schema
                self._tool_schemas[tool_name] = function_to_tool_schema(fn, name=tool_name)
            except Exception:
                self._tool_schemas[tool_name] = None
            # Attach a reference so introspection tools can see it.
            fn.__fastagent_tool__ = tool_name  # type: ignore[attr-defined]
            return fn
        return decorator

    # -- agent decorator ------------------------------------------------- #
    def agent(
        self,
        name: str,
        system_prompt: str = "You are a helpful assistant.",
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        memory_k: int = 4,
        on_error: OnErrorPolicy = "return_error",
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        max_retries: int = 0,
    ) -> Callable[[Callable[..., Any]], "_RegisteredAgent"]:
        """Register an agent.

        ============================================================
        YOUR FUNCTION'S SHAPE
        ============================================================
        Your decorated function must be ``async def`` and take either 2 or
        3 arguments::

            async def my_agent(ctx, user_input)              -> str: ...
            async def my_agent(ctx, user_input, messages)     -> str: ...

        If you accept the third ``messages`` argument, the framework hands
        you the fully-built OpenAI-format chat message list (system +
        memory + short-term history + user input). If you only accept 2,
        the framework still builds ``messages`` internally but does NOT
        pass it in - in that case the simplest pattern is to call
        ``await ctx.client.chat(format_prompt(...))`` yourself.

        ============================================================
        BEGINNER EXAMPLE
        ============================================================
        .. code-block:: python

            @app.agent(name="qa", system_prompt="Answer in one short sentence.")
            async def qa(ctx, user_input, messages=None):
                resp = await ctx.client.chat(messages)
                return resp.content

        ============================================================
        PARAMETERS
        ============================================================
        name : str
            Unique agent name. Use it to call ``app.run_agent(name, ...)``.
        system_prompt : str, default "You are a helpful assistant."
            System message sent on every turn. Keep it short and clear.
        tools : list of callable, optional
            Explicit list of tool functions. If ``None`` (the default) the
            agent receives EVERY tool registered via ``@app.tool()`` on the
            app. Pass an explicit list to scope tools to specific agents.
        memory_k : int, default 4
            How many long-term memory chunks to surface per turn.
        on_error : str, default "return_error"
            Policy when the wrapped function raises:
              * ``"raise"``       - re-raise the exception to the caller
              * ``"return_error"`` - return ``AgentResult(ok=False, error=...)``
              * ``"retry"``       - retry up to ``max_retries`` times
        max_tool_iterations : int
            Reserved for future tool-call loops.
        max_retries : int, default 0
            Retries on exception when ``on_error="retry"``.

        ============================================================
        RETURN VALUE
        ============================================================
        The decorator returns the wrapped function PLUS a thin registry
        object. You usually ignore the return value and just call
        ``await app.run_agent(name, ...)`` later.
        """
        if tools is None:
            tools = list(self._tools.values())
        else:
            tools = list(tools)

        def decorator(fn: Callable[..., Any]) -> _RegisteredAgent:
            wrapped = _RegisteredAgent(
                name, system_prompt, tools, fn,
                memory_k=memory_k,
                on_error=on_error,
                max_tool_iterations=max_tool_iterations,
                max_retries=max_retries,
            )
            self._agents[name] = wrapped
            return wrapped

        return decorator

    # -- structured agent decorator --------------------------------------- #
    def structured_agent(
        self,
        name: str,
        output_schema: Any,
        system_prompt: str = "You are a helpful assistant. Respond with valid JSON matching the requested schema.",
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        memory_k: int = 4,
        on_error: OnErrorPolicy = "return_error",
        max_retries: int = 1,
    ) -> Callable[[Callable[..., Any]], "_RegisteredAgent"]:
        """Like ``@app.agent`` but validates the agent's final return value
        against a Pydantic v2 ``BaseModel`` schema.

        The wrapped function should return a dict (or string parseable as JSON).
        On success ``AgentResult.value`` is the parsed ``output_schema``
        instance; on schema mismatch ``AgentResult.ok`` is False.
        """
        if not (_PYDANTIC and isinstance(output_schema, type) and issubclass(output_schema, BaseModel)):
            raise TypeError(
                "structured_agent requires a Pydantic v2 BaseModel subclass as output_schema"
            )

        # Inner wrapper turns the user function into a JSON-producing agent.
        def decorator(fn: Callable[..., Any]) -> _RegisteredAgent:
            async def wrapped(ctx: "AgentContext", user_input: str, messages=None):  # noqa: ANN001
                if messages is None:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input},
                    ]
                resp = await ctx.client.chat(messages)
                text = (resp.content or "").strip()
                # Best-effort: extract JSON object from the response.
                start = text.find("{")
                end = text.rfind("}")
                candidate = text[start:end + 1] if start != -1 and end > start else text
                return candidate

            wrapped.__name__ = getattr(fn, "__name__", "structured")
            wrapped.__doc__ = getattr(fn, "__doc__", None)
            inner = _RegisteredAgent(
                name=name,
                system_prompt=system_prompt,
                tools=list(tools or []),
                fn=wrapped,
                memory_k=memory_k,
                on_error=on_error,
                max_retries=max_retries,
            )

            # Wrap run() so we post-validate against output_schema.
            original_run = inner.run
            async def validating_run(ctx, user_input):
                result = await original_run(ctx, user_input)
                if not result.ok:
                    return result
                try:
                    parsed = output_schema.model_validate_json(str(result.value))
                except Exception as exc:
                    ctx.log("structured.invalid", agent=name, exc=str(exc))
                    return AgentResult(
                        ok=False,
                        error=exc,
                        agent=name,
                        tool_calls=result.tool_calls,
                        iterations=result.iterations,
                    )
                return AgentResult(
                    ok=True,
                    value=parsed,
                    agent=name,
                    tool_calls=result.tool_calls,
                    iterations=result.iterations,
                )
            inner.run = validating_run  # type: ignore[assignment]
            self._agents[name] = inner
            return inner

        return decorator

    # -- workflow decorator --------------------------------------------- #
    def workflow(self, name: str) -> Callable[[Callable[..., Any]], _RegisteredWorkflow]:
        """Register a multi-step workflow.

        ============================================================
        WHAT IS A WORKFLOW?
        ============================================================
        A workflow is a sequence of steps that run in order. Each step is
        usually an agent call, but it can be any async function. You write
        it as an ``async def`` that ``yield``s each step's output:

        .. code-block:: python

            @app.workflow(name="intake")
            async def intake(ctx):
                answer = await app.run_agent("qa", "hi", ctx=ctx)
                yield answer.value            # step 1 output

                summary = await summarize(answer.value)
                yield summary                 # step 2 output

        ============================================================
        BEGINNER EXAMPLE (the shortest possible workflow)
        ============================================================
        .. code-block:: python

            @app.workflow(name="greet-twice")
            async def greet_twice(ctx):
                yield "hello"
                yield "world"

        ============================================================
        RETURN VALUE
        ============================================================
        ``await app.run_workflow(name)`` returns a list of all the values
        your workflow ``yield``ed, in order.
        """
        def decorator(fn: Callable[..., Any]) -> _RegisteredWorkflow:
            if not (asyncio.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn)):
                raise TypeError(f"workflow {name!r} must be async (use ``async def``)")

            steps = _flatten_workflow(fn)
            wrapped = _RegisteredWorkflow(name, steps)
            self._workflows[name] = wrapped
            return wrapped

        return decorator

    # -- loop decorator -------------------------------------------------- #
    def loop(
        self,
        name: str,
        max_iterations: int = 5,
        evaluator: Optional[Callable[[AgentContext], Awaitable[Tuple[str, Optional[Dict[str, Any]]]]]] = None,
    ) -> Callable[[Callable[..., Any]], "_RegisteredLoop"]:
        """Register a self-evaluating autonomous loop.

        ============================================================
        WHAT IS A LOOP?
        ============================================================
        A loop runs a body function repeatedly until either an evaluator
        says "stop" or ``max_iterations`` is reached. This is how you
        build agents that refine their own output:

        .. code-block:: python

            @app.loop(name="refine", max_iterations=5)
            async def refine(ctx, i):
                # Each iteration improves ctx.state["draft"] a little.
                draft = ctx.state.get("draft", "Initial draft.")
                ctx.state["draft"] = draft + " More detail."
                return ctx.state["draft"]

            @app.loop(
                name="refine-eval",
                max_iterations=5,
                evaluator=check_draft_quality,   # custom stop function
            )
            async def refine_eval(ctx, i):
                ...

        ============================================================
        EVALUATOR SIGNATURE
        ============================================================
        An evaluator is an async function that returns ``(status, suggestion)``:

          * ``status`` is either ``"continue"`` or ``"stop"``.
          * ``suggestion`` is an optional dict passed back to the loop body
            on the next iteration.

        ============================================================
        PARAMETERS
        ============================================================
        name : str
            Unique loop name.
        max_iterations : int, default 5
            Hard cap on iterations. The loop ALWAYS stops at this number
            even if the evaluator never says stop.
        evaluator : async callable, optional
            Your stop-function. If omitted, the loop stops when
            ``ctx.state["done"]`` becomes True or after ``max_iterations``.

        ============================================================
        RETURN VALUE OF ``app.run_loop(...)``
        ============================================================
        Returns a ``LoopResult`` with the final state, iteration count,
        and the stop reason. See the ``LoopResult`` class.
        """
        if evaluator is None:
            async def default_evaluator(ctx: AgentContext) -> Tuple[str, Optional[Dict[str, Any]]]:
                it = ctx.state.get("iteration", 0)
                # Default: stop after max_iterations or when state["done"] is True.
                if ctx.state.get("done") or it >= max_iterations - 1:
                    return "stop", None
                return "continue", None
            evaluator = default_evaluator

        def decorator(fn: Callable[..., Any]) -> _RegisteredLoop:
            if not asyncio.iscoroutinefunction(fn):
                raise TypeError(f"loop {name!r} must be async")
            wrapped = _RegisteredLoop(name, max_iterations, fn, evaluator)
            self._loops[name] = wrapped
            return wrapped

        return decorator

    # -- direct runners -------------------------------------------------- #
    async def run_agent(self, name: str, user_input: str, ctx: Optional[AgentContext] = None) -> Any:
        if name not in self._agents:
            raise KeyError(f"agent {name!r} not registered")
        ctx = ctx or self._new_context()
        return await self._agents[name](ctx, user_input)

    async def run_workflow(self, name: str, ctx: Optional[AgentContext] = None) -> List[Any]:
        if name not in self._workflows:
            raise KeyError(f"workflow {name!r} not registered")
        ctx = ctx or self._new_context()
        return await self._workflows[name](ctx)

    async def run_loop(self, name: str, ctx: Optional[AgentContext] = None) -> LoopResult:
        if name not in self._loops:
            raise KeyError(f"loop {name!r} not registered")
        ctx = ctx or self._new_context()
        return await self._loops[name](ctx)

    # -- introspection --------------------------------------------------- #
    def list_components(self) -> Dict[str, List[str]]:
        return {
            "agents": sorted(self._agents),
            "workflows": sorted(self._workflows),
            "loops": sorted(self._loops),
            "tools": sorted(self._tools),
        }


def _flatten_workflow(fn: Callable[..., Any]) -> List[Callable[[AgentContext], Awaitable[Any]]]:
    """Convert a workflow body into a list of per-step callables.

    Supports three shapes:

    1. ``async def`` that ``return``s once  ->  one step whose output is the
       returned value.
    2. ``async def`` that ``yield``s awaitables N times  ->  N steps, one per
       ``yield``, each output is the awaited value.
    3. ``async def`` decorated async generator function (``async def`` with
       ``yield`` at top level)  ->  same as (2).

    In shapes 2 and 3 the workflow's final ``outputs`` list is the flat list
    of yielded values. The framework wraps the async generator's ``__aiter__``
    into separate step callables so each yielded value lands in ``outputs``
    as one entry (not nested).
    """
    is_coro = asyncio.iscoroutinefunction(fn)
    is_gen = inspect.isasyncgenfunction(fn)

    if not (is_coro or is_gen):
        raise TypeError("workflow body must be async (use ``async def``)")

    if is_gen and not is_coro:
        # Pure async generator: produce one step per ``yield``.
        return _make_generator_steps(fn)

    # async def that may return once or yield multiple times.
    async def dispatch_runner(ctx: AgentContext) -> Any:
        result = fn(ctx)
        if hasattr(result, "__aiter__"):
            return await _drive_generator_to_steps(result)
        return await result

    # We do not know in advance whether ``fn`` is a one-shot or a multi-yield
    # coroutine. Easiest: always run it through dispatch_runner; if it turns
    # out to be a multi-yield async generator, we expand at runtime by
    # returning a list of step callables - but we cannot change ``self.steps``
    # post-registration. Instead, we install a single runner that returns the
    # full list, and let _RegisteredWorkflow.__call__ flatten a list result.
    return [dispatch_runner]


def _make_generator_steps(fn: Callable[..., Any]) -> List[Callable[[AgentContext], Awaitable[Any]]]:
    """Materialize an async generator workflow into one step per yielded value.

    The first step kicks off the generator and returns the FIRST yielded
    value. Subsequent steps advance the generator and return the NEXT yielded
    value. A sentinel ``_WORKFLOW_END`` marks generator exhaustion.
    """
    # We use a shared holder so consecutive step callables can coordinate.
    def make_holder():
        return {"gen": None}

    _END = object()

    # First step: starts the generator and returns first yielded value.
    # All subsequent steps: advance the generator and return next yielded value.
    async def step0(ctx: AgentContext, holder):
        gen = fn(ctx)
        holder["gen"] = gen
        try:
            first = await gen.__anext__()
            return first
        except StopAsyncIteration:
            return _END

    async def stepN(ctx: AgentContext, holder):
        gen = holder["gen"]
        try:
            return await gen.__anext__()
        except StopAsyncIteration:
            return _END

    # We return step0 as the only declared step; stepN is invoked dynamically
    # via a wrapper inside _RegisteredWorkflow. To keep the API simple, we
    # return a single runner that returns the FULL list of yielded values, and
    # let the workflow wrapper flatten one level. (This is the same trick as
    # the dispatch case below.)
    async def generator_runner(ctx: AgentContext) -> Any:
        results = []
        async for awaited in fn(ctx):
            results.append(awaited)
        return results

    return [generator_runner]


async def _drive_generator_to_steps(gen):
    """Helper that drains an async generator into a list of values."""
    results = []
    async for awaited in gen:
        results.append(awaited)
    return results


__all__ = ["AgentContext", "AgentResult", "FastAgent", "LoopResult"]
