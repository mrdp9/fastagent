"""FastAgent framework test suite.

Run with::

    pytest -v tests/

The suite is intentionally comprehensive: every public class and every
decorator is exercised at least once, and the integration tests mirror the
flow shown in app.py.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Make the project root importable regardless of where pytest is run from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastagent import (  # noqa: E402
    AgentContext,
    AgentResult,
    FastAgent,
    LoopResult,
    MemoryHit,
    MemoryStore,
    ShortTermContext,
    Message,
    LLMClient,
    ChatResponse,
    function_to_tool_schema,
    format_prompt,
    safe_run,
)
from fastagent.memory import _Matrix  # noqa: E402


# --------------------------------------------------------------------------- #
# Memory layer
# --------------------------------------------------------------------------- #
class TestShortTermContext:
    def test_basic_add_and_messages(self):
        st = ShortTermContext()
        st.add("user", "hello")
        st.add("assistant", "hi")
        msgs = st.messages()
        assert len(msgs) == 2
        assert msgs[0].role == "user" and msgs[0].content == "hello"
        assert msgs[1].role == "assistant" and msgs[1].content == "hi"

    def test_invalid_role_raises(self):
        st = ShortTermContext()
        with pytest.raises(ValueError):
            st.add("bot", "nope")

    def test_max_messages_trims_oldest(self):
        st = ShortTermContext(max_messages=3)
        for i in range(5):
            st.add("user", "msg-{}".format(i))
        msgs = st.messages()
        assert len(msgs) == 3
        assert [m.content for m in msgs] == ["msg-2", "msg-3", "msg-4"]

    def test_clear(self):
        st = ShortTermContext()
        st.add("user", "x")
        st.clear()
        assert len(st) == 0

    def test_thread_safety(self):
        # big enough to hold all 800 messages from the workers below.
        st = ShortTermContext(max_messages=10_000)
        errors = []

        def worker():
            try:
                for _ in range(100):
                    st.add("user", "x")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(st) == 800


class TestMemoryStore:
    @pytest.mark.asyncio
    async def test_add_returns_record_id(self):
        s = MemoryStore()
        rid = await s.add("The CEO is Priya.", {"tag": "people"})
        assert isinstance(rid, str) and len(rid) > 0
        rec = s.get(rid)
        assert rec is not None
        assert rec["text"] == "The CEO is Priya."
        assert rec["metadata"]["tag"] == "people"

    @pytest.mark.asyncio
    async def test_empty_text_rejected(self):
        s = MemoryStore()
        with pytest.raises(ValueError):
            await s.add("")
        with pytest.raises(ValueError):
            await s.add("   ")

    @pytest.mark.asyncio
    async def test_search_empty_store_returns_empty(self):
        s = MemoryStore()
        assert await s.search("anything") == []

    @pytest.mark.asyncio
    async def test_search_empty_query_returns_empty(self):
        s = MemoryStore()
        await s.add("hello world")
        result = await s.search("")
        assert result == []

    @pytest.mark.asyncio
    async def test_search_returns_relevant_top_hit(self):
        # The offline (no-API) embedder is token-overlap driven, so we pick
        # queries that share distinctive tokens with the answer.
        s = MemoryStore()
        docs = [
            "The CEO of Acme is Priya Sharma.",
            "The CTO of Acme is Daniel Okafor.",
            "Q3 revenue was 142 million USD.",
            "Refund window is 30 days.",
            "Office hours are 9am-6pm IST.",
        ]
        for d in docs:
            await s.add(d)
        for q, expected_substring in [
            ("Who is the CEO?", "CEO"),
            ("Who is the CTO?", "CTO"),
            ("What was Q3 revenue?", "revenue"),
            ("Refund window length?", "Refund"),
            ("Office hours today?", "Office"),
        ]:
            hits = await s.search(q, limit=3)
            assert hits, "no hits for {!r}".format(q)
            assert expected_substring in hits[0].text, \
                "query {!r} top hit was {!r}".format(q, hits[0].text)

    @pytest.mark.asyncio
    async def test_search_score_is_finite(self):
        s = MemoryStore()
        await s.add("alpha bravo charlie")
        hits = await s.search("alpha", limit=1)
        assert len(hits) == 1
        assert isinstance(hits[0].score, float)
        assert hits[0].score == hits[0].score  # not NaN

    @pytest.mark.asyncio
    async def test_clear_long_term(self):
        s = MemoryStore()
        await s.add("alpha")
        await s.add("bravo")
        assert len(s) == 2
        await s.clear_long_term()
        assert len(s) == 0
        assert await s.search("alpha") == []

    @pytest.mark.asyncio
    async def test_custom_embed_fn(self):
        def embed(t):
            return [float(len(t)), 1.0, 0.0, 0.0]
        s = MemoryStore(embed_fn=embed)
        await s.add("ab")
        await s.add("xyz")
        hits = await s.search("abc", limit=2)
        assert len(hits) == 2

    @pytest.mark.asyncio
    async def test_async_embed_fn_supported(self):
        async def embed(t):
            return [float(len(t)), 0.0, 0.0, 0.0]
        s = MemoryStore(embed_fn=embed)
        await s.add("alpha")
        hits = await s.search("a", limit=1)
        assert len(hits) == 1

    def test_short_term_is_accessible(self):
        s = MemoryStore()
        s.short_term.add("user", "x")
        assert len(s.short_term) == 1


class TestMatrixShim:
    def test_shape_empty(self):
        m = _Matrix()
        assert m.shape == (0, 0)

    def test_shape_zeros(self):
        m = _Matrix()
        m.append([0.0] * 4)
        m.append([0.0] * 4)
        assert m.shape == (2, 4)

    def test_transpose(self):
        m = _Matrix()
        m.append([1.0, 2.0, 3.0])
        m.append([4.0, 5.0, 6.0])
        T = m.T
        assert T.shape == (3, 2)
        assert T[0] == [1.0, 4.0]
        assert T[1] == [2.0, 5.0]
        assert T[2] == [3.0, 6.0]

    def test_flatten(self):
        m = _Matrix()
        m.append([1.0, 2.0])
        m.append([3.0, 4.0])
        f = m.flatten()
        assert f.shape == (1, 4)
        assert f[0] == [1.0, 2.0, 3.0, 4.0]


# --------------------------------------------------------------------------- #
# Utils layer
# --------------------------------------------------------------------------- #
class TestFunctionToToolSchema:
    def test_basic_function(self):
        def get_weather(city: str) -> dict:
            """Return weather for a city."""
            return {}
        schema = function_to_tool_schema(get_weather)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert "city" in schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["required"] == ["city"]

    def test_defaults_and_types(self):
        def search(q: str, limit: int = 10, flag: bool = False, score: float = 0.5):
            """Search things."""
            return []
        s = function_to_tool_schema(search)
        props = s["function"]["parameters"]["properties"]
        assert props["q"]["type"] == "string"
        assert props["limit"]["type"] == "integer"
        assert props["limit"]["default"] == 10
        assert props["flag"]["type"] == "boolean"
        assert props["flag"]["default"] is False
        assert props["score"]["type"] == "number"
        assert s["function"]["parameters"]["required"] == ["q"]

    def test_google_docstring_param_descriptions(self):
        def schedule(attendees: list, topic: str, duration_min: int = 30) -> dict:
            """Schedule a meeting.

            Args:
                attendees: list of people
                topic: meeting subject
                duration_min: length in minutes
            """
            return {}
        s = function_to_tool_schema(schedule)
        props = s["function"]["parameters"]["properties"]
        assert props["attendees"]["description"] == "list of people"
        assert props["topic"]["description"] == "meeting subject"
        assert props["duration_min"]["description"] == "length in minutes"

    def test_sphinx_docstring_param_descriptions(self):
        def f(x: int, y: str) -> None:
            """Do thing.

            :param x: an int
            :param y: a string
            """
            return None
        s = function_to_tool_schema(f)
        props = s["function"]["parameters"]["properties"]
        assert props["x"]["description"] == "an int"
        assert props["y"]["description"] == "a string"

    def test_self_and_cls_skipped(self):
        class C:
            def m(self, x: int) -> int:
                """Method m."""
                return x
        s = function_to_tool_schema(C.m)
        assert "self" not in s["function"]["parameters"]["properties"]
        assert s["function"]["parameters"]["required"] == ["x"]

    def test_optional_type(self):
        from typing import Optional
        def f(x: Optional[int] = None) -> int:
            """Maybe-int input."""
            return 0
        s = function_to_tool_schema(f)
        assert s["function"]["parameters"]["properties"]["x"]["type"] == "integer"

    def test_list_of_strings(self):
        from typing import List
        def f(items: List[str]) -> int:
            """Count items."""
            return 0
        s = function_to_tool_schema(f)
        assert s["function"]["parameters"]["properties"]["items"]["type"] == "array"

    def test_no_docstring_ok(self):
        def f(x: int) -> int:
            return x
        s = function_to_tool_schema(f)
        assert s["function"]["description"].startswith("Tool:")

    def test_non_callable_raises(self):
        with pytest.raises(TypeError):
            function_to_tool_schema(42)


class TestFormatPrompt:
    def test_system_only(self):
        msgs = format_prompt(system_prompt="be helpful")
        assert msgs == [{"role": "system", "content": "be helpful"}]

    def test_user_only(self):
        msgs = format_prompt(system_prompt="", user_input="hi")
        assert msgs[-1] == {"role": "user", "content": "hi"}

    def test_memory_block(self):
        hits = [MemoryHit(id="x", text="CEO is Priya", metadata={}, score=0.7)]
        msgs = format_prompt(system_prompt="sys", memory_hits=hits)
        assert any("long-term memory" in m["content"] for m in msgs)
        assert any("Priya" in m["content"] for m in msgs)

    def test_short_term_passthrough(self):
        st = [Message(role="user", content="u1"), Message(role="assistant", content="a1")]
        msgs = format_prompt(system_prompt="", short_term=st)
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant"]

    def test_order(self):
        st = [Message(role="user", content="u1")]
        hits = [MemoryHit(id="x", text="h1", metadata={}, score=0.5)]
        msgs = format_prompt(system_prompt="S", short_term=st, memory_hits=hits, user_input="U")
        assert msgs[0]["role"] == "system"
        assert "long-term" in msgs[1]["content"]
        assert msgs[2]["role"] == "user" and msgs[2]["content"] == "u1"
        assert msgs[3]["role"] == "user" and msgs[3]["content"] == "U"


class TestSafeRun:
    @pytest.mark.asyncio
    async def test_sync_success(self):
        ok, val = await safe_run(lambda: 42)
        assert ok is True and val == 42

    @pytest.mark.asyncio
    async def test_async_success(self):
        async def f():
            return "hi"
        ok, val = await safe_run(f)
        assert ok is True and val == "hi"

    @pytest.mark.asyncio
    async def test_exception_caught(self):
        def bad():
            raise RuntimeError("boom")
        ok, val = await safe_run(bad)
        assert ok is False
        assert isinstance(val, RuntimeError)


# --------------------------------------------------------------------------- #
# LLM layer
# --------------------------------------------------------------------------- #
class TestLLMClient:
    @pytest.mark.asyncio
    async def test_mock_provider_chat(self):
        c = LLMClient(provider="mock")
        r = await c.chat([{"role": "user", "content": "hello"}])
        assert isinstance(r, ChatResponse)
        assert "hello" in r.content
        assert r.provider == "mock"

    @pytest.mark.asyncio
    async def test_mock_provider_uses_memory_context(self):
        c = LLMClient(provider="mock")
        r = await c.chat([
            {"role": "system", "content": "Relevant long-term memory:\n  [1] (score=0.9) CEO is Priya."},
            {"role": "user", "content": "Who is the CEO?"},
        ])
        assert "Priya" in r.content or "memory" in r.content.lower()

    @pytest.mark.asyncio
    async def test_mock_embed_dim(self):
        from fastagent.llm import EmbeddingResponse
        c = LLMClient(provider="mock")
        v = await c.embed("anything")
        assert isinstance(v, EmbeddingResponse)
        assert len(v.vectors) == 1
        assert len(v.vectors[0]) == 256

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            LLMClient(provider="no-such-provider")

    @pytest.mark.asyncio
    async def test_http_provider_degrades_when_unreachable(self):
        c = LLMClient(provider="openai", base_url="http://127.0.0.1:1/v1")
        r = await c.chat([{"role": "user", "content": "ping"}])
        assert isinstance(r, ChatResponse)
        assert "ping" in r.content

    @pytest.mark.asyncio
    async def test_chat_supports_tools_argument(self):
        c = LLMClient(provider="mock")
        tools = [{
            "type": "function",
            "function": {"name": "foo", "description": "do foo",
                         "parameters": {"type": "object", "properties": {}}}
        }]
        r = await c.chat([{"role": "user", "content": "use the tool"}], tools=tools)
        assert isinstance(r, ChatResponse)


# --------------------------------------------------------------------------- #
# Core / decorator layer
# --------------------------------------------------------------------------- #
class TestAgentContext:
    def test_default_construction(self):
        ctx = AgentContext()
        assert ctx.state == {}
        assert ctx.history == []
        assert isinstance(ctx.memory, MemoryStore)
        assert isinstance(ctx.client, LLMClient)

    def test_log_appends_history(self):
        ctx = AgentContext()
        ctx.log("x", y=1)
        ctx.log("y")
        assert len(ctx.history) == 2
        assert ctx.history[0]["event"] == "x"
        assert ctx.history[0]["y"] == 1
        assert ctx.history[1]["event"] == "y"

    def test_snapshot(self):
        ctx = AgentContext()
        ctx.log("e")
        snap = ctx.snapshot()
        assert snap["history_len"] == 1
        assert snap["state"] == {}

    def test_thread_id_unique(self):
        a = AgentContext()
        b = AgentContext()
        assert a.thread_id != b.thread_id


class TestFastAgentAgents:
    @pytest.mark.asyncio
    async def test_simple_echo_agent(self):
        app = FastAgent(name="t1")

        @app.agent(name="echo", system_prompt="Echo.")
        async def echo(ctx, user_input, messages=None):
            return "echo:{}".format(user_input)

        ctx = AgentContext(memory=MemoryStore(), client=app.client)
        out = await app.run_agent("echo", "hi", ctx=ctx)
        assert isinstance(out, AgentResult)
        assert out.ok
        assert out.value == "echo:hi"
        assert ctx.agent_name == "echo"
        msgs = ctx.memory.short_term.messages()
        assert [m.role for m in msgs] == ["user", "assistant"]
        events = [h["event"] for h in ctx.history]
        assert "agent.start" in events and "agent.end" in events

    @pytest.mark.asyncio
    async def test_agent_signature_two_args(self):
        app = FastAgent(name="t2")

        @app.agent(name="a2")
        async def a2(ctx, user_input):
            return user_input.upper()

        ctx = AgentContext(memory=MemoryStore(), client=app.client)
        r = await app.run_agent("a2", "hi", ctx=ctx)
        assert r.ok and r.value == "HI"

    @pytest.mark.asyncio
    async def test_agent_uses_long_term_memory(self):
        app = FastAgent(name="t3")

        @app.agent(name="qa", system_prompt="Use memory.", memory_k=2)
        async def qa(ctx, user_input, messages=None):
            sys_blocks = [m["content"] for m in messages if m["role"] == "system"]
            joined = "\n".join(sys_blocks)
            assert "long-term memory" in joined
            return messages[1]["content"].splitlines()[1] if len(messages) > 1 else ""

        ctx = AgentContext(memory=MemoryStore(), client=app.client)
        await ctx.memory.add("The CEO is Priya.")
        await ctx.memory.add("The CTO is Daniel.")
        out = await app.run_agent("qa", "Who is the CEO?", ctx=ctx)
        assert out.ok and "Priya" in out.value

    @pytest.mark.asyncio
    async def test_agent_with_tools(self):
        app = FastAgent(name="t4")

        def lookup(name: str) -> str:
            """Look up an employee."""
            return {"priya": "priya@x"}.get(name.lower(), "unknown")

        @app.agent(name="t4a", tools=[lookup])
        async def t4a(ctx, user_input, messages=None):
            return "have {} short-term msgs".format(len(ctx.memory.short_term))

        ctx = AgentContext(memory=MemoryStore(), client=app.client)
        out = await app.run_agent("t4a", "hi", ctx=ctx)
        assert out.ok and "short-term" in out.value

    @pytest.mark.asyncio
    async def test_run_unknown_agent_raises(self):
        app = FastAgent(name="t5")
        with pytest.raises(KeyError):
            await app.run_agent("nope", "x")

    def test_list_components(self):
        app = FastAgent(name="t6")

        @app.agent(name="a")
        async def a(ctx, x):
            return x

        @app.workflow(name="w")
        async def w(ctx):
            return None

        @app.loop(name="l")
        async def l(ctx, i):
            return None

        comps = app.list_components()
        assert comps == {"agents": ["a"], "workflows": ["w"], "loops": ["l"], "tools": []}


class TestFastAgentWorkflows:
    @pytest.mark.asyncio
    async def test_workflow_records_state(self):
        app = FastAgent(name="tw1")

        async def step_a(ctx):
            ctx.state["a"] = 1
            return 1

        async def step_b(ctx):
            ctx.state["b"] = 2
            return 2

        @app.workflow(name="two-step")
        async def wf(ctx):
            yield await step_a(ctx)
            yield await step_b(ctx)

        ctx = AgentContext(memory=MemoryStore(), client=app.client)
        out = await app.run_workflow("two-step", ctx=ctx)
        assert out == [1, 2]
        events = [h["event"] for h in ctx.history]
        assert "workflow.start" in events
        assert "workflow.end" in events
        # The two-step generator is run as a single step whose output is the
        # full list; the framework flattens it and stores the yielded values
        # under "step_0_0_output" and "step_0_1_output".
        assert ctx.state["step_0_0_output"] == 1
        assert ctx.state["step_0_1_output"] == 2

    @pytest.mark.asyncio
    async def test_workflow_chain_agents(self):
        app = FastAgent(name="tw2")

        @app.agent(name="a1")
        async def a1(ctx, x, messages=None):
            # unwrap AgentResult from previous step so we can string-concat
            if hasattr(x, "value"):
                x = x.value
            return x + "+a1"

        @app.workflow(name="chain")
        async def chain(ctx):
            x = await app.run_agent("a1", "start", ctx=ctx)
            yield x
            y = await app.run_agent("a1", x, ctx=ctx)
            yield y

        ctx = AgentContext(memory=MemoryStore(), client=app.client)
        out = await app.run_workflow("chain", ctx=ctx)
        # Generator-style workflow returns the list of yielded step values.
        # Each yielded value is an AgentResult now.
        assert [r.value for r in out] == ["start+a1", "start+a1+a1"]

    def test_workflow_must_be_async(self):
        app = FastAgent(name="tw3")
        with pytest.raises(TypeError):
            @app.workflow(name="bad")
            def bad(ctx):
                return None


class TestFastAgentLoops:
    @pytest.mark.asyncio
    async def test_loop_stops_on_evaluator_stop(self):
        app = FastAgent(name="tl1")

        async def ev(ctx):
            if ctx.state.get("count", 0) >= 2:
                return "stop", {"reason": "got 2"}
            return "continue", None

        @app.loop(name="refine", max_iterations=10, evaluator=ev)
        async def refine(ctx, i):
            ctx.state["count"] = ctx.state.get("count", 0) + 1

        ctx = AgentContext(memory=MemoryStore(), client=app.client)
        result = await app.run_loop("refine", ctx=ctx)
        assert isinstance(result, LoopResult)
        # iter 0 -> count=1 (continue, 1 < 2); iter 1 -> count=2 (stop, 2 >= 2). 2 iterations total.
        assert result.iterations == 2
        assert result.stopped_reason == "stop"
        assert result.final_state["count"] == 2

    @pytest.mark.asyncio
    async def test_loop_respects_max_iterations(self):
        app = FastAgent(name="tl2")

        @app.loop(name="inf", max_iterations=4)
        async def inf(ctx, i):
            ctx.state.setdefault("iters", []).append(i)

        ctx = AgentContext(memory=MemoryStore(), client=app.client)
        result = await app.run_loop("inf", ctx=ctx)
        assert result.iterations == 4
        assert result.final_state["iters"] == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_loop_default_evaluator_stops_at_max(self):
        app = FastAgent(name="tl3")

        @app.loop(name="d", max_iterations=2)
        async def d(ctx, i):
            return None

        ctx = AgentContext(memory=MemoryStore(), client=app.client)
        result = await app.run_loop("d", ctx=ctx)
        assert result.iterations == 2
        assert result.stopped_reason == "stop"

    def test_loop_must_be_async(self):
        app = FastAgent(name="tl4")
        with pytest.raises(TypeError):
            @app.loop(name="bad")
            def bad(ctx, i):
                return None


# --------------------------------------------------------------------------- #
# Integration tests
# --------------------------------------------------------------------------- #
@pytest.fixture
def knowledge_base():
    return [
        ("The CEO of Acme Corp is Priya Sharma.", {"tag": "people"}),
        ("The CTO of Acme Corp is Daniel Okafor.", {"tag": "people"}),
        ("Q3 revenue was 142 million USD.", {"tag": "finance"}),
        ("All employees get 25 days of paid leave.", {"tag": "policy"}),
    ]


@pytest.mark.asyncio
async def test_end_to_end_seed_and_ask(knowledge_base):
    app = FastAgent(name="int1")

    @app.agent(name="qa", system_prompt="Use memory.", memory_k=3)
    async def qa(ctx, user_input, messages=None):
        ctx_block = next((m["content"] for m in messages if "long-term memory" in m["content"]), "")
        return ctx_block

    ctx = AgentContext(memory=MemoryStore(), client=app.client)
    for text, meta in knowledge_base:
        await ctx.memory.add(text, meta)
    out = await app.run_agent("qa", "Who is the CEO?", ctx=ctx)
    assert out.ok and "Priya" in out.value


@pytest.mark.asyncio
async def test_end_to_end_workflow_with_loop(knowledge_base):
    app = FastAgent(name="int2")

    @app.agent(name="qa", system_prompt="Use memory.", memory_k=2)
    async def qa(ctx, user_input, messages=None):
        ctx_block = next((m["content"] for m in messages if "long-term memory" in m["content"]), "")
        return ctx_block

    async def ev(ctx):
        return ("stop", None) if ctx.state.get("count", 0) >= 1 else ("continue", None)

    @app.loop(name="loop", max_iterations=3, evaluator=ev)
    async def loop_fn(ctx, i):
        ctx.state["count"] = ctx.state.get("count", 0) + 1

    @app.workflow(name="combo")
    async def combo(ctx):
        a = await app.run_agent("qa", "CEO?", ctx=ctx)
        r = await app.run_loop("loop", ctx=ctx)
        return [a, r.iterations]

    ctx = AgentContext(memory=MemoryStore(), client=app.client)
    for text, meta in knowledge_base:
        await ctx.memory.add(text, meta)
    out = await app.run_workflow("combo", ctx=ctx)
    assert out[0].ok and "Priya" in out[0].value
    # count goes 0 -> 1, then stop. 1 iteration total.
    assert out[1] == 1


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_concurrent_writes_to_memory():
    s = MemoryStore()

    async def writer(i):
        await s.add("document number {} about topic {}".format(i, i % 5))

    await asyncio.gather(*(writer(i) for i in range(50)))
    assert len(s) == 50
    hits = await s.search("topic 3", limit=5)
    assert len(hits) >= 1
