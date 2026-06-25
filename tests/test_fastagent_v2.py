"""Tests for FastAgent v2 features: tool decorator, error envelope, structured agents, memory LRU + persistence."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastagent import (  # noqa: E402
    AgentContext,
    AgentResult,
    FastAgent,
    MemoryStore,
    ShortTermContext,
)


# --------------------------------------------------------------------------- #
# @app.tool decorator
# --------------------------------------------------------------------------- #
class TestToolDecorator:
    def test_register_single_tool(self):
        app = FastAgent(name="tool1")

        @app.tool()
        def lookup_user(user_id: str) -> dict:
            """Look up a user."""
            return {"id": user_id}

        assert "lookup_user" in app._tools
        assert "lookup_user" in app._tool_schemas
        schema = app._tool_schemas["lookup_user"]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "lookup_user"

    def test_tool_with_custom_name(self):
        app = FastAgent(name="tool2")

        @app.tool(name="get_employee")
        def _internal(user_id: str) -> dict:
            """Look up an employee."""
            return {"id": user_id}

        # Both the registered name and the underlying function are accessible.
        assert "get_employee" in app._tools
        assert app._tools["get_employee"].__name__ == "_internal"

    def test_agent_auto_pulls_app_tools(self):
        app = FastAgent(name="tool3")

        @app.tool()
        def alpha(x: str) -> str:
            """Alpha tool."""
            return x

        @app.tool()
        def beta(x: str) -> str:
            """Beta tool."""
            return x

        @app.agent(name="a")
        async def a(ctx, user_input, messages=None):
            return "ran"

        agent = app._agents["a"]
        tool_names = sorted(t.__name__ for t in agent.tools)
        assert tool_names == ["alpha", "beta"]

    def test_agent_explicit_tools_override_app_tools(self):
        app = FastAgent(name="tool4")

        @app.tool()
        def app_tool(x: str) -> str:
            """An app-level tool."""
            return x

        def my_only_tool(x: str) -> str:
            """The only tool this agent should see."""
            return x

        @app.agent(name="a", tools=[my_only_tool])
        async def a(ctx, user_input, messages=None):
            return "ran"

        agent = app._agents["a"]
        tool_names = sorted(t.__name__ for t in agent.tools)
        assert tool_names == ["my_only_tool"]

    def test_list_components_includes_tools(self):
        app = FastAgent(name="tool5")

        @app.tool()
        def t1():
            pass

        @app.tool()
        def t2():
            pass

        comps = app.list_components()
        assert "tools" in comps
        assert sorted(comps["tools"]) == ["t1", "t2"]

    @pytest.mark.asyncio
    async def test_tool_can_be_called_directly(self):
        app = FastAgent(name="tool6")

        @app.tool()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert app._tools["add"](2, 3) == 5


# --------------------------------------------------------------------------- #
# AgentResult envelope + on_error policy
# --------------------------------------------------------------------------- #
class TestAgentResultEnvelope:
    @pytest.mark.asyncio
    async def test_successful_run_returns_ok_envelope(self):
        app = FastAgent(name="env1")

        @app.agent(name="good")
        async def good(ctx, user_input, messages=None):
            return "ok"

        ctx = AgentContext(memory=MemoryStore(), client=app.client)
        r = await app.run_agent("good", "hi", ctx=ctx)
        assert isinstance(r, AgentResult)
        assert r.ok is True
        assert r.value == "ok"
        assert r.error is None
        assert r.agent == "good"
        assert bool(r) is True

    @pytest.mark.asyncio
    async def test_exception_returned_as_envelope_by_default(self):
        app = FastAgent(name="env2")

        @app.agent(name="bad")
        async def bad(ctx, user_input, messages=None):
            raise ValueError("intentional")

        ctx = AgentContext(memory=MemoryStore(), client=app.client)
        r = await app.run_agent("bad", "hi", ctx=ctx)
        assert isinstance(r, AgentResult)
        assert r.ok is False
        assert isinstance(r.error, ValueError)
        assert "intentional" in str(r.error)
        assert bool(r) is False

    @pytest.mark.asyncio
    async def test_on_error_raise_re_raises(self):
        app = FastAgent(name="env3")

        @app.agent(name="bad", on_error="raise")
        async def bad(ctx, user_input, messages=None):
            raise ValueError("nope")

        ctx = AgentContext(memory=MemoryStore(), client=app.client)
        with pytest.raises(ValueError, match="nope"):
            await app.run_agent("bad", "hi", ctx=ctx)

    @pytest.mark.asyncio
    async def test_on_error_retries_then_returns_error(self):
        app = FastAgent(name="env4")

        @app.agent(name="flaky", on_error="retry", max_retries=2)
        async def flaky(ctx, user_input, messages=None):
            raise RuntimeError("boom")

        ctx = AgentContext(memory=MemoryStore(), client=app.client)
        r = await app.run_agent("flaky", "hi", ctx=ctx)
        assert r.ok is False
        # The error log records 1 initial attempt + 2 retries = 3 entries.
        err_events = [h for h in ctx.history if h["event"] == "agent.error"]
        assert len(err_events) == 3

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_first_failure(self):
        app = FastAgent(name="env5")

        attempts = {"n": 0}

        @app.agent(name="sometimes", on_error="retry", max_retries=2)
        async def sometimes(ctx, user_input, messages=None):
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise RuntimeError("first try fails")
            return "second try works"

        ctx = AgentContext(memory=MemoryStore(), client=app.client)
        r = await app.run_agent("sometimes", "hi", ctx=ctx)
        assert r.ok is True
        assert r.value == "second try works"
        assert attempts["n"] == 2

    def test_unwrap_helper(self):
        ok = AgentResult(ok=True, value=42)
        assert ok.unwrap() == 42

        bad = AgentResult(ok=False, error=ValueError("x"))
        with pytest.raises(ValueError, match="x"):
            bad.unwrap()


# --------------------------------------------------------------------------- #
# Tool-call execution loop
# --------------------------------------------------------------------------- #
class TestToolExecution:
    @pytest.mark.asyncio
    async def test_execute_tool_calls_runs_registered_tools(self):
        app = FastAgent(name="texec1")

        @app.tool()
        def square(n: int) -> int:
            """Square a number."""
            return n * n

        @app.agent(name="calc", tools=[app._tools["square"]])
        async def calc(ctx, user_input, messages=None):
            # Manually invoke the executor with synthetic tool_calls.
            tcs = [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "square", "arguments": json.dumps({"n": 7})},
            }]
            results = await app._agents["calc"].execute_tool_calls(ctx, "x", tcs)
            assert len(results) == 1
            assert results[0]["role"] == "tool"
            assert results[0]["tool_call_id"] == "call_1"
            assert "49" in results[0]["content"]
            return "done"

        ctx = AgentContext(memory=MemoryStore(), client=app.client)
        r = await app.run_agent("calc", "test", ctx=ctx)
        assert r.ok
        # The tool execution should have been logged in history.
        tool_events = [h for h in ctx.history if h["event"] == "tool.call"]
        assert len(tool_events) == 1
        assert tool_events[0]["tool"] == "square"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_envelope(self):
        app = FastAgent(name="texec2")

        @app.agent(name="a")
        async def a(ctx, user_input, messages=None):
            tcs = [{"id": "x", "type": "function",
                    "function": {"name": "nope", "arguments": "{}"}}]
            results = await app._agents["a"].execute_tool_calls(ctx, "x", tcs)
            assert "error" in results[0]["content"]
            return results

        ctx = AgentContext(memory=MemoryStore(), client=app.client)
        r = await app.run_agent("a", "x", ctx=ctx)
        assert r.ok

    @pytest.mark.asyncio
    async def test_tool_exception_caught(self):
        app = FastAgent(name="texec3")

        @app.tool()
        def bad_tool(x: int) -> int:
            """Always raises."""
            raise RuntimeError("tool boom")

        @app.agent(name="a")
        async def a(ctx, user_input, messages=None):
            tcs = [{"id": "x", "type": "function",
                    "function": {"name": "bad_tool", "arguments": json.dumps({"x": 1})}}]
            results = await app._agents["a"].execute_tool_calls(ctx, "x", tcs)
            assert "tool boom" in results[0]["content"]
            return "ok"

        ctx = AgentContext(memory=MemoryStore(), client=app.client)
        r = await app.run_agent("a", "x", ctx=ctx)
        assert r.ok


# --------------------------------------------------------------------------- #
# @app.structured_agent
# --------------------------------------------------------------------------- #
class TestStructuredAgent:
    @pytest.mark.asyncio
    async def test_parses_valid_json_response(self):
        app = FastAgent(name="struct1")

        class Answer(BaseModel):
            name: str
            value: int

        # Patch the mock LLM client to return a valid JSON string.
        original_chat = app.client.chat

        async def chat_with_json(messages, tools=None, **kwargs):
            from fastagent.llm import ChatResponse
            return ChatResponse(
                content=json.dumps({"name": "answer", "value": 42}),
                tool_calls=[],
                raw=None,
                model=app.client.model,
                provider="mock",
            )
        app.client.chat = chat_with_json

        @app.structured_agent(name="a", output_schema=Answer)
        async def a(ctx, user_input, messages=None):
            return None  # structured_agent uses the LLM directly, not the body

        ctx = AgentContext(memory=MemoryStore(), client=app.client)
        r = await app.run_agent("a", "test", ctx=ctx)
        assert r.ok
        assert isinstance(r.value, Answer)
        assert r.value.name == "answer"
        assert r.value.value == 42

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error_envelope(self):
        app = FastAgent(name="struct2")

        class Answer(BaseModel):
            name: str

        from fastagent.llm import ChatResponse
        async def chat_bad(messages, tools=None, **kwargs):
            return ChatResponse(content="not json at all",
                                tool_calls=[], raw=None,
                                model=app.client.model, provider="mock")
        app.client.chat = chat_bad

        @app.structured_agent(name="a", output_schema=Answer)
        async def a(ctx, user_input, messages=None):
            return None

        ctx = AgentContext(memory=MemoryStore(), client=app.client)
        r = await app.run_agent("a", "test", ctx=ctx)
        assert r.ok is False

    def test_requires_basemodel(self):
        app = FastAgent(name="struct3")
        with pytest.raises(TypeError):
            @app.structured_agent(name="a", output_schema=dict)
            async def a(ctx, user_input, messages=None):
                return None


# --------------------------------------------------------------------------- #
# Memory: max_records LRU eviction + JSONL persistence
# --------------------------------------------------------------------------- #
class TestMemoryLRU:
    @pytest.mark.asyncio
    async def test_evicts_oldest_when_cap_exceeded(self):
        s = MemoryStore(max_records=3)
        for i in range(5):
            await s.add("doc {} about alpha".format(i))
        assert len(s) == 3

    @pytest.mark.asyncio
    async def test_search_bumps_lru(self):
        s = MemoryStore(max_records=3)
        ids = []
        for i in range(3):
            rid = await s.add("doc {} about alpha".format(i))
            ids.append(rid)
        # Search for something that matches the first document; it should
        # move to most-recently-used.
        await s.search("doc 0 alpha", limit=1)
        # Add two more - should evict the SECOND-OLDEST (doc 1), not doc 0.
        await s.add("doc 3")
        await s.add("doc 4")
        remaining = sorted(rec["text"] for rec in s._records.values())
        assert any("doc 0" in t for t in remaining), "doc 0 should still be present"
        assert not any("doc 1" in t for t in remaining), "doc 1 should be evicted"

    @pytest.mark.asyncio
    async def test_unbounded_when_max_records_none(self):
        s = MemoryStore()
        for i in range(100):
            await s.add("doc {}".format(i))
        assert len(s) == 100


class TestMemoryJSONLPersistence:
    @pytest.mark.asyncio
    async def test_save_and_load_round_trip(self):
        s1 = MemoryStore()
        await s1.add("the CEO is Priya", {"tag": "people"})
        await s1.add("Q3 revenue 142M", {"tag": "finance"})
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            n = await s1.save_jsonl(path)
            assert n == 2
            s2 = MemoryStore()
            loaded = await s2.load_jsonl(path)
            assert loaded == 2
            assert len(s2) == 2
            hits = await s2.search("Priya", limit=1)
            assert hits and "Priya" in hits[0].text
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_load_missing_file_returns_zero(self):
        s = MemoryStore()
        n = await s.load_jsonl("/nonexistent/path/file.jsonl")
        assert n == 0

    @pytest.mark.asyncio
    async def test_load_appends_incremental(self):
        s = MemoryStore()
        await s.add("doc 1")
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            path = f.name
            f.write(json.dumps({"text": "doc 2", "metadata": {}}) + chr(10))
            f.write(json.dumps({"text": "doc 3", "metadata": {}}) + chr(10))
        try:
            n = await s.load_jsonl(path)
            assert n == 2
            assert len(s) == 3  # original + 2 loaded
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_save_creates_valid_jsonl(self):
        s = MemoryStore()
        await s.add("hello", {"tag": "greeting"})
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            await s.save_jsonl(path)
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
            assert len(lines) == 1
            row = json.loads(lines[0])
            assert row["text"] == "hello"
            assert row["metadata"]["tag"] == "greeting"
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
