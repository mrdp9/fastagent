"""fastagent - decorator-driven AI & memory framework.

============================================================
WHAT IS FASTAGENT? (read this first if you are new)
============================================================

FastAgent is a tiny Python framework for building AI agents. The whole
point is to let you write agents like normal Python functions - no YAML,
no DSL, no special "agent runtime" to learn.

In five minutes you can write::

    import asyncio
    from fastagent import FastAgent, AgentContext, MemoryStore

    app = FastAgent(name="hello")

    @app.agent(name="greeter", system_prompt="Greet the user warmly.")
    async def greeter(ctx, user_input, messages=None):
        return "Hello, " + user_input + "!"

    print(asyncio.run(app.run_agent("greeter", "world")).value)

============================================================
WHAT YOU CAN IMPORT (the entire public API)
============================================================

Beginner essentials::

    FastAgent       - the main app object. Instantiate ONE per app.
    AgentContext    - the shared state object every agent sees.
    AgentResult     - the envelope every agent returns.
    LoopResult      - the envelope every loop returns.
    MemoryStore     - short + long term memory in one object.

Building blocks (you will use these too)::

    Message         - one message in short-term history
    MemoryHit       - one result from long-term search
    ShortTermContext- just the chat-history part (advanced)
    LLMClient       - the LLM router (default = offline mock)
    ChatResponse    - what ``await client.chat(...)`` returns
    EmbeddingResponse - what ``await client.embed(...)`` returns

Helpers::

    function_to_tool_schema - turn a function into a JSON tool schema
    format_prompt           - build a chat messages list (or substitute a template)
    safe_run                - run a function, return (ok, value)
    SkipSchema              - marker: hide a parameter from the JSON schema

============================================================
TYPING-FRIENDLY IMPORTS
============================================================

You can import in either style::

    from fastagent import FastAgent               # flat (recommended for beginners)
    from fastagent.core import FastAgent          # explicit module path

Both work. The flat style is shorter; the explicit style makes it clear
where each symbol lives when you read code later.

============================================================
DEPENDENCIES
============================================================

* pydantic v2   - the only required runtime dep (used by AgentContext)
* numpy         - OPTIONAL. Makes the vector index ~50x faster on large corpora.

Install with::

    pip install pydantic            # required
    pip install numpy               # optional

Or use the project helper::

    pip install -e .                # from the project root
"""
from __future__ import annotations

from .memory import (
    Message,
    MemoryHit,
    ShortTermContext,
    MemoryStore,
    default_store,
)
from .llm import LLMClient, ChatResponse, EmbeddingResponse
from .utils import function_to_tool_schema, format_prompt, safe_run, SkipSchema
from .core import (
    AgentContext,
    AgentResult,
    FastAgent,
    LoopResult,
    RunContext,
)

__version__ = "0.2.0"
__all__ = [
    # Beginner essentials
    "FastAgent",
    "AgentContext",
    "AgentResult",
    "LoopResult",
    "MemoryStore",
    # Building blocks
    "Message",
    "MemoryHit",
    "ShortTermContext",
    "LLMClient",
    "ChatResponse",
    "EmbeddingResponse",
    "RunContext",
    "default_store",
    # Helpers
    "function_to_tool_schema",
    "format_prompt",
    "safe_run",
    "SkipSchema",
]