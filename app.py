"""FastAgent framework validation demo.

Exercises every public surface of the FastAgent framework and produces visible
proof that it works end-to-end:

1. Loads corporate knowledge into the long-term memory store.
2. Registers an @app.agent that consults that memory before answering.
3. Registers a sequential @app.workflow that chains two agents together.
4. Registers an autonomous @app.loop that uses an evaluator to decide
   when the draft is good enough and to stop the loop.

The default provider is the offline mock so the demo runs with no API key.
Set FASTAGENT_PROVIDER=openai|minimax|ollama and the matching env vars to
exercise a real model.

Run:
    python app.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from fastagent import FastAgent, AgentContext, MemoryStore, LLMClient, LoopResult
from fastagent.utils import function_to_tool_schema


def banner(title: str) -> None:
    """Pretty section header."""
    print()
    print("=" * 72)
    print(" " + title)
    print("=" * 72)


def get_employee_email(name: str) -> str:
    """Look up the corporate email of an employee by full or partial name."""
    directory = {
        "priya sharma": "priya.sharma@acme.example",
        "daniel okafor": "daniel.okafor@acme.example",
        "mridul pandey": "mridul.pandey@acme.example",
    }
    return directory.get(name.lower(), "not found")


def schedule_meeting(attendees: List[str], topic: str, duration_minutes: int = 30) -> Dict[str, Any]:
    """Schedule a meeting with the given attendees.

    Args:
        attendees: list of employee names
        topic: meeting subject
        duration_minutes: length in minutes (default 30)
    """
    return {
        "scheduled": True,
        "attendees": attendees,
        "topic": topic,
        "duration_minutes": duration_minutes,
    }


provider = os.environ.get("FASTAGENT_PROVIDER", "mock")
client = LLMClient(provider=provider)
app = FastAgent(name="acme-corp-demo", client=client)


@app.agent(
    name="support-agent",
    system_prompt=(
        "You are the Acme Corp internal support agent. Answer questions "
        "using ONLY the relevant long-term memory chunks provided above. "
        "If the answer is not in memory, say you do not know."
    ),
    tools=[get_employee_email, schedule_meeting],
    memory_k=3,
)
async def support_agent(ctx: AgentContext, user_input: str, messages: Optional[List[Dict[str, Any]]] = None) -> str:
    """A memory-grounded agent. The wrapper has already pulled the top-k
    memory hits and assembled them into messages for us."""
    resp = await ctx.client.chat(messages)
    return resp.content


async def summarize(ctx: AgentContext, text: str) -> str:
    """Trivial one-shot summarizer used as workflow step 2."""
    if hasattr(text, "value"):
        text = text.value
    msgs = [
        {"role": "system", "content": "You are a concise summarizer. Output one sentence."},
        {"role": "user", "content": "Summarize: " + text},
    ]
    resp = await ctx.client.chat(msgs)
    return resp.content


@app.workflow(name="intake-and-summarize")
async def intake_workflow(ctx: AgentContext) -> List[Any]:
    """Step 1: ask the support agent. Step 2: summarize."""
    user_query = ctx.state.get("user_query", "Who is the CEO of Acme Corp?")
    ctx.log("workflow.intake", query=user_query)
    answer = await app.run_agent("support-agent", user_query, ctx=ctx)
    answer_value = answer.value if hasattr(answer, "value") else answer
    ctx.state["raw_answer"] = answer_value
    summary = await summarize(ctx, answer_value)
    ctx.state["summary"] = summary
    return [answer_value, summary]


async def quality_evaluator(ctx: AgentContext) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Decide whether the draft is good enough to stop.

    Uses a simple heuristic: the draft is good once it has >= 3 sentences.
    In a real deployment swap this for an LLM-as-judge call or a scorer.
    """
    drafts = ctx.state.get("drafts", [])
    current = drafts[-1] if drafts else ""
    sentences = [s for s in current.split(".") if s.strip()]
    if len(sentences) >= 3:
        return "stop", {"reason": "draft has >= 3 sentences", "sentences": len(sentences)}
    if len(drafts) >= 5:
        return "stop", {"reason": "safety cap reached"}
    return "continue", {"reason": "draft too short", "sentences": len(sentences)}


@app.loop(name="refine-draft", max_iterations=5, evaluator=quality_evaluator)
async def refine_draft(ctx: AgentContext, iteration: int) -> None:
    """Each iteration produces a longer draft. The evaluator decides when to stop."""
    drafts: List[str] = ctx.state.setdefault("drafts", [])
    topic = ctx.state.get("topic", "Why Acme Corp is a great place to work")
    if not drafts:
        drafts.append(topic + ".")
    else:
        pool = [
            "We ship fast and measure outcomes.",
            "Engineering teams own their services end to end.",
            "We invest in on-call ergonomics and incident review.",
            "Customer empathy is built into every release.",
        ]
        addition = pool[iteration % len(pool)]
        drafts.append(drafts[-1] + " " + addition)
    ctx.log("refine.iter", iteration=iteration, length=len(drafts[-1].split()))


KNOWLEDGE_BASE = [
    ("Acme Corp is headquartered in Bengaluru, India, with a European office in London.", {"tag": "company"}),
    ("The CEO of Acme Corp is Priya Sharma.", {"tag": "people"}),
    ("The CTO of Acme Corp is Daniel Okafor.", {"tag": "people"}),
    ("Acme Corp uses Kubernetes for all production workloads and Postgres for transactional data.", {"tag": "tech"}),
    ("All employees get 25 days of paid leave per year plus 10 company holidays.", {"tag": "policy"}),
    ("The on-call rotation is one week long and includes a USD 150 stipend.", {"tag": "policy"}),
    ("Q3 revenue was 142 million USD, up 18 percent year over year.", {"tag": "finance"}),
    ("Customer support response SLA is 4 hours during business hours, 24 hours otherwise.", {"tag": "policy"}),
]


async def seed_memory(ctx: AgentContext) -> None:
    """Add the knowledge base into the context long-term memory."""
    for text, meta in KNOWLEDGE_BASE:
        await ctx.memory.add(text, meta)


async def run_demo() -> int:
    print("FastAgent framework validation demo")
    print("  provider:", app.client.provider, " model:", app.client.model, " base:", app.client.base_url or "-")
    print("  components registered:", app.list_components())

    banner("Auto-generated tool schemas (from plain Python functions)")
    for tool in [function_to_tool_schema(get_employee_email), function_to_tool_schema(schedule_meeting)]:
        print(json.dumps(tool, indent=2))

    banner("Step 1: agent consults long-term memory")
    ctx_agent = app._new_context()
    await seed_memory(ctx_agent)
    print("Seeded", len(ctx_agent.memory), "chunks into long-term memory.")
    print()
    print("Tool schemas the agent has access to:")
    for schema in app._agents["support-agent"].tool_schemas:
        print("  -", schema["function"]["name"], ":", schema["function"]["description"][:80])
    print()
    for q in [
        "Who is the CEO of Acme Corp?",
        "How much paid leave do employees get?",
        "What was Q3 revenue?",
    ]:
        print("> User:", q)
        out = await app.run_agent("support-agent", q, ctx=ctx_agent)
        print("< Agent:", out)
        last = [h for h in ctx_agent.history if h["event"] == "agent.end"][-1]
        print("  (used memory ids:", last.get("used_memory"), ")")
        print()

    banner("Step 2: sequential workflow (agent -> summarizer)")
    ctx_wf = app._new_context()
    await seed_memory(ctx_wf)
    ctx_wf.state["user_query"] = "What is the on-call stipend at Acme Corp?"
    out = await app.run_workflow("intake-and-summarize", ctx=ctx_wf)
    print("Step outputs:")
    for i, v in enumerate(out):
        print("  [{}] {}".format(i, v))
    print("Final workflow state:")
    for k, v in ctx_wf.state.items():
        print("  {} = {}".format(k, v))

    banner("Step 3: autonomous loop (refine-draft)")
    ctx_loop = app._new_context()
    ctx_loop.state["topic"] = "Acme Corp is a great place to work"
    result = await app.run_loop("refine-draft", ctx=ctx_loop)
    print("Stopped after", result.iterations, "iterations, reason:", result.stopped_reason)
    print("Final draft:")
    print(" ", result.final_state["drafts"][-1])
    print()
    print("Per-iteration log (last 6 events):")
    for h in result.history[-6:]:
        print(" ", h)

    banner("All steps passed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run_demo()) or 0)
