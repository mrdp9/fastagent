"""FastAgent workflow - chain two agents into a pipeline.

Save as app.py and run:

    python app.py

Step 1: a support agent answers the question using memory.
Step 2: a summarizer condenses the answer into one sentence.

Each step's output is yielded; the workflow returns the list.
"""
import asyncio
import os as _os
import sys as _sys

# Make fastagent importable when this file is run directly without pip install.
for _cand in (
    r"C:/Users/Administrator/fastagent_project",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
):
    if _os.path.isdir(_os.path.join(_cand, "fastagent")):
        if _cand not in _sys.path:
            _sys.path.insert(0, _cand)
        break

from fastagent import FastAgent, AgentContext, MemoryStore

app = FastAgent(name="workflow-demo")


@app.agent(name="support-agent", system_prompt="You are a helpful support agent.")
async def support_agent(ctx: AgentContext, user_input: str, messages=None):
    """Seed facts on first turn, then answer using long-term memory."""
    if len(ctx.memory) == 0:
        await ctx.memory.add("Refunds are processed within 5 business days.", {"topic": "policy"})
        await ctx.memory.add("Office hours are 9am-6pm IST, Monday to Friday.", {"topic": "policy"})
        await ctx.memory.add("The support email is help@acme.com.", {"topic": "policy"})
    hits = await ctx.memory.search(user_input, limit=1)
    return hits[0].text if hits else "I don't know yet."


@app.agent(name="summarizer", system_prompt="Compress the user's message into one short sentence.")
async def summarizer(ctx: AgentContext, user_input: str, messages=None):
    """Trivial one-shot summarizer for the demo."""
    if hasattr(user_input, "value"):
        user_input = user_input.value
    return user_input.split(".")[0].strip() + "."


@app.workflow(name="intake-and-summarize")
async def intake_workflow(ctx: AgentContext):
    """Step 1: ask the support agent. Step 2: summarize."""
    user_query = ctx.state.get("user_query", "How long do refunds take?")
    ctx.log("workflow.intake", query=user_query)
    answer = await app.run_agent("support-agent", user_query, ctx=ctx)
    answer_value = answer.value if hasattr(answer, "value") else answer
    ctx.state["raw_answer"] = answer_value
    summary_result = await app.run_agent("summarizer", answer_value, ctx=ctx)
    summary_value = summary_result.value if hasattr(summary_result, "value") else summary_result
    ctx.state["summary"] = summary_value
    yield answer_value
    yield summary_value


async def main():
    ctx = AgentContext(memory=MemoryStore(), client=app.client)
    outputs = await app.run_workflow("intake-and-summarize", ctx=ctx)
    print("Step outputs:")
    for i, v in enumerate(outputs):
        print("  [{}] {}".format(i, v))


if __name__ == "__main__":
    asyncio.run(main())