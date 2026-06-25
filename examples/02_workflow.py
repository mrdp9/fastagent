"""Example 2: Workflow chaining two agents.

Run:
    python examples/02_workflow.py

Step 1: support-agent answers a question using memory.
Step 2: summarizer condenses the answer into one sentence.

Each step's output is yielded; the workflow returns a list of all of them.
"""
import asyncio
from fastagent import FastAgent, AgentContext, MemoryStore

app = FastAgent(name="example-02")


@app.agent(name="support-agent", system_prompt="You are a helpful support agent.")
async def support_agent(ctx: AgentContext, user_input: str, messages=None):
    if len(ctx.memory) == 0:
        await ctx.memory.add("Refunds are processed within 5 business days.", {"topic": "policy"})
        await ctx.memory.add("Office hours are 9am-6pm IST, Monday to Friday.", {"topic": "policy"})
        await ctx.memory.add("The support email is help@acme.com.", {"topic": "policy"})
    hits = await ctx.memory.search(user_input, limit=1)
    return hits[0].text if hits else "I don't know yet."


@app.agent(name="summarizer", system_prompt="Compress the user's message into one short sentence.")
async def summarizer(ctx: AgentContext, user_input: str, messages=None):
    if hasattr(user_input, "value"):
        user_input = user_input.value
    return user_input.split(".")[0].strip() + "."


@app.workflow(name="intake-and-summarize")
async def intake(ctx: AgentContext):
    user_query = ctx.state.get("user_query", "How long do refunds take?")
    answer = await app.run_agent("support-agent", user_query, ctx=ctx)
    yield answer.value if hasattr(answer, "value") else answer

    summary = await app.run_agent("summarizer", answer, ctx=ctx)
    yield summary.value if hasattr(summary, "value") else summary


async def main():
    ctx = AgentContext(memory=MemoryStore(), client=app.client)
    outputs = await app.run_workflow("intake-and-summarize", ctx=ctx)
    print("Step outputs:")
    for i, v in enumerate(outputs):
        print(f"  [{i}] {v}")


if __name__ == "__main__":
    asyncio.run(main())
