"""Example 1: Agent with long-term memory.

Run:
    python examples/01_memory_agent.py

Seeds a corporate knowledge base on first call, then answers user queries
grounded in memory. Demonstrates ctx.memory.search() and ctx.memory.add().
"""
import asyncio
from fastagent import FastAgent, AgentContext, MemoryStore

app = FastAgent(name="example-01")

KNOWLEDGE = [
    ("The CEO of Acme Corp is Priya Sharma.", "people"),
    ("The CTO of Acme Corp is Daniel Park.", "people"),
    ("Acme Corp is headquartered in Bengaluru, India.", "office"),
    ("Acme offers 25 days of paid leave per year.", "policy"),
    ("The Q3 2026 revenue was 142 million USD, up 18% YoY.", "finance"),
]


@app.agent(name="qa", system_prompt="Answer using only the context provided.")
async def qa(ctx: AgentContext, user_input: str, messages=None):
    # Seed once (or persist with ctx.memory.save_jsonl(...)).
    if len(ctx.memory) == 0:
        for text, topic in KNOWLEDGE:
            await ctx.memory.add(text, {"topic": topic})

    # Search long-term memory for the top hit.
    hits = await ctx.memory.search(user_input, limit=1)
    if not hits:
        return "I don't know yet."
    best = hits[0]
    return f"[{best.metadata.get('topic', '?')}] (score={best.score:.2f}) {best.text}"


async def main():
    ctx = AgentContext(memory=MemoryStore(), client=app.client)
    questions = [
        "Who is the head of the company?",
        "How much paid time off do we get?",
        "What was last quarter's revenue?",
        "Where is the main office?",
    ]
    for q in questions:
        result = await app.run_agent("qa", q, ctx=ctx)
        print(f"Q: {q}")
        print(f"A: {result.value}\n")


if __name__ == "__main__":
    asyncio.run(main())
