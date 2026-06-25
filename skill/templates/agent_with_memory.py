"""FastAgent agent with long-term memory - seed facts, then ask questions.

Save as app.py and run:

    python app.py

The agent seeds a corporate knowledge base, then asks three questions
that should retrieve the right grounded memory chunk each time.
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

app = FastAgent(name="memory-demo")

# Corporate knowledge base - seed these once.
KNOWLEDGE = [
    ("The CEO of Acme Corp is Priya Sharma.", "people"),
    ("The CTO of Acme Corp is Daniel Park.", "people"),
    ("Acme Corp is headquartered in Bengaluru, India.", "office"),
    ("Acme offers 25 days of paid leave per year.", "policy"),
    ("The Q3 2026 revenue was 142 million USD, up 18% YoY.", "finance"),
    ("The on-call rotation is one week long and includes a USD 150 stipend.", "policy"),
]


@app.agent(name="qa", system_prompt="Answer using only the context provided.")
async def qa(ctx: AgentContext, user_input: str, messages=None):
    """Seed memory on first call, then answer using semantic search."""
    # Seed once: the short-term history is empty on the very first turn.
    if len(ctx.memory) == 0:
        for text, topic in KNOWLEDGE:
            await ctx.memory.add(text, {"topic": topic})

    # Search long-term memory for the top hit.
    hits = await ctx.memory.search(user_input, limit=1)
    if not hits:
        return "I don't know yet."
    best = hits[0]
    return "[{}] (score={:.2f}) {}".format(
        best.metadata.get("topic", "?"), best.score, best.text
    )


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
        print("Q:", q)
        print("A:", result.value)
        print()


if __name__ == "__main__":
    asyncio.run(main())