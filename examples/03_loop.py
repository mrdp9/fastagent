"""Example 3: Self-improving loop.

Run:
    python examples/03_loop.py

The loop appends a new sentence to a draft each iteration, stopping when
the draft has 3 or more sentences (or max_iterations is reached).

Demonstrates @app.loop + a custom evaluator returning (status, suggestion).
"""
import asyncio
from fastagent import FastAgent, AgentContext


async def draft_evaluator(ctx: AgentContext):
    """Stop when ctx.state['draft'] has >= 3 sentences."""
    draft = ctx.state.get("draft", "")
    sentences = draft.count(".") + draft.count("!") + draft.count("?")
    if sentences >= 3:
        return ("stop", {"reason": "draft has >= 3 sentences", "sentences": sentences})
    return ("continue", {"reason": "draft too short", "sentences": sentences})


app = FastAgent(name="example-03")


@app.loop(name="refine-draft", max_iterations=5, evaluator=draft_evaluator)
async def refine(ctx: AgentContext, i: int):
    """Append one sentence per iteration."""
    sentences = [
        "Acme Corp is a great place to work.",
        "Engineering teams own their services end to end.",
        "We invest in on-call ergonomics and incident review.",
        "Our culture prizes written communication.",
    ]
    addition = sentences[i] if i < len(sentences) else "More context can be added."
    current = ctx.state.get("draft", "")
    ctx.state["draft"] = (current + " " + addition).strip() if current else addition
    return ctx.state["draft"]


async def main():
    ctx = AgentContext(client=app.client)
    result = await app.run_loop("refine-draft", ctx=ctx)
    print(f"Stopped after {result.iterations} iterations")
    print(f"Reason: {result.stopped_reason}")
    print(f"Final draft:\n  {result.final_state.get('draft', '(empty)')}")


if __name__ == "__main__":
    asyncio.run(main())
