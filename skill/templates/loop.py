"""FastAgent loop - a self-evaluating refine loop.

Save as app.py and run:

    python app.py

The loop repeatedly extends a draft sentence until it has at least 3
sentences, then stops. The evaluator returns ("stop", ...) or
("continue", ...) on each iteration.
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

from fastagent import FastAgent, AgentContext


async def draft_evaluator(ctx: AgentContext):
    """Stop when the draft has 3 or more sentences."""
    draft = ctx.state.get("draft", "")
    sentences = draft.count(".") + draft.count("!") + draft.count("?")
    if sentences >= 3:
        return ("stop", {"reason": "draft has >= 3 sentences", "sentences": sentences})
    return ("continue", {"reason": "draft too short", "sentences": sentences})


app = FastAgent(name="loop-demo")


@app.loop(name="refine-draft", max_iterations=5, evaluator=draft_evaluator)
async def refine_draft(ctx: AgentContext, i: int):
    """Append one more sentence to the draft on each iteration."""
    sentences = [
        "Acme Corp is a great place to work.",
        "Engineering teams own their services end to end.",
        "We invest in on-call ergonomics and incident review.",
        "Our culture prizes written communication and short meetings.",
    ]
    current = ctx.state.get("draft", "")
    if i < len(sentences):
        addition = sentences[i]
    else:
        addition = "More context can be added here."
    ctx.state["draft"] = (current + " " + addition).strip() if current else addition
    return ctx.state["draft"]


async def main():
    ctx = AgentContext(client=app.client)
    result = await app.run_loop("refine-draft", ctx=ctx)
    print("Stopped after", result.iterations, "iterations, reason:", result.status)
    print("Final draft:")
    print(" ", result.state.get("draft", "(empty)"))


if __name__ == "__main__":
    asyncio.run(main())