"""FastAgent hello world - the shortest possible working app.

Save as app.py and run:

    python app.py

No API key required. Uses the offline mock LLM.
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

from fastagent import FastAgent

app = FastAgent(name="hello-app")


@app.agent(name="greeter", system_prompt="Greet the user warmly.")
async def greeter(ctx, user_input, messages=None):
    """Trivial agent: echoes the user input prefixed with a greeting."""
    return "Hello, " + user_input + "! Nice to meet you."


async def main():
    # Run the agent and print the result.
    result = await app.run_agent("greeter", "world")
    print("result.ok:", result.ok)
    print("result.value:", result.value)
    print("result.agent:", result.agent)


if __name__ == "__main__":
    asyncio.run(main())