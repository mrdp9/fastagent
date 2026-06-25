"""FastAgent structured_agent - validate LLM output against a Pydantic model.

Save as app.py and run:

    python app.py

The agent asks the LLM to extract a structured Answer from free text.
The framework validates the response and returns the parsed Pydantic
instance in result.value.
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

import json
from pydantic import BaseModel, Field
from fastagent import FastAgent, AgentContext


class Answer(BaseModel):
    """The structured output we want from the LLM."""
    name: str = Field(description="The person's name mentioned in the text")
    role: str = Field(description="Their role, e.g. 'CEO', 'CTO', 'engineer'")


app = FastAgent(name="structured-demo")


@app.structured_agent(name="extract-person", output_schema=Answer)
async def extract_person(ctx: AgentContext, user_input: str, messages=None):
    """Body is unused - structured_agent calls the LLM directly."""
    pass


async def main():
    # Patch the mock LLM to return a valid JSON for the demo.
    from fastagent.llm import ChatResponse

    async def chat_with_json(messages, tools=None, **kwargs):
        return ChatResponse(
            content=json.dumps({"name": "Priya Sharma", "role": "CEO"}),
            tool_calls=[],
            raw=None,
            model=app.client.model,
            provider="mock",
        )

    original = app.client.chat
    app.client.chat = chat_with_json

    result = await app.run_agent("extract-person", "Priya Sharma is the CEO of Acme Corp.")
    if result.ok:
        ans = result.value
        print("Got a structured Answer:")
        print("  name :", ans.name)
        print("  role :", ans.role)
        print("  type :", type(ans).__name__)
    else:
        print("Failed:", result.error)

    # Restore the original chat so other tests aren't affected.
    app.client.chat = original


if __name__ == "__main__":
    asyncio.run(main())