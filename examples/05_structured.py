"""Example 5: Structured agent with Pydantic output schema.

Run:
    python examples/05_structured.py

The LLM is asked to extract a structured Answer from free text. The
framework validates the response against the Pydantic model and returns
the parsed instance in result.value.

This example patches the LLM client with a deterministic stub so it runs
offline. To use a real LLM, remove the stub and set FASTAGENT_PROVIDER.
"""
import asyncio
import json
from pydantic import BaseModel, Field
from fastagent import FastAgent, AgentContext
from fastagent.llm import ChatResponse


class Answer(BaseModel):
    """A structured extraction from free text."""
    name: str = Field(description="The person's name")
    role: str = Field(description="Their role, e.g. CEO, CTO, engineer")


app = FastAgent(name="example-05")


@app.structured_agent(name="extract-person", output_schema=Answer)
async def extract(ctx: AgentContext, user_input: str, messages=None):
    pass  # body is unused; framework calls the LLM directly


async def main():
    # Stub the LLM so the demo runs offline.
    async def stub_chat(messages, tools=None, **kwargs):
        return ChatResponse(
            content=json.dumps({"name": "Priya Sharma", "role": "CEO"}),
            tool_calls=[],
            raw=None,
            model=app.client.model,
            provider="mock",
        )
    app.client.chat = stub_chat

    result = await app.run_agent(
        "extract-person",
        "Priya Sharma is the CEO of Acme Corp.",
    )
    if result.ok:
        ans = result.value
        print(f"Got a structured Answer:")
        print(f"  name : {ans.name}")
        print(f"  role : {ans.role}")
        print(f"  type : {type(ans).__name__}")
    else:
        print(f"Failed: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
