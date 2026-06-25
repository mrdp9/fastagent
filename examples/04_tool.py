"""Example 4: Tool with auto-generated JSON schema.

Run:
    python examples/04_tool.py

Writes a typed Python function with a Google-style docstring, decorates it
with @app.tool(), and prints the JSON schema FastAgent generated for it.
The LLM would receive this schema and call your function directly.
"""
import json
from fastagent import FastAgent, function_to_tool_schema

app = FastAgent(name="example-04")


@app.tool()
def get_weather(city: str, units: str = "metric") -> dict:
    """Look up current weather for a city.

    Args:
        city: City name, e.g. "Mumbai".
        units: "metric" or "imperial".
    """
    # In real life: call a weather API. For the demo we return a stub.
    return {"city": city, "temp": 28, "units": units}


def main():
    fn = app._tools["get_weather"]
    schema = function_to_tool_schema(fn, name="get_weather")
    print("Auto-generated JSON tool schema:")
    print(json.dumps(schema, indent=2))
    print()

    # Call the tool directly to verify it works.
    result = fn(city="Mumbai")
    print("Direct invocation:")
    print(f"  {result}")


if __name__ == "__main__":
    main()
