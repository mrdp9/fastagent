# LLM Providers

How to wire FastAgent to a real LLM. The framework ships with a mock
provider that works offline; switch to a real one with an env var.

## Provider matrix

| Provider | Needs API key | Base URL | Default model | Setup |
|---|---|---|---|---|
| `mock` | no | (none) | `mock-0` | Default. Works offline. |
| `openai` | yes | `https://api.openai.com/v1` | `gpt-4o-mini` | `export OPENAI_API_KEY=*** |
| `minimax` | yes | `https://api.minimaxi.com/v1` | `minimax-chat` | `export MINIMAX_API_KEY=*** |
| `ollama` | no | `http://localhost:11434/v1` | `llama3.2` | Install Ollama, `ollama pull llama3.2` |

## Switching providers

Three ways, in priority order:

1. **Env var** (best for scripts):
   ```bash
   export FASTAGENT_PROVIDER=openai
   python my_app.py
   ```

2. **Constructor arg** (best for code):
   ```python
   from fastagent import FastAgent
   app = FastAgent(name="x", client=LLMClient(provider="openai"))
   ```

3. **Per-call** (advanced):
   ```python
   from fastagent.llm import LLMClient
   other = LLMClient(provider="ollama")
   resp = await other.chat(messages)
   ```

## Auto-fallback

If a real-provider call fails (network error, bad key, rate limit),
FastAgent **automatically falls back to the mock provider** so demos keep
working. The mock's reply will include a note like
`(falling back to mock: provider 'openai' failed: ...)`.

To disable auto-fallback:

```python
from fastagent.llm import LLMClient
client = LLMClient(provider="openai")
# Replace client.chat directly if you need full control:
async def safe_chat(messages, tools=None, **kwargs):
    try:
        return await client._openai_chat(messages, tools, 0.7, None, **kwargs)
    except Exception:
        raise  # no fallback
client.chat = safe_chat
```

## Per-provider gotchas

### OpenAI

- Set `OPENAI_API_KEY`. No other env var needed.
- Streaming not supported in this version (each call returns the full reply).
- Tool-call loop is implemented but not yet wired through `@app.agent` by
  default - see TODO list in the framework's CHANGELOG.

### MiniMax

- Set `MINIMAX_API_KEY`. MiniMax exposes an OpenAI-compatible API; FastAgent
  uses the same code path as OpenAI but with the MiniMax base URL.

### Ollama

- No API key.
- Start the Ollama server before running your app:
  ```bash
  ollama serve     # in one terminal
  ollama pull llama3.2
  ```
- FastAgent talks to Ollama via its OpenAI-compat layer at `/v1`.
- Embeddings: FastAgent falls back to the offline hash embedder when the
  provider is `ollama` because Ollama's embeddings endpoint differs from
  OpenAI's. If you need real embeddings, plug in `embed_fn=`.

## Setting env vars in code

If you don't want to use shell env vars:

```python
import os
os.environ["FASTAGENT_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "***"
from fastagent import FastAgent
app = FastAgent(name="x")  # picks up the env vars
```

Do NOT hardcode keys in source files you commit. Use a `.env` file with
`python-dotenv`, or shell env vars.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `[mock answer]` always in output | Provider not set | `export FASTAGENT_PROVIDER=openai` |
| `RuntimeError: HTTP 401` | Wrong API key | Re-export the right key |
| `RuntimeError: could not reach https://...` | Network/firewall | Try Ollama or mock |
| Provider fell back to mock | Real call failed | Look at the `(fallback note: ...)` in the response |
| Tool calls don't fire | Model doesn't know about the tools | Check the JSON schema was generated (use `function_to_tool_schema(fn)`) |