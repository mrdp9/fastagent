name: Bug report
about: Something is broken in FastAgent
title: "[BUG] "
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**Minimal reproduction**
```python
# Paste the smallest possible script that reproduces the issue.
import asyncio
from fastagent import FastAgent

app = FastAgent(name="repro")

@app.agent(name="my-agent")
async def my_agent(ctx, user_input, messages=None):
    return "hello"

asyncio.run(app.run_agent("my-agent", "hi"))
```

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened. Include the full traceback if it's a crash.

**Environment**
- FastAgent version (`python -c "import fastagent; print(fastagent.__version__)"`):
- Python version (`python --version`):
- OS (Windows / macOS / Linux):
- LLM provider (`mock` / `openai` / `ollama` / etc.):

**Anything else?**
Screenshots, logs, related issues, etc.
