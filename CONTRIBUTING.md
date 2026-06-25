# Contributing to FastAgent

First off: thank you for considering a contribution. FastAgent is small on
purpose, and we want to keep it that way. Every PR should make the framework
**easier to understand**, not more powerful in a way that costs readability.

## Quick start

```bash
git clone https://github.com/mrdp9/fastagent.git
cd fastagent
python -m pip install -e ".[dev]"
python -m pytest tests/ -v
```

You should see **87 tests pass** in ~3 seconds.

## Where things live

| Path | What's there |
|---|---|
| `fastagent/core.py` | The 5 decorators + `AgentContext`, `AgentResult`, `LoopResult`, `RunContext` |
| `fastagent/memory.py` | `MemoryStore`, `ShortTermContext`, the offline vector index |
| `fastagent/llm.py` | `LLMClient` + 4 provider implementations |
| `fastagent/utils.py` | `function_to_tool_schema`, `format_prompt`, `safe_run`, `SkipSchema` |
| `tests/` | 87 tests, mirrored against each module's surface |
| `skill/` | The `/fastagent` slash-command skill (Claude Code, OpenCode, Hermes) |
| `examples/` | 5 copy-paste-ready starter apps |
| `app.py` | End-to-end demo |
| `docs/` | Deep-dive documentation |

## How to add a feature

1. **Open an issue first.** Even for small changes. The framework is small on
   purpose; we want to discuss API shape before code lands.
2. **Write the test first.** Add a test in `tests/` that demonstrates the
   new behavior. The test should fail without your change.
3. **Make the smallest change that passes the test.** Don't refactor unrelated
   code in the same PR.
4. **Update `docs/` if the public API changed.**
5. **Run the full suite + lint.** `python -m pytest tests/ -v` and confirm
   you haven't regressed any of the 87 existing tests.

## Coding style

- **Type hints everywhere.** Public functions should have full type annotations.
- **Docstrings on every public class and function.** Use the same format as
  the existing modules: a "WHAT IS THIS?" opener, PARAMETERS, RETURNS, RAISES,
  and a BEGINNER EXAMPLE where useful.
- **No new top-level dependencies without discussion.** Pydantic is the only
  hard dep; numpy is the only soft dep. Everything else should be stdlib.
- **Async by default.** All decorated functions must be `async def`.
- **Backward compatible.** If a change is breaking, it goes behind a new
  parameter or a new decorator, not by changing existing signatures.

## Good first contributions

We have an active list of beginner-friendly issues. Some good starters:

- **Add a new LLM provider** (Anthropic, Gemini, Mistral, Cohere). The pattern
  is in `fastagent/llm.py` — copy `_openai_chat`, swap the URL and key env
  var, write tests.
- **Improve the offline embedder.** Real BM25 instead of bag-of-words, or a
  smarter tokenizer that handles stemming.
- **Add a vector DB backend.** `MemoryStore` should accept a `backend=` kwarg
  that dispatches to Chroma / Qdrant / Weaviate for users who outgrow the
  in-memory store.
- **Write more `examples/`.** A RAG-over-a-folder example, a code-review
  agent, a CLI chatbot. The templates in `skill/templates/` are the right
  shape.
- **Better error messages.** When something fails, the message should tell
  the user what to do next, not just what broke.

## Pull request process

1. Fork the repo and create a branch (`git checkout -b feature/my-thing`).
2. Write tests + implementation.
3. Run `python -m pytest tests/ -v` — all tests must pass.
4. Update `CHANGELOG.md` under an "Unreleased" section.
5. Push your branch and open a PR.
6. Wait for review. We aim to respond within 2 business days.

## Reporting bugs

Open an issue with:

- A minimal repro (5-10 lines)
- The expected vs actual behavior
- Your Python version (`python --version`)
- Your OS

If it's a framework crash, include the full traceback.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating, you agree to abide by its terms.

## Questions?

Open a GitHub Discussion — https://github.com/mrdp9/fastagent/discussions

Or DM the maintainer — [@mrdp9](https://github.com/mrdp9).

## License

By contributing, you agree that your contributions will be licensed under
the MIT License. See [`LICENSE`](LICENSE) for the full text.
