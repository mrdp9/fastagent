# Discoverability playbook — how to get eyeballs on FastAgent

The repo is at **https://github.com/mrdp9/fastagent**. Shipping code is
the easy part. Getting developers to actually FIND it is the hard part.
This playbook lists every channel that worked for other small open-source
AI frameworks and exactly what to post where.

## The funnel

```
GitHub search / Google  →  README →  pip install →  GitHub star / fork
       ↑                                              ↓
   SEO + posts on Hacker News, Reddit, dev.to, Twitter/X
                                                      ↓
                                          word of mouth, blog citations
```

Each layer feeds the next. Without traffic from layer 1, layer 3 never
activates. Plan to spend ~30% of your time on code, ~70% on promotion
for the first 3 months. That's normal for solo open-source projects.

---

## Tier 1: SEO — make GitHub search / Google surface the repo

### ✅ Done already (in this commit)

- Repo name `fastagent` — short, searchable, no `-` clutter
- Description is filled in (in `docs/github-settings.md` — paste into UI)
- Topics set: `python ai agents llm decorator framework memory ...`
- README's first 200 chars include the search terms:
  > "Decorator-driven AI & memory framework for Python. Zero boilerplate.
  > 5 decorators. Works offline."
- License: MIT (auto-detected)
- CHANGELOG, CONTRIBUTING, CODE_OF_CONDUCT, LICENSE all present
- 87 passing tests with a green CI badge
- 5/5 GitHub repo signals filled

### 🔲 Still TODO (do these in the GitHub UI)

- [ ] Pin the repo on your profile (Settings → Profile → Pinned)
- [ ] Add social preview image (1280×640)
- [ ] Add a GitHub Pages site (optional but powerful)
- [ ] Enable Discussions + Issues

### 🔲 Search keywords to write into docs (so Google picks up)

The framework README already includes:

- "decorator-driven agent" (vs FastMCP's `@mcp.tool`)
- "AI agent framework Python"
- "Pydantic AI" (mentioned in comparisons/inspiration)
- "LangChain alternative"
- "smolagents alternative"
- "AutoGen alternative"
- "CrewAI alternative"
- "OpenAI function calling"
- "vector memory"
- "offline LLM agent"

These are what people actually search for when they want what FastAgent does.

---

## Tier 2: Show HN / Hacker News

Hacker News is THE channel for a developer framework. One good "Show HN"
post can drive 5,000+ stars in a week.

### The post (target 200-400 words)

Title: **Show HN: FastAgent – decorator-driven AI agents in 30 seconds**

> I built FastAgent because every Python AI agent framework I tried
> needed 200MB of deps and three abstractions before I could write
> "hello world". FastAgent is **~3,500 LOC**, depends only on Pydantic,
> and ships with an offline mock LLM so you can write a working agent
> with no API key.
>
> The whole framework is **five decorators**:
>
>     @app.tool()                  # register a function as an LLM tool
>     @app.agent(name, prompt)     # one agent run, returns AgentResult
>     @app.workflow(name)          # chain via yield, returns list
>     @app.loop(name, evaluator)   # self-improving loop
>     @app.structured_agent(name, output_schema=Model)  # typed output
>
> Memory is built in (short-term chat history + long-term semantic
> search), no DB needed. Switching to a real LLM is one env var.
>
> It also ships as a slash-command skill for Claude Code, OpenCode, and
> Hermes — so you can type `/fastagent build me X` and the agent writes
> the code for you.
>
> GitHub: https://github.com/mrdp9/fastagent
>
> What I'd love feedback on:
> - The five-decorator API — too small? Just right?
> - The memory layer — would you rather plug in Chroma/Qdrant?
> - The slash-command skill pattern — useful or gimmick?
>
> (87 tests, MIT licensed, no dependencies beyond Pydantic.)

### When to post

- **Tuesday, Wednesday, or Thursday**, 8–10am US Eastern (peak HN traffic)
- Don't post on Mondays or Fridays (gets buried)
- If it doesn't hit the front page in 2 hours, that's OK — leave it up

### Don't do

- Don't mass-DM people to upvote
- Don't post on Reddit at the same time (cannibalizes)
- Don't fake accounts

---

## Tier 3: Reddit

| Subreddit | Format | Best timing |
|---|---|---|
| r/Python | Text post, "Show & Tell" flair | Tuesday morning |
| r/MachineLearning | Discussion / link post | Wed/Thu |
| r/LocalLLaMA | "Show your project" | Any time |
| r/ChatGPTCoding | Text post | Any time |

### r/Python post (target 300-500 words)

Title: **FastAgent – decorator-driven AI agent framework, ~3.5K LOC, 1 dep**

Body:

> Hey r/Python — I built a tiny AI agent framework and I'd love feedback.
>
> **TL;DR:** `@app.tool`, `@app.agent`, `@app.workflow`, `@app.loop`,
> `@app.structured_agent`. That's the entire API surface. Pydantic is
> the only required dep. Works offline out of the box (mock LLM
> provider built in).
>
> **Why another framework?** I was tired of importing LangChain to write
> `await client.chat(...)`. FastAgent is ~3,500 LOC, decorator-driven
> (think FastMCP's `@mcp.tool`), with built-in memory and a slash-
> command skill for Claude Code / OpenCode.
>
> Quickstart:
>
>     pip install pydantic
>     pip install git+https://github.com/mrdp9/fastagent.git
>
>     from fastagent import FastAgent
>
>     app = FastAgent(name="hello")
>
>     @app.agent(name="greeter", system_prompt="Greet warmly.")
>     async def greeter(ctx, user_input, messages=None):
>         return "Hello, " + user_input + "!"
>
>     asyncio.run(app.run_agent("greeter", "world")).value
>     # -> 'Hello, world!'
>
> **What's different from LangChain / AutoGen / smolagents?**
>
> - ~3,500 LOC vs 100K+
> - 1 required dep (Pydantic) vs 30+
> - Built-in memory (chat history + semantic search) without a separate package
> - Slash-command skill for AI CLIs (you can `/fastagent build me X` in Claude Code)
>
> **What's NOT done yet**
>
> - No Anthropic / Gemini provider yet (only OpenAI, MiniMax, Ollama)
> - Streaming responses (each call returns the full reply)
> - Tool-call loop wired through `@app.agent` (the plumbing is there, the loop isn't)
>
> GitHub: https://github.com/mrdp9/fastagent
>
> 87 tests, MIT licensed, contributions welcome.

---

## Tier 4: dev.to / Hashnode / Medium

Write 2–3 long-form posts over the first month:

1. **"I built a 3,500-line AI agent framework. Here's what I learned about decorator APIs."**
   - The design decisions behind the 5-decorator API
   - Why no async generators in `@app.agent` body
   - The trade-off between "5 decorators" and "100 classes"
   - Target: r/Python, dev.to, your own blog

2. **"Memory in FastAgent: a deep dive"**
   - How the offline vector index works (deterministic hash embedding)
   - Cosine + BM25 hybrid rerank
   - When to upgrade to a real vector DB
   - Target: MLOps Substack, r/MachineLearning

3. **"How to ship a slash-command skill that 100% of dev tools can use"**
   - The SKILL.md spec
   - Why the same skill works in Hermes / Claude Code / OpenCode
   - How to design triggers that actually fire
   - Target: dev.to

Each post links back to the repo at least 3 times.

---

## Tier 5: Twitter / X / Mastodon / Bluesky

Tweet from your personal account (better engagement than a project account):

### Tweet 1 (launch day)

> Shipped FastAgent — a decorator-driven AI agent framework.
>
> 5 decorators. 1 required dep. Works offline.
>
> ```
> @app.agent(name="qa")
> async def qa(ctx, user_input, messages=None):
>     return "Hello, " + user_input + "!"
> ```
>
> https://github.com/mrdp9/fastagent

### Tweet 2 (1 day later)

> Things FastAgent does that LangChain doesn't:
>
> 1. Runs with `pip install pydantic` and nothing else
> 2. Has a working demo with no API key
> 3. Doubles as a slash-command skill for AI coding CLIs
>
> https://github.com/mrdp9/fastagent

### Tweet 3 (1 week later)

> Why I built a 3,500-line AI framework instead of using LangChain:
>
> https://github.com/mrdp9/fastagent

Tag relevant people (Hamel Husain, Simon Willison, swyx, etc.) — but only
if your post adds value, not just to farm attention.

---

## Tier 6: Newsletters

- **TLDR AI** (tldr.tech/ai) — submit via their form, ~3-day lead time
- **Python Weekly** — submit via pyweekly.org
- **Console** by Matt Rickard (console.dev) — submit via console.dev/submit
- **Hacker Newsletter** (hackernewsletter.com) — email the editor
- **Awesome-Python** GitHub repo — open a PR adding FastAgent
- **awesome-llm** GitHub repo — open a PR
- **awesome-ai** GitHub repo — open a PR
- **awesome-rag** GitHub repo — open a PR

---

## Tier 7: Conferences & meetups

- **PyCon US / EU / India** — submit a talk: "Building a tiny AI agent framework: lessons from FastAgent"
- **AI Engineer Summit** — same talk, AIEA track
- **Local Python meetups** — short lightning talk (5 min): live-code an agent in front of the audience
- **Mumbai Python Users Group** (since you're in Mumbai) — easy first talk, builds local community

The talk outline is the same as the dev.to post: "I built a 3,500-line framework, here's what I learned."

---

## Tier 8: SEO long game

These take months but compound:

- **Link to FastAgent from your other projects' READMEs** ("Built with FastAgent" badge)
- **Write a tutorial on Real Python** — Real Python's audience is huge
- **Cross-post to r/Python, r/ML, dev.to simultaneously** when you hit a milestone (e.g. v1.0 with streaming)
- **Get listed on python.libhunt.com** — submit via their form
- **Get listed on pypi.org's "trending this week"** — easier said than done, but release cadence matters
- **Add a "Built with FastAgent" section to the README** and seed it with 3 examples (one being a real public project, even if small)
- **PyPI package**: upload with `python -m twine upload dist/*` so `pip install fastagent` works

---

## Tier 9: SEO helpers (cheat sheet)

If you write blog posts or tutorials, sprinkle these exact phrases:

- "AI agent framework"
- "LangChain alternative"
- "AutoGen alternative"
- "OpenAI function calling"
- "Pydantic AI agent"
- "agent with memory"
- "self-evaluating agent loop"
- "decorator-driven agents"
- "FastMCP-style decorators"
- "agent with vector memory"

These are the phrases developers actually search for.

---

## The 90-day plan

| Week | Action | Expected outcome |
|---|---|---|
| 1 | Push code, fill in repo metadata, post Show HN | 100–500 stars |
| 2 | Post on r/Python + dev.to, write blog post 1 | 500–1500 stars |
| 3 | Submit to Python Weekly, awesome-python PR | 1500–3000 stars |
| 4 | Add 3 contributors, write blog post 2 | 3000–5000 stars |
| 5–8 | PyPI release, Real Python pitch | 5000–10K stars |
| 9–12 | PyCon talk (if accepted), Hacker Newsletter | 10K+ stars |

> The numbers are rough. Some projects blow up in week 1, others take
> 6 months. The variables are: post timing, post quality, and whether
> you get one early adopter who's excited enough to evangelize.

---

## What NOT to do

- **Don't spam.** One post per channel per week, max.
- **Don't fake engagement.** Don't buy stars, don't use upvote rings.
- **Don't overpromise.** The README and HN post are honest about
  limitations (no streaming yet, no Anthropic yet). Stick to that.
- **Don't vanish.** If someone opens an issue, reply within 48 hours
  for the first month. That alone determines whether the project lives.

---

## TL;DR

1. Fill in repo metadata (description, topics, social preview).
2. Post on Hacker News (Tuesday morning).
3. Post on r/Python + dev.to (week after).
4. Submit to Python Weekly + awesome-python.
5. PyPI release + 2–3 long-form blog posts over the next 3 months.
6. Conference talk when you have something to say (PyCon is the obvious one).

Each step is 30–60 minutes of work. The whole thing fits in a weekend.
