# GitHub repo settings — paste these into the UI

After the first push, go to https://github.com/mrdp9/fastagent/settings
and apply each section below. None of these are blocking — the repo is
fully functional without them — but together they 5–10× the chance that
random GitHub visitors actually click into the code.

## About (top-right of the repo page → gear icon)

**Description** (160 chars max):

```
Decorator-driven AI & memory framework for Python. Zero boilerplate. 5 decorators. Works offline. ~3.5K LOC. Inspired by FastMCP, Pydantic AI, smolagents.
```

**Website** (optional — link to docs site if you set one up later):

```
https://fastagent.dev
```

**Topics** (click "Add topics" — pick from these, max 20):

```
python ai agents llm decorator framework openai ollama
memory vector-search rag pydantic offline-first
cli-tool developer-tools machine-learning chatgpt
gpt-4 gpt-4o anthropic-cohere
```

**Releases**: leave "Only show releases that point to a tag" UNCHECKED
initially so v0.2.0 is visible.

## Social preview (Settings → General → Social preview)

Upload a 1280×640 PNG. Quick recipe:

1. Open `https://github.com/mrdp9/fastagent/blob/main/README.md`
2. Screenshot the top header + the "30-second quickstart" section
3. Crop to 1280×640, save as `social-preview.png`
4. Upload via Settings → Social preview → Upload an image

If you don't want to design one, the GitHub default repo card works.

## GitHub Pages (Settings → Pages)

Optional but great for discoverability. Turn on "Deploy from a branch" →
`main` / `docs/` if you want GitHub to host `docs/` as a static site.

Alternatively, use the README as your landing page (default behavior).

## Branch protection (Settings → Branches)

If you intend to accept PRs from strangers:

1. Add rule for `main`
2. ✅ Require a pull request before merging
3. ✅ Require approvals (1)
4. ✅ Require status checks to pass before merging — pick `tests`
5. ✅ Include administrators
6. ❌ Allow force pushes — DO NOT allow

## Issues (Settings → General → Features)

- ✅ Issues (required for bug reports)
- ✅ Sponsorship (optional — set up GitHub Sponsors later if you want)
- ❌ Wiki (use `docs/` instead)
- ✅ Discussions (good for Q&A)

## GitHub Social links (top-right of repo page → click your avatar → Profile)

Make sure your GitHub profile has:

- A bio that mentions "FastAgent"
- A website link
- A pinned repo card for fastagent (Settings → Profile → Pinned)

## Repository metadata checklist

Before announcing anywhere, confirm:

- [ ] Description is filled in
- [ ] 5–10 topics are set
- [ ] The `tests` CI badge is green (wait for first Actions run)
- [ ] `LICENSE` is detected by GitHub (auto-detects MIT)
- [ ] `requirements.txt` shows up in the "Used by" / dependency graph

## Suggested first release (tag after first push)

```bash
git tag -a v0.2.0 -m "FastAgent 0.2.0 - decorator-driven AI framework"
git push origin v0.2.0
```

GitHub will auto-generate release notes from your commits.

## First Issue templates (already in `.github/ISSUE_TEMPLATE/`)

The repo ships with:

- `bug_report.md`
- `feature_request.md`

If you want to add more:

- `question.md` — for "how do I do X?" style questions (point to Discussions)
- `documentation.md` — for "this doc section is wrong/unclear"
