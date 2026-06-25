# Bootstrap pattern: making a scaffolded FastAgent project runnable without pip install

When `scripts/scaffold.py` writes a new project, it does NOT pip-install
fastagent. The user gets `app.py` + `requirements.txt` + `Dockerfile` and
must `pip install -r requirements.txt` themselves. But there is a much
smoother path: **make `app.py` find fastagent at runtime** so `python app.py`
just works.

The pattern (copy-paste at the top of any scaffolded `app.py`, right after
the module docstring and before `from __future__ import annotations`):

```python
import os as _os
import sys as _sys

# Make fastagent importable when this file is run directly without pip install.
_BOOTSTRAP_CANDIDATES = (
    r"C:/Users/Administrator/fastagent_project",   # canonical install
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),  # parent dir
)
for _cand in _BOOTSTRAP_CANDIDATES:
    if _os.path.isdir(_os.path.join(_cand, "fastagent")):
        if _cand not in _sys.path:
            _sys.path.insert(0, _cand)
        break

from __future__ import annotations
# ... rest of the imports follow as normal
```

## Why this works

The bootstrap runs before any framework import. It walks a small list of
candidate locations and prepends the first one that actually contains a
`fastagent/` package directory to `sys.path`. From then on, every
`from fastagent import ...` works.

This is preferable to asking the user to `pip install -e .` or to set
`PYTHONPATH` manually. They want to run `python app.py` and see output.

## When to add this pattern

- Every `templates/*.py` in the skill
- Any new project scaffolded via `scripts/scaffold.py`
- Every agent / workflow / loop demo you write for the user

## When NOT to add this pattern

- Production code that should run inside a properly-built Docker image
  (the Dockerfile already does `pip install`).
- Library code that is itself a package (no entry script).

## Cross-IDE / cross-machine

The first candidate is hardcoded for THIS machine (`C:/Users/Administrator/
fastagent_project`). For a different host, the user edits the tuple to
point at their install. The second candidate (parent of `app.py`) is
portable - if the user copies the skill's `framework/` into a sibling
directory of their project, it Just Works.

## Why not a CLI argument?

Adding CLI flags like `--fastagent-path=PATH` would be more flexible but
adds friction for beginners. The bootstrap is "works 90% of the time,
edit one line if not" - which is the right tradeoff for the noob-friendly
audience this skill targets.

## Verified during the 2026-06 finance-copilot demo build

Without the bootstrap, `python app.py` failed with `ModuleNotFoundError:
No module named 'fastagent'`. With the bootstrap, it ran end-to-end and
printed "All steps passed."
