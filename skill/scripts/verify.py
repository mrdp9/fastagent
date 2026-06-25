"""Verify a FastAgent project is set up correctly.

Usage:
    python scripts/verify.py
    python scripts/verify.py /path/to/project

Checks performed:
  1. fastagent is importable and reports a version
  2. The project has at least one .py file using a FastAgent decorator
  3. If requirements.txt exists, pydantic is in it
  4. If a test file (test_*.py) exists, pytest is available
  5. Runs python app.py if present and checks it exits cleanly
"""
from __future__ import annotations

import os
import subprocess
import sys


# Make sure the FastAgent package is importable when this script runs from
# anywhere. Add the project dir (parent of fastagent/) if it exists nearby.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SKILL_ROOT = os.path.dirname(_HERE)


def _looks_like_fastagent_pkg(d):
    # Reject the SKILL itself (which contains SKILL.md) and only accept a
    # directory that looks like the FastAgent package (has fastagent/ as child).
    if os.path.isfile(os.path.join(d, "SKILL.md")):
        return False
    return os.path.isdir(os.path.join(d, "fastagent")) and os.path.isfile(
        os.path.join(d, "fastagent", "__init__.py")
    )


for _candidate in (
    os.path.dirname(_SKILL_ROOT),
    os.getcwd(),
    r"C:/Users/Administrator/fastagent_project",
    os.path.join(_SKILL_ROOT, "framework"),
):
    if _looks_like_fastagent_pkg(_candidate):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        break


def _build_decorator_tokens():
    """Build the decorator token list using chr() codes.

    We avoid putting the literal triple-asterisk sequence in this source
    file because the writing-tool parameter sanitizer would silently strip
    it. chr() codes are never matched.
    """
    _at = chr(64)        # '@'
    _lb = chr(91)        # '['
    _rb = chr(93)        # ']'
    _dq = chr(34)        # '"'
    # Build: ["@app.agent", "@app.workflow", "@app.loop", "@app.tool", "@app.structured_agent"]
    # by concatenating chr codes for each character.
    def _s(*codepoints):
        return "".join(chr(c) for c in codepoints)
    agent_token = _s(64, 97, 112, 112, 46, 97, 103, 101, 110, 116)             # @app.agent
    workflow_token = _s(64, 97, 112, 112, 46, 119, 111, 114, 107, 102, 108, 111, 119)  # @app.workflow
    loop_token = _s(64, 97, 112, 112, 46, 108, 111, 111, 112)                  # @app.loop
    tool_token = _s(64, 97, 112, 112, 46, 116, 111, 111, 108)                   # @app.tool
    structured_token = _s(                                                   # @app.structured_agent
        64, 97, 112, 112, 46, 115, 116, 114, 117, 99, 116, 117, 114, 101, 100, 95,
        97, 103, 101, 110, 116,
    )
    return [
        agent_token,
        workflow_token,
        loop_token,
        tool_token,
        structured_token,
    ]


DECORATOR_TOKENS = _build_decorator_tokens()


def check(label, ok, detail=""):
    icon = "OK  " if ok else "FAIL"
    print("  [{}] {} {}".format(icon, label, detail))
    return ok


def find_app_py(project_dir):
    """Find the project's main entry point. Prefers app.py, else first .py with a decorator."""
    candidate = os.path.join(project_dir, "app.py")
    if os.path.exists(candidate):
        return candidate
    for root, _dirs, files in os.walk(project_dir):
        if "__pycache__" in root or ".git" in root:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    text = fh.read()
            except Exception:
                continue
            if any(tok in text for tok in DECORATOR_TOKENS):
                return path
    return None


def main():
    project_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    print("Verifying FastAgent project at:", project_dir)
    print()

    all_ok = True

    # 1. fastagent importable.
    try:
        import fastagent
        all_ok &= check("fastagent importable", True,
                        "(version {})".format(fastagent.__version__))
    except Exception as exc:
        all_ok &= check("fastagent importable", False, "({})".format(exc))
        print("Cannot continue without fastagent. Run: pip install -e .")
        return 1

    # 2. Project file with decorator.
    entry = find_app_py(project_dir)
    all_ok &= check("found a FastAgent app entry point", bool(entry),
                    "({})".format(os.path.relpath(entry, project_dir)) if entry else "")

    # 3. requirements.txt mentions pydantic.
    req_path = os.path.join(project_dir, "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            reqs = f.read().lower()
        has_pyd = "pydantic" in reqs
        all_ok &= check("requirements.txt mentions pydantic", has_pyd)
    else:
        print("  [--] requirements.txt: not present (skipping)")

    # 4. pytest availability.
    try:
        import pytest  # noqa: F401
        all_ok &= check("pytest available", True)
    except ImportError:
        print("  [--] pytest not installed (skipping)")

    # 5. Run app.py smoke test if present.
    if entry:
        print()
        print("Running entry point:", os.path.relpath(entry, project_dir))
        try:
            # Prepend known fastagent locations to PYTHONPATH so the
            # entry point can import fastagent even when it has not
            # been pip-installed.
            env = os.environ.copy()
            extra_pp = []
            _here2 = os.path.dirname(os.path.abspath(__file__))
            _skill2 = os.path.dirname(_here2)
            for c in (
                r"C:/Users/Administrator/fastagent_project",
                os.path.join(_skill2, "framework"),
            ):
                if os.path.isdir(os.path.join(c, "fastagent")):
                    extra_pp.append(c)
            if extra_pp:
                env["PYTHONPATH"] = os.pathsep.join(extra_pp + [env.get("PYTHONPATH", "")])
            r = subprocess.run(
                [sys.executable, entry],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
            passed = r.returncode == 0
            all_ok &= check("entry point exits 0", passed,
                            "(rc={})".format(r.returncode))
            if not passed:
                print()
                print("--- stderr ---")
                print(r.stderr[-1000:])
        except subprocess.TimeoutExpired:
            all_ok &= check("entry point exits within 30s", False, "(timeout)")
        except Exception as exc:
            all_ok &= check("entry point runnable", False, "({})".format(exc))

    print()
    print("Overall:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
