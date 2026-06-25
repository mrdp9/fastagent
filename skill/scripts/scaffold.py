"""Scaffold a new FastAgent project from this skill's templates.

Usage:
    python scripts/scaffold.py my-new-app          # creates ./my-new-app/
    python scripts/scaffold.py my-new-app hello    # picks templates/hello.py

Without args, lists the available templates and exits.
"""
from __future__ import annotations

import os
import shutil
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

# Skill root = two directories up from this script (scripts/scaffold.py -> fastagent/)
SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = os.path.join(SKILL_ROOT, "templates")

TEMPLATE_CHOICES = {
    "hello": "hello.py - minimal hello-world agent",
    "memory": "agent_with_memory.py - agent with seeded long-term memory",
    "agent_with_memory": "agent_with_memory.py - alias of `memory`",
    "workflow": "workflow.py - two-agent workflow",
    "loop": "loop.py - self-evaluating refine loop",
    "structured": "structured.py - structured_agent with Pydantic output",
}


def list_templates():
    print("Available templates:")
    for key, desc in TEMPLATE_CHOICES.items():
        print("  {:12s}  {}".format(key, desc))
    print()
    print("Usage: python scripts/scaffold.py <project-name> [template]")
    print("  e.g. python scripts/scaffold.py my-app hello")


# Map template keys to the actual file basenames (in case key != filename).
TEMPLATE_FILES = {
    "hello": "hello.py",
    "memory": "agent_with_memory.py",
    "agent_with_memory": "agent_with_memory.py",
    "workflow": "workflow.py",
    "loop": "loop.py",
    "structured": "structured.py",
}


def copy_template(template_key, dest_dir):
    basename = TEMPLATE_FILES.get(template_key, template_key + ".py")
    src = os.path.join(TEMPLATES, basename)
    if not os.path.exists(src):
        raise FileNotFoundError(
            "Template {!r} not found at {}".format(template_key, src)
        )
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(src, os.path.join(dest_dir, "app.py"))
    # Always copy the Dockerfile + requirements too.
    for extra in ("Dockerfile", "requirements.txt"):
        extra_src = os.path.join(TEMPLATES, extra)
        if os.path.exists(extra_src):
            shutil.copy(extra_src, os.path.join(dest_dir, extra))
    # Drop a tiny README so the user knows what they got.
    with open(os.path.join(dest_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# {}\n\nFastAgent project scaffolded from the 'fastagent' skill.\n\n"
            "Run with:\n\n    pip install -r requirements.txt\n    python app.py\n\n"
            "For more, see the FastAgent skill or visit the framework repo.\n".format(
                os.path.basename(dest_dir)
            )
        )


def main():
    if len(sys.argv) < 2:
        list_templates()
        return 0
    project_name = sys.argv[1]
    template_key = sys.argv[2] if len(sys.argv) > 2 else "hello"
    if template_key not in TEMPLATE_CHOICES:
        print("Unknown template: {!r}".format(template_key))
        list_templates()
        return 1
    dest = os.path.join(os.getcwd(), project_name)
    if os.path.exists(dest):
        print("Refusing to overwrite existing directory:", dest)
        return 1
    copy_template(template_key, dest)
    print("Scaffolded {} -> {}/".format(template_key, dest))
    print()
    print("Next:")
    print("  cd", project_name)
    print("  pip install -r requirements.txt")
    print("  python app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())