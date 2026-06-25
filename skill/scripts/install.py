"""Install the FastAgent package into the current Python environment.

Usage:
    python scripts/install.py         # pip install -e ./framework
    python scripts/install.py pypi    # pip install fastagent (placeholder)

The default mode does an editable install from the bundled framework/ folder
inside this skill, so you can pip install without needing to clone anything.
"""
from __future__ import annotations

import os
import subprocess
import sys

SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMEWORK = os.path.join(SKILL_ROOT, "framework")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "local"
    if mode == "local":
        if not os.path.isdir(FRAMEWORK):
            print("Bundled framework not found at", FRAMEWORK)
            return 1
        print("Installing editable from", FRAMEWORK)
        cmd = [sys.executable, "-m", "pip", "install", "-e", FRAMEWORK]
    elif mode == "pypi":
        print("PyPI install not yet available; using editable from bundled framework instead.")
        cmd = [sys.executable, "-m", "pip", "install", "-e", FRAMEWORK]
    else:
        print("Unknown mode:", mode)
        return 1
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
