"""Dump a MemoryStore's contents as readable text.

Usage:
    python scripts/memory_dump.py memory.jsonl
    python scripts/memory_dump.py                       (uses default in-memory store)

This is useful for debugging what the agent actually has stored in long-term
memory. Useful when an agent answers wrong - dump memory first to see if it
ever had the right fact.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

# Make sure the FastAgent package is importable when this script runs from
# anywhere. Add the project dir (parent of fastagent/) if it exists nearby.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SKILL_ROOT = os.path.dirname(_HERE)


def _looks_like_fastagent_pkg(d):
    return os.path.isdir(os.path.join(d, "fastagent")) and os.path.isfile(
        os.path.join(d, "fastagent", "__init__.py")
    ) and os.path.basename(d) != "fastagent"


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


async def dump_store_from_jsonl(path):
    from fastagent import MemoryStore
    if not os.path.exists(path):
        print("File not found:", path)
        return
    store = MemoryStore()
    await store.load_jsonl(path)
    print_memory(store, source=path)


def dump_default_store():
    """No args: print whatever default_store contains (often empty)."""
    from fastagent import default_store
    print_memory(default_store, source="default_store (in-process)")


def print_memory(store, source=""):
    print("=" * 70)
    print("MemoryStore dump (source: {})".format(source))
    print("=" * 70)
    short = list(store.short_term.messages())
    print("Short-term memory: {} messages".format(len(short)))
    for msg in short[-10:]:
        print("  [{}] {}".format(msg.role, (msg.content or "")[:120]))
    print()
    print("Long-term memory: {} records".format(len(store)))
    for record in store._records.values():
        meta = record.get("metadata") or {}
        meta_str = ", ".join("{}={}".format(k, v) for k, v in meta.items())
        print("  [{}] {}".format(record.get("id", "?"), meta_str))
        print("      text: {}".format(record.get("text", "")[:200]))
    print("=" * 70)


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        asyncio.run(dump_store_from_jsonl(path))
    else:
        dump_default_store()


if __name__ == "__main__":
    main()
