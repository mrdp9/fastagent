"""fastagent.utils - tiny helpers that keep FastAgent zero-boilerplate.

============================================================
WHAT IS THIS FILE? (read this first if you are new)
============================================================

This file is mostly "magic glue". When you write ``@app.tool()`` and
hand it a plain Python function, FastAgent has to convert that function
into a JSON Schema description that an LLM (like GPT-4 or Llama) can
understand. Doing that by hand is awful - lots of nested dicts, lots
of edge cases.

This file does it for you, automatically, by READING your function:

  * the function NAME          -> the tool name
  * its DOCSTRING               -> the tool description (and per-arg docs)
  * its PARAMETER NAMES         -> the JSON Schema property names
  * its PARAMETER TYPE HINTS    -> the JSON Schema types
  * its PARAMETER DEFAULTS      -> the JSON Schema defaults
  * its RETURN TYPE ANNOTATION  -> the output schema (optional)

You write a normal function. FastAgent does the rest.

============================================================
BEGINNER EXAMPLE - 60 seconds
============================================================

.. code-block:: python

    from fastagent.utils import function_to_tool_schema

    def get_weather(city, units='metric'):
        '''Look up the current weather for a city.

        Args:
            city: Name of the city, e.g. "Mumbai".
            units: "metric" for Celsius, "imperial" for Fahrenheit.
        '''
        return f'the weather in {city} is...'

    schema = function_to_tool_schema(get_weather)
    print(schema)

Output (simplified)::

    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Look up the current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string",
                     "description": "Name of the city, e.g. Mumbai."},
            "units": {"type": "string", "default": "metric"}
          },
          "required": ["city"]
        }
      }
    }

============================================================
PUBLIC API (what you can import from fastagent)
============================================================

  function_to_tool_schema   - the main function described above
  format_prompt             - interpolate {placeholders} in a template string
  safe_run                  - run an async coroutine and return a default on failure
  SkipSchema                - marker: hide a parameter from the JSON schema

============================================================
REQUIREMENTS
============================================================

Zero hard dependencies. Pure stdlib (inspect, typing).
"""
from __future__ import annotations

import inspect
import re
import typing
from typing import Any, Callable, Dict, List, Optional


# ============================================================================ #
# Marker class: SkipSchema
# ============================================================================ #
class SkipSchema:
    """Mark a parameter as "inject at call time, hide from JSON schema".

    HOW TO USE
    ----------
    When you write a tool, the framework automatically inspects your function
    signature and turns every parameter into a JSON Schema property the LLM
    must supply. But sometimes you want the framework to inject a value
    itself (memory handle, current user, logger) - and you do NOT want the
    LLM to ever see it. Wrap that parameter's type annotation in
    ``SkipSchema[...]`` to hide it.

    EXAMPLE
    -------
    .. code-block:: python

        from fastagent.utils import SkipSchema
        from fastagent.core import RunContext

        @app.tool()
        def lookup_user(user_id, ctx=None):
            '''Look up a user, with access to the current agent context.'''
            return {"id": user_id, "agent": ctx.agent_name}

    The ``ctx`` parameter will NOT appear in the JSON tool schema - the LLM
    never sees it, never has to supply it, and never knows it exists. The
    framework fills it in for you when the tool runs.
    """

    def __class_getitem__(cls, item):
        # Allow ``SkipSchema[RunContext]`` to be used as a type marker.
        # We just return the class; the framework detects SkipSchema by name.
        return cls


# ============================================================================ #
# Skip detection
# ============================================================================ #
def _is_skip_param(annotation, name):
    """True if this parameter should be excluded from the JSON tool schema.

    A parameter is skipped when:
      * its annotation (when resolved to a class) IS or subclasses
        ``fastagent.core.RunContext``,
      * OR its annotation is a string mentioning ``RunContext`` (because
        ``from __future__ import annotations`` makes annotations be plain
        strings instead of real type objects),
      * OR its annotation is ``SkipSchema`` or ``SkipSchema[X]``.
    """
    # Direct class check.
    try:
        from .core import RunContext as _RC
        if isinstance(annotation, type) and issubclass(annotation, _RC):
            return True
    except Exception:
        pass
    # String annotation (PEP 563).
    if isinstance(annotation, str):
        return "RunContext" in annotation or "SkipSchema" in annotation
    # Direct SkipSchema class.
    try:
        if annotation is SkipSchema:
            return True
    except Exception:
        pass
    return False


# ============================================================================ #
# Python type -> JSON Schema type
# ============================================================================ #
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _json_type(annotation):
    """Convert a Python type annotation into a JSON-Schema type fragment.

    BEGINNER NOTE
    -------------
    JSON Schema is a standard format for describing "what shape is this data?"
    The LLMs use it to know which arguments to pass to your tools. If you say
    ``name: str`` in Python, the LLM sees ``{"type": "string"}``.

    HANDLES
    -------
      * Plain types (str, int, float, bool, list, dict)
      * ``Optional[X]`` and ``Union[X, Y, ...]``
      * ``List[X]``, ``Dict[K, V]``
      * Anything unknown - falls back to ``{"type": "string"}``
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if isinstance(annotation, type) and annotation in _TYPE_MAP:
        return {"type": _TYPE_MAP[annotation]}
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _json_type(non_none[0])
        return {"type": "string"}
    if origin in (list, List):
        if args:
            return {"type": "array", "items": _json_type(args[0])}
        return {"type": "array"}
    if origin in (dict, Dict):
        return {"type": "object"}
    return {"type": "string"}


# ============================================================================ #
# Defaults
# ============================================================================ #
def _python_default(d):
    """Convert a Python default value into a JSON-Schema-friendly form.

    Most defaults pass through unchanged. We only special-case things that
    JSON does not natively support, like ``set`` (we coerce to list).
    """
    if isinstance(d, (str, int, float, bool)) or d is None:
        return d
    if isinstance(d, (list, tuple)):
        return list(d)
    if isinstance(d, dict):
        return dict(d)
    if isinstance(d, set):
        return list(d)
    return str(d)


# ============================================================================ #
# Docstring parsing
# ============================================================================ #
def _extract_param_doc(doc, param_name):
    """Pull a parameter's description out of the function's docstring.

    SUPPORTS
    --------
    Google style (the most common, recommended by FastAgent)::

        def f(x):
            '''Do something.

            Args:
                x: how many things.
            '''

    Sphinx style::

        def f(x):
            '''Do something.

            :param x: how many things.
            '''

    If neither style is found, returns ``None``.
    """
    if not doc:
        return None
    m = re.search(
        r"^\s*Args?\s*:\s*\n(.*?)(?=\n\s*(?:Returns?|Raises?|Examples?|Notes?|Yields?)\s*:|\Z)",
        doc,
        flags=re.DOTALL | re.MULTILINE,
    )
    if m:
        block = m.group(1)
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith(param_name + ":") or stripped.startswith(param_name + " "):
                _, _, after = stripped.partition(":")
                return after.strip() if after else None
    m = re.search(r":param\s+`?" + re.escape(param_name) + r"`?\s*:\s*(.+)", doc)
    if m:
        return m.group(1).strip()
    return None


# ============================================================================ #
# The MAIN public function: function_to_tool_schema
# ============================================================================ #
def function_to_tool_schema(fn, name=None):
    """Build an OpenAI-compatible JSON tool schema from a Python function.

    PARAMETERS
    ----------
    fn : callable
        The function (or method) you want to expose as a tool. The framework
        inspects its signature, type hints, defaults, and docstring.
    name : str, optional
        Override the tool name. Defaults to ``fn.__name__``. Useful when you
        register ``@app.tool(name="custom_name")`` on a function whose Python
        name is ugly (e.g. starts with an underscore).

    RETURNS
    -------
    dict
        A JSON Schema in the exact shape OpenAI / Anthropic / Ollama expect::

            {
              "type": "function",
              "function": {
                "name": "get_weather",
                "description": "...",
                "parameters": {
                  "type": "object",
                  "properties": {...},
                  "required": [...]
                }
              }
            }

    RAISES
    ------
    TypeError
        If ``fn`` is not callable.

    BEGINNER EXAMPLE
    ----------------
    .. code-block:: python

        def add(a, b=0):
            '''Add two numbers.

            Args:
                a: first number.
                b: second number.
            '''
            return a + b

        function_to_tool_schema(add)

    BEGINNER TIP
    ------------
    You almost never need to call this yourself - the ``@app.tool()``
    decorator does it for you when you register a tool.
    """
    if not callable(fn):
        raise TypeError(
            "function_to_tool_schema: argument must be callable, got %r. "
            "Did you pass the function itself, not a result of calling it?" % (fn,)
        )

    name = name or getattr(fn, "__name__", "anonymous_tool")
    doc = inspect.getdoc(fn) or ""
    description = ""
    if doc.strip():
        first_block = doc.split("\n\n", 1)[0]
        first_line = first_block.strip().splitlines()[0]
        description = first_line.strip()
    if not description:
        # No docstring (or docstring is empty/whitespace) - generate a
        # beginner-friendly placeholder so the LLM still has something to read.
        description = "Tool: %s. (No description provided - add a docstring to your function.)" % name

    properties = {}
    required = []

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None

    if sig is not None:
        try:
            resolved_hints = typing.get_type_hints(fn)
        except Exception:
            resolved_hints = {}

        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                properties[pname] = {
                    "type": "object",
                    "description": "Variadic %s." % pname,
                }
                continue
            ann = resolved_hints.get(pname, param.annotation)
            if _is_skip_param(ann, pname):
                continue
            schema = _json_type(ann)
            if param.default is not inspect.Parameter.empty:
                schema["default"] = _python_default(param.default)
            else:
                required.append(pname)
            param_desc = _extract_param_doc(doc, pname)
            if param_desc:
                schema["description"] = param_desc
            properties[pname] = schema

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# ============================================================================ #
# format_prompt - dual-purpose helper for beginners
# ============================================================================ #
def format_prompt(template_or_system=None, **kwargs):
    """Build a chat messages list OR substitute placeholders in a template.

    This function has TWO modes for beginner ergonomics:

    ============================================================
    MODE 1: CHAT MESSAGE BUILDER (the common case)
    ============================================================
    Pass chat-related kwargs and you get back a list of OpenAI-format
    message dicts ready to send to an LLM.

    .. code-block:: python

        msgs = format_prompt(
            system_prompt="You are a helpful assistant.",
            user_input="What is the weather in Mumbai?",
            memory_hits=[MemoryHit(id="x", text="...", metadata={}, score=0.8)],
            short_term=[Message(role="user", content="hi")],
        )

    The returned list looks like::

        [
          {"role": "system", "content": "You are a helpful assistant.\\n\\nRelevant long-term memory:\\n  ..."},
          {"role": "user", "content": "hi"},
          {"role": "user", "content": "What is the weather in Mumbai?"},
        ]

    ============================================================
    MODE 2: TEMPLATE SUBSTITUTION (for prompt strings)
    ============================================================
    Pass a string with {placeholders} and you get the rendered string.

    .. code-block:: python

        format_prompt("Hello {name}, score={score}/{total}.",
                      name="Alice", score=8, total=10)
        # -> "Hello Alice, score=8/10."

    Which mode is used depends on the kwargs you pass. If you pass
    ``system_prompt=`` or ``user_input=`` etc., you get MODE 1.
    Otherwise you get MODE 2 (template substitution).
    """
    # MODE 1: chat message builder
    chat_kws = {"system_prompt", "user_input", "memory_hits", "short_term"}
    if any(k in kwargs for k in chat_kws) or template_or_system is None:
        return _build_chat_messages(
            system_prompt=kwargs.get("system_prompt", template_or_system or ""),
            user_input=kwargs.get("user_input", ""),
            memory_hits=kwargs.get("memory_hits"),
            short_term=kwargs.get("short_term"),
        )
    # MODE 2: template substitution
    template = template_or_system
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError):
        def repl(match):
            key = match.group(1)
            return str(kwargs.get(key, ""))
        return re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", repl, template)


def _build_chat_messages(system_prompt="", user_input="",
                        memory_hits=None, short_term=None):
    """Internal: build an OpenAI-format chat messages list.

    OUTPUT ORDER
    ------------
    1. The system prompt (if non-empty)
    2. A second system message containing the long-term memory bullets (if any)
    3. Every message from the short-term context (in order)
    4. The user's input (if provided)

    Splitting memory into its OWN system message (instead of gluing it onto
    the user's system_prompt) makes it easier for the model to focus on the
    memory section, and matches the convention used by other agent frameworks.
    """
    msgs = []
    system_content = system_prompt or ""
    if system_content.strip():
        msgs.append({"role": "system", "content": system_content})
    if memory_hits:
        bullet_lines = []
        for i, h in enumerate(memory_hits, 1):
            try:
                score = float(h.score)
            except Exception:
                score = 0.0
            bullet_lines.append("  [{}] (score={:.2f}) {}".format(i, score, h.text))
        memory_msg = "Relevant long-term memory:\n" + "\n".join(bullet_lines)
        msgs.append({"role": "system", "content": memory_msg})
    if short_term:
        for m in short_term:
            msgs.append({"role": m.role, "content": m.content})
    if user_input:
        msgs.append({"role": "user", "content": user_input})
    return msgs


# ============================================================================ #
# safe_run - run a callable, return (ok, value) tuple
# ============================================================================ #
async def safe_run(fn, *args, **kwargs):
    """Run a sync OR async function safely, return ``(ok, value)``.

    RETURN VALUE
    ------------
    Returns a 2-tuple ``(ok, value)``:
      * If ``fn`` ran successfully: ``(True, fn(*args, **kwargs))``
      * If it raised: ``(False, the_exception)``

    BEGINNER EXAMPLE
    ----------------
    .. code-block:: python

        ok, val = await safe_run(my_function, arg1, arg2)
        if ok:
            print("got", val)
        else:
            print("failed:", val)
    """
    try:
        result = fn(*args, **kwargs)
        if hasattr(result, "__await__"):
            result = await result
        return (True, result)
    except Exception as exc:
        return (False, exc)


__all__ = [
    "function_to_tool_schema",
    "format_prompt",
    "safe_run",
    "SkipSchema",
]