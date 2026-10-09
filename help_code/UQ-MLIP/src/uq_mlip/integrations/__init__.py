"""Integrations that expose uq-mlip to external orchestrators.

The :mod:`uq_mlip.integrations.scilink` module adapts the uq-mlip workflow to
SciLink's tool-registry contract (``TOOL_SPEC`` / ``TOOL_SPECS`` plus callables
named to match). The same callables are usable by any tool-calling
orchestrator (Claude Code, Codex, Cline, ...) through the OpenAI function-call
schemas emitted by :func:`uq_mlip.integrations.scilink.openai_tool_schemas`.
"""

from uq_mlip.integrations import scilink

__all__ = ["scilink"]
