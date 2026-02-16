"""Bash tool — lets the agent run shell commands on Linux hosts.

This tool is only registered when running on a Linux platform
(``sys.platform == "linux"``).  On all other platforms the
``register()`` call is a silent no-op.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from pydantic_ai import Agent, RunContext

from newton.tracing import get_tracer

log = logging.getLogger(__name__)
tracer = get_tracer("newton.tools.bash")

TYPE_CHECKING = False
if TYPE_CHECKING:
    from newton.agent import AgentDeps


def register(agent: Agent) -> None:
    """Register the bash tool on the given agent (Linux only)."""
    if sys.platform != "linux":
        log.info("bash tool skipped — platform is %s, not linux", sys.platform)
        return
    from newton.agent import AgentDeps
    globals()["AgentDeps"] = AgentDeps
    agent.tool(run_bash)


async def run_bash(
    ctx: RunContext[AgentDeps],
    command: str,
    timeout_seconds: int = 60,
) -> str:
    """Run a bash command and return its output.

    Args:
        command: The shell command to execute (passed to ``bash -c``).
        timeout_seconds: Max wall-clock seconds before the process is killed
                         (default 60, max controlled by config).
    """
    with tracer.start_as_current_span(
        "tools.run_bash",
        attributes={"command": command, "timeout": timeout_seconds},
    ) as span:
        bash_cfg = ctx.deps.cfg.tools.bash
        max_timeout = int(bash_cfg.get("max_timeout", 120))
        timeout_seconds = min(max(timeout_seconds, 1), max_timeout)

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            span.set_attribute("timed_out", True)
            return f"Command timed out after {timeout_seconds}s."

        exit_code = proc.returncode
        span.set_attribute("exit_code", exit_code)

        out = stdout.decode(errors="replace")
        err = stderr.decode(errors="replace")

        max_output = int(bash_cfg.get("max_output_chars", 10_000))
        if len(out) > max_output:
            out = out[:max_output] + f"\n... (truncated at {max_output} chars)"
        if len(err) > max_output:
            err = err[:max_output] + f"\n... (truncated at {max_output} chars)"

        parts: list[str] = [f"Exit code: {exit_code}"]
        if out.strip():
            parts.append(f"--- stdout ---\n{out.strip()}")
        if err.strip():
            parts.append(f"--- stderr ---\n{err.strip()}")

        from newton.agent import _step_tag
        return "\n\n".join(parts) + _step_tag(ctx.deps)
