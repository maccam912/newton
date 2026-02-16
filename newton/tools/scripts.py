"""Script runner tool — lets the agent write and execute Python scripts.

Scripts are run with ``uv run``, which auto-installs any dependencies
declared via PEP 723 inline metadata at the top of the file:

    # /// script
    # dependencies = [
    #   "requests<3",
    #   "rich",
    # ]
    # ///

You can also pin the Python version:

    # /// script
    # requires-python = ">=3.12"
    # dependencies = ["numpy"]
    # ///

When inline metadata is present, ``uv run`` creates an isolated
environment with exactly those dependencies — the project's own
virtualenv is ignored for resolution, so the script is fully
self-contained and reproducible.
"""

from __future__ import annotations

import asyncio
import tempfile
import os
from pathlib import Path

from pydantic_ai import Agent, RunContext

from newton.tracing import get_tracer

tracer = get_tracer("newton.tools.scripts")

TYPE_CHECKING = False
if TYPE_CHECKING:
    from newton.agent import AgentDeps


def register(agent: Agent) -> None:
    """Register the script-runner tool on the given agent."""
    from newton.agent import AgentDeps
    globals()["AgentDeps"] = AgentDeps
    agent.tool(run_python_script)


async def run_python_script(
    ctx: RunContext[AgentDeps],
    script: str,
    timeout_seconds: int = 60,
) -> str:
    """Write a Python script to a temp file and execute it with ``uv run``.

    The script can use PEP 723 inline metadata to declare dependencies
    that ``uv`` will automatically install in an isolated environment::

        # /// script
        # dependencies = ["requests<3", "rich"]
        # ///

    You can also set a minimum Python version::

        # /// script
        # requires-python = ">=3.12"
        # dependencies = ["numpy"]
        # ///

    Args:
        script: The full Python source code to execute.
        timeout_seconds: Max wall-clock seconds before the process is killed
                         (default 60, max 300).
    """
    with tracer.start_as_current_span(
        "tools.run_python_script",
        attributes={"script_len": len(script), "timeout": timeout_seconds},
    ) as span:
        scripts_cfg = ctx.deps.cfg.tools.scripts
        max_timeout = int(scripts_cfg.get("max_timeout", 300))
        timeout_seconds = min(max(timeout_seconds, 1), max_timeout)

        # Write script to a temp file
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="newton_script_",
            delete=False,
        )
        try:
            tmp.write(script)
            tmp.close()

            proc = await asyncio.create_subprocess_exec(
                "uv", "run", tmp.name,
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
                return f"Script timed out after {timeout_seconds}s."

            exit_code = proc.returncode
            span.set_attribute("exit_code", exit_code)

            out = stdout.decode(errors="replace")
            err = stderr.decode(errors="replace")

            # Truncate to keep context manageable
            max_output = int(scripts_cfg.get("max_output_chars", 10_000))
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

        finally:
            Path(tmp.name).unlink(missing_ok=True)
