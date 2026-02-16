"""Integration tests for the script runner — actually runs uv.

These require uv and Python to be available, and are skipped in CI
(unless explicitly opted in with -m integration).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from newton.config import Config
from newton.tools.scripts import run_python_script


pytestmark = pytest.mark.integration


def _make_ctx(
    max_timeout: int = 30,
    max_output_chars: int = 10_000,
    max_steps: int = 50,
) -> MagicMock:
    cfg = Config(
        tools={"scripts": {"max_timeout": max_timeout, "max_output_chars": max_output_chars}},
        agent={"max_steps": max_steps},
    )
    deps = MagicMock()
    deps.cfg = cfg
    deps.step = 0
    deps.max_steps = max_steps
    ctx = MagicMock()
    ctx.deps = deps
    return ctx


class TestScriptRunnerIntegration:
    async def test_hello_world(self):
        ctx = _make_ctx()
        result = await run_python_script(ctx, 'print("hello from uv")')
        assert "Exit code: 0" in result
        assert "hello from uv" in result

    async def test_exit_code_nonzero(self):
        ctx = _make_ctx()
        result = await run_python_script(ctx, "import sys; sys.exit(42)")
        assert "Exit code: 42" in result

    async def test_inline_metadata_deps(self):
        """Script with PEP 723 inline deps — uv should auto-install."""
        script = '''\
# /// script
# dependencies = ["markupsafe"]
# ///

from markupsafe import escape
print(escape("<b>safe</b>"))
'''
        ctx = _make_ctx(max_timeout=60)
        result = await run_python_script(ctx, script, timeout_seconds=60)
        assert "Exit code: 0" in result
        assert "&lt;b&gt;safe&lt;/b&gt;" in result

    async def test_stderr_captured(self):
        ctx = _make_ctx()
        script = 'import sys; print("err", file=sys.stderr)'
        result = await run_python_script(ctx, script)
        assert "stderr" in result
        assert "err" in result
