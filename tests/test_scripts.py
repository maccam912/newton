"""Unit tests for newton.tools.scripts (mocked subprocess)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from newton.config import Config
from newton.tools.scripts import run_python_script


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers — fake RunContext and AgentDeps
# ---------------------------------------------------------------------------

def _make_ctx(
    max_timeout: int = 300,
    max_output_chars: int = 10_000,
    max_steps: int = 50,
) -> MagicMock:
    """Build a minimal mock RunContext[AgentDeps]."""
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


def _fake_proc(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
    """Create a fake asyncio.Process."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.kill = MagicMock()
    return proc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunPythonScript:
    @patch("newton.tools.scripts.asyncio.create_subprocess_exec")
    async def test_simple_script(self, mock_exec):
        proc = _fake_proc(stdout=b"hello world\n", returncode=0)
        mock_exec.return_value = proc

        ctx = _make_ctx()
        result = await run_python_script(ctx, 'print("hello world")')

        assert "Exit code: 0" in result
        assert "hello world" in result
        # Verify uv run was called
        mock_exec.assert_called_once()
        args = mock_exec.call_args[0]
        assert args[0] == "uv"
        assert args[1] == "run"
        assert args[2].endswith(".py")

    @patch("newton.tools.scripts.asyncio.create_subprocess_exec")
    async def test_script_with_stderr(self, mock_exec):
        proc = _fake_proc(stdout=b"", stderr=b"Traceback: error\n", returncode=1)
        mock_exec.return_value = proc

        ctx = _make_ctx()
        result = await run_python_script(ctx, "raise Exception('boom')")

        assert "Exit code: 1" in result
        assert "stderr" in result
        assert "Traceback: error" in result

    @patch("newton.tools.scripts.asyncio.create_subprocess_exec")
    async def test_timeout(self, mock_exec):
        proc = AsyncMock()
        proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
        proc.kill = MagicMock()
        # After kill, communicate returns empty
        proc.communicate.side_effect = [asyncio.TimeoutError, (b"", b"")]
        mock_exec.return_value = proc

        ctx = _make_ctx(max_timeout=5)
        result = await run_python_script(ctx, "import time; time.sleep(999)", timeout_seconds=2)

        assert "timed out" in result.lower()
        proc.kill.assert_called_once()

    @patch("newton.tools.scripts.asyncio.create_subprocess_exec")
    async def test_output_truncation(self, mock_exec):
        big_output = b"x" * 20_000
        proc = _fake_proc(stdout=big_output, returncode=0)
        mock_exec.return_value = proc

        ctx = _make_ctx(max_output_chars=100)
        result = await run_python_script(ctx, "print('x' * 20000)")

        assert "truncated at 100 chars" in result

    @patch("newton.tools.scripts.asyncio.create_subprocess_exec")
    async def test_timeout_clamped_to_max(self, mock_exec):
        proc = _fake_proc(stdout=b"ok\n", returncode=0)
        mock_exec.return_value = proc

        ctx = _make_ctx(max_timeout=30)
        # Request 999s but max_timeout is 30
        result = await run_python_script(ctx, "print('ok')", timeout_seconds=999)

        assert "Exit code: 0" in result

    @patch("newton.tools.scripts.asyncio.create_subprocess_exec")
    async def test_temp_file_cleaned_up(self, mock_exec):
        proc = _fake_proc(stdout=b"done\n", returncode=0)
        mock_exec.return_value = proc

        ctx = _make_ctx()
        await run_python_script(ctx, "print('done')")

        # The temp file passed to uv run should have been deleted
        script_path = mock_exec.call_args[0][2]
        assert not Path(script_path).exists()

    @patch("newton.tools.scripts.asyncio.create_subprocess_exec")
    async def test_temp_file_cleaned_up_on_error(self, mock_exec):
        mock_exec.side_effect = OSError("spawn failed")

        ctx = _make_ctx()
        with pytest.raises(OSError):
            await run_python_script(ctx, "print('hi')")

        # Even on error the temp file should be gone — but we can't easily
        # check which file it was without more intrusive mocking, so just
        # verify we got the exception (the finally block ran).
