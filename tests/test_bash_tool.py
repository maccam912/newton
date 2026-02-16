import pytest
import sys
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from newton.tools.bash import run_bash, register
from newton.config import Config, ToolsConfig


@pytest.fixture
def mock_step_tag():
    """Patch _step_tag in newton.agent."""
    try:
        import newton.agent
    except (ImportError, ModuleNotFoundError):
        mock_mod = MagicMock()
        mock_mod._step_tag = lambda deps: f" [step {deps.step+1}/{deps.max_steps}]"
        with patch.dict(sys.modules, {"newton.agent": mock_mod}):
            yield
        return

    with patch(
        "newton.agent._step_tag",
        side_effect=lambda deps: f" [step {deps.step+1}/{deps.max_steps}]",
    ):
        yield


class MockDeps:
    def __init__(self, bash_cfg=None):
        self.cfg = Config(
            tools=ToolsConfig(bash=bash_cfg or {"max_timeout": 120, "max_output_chars": 10000})
        )
        self.step = 1
        self.max_steps = 10
        self.bus = MagicMock()
        self.memory = MagicMock()
        self.current_message = ""
        self.turn_ended = False
        self.event_metadata = {}


@pytest.fixture
def mock_ctx():
    ctx = MagicMock()
    ctx.deps = MockDeps()
    return ctx


def test_register_on_linux():
    """Tool is registered when platform is linux."""
    agent = MagicMock()
    # Mock both the platform check and the AgentDeps import inside register()
    mock_agent_mod = MagicMock()
    mock_agent_mod.AgentDeps = type("AgentDeps", (), {})
    with patch("newton.tools.bash.sys") as mock_sys, \
         patch.dict(sys.modules, {"newton.agent": mock_agent_mod}):
        mock_sys.platform = "linux"
        register(agent)
    agent.tool.assert_called_once_with(run_bash)


def test_register_skipped_on_darwin():
    """Tool is NOT registered when platform is darwin."""
    agent = MagicMock()
    with patch("newton.tools.bash.sys") as mock_sys:
        mock_sys.platform = "darwin"
        register(agent)
    agent.tool.assert_not_called()


def test_register_skipped_on_windows():
    """Tool is NOT registered when platform is win32."""
    agent = MagicMock()
    with patch("newton.tools.bash.sys") as mock_sys:
        mock_sys.platform = "win32"
        register(agent)
    agent.tool.assert_not_called()


@pytest.mark.asyncio
async def test_bash_success(mock_ctx, mock_step_tag):
    """Successful command returns exit code and stdout."""
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"hello world\n", b"")
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_shell", return_value=mock_proc):
        result = await run_bash(mock_ctx, "echo hello world")

    assert "Exit code: 0" in result
    assert "hello world" in result
    assert "[step 2/10]" in result


@pytest.mark.asyncio
async def test_bash_stderr(mock_ctx, mock_step_tag):
    """Command with stderr includes it in output."""
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"", b"error msg\n")
    mock_proc.returncode = 1

    with patch("asyncio.create_subprocess_shell", return_value=mock_proc):
        result = await run_bash(mock_ctx, "bad_cmd")

    assert "Exit code: 1" in result
    assert "--- stderr ---" in result
    assert "error msg" in result


@pytest.mark.asyncio
async def test_bash_timeout(mock_ctx, mock_step_tag):
    """Command that exceeds timeout is killed."""
    mock_proc = AsyncMock()
    # First communicate() call will be wrapped by wait_for which raises TimeoutError
    # Second communicate() call (after kill) returns empty
    mock_proc.communicate.return_value = (b"", b"")
    mock_proc.kill = MagicMock()

    async def fake_wait_for(coro, *, timeout):
        # Consume the coroutine to avoid warning
        coro.close()
        raise asyncio.TimeoutError()

    with patch("asyncio.create_subprocess_shell", return_value=mock_proc), \
         patch("newton.tools.bash.asyncio.wait_for", side_effect=fake_wait_for):
        result = await run_bash(mock_ctx, "sleep 999", timeout_seconds=1)

    assert "timed out" in result
    assert "1s" in result


@pytest.mark.asyncio
async def test_bash_output_truncation(mock_step_tag):
    """Long output is truncated per config."""
    ctx = MagicMock()
    ctx.deps = MockDeps(bash_cfg={"max_timeout": 120, "max_output_chars": 20})

    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"A" * 100, b"")
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_shell", return_value=mock_proc):
        result = await run_bash(ctx, "echo lots")

    assert "truncated at 20 chars" in result
