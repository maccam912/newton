import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from newton.agent import process_turn, SessionArchivalResult, _run_session_archival
from newton.events import Event, EventKind, EventBus
from newton.config import Config
from newton.memory import MemoryStore

pytestmark = pytest.mark.unit

@pytest.fixture
def cfg():
    return Config()


@pytest.mark.asyncio
async def test_process_turn_no_longer_runs_archival(cfg):
    """After the rework, process_turn should NOT run archival reflection."""
    mock_memory = AsyncMock(spec=MemoryStore)
    mock_bus = AsyncMock(spec=EventBus)
    mock_bus.put_outbox = AsyncMock()

    mock_agent = AsyncMock()
    def _fake_run(prompt, deps=None):
        if deps is not None:
            deps.response_count = 1
        return MagicMock(output="Main agent response")
    mock_agent.run.side_effect = _fake_run

    with patch("newton.agent.atracer") as mock_tracer:
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        event = Event(source="test_src", kind=EventKind.MESSAGE, payload="User message")
        await process_turn(event, mock_memory, mock_bus, mock_agent, cfg)

        # Main agent was called
        mock_agent.run.assert_called_once()
        # recall_save was called (inbound + assistant output)
        assert mock_memory.recall_save.await_count == 2
        # archival_insert should NOT have been called (no per-turn archival)
        mock_memory.archival_insert.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_session_archival(cfg):
    """Test that _run_session_archival summarizes, archives facts, and clears recall."""
    mock_memory = AsyncMock(spec=MemoryStore)
    mock_bus = AsyncMock(spec=EventBus)

    mock_memory.recall_get_all.return_value = [
        {"role": "user", "content": "Hello", "channel": "telegram", "timestamp": "2026-01-01T00:00:00", "handled": "1"},
        {"role": "assistant", "content": "Hi there!", "channel": "telegram", "timestamp": "2026-01-01T00:00:01", "handled": "1"},
    ]
    mock_memory.archival_insert.return_value = 1
    mock_memory.session_summary_save.return_value = 1
    mock_memory.recall_clear.return_value = 2

    mock_result = SessionArchivalResult(
        summary="User greeted the assistant. Brief exchange.",
        archival_facts=["User prefers telegram channel."],
    )
    mock_agent_result = MagicMock()
    mock_agent_result.output = mock_result

    mock_archival_agent = AsyncMock()
    mock_archival_agent.run.return_value = mock_agent_result

    with patch("newton.agent.atracer") as mock_tracer:
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        with patch("newton.agent._get_session_archival_agent", return_value=mock_archival_agent):
            await _run_session_archival(mock_memory, cfg, mock_bus)

    # Verify archival_insert was called with the fact
    mock_memory.archival_insert.assert_awaited_once_with("User prefers telegram channel.")
    # Verify session summary was saved
    mock_memory.session_summary_save.assert_awaited_once_with(
        summary="User greeted the assistant. Brief exchange.",
        channels="telegram",
        msg_count=2,
    )
    # Verify recall was cleared
    mock_memory.recall_clear.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_session_archival_empty_recall(cfg):
    """If recall is empty, _run_session_archival should be a no-op."""
    mock_memory = AsyncMock(spec=MemoryStore)
    mock_bus = AsyncMock(spec=EventBus)
    mock_memory.recall_get_all.return_value = []

    with patch("newton.agent.atracer") as mock_tracer:
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        await _run_session_archival(mock_memory, cfg, mock_bus)

    mock_memory.archival_insert.assert_not_awaited()
    mock_memory.session_summary_save.assert_not_awaited()
    mock_memory.recall_clear.assert_not_awaited()
