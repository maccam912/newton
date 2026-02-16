import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from newton.agent import process_turn, ArchivalDecision
from newton.events import Event, EventKind, EventBus
from newton.config import Config
from newton.memory import MemoryStore

pytestmark = pytest.mark.unit

@pytest.fixture
def cfg():
    return Config()

@pytest.mark.asyncio
async def test_process_turn_archives_content(cfg):
    # Setup mocks
    mock_memory = AsyncMock(spec=MemoryStore)
    mock_bus = AsyncMock(spec=EventBus)
    mock_bus.put_outbox = AsyncMock()

    # Mock the main agent — simulate that it sent a response (increments response_count)
    mock_agent = AsyncMock()
    def _fake_run(prompt, deps=None):
        if deps is not None:
            deps.response_count = 1
        return MagicMock(output="Main agent response")
    mock_agent.run.side_effect = _fake_run

    # Mock atracer to avoid OTEL issues
    with patch("newton.agent.atracer") as mock_tracer:
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        mock_tracer.current_span.return_value = mock_span

        # Mock the archival agent via the lazy getter
        mock_archival_agent = AsyncMock()
        mock_archival_result = MagicMock()
        mock_archival_result.output = ArchivalDecision(should_archive=True, content="Save this fact")
        mock_archival_agent.run.return_value = mock_archival_result

        with patch("newton.agent._get_archival_agent", return_value=mock_archival_agent):
            # Create an event
            event = Event(source="test_src", kind=EventKind.MESSAGE, payload="User message")

            # Run process_turn
            await process_turn(event, mock_memory, mock_bus, mock_agent, cfg)

            # Verify main agent was called
            mock_agent.run.assert_called_once()

            # Verify archival agent was called with transcript
            mock_archival_agent.run.assert_called_once()
            transcript = mock_archival_agent.run.call_args[0][0]
            assert "User: User message" in transcript
            assert "Assistant: Main agent response" in transcript

            # Verify archival_insert was called
            mock_memory.archival_insert.assert_called_once_with("Save this fact")

@pytest.mark.asyncio
async def test_process_turn_skips_archive(cfg):
    # Setup mocks
    mock_memory = AsyncMock(spec=MemoryStore)
    mock_bus = AsyncMock(spec=EventBus)

    # Mock the main agent — simulate that it sent a response
    mock_agent = AsyncMock()
    def _fake_run(prompt, deps=None):
        if deps is not None:
            deps.response_count = 1
        return MagicMock(output="Main agent response")
    mock_agent.run.side_effect = _fake_run

    # Mock atracer
    with patch("newton.agent.atracer") as mock_tracer:
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        # Mock the archival agent via the lazy getter
        mock_archival_agent = AsyncMock()
        mock_archival_result = MagicMock()
        mock_archival_result.output = ArchivalDecision(should_archive=False, content=None)
        mock_archival_agent.run.return_value = mock_archival_result

        with patch("newton.agent._get_archival_agent", return_value=mock_archival_agent):
            event = Event(source="test_src", kind=EventKind.MESSAGE, payload="User message")

            await process_turn(event, mock_memory, mock_bus, mock_agent, cfg)

            # Verify archival_insert was NOT called
            mock_memory.archival_insert.assert_not_awaited()
