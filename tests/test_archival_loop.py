import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from newton.agent import process_turn, ArchivalDecision
from newton.events import Event, EventKind, EventBus
from newton.config import Config
from newton.memory import MemoryStore

@pytest.fixture
def cfg():
    return Config()

@pytest.mark.asyncio
async def test_process_turn_archives_content(cfg):
    # Setup mocks
    mock_memory = AsyncMock(spec=MemoryStore)
    mock_bus = AsyncMock(spec=EventBus)
    mock_bus.put_outbox = AsyncMock()
    
    # Mock the main agent
    mock_agent = AsyncMock()
    mock_agent.run.return_value = MagicMock(output="Main agent response")
    
    # Mock atracer to avoid OTEL issues
    with patch("newton.agent.atracer") as mock_tracer:
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        mock_tracer.current_span.return_value = mock_span

        # Mock the archival agent (use patch since it's a global in agent.py)
        with patch("newton.agent.archival_agent", new_callable=AsyncMock) as mock_archival_agent:
            # Configure archival agent to decide YES
            mock_archival_result = MagicMock()
            mock_archival_result.data = ArchivalDecision(should_archive=True, content="Save this fact")
            mock_archival_agent.run.return_value = mock_archival_result

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
    
    # Mock the main agent
    mock_agent = AsyncMock()
    mock_agent.run.return_value = MagicMock(output="Main agent response")
    
    # Mock atracer
    with patch("newton.agent.atracer") as mock_tracer:
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        # Mock the archival agent
        with patch("newton.agent.archival_agent", new_callable=AsyncMock) as mock_archival_agent:
            # Configure archival agent to decide NO
            mock_archival_result = MagicMock()
            mock_archival_result.data = ArchivalDecision(should_archive=False, content=None)
            mock_archival_agent.run.return_value = mock_archival_result

            event = Event(source="test_src", kind=EventKind.MESSAGE, payload="User message")

            await process_turn(event, mock_memory, mock_bus, mock_agent, cfg)

            # Verify archival_insert was NOT called
            mock_memory.archival_insert.assert_not_awaited()
