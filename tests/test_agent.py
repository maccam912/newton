"""Unit tests for newton.agent — structure and registration, no LLM calls."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic_ai import ImageUrl

from newton.config import Config
from newton.agent import (
    AgentDeps,
    _build_last_chance_prompt,
    _build_run_input,
    _step_tag,
    create_agent,
)
from newton.events import Event, EventBus, EventKind


pytestmark = pytest.mark.unit


class TestAgentDeps:
    def test_init(self, cfg: Config):
        memory = MagicMock()
        bus = EventBus()
        deps = AgentDeps(memory=memory, cfg=cfg, bus=bus)
        assert deps.step == 0
        assert deps.max_steps == cfg.agent.max_steps
        assert deps.turn_ended is False


class TestStepTag:
    def test_increments(self, cfg: Config):
        deps = AgentDeps(memory=MagicMock(), cfg=cfg, bus=EventBus())
        tag = _step_tag(deps)
        assert deps.step == 1
        assert "[step 1/" in tag

    def test_limit_reached(self):
        cfg = Config(agent={"max_steps": 2})
        deps = AgentDeps(memory=MagicMock(), cfg=cfg, bus=EventBus())
        deps.step = 1  # one before limit
        tag = _step_tag(deps)
        assert deps.step == 2
        assert "LIMIT REACHED" in tag


class TestCreateAgent:
    def test_agent_has_tools(self, cfg: Config):
        agent = create_agent(cfg)
        # Collect tool names from all toolsets that expose them eagerly.
        # The MCP toolset (browser) is lazy — tools are only known after
        # the server is started, so we check it separately.
        tool_names: set[str] = set()
        for ts in agent.toolsets:
            tools = getattr(ts, "tools", None)
            if isinstance(tools, dict):
                tool_names.update(tools.keys())

        # Core tools
        assert "end_turn" in tool_names
        assert "respond_to_user" in tool_names
        # Memory tools
        assert "core_memory_read" in tool_names
        assert "core_memory_update" in tool_names
        assert "archival_memory_insert" in tool_names
        assert "archival_memory_search" in tool_names
        # User tools
        assert "user_upsert" in tool_names
        assert "user_details" in tool_names
        # External tools
        assert "web_search" in tool_names
        assert "run_python_script" in tool_names

    def test_agent_has_browser_toolset(self, cfg: Config):
        from pydantic_ai.mcp import MCPServerStdio
        agent = create_agent(cfg)
        mcp_toolsets = [ts for ts in agent.toolsets if isinstance(ts, MCPServerStdio)]
        assert len(mcp_toolsets) == 1
        assert mcp_toolsets[0].command == "npx"


class TestRunInputBuilder:
    def test_text_only(self):
        event = Event(source="telegram", kind=EventKind.MESSAGE, payload="hello")
        run_input, summary = _build_run_input(event)
        assert run_input == "hello"
        assert summary == "hello"

    def test_image_builds_multimodal_input(self):
        event = Event(
            source="telegram",
            kind=EventKind.MESSAGE,
            payload="[User sent a photo.]",
            metadata={"chat_id": "1", "image_url": "https://example.com/file.jpg"},
        )
        run_input, summary = _build_run_input(event)
        assert isinstance(run_input, list)
        assert run_input[0] == "[User sent a photo.]"
        assert isinstance(run_input[1], ImageUrl)
        assert run_input[1].url == "https://example.com/file.jpg"
        assert summary == "[User sent a photo.]"

    def test_last_chance_keeps_image(self):
        event = Event(
            source="telegram",
            kind=EventKind.MESSAGE,
            payload="[User sent a photo.]",
            metadata={"chat_id": "1", "image_url": "https://example.com/file.jpg"},
        )
        prompt = _build_last_chance_prompt(event, "[User sent a photo.]")
        assert isinstance(prompt, list)
        assert isinstance(prompt[1], ImageUrl)
