"""Unit tests for newton.tools.subagent — structure and caching, no LLM calls."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from newton.config import Config
from newton.agent import AgentDeps, create_agent
from newton.events import EventBus
from newton.tools.subagent import _get_or_create_subagent, _subagent_cache


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clear_cache():
    """Reset the sub-agent cache before each test."""
    _subagent_cache.clear()
    yield
    _subagent_cache.clear()


class TestSubagentCaching:
    def test_builds_and_caches_on_first_call(self, cfg: Config):
        sub1 = _get_or_create_subagent(cfg)
        assert sub1 is not None
        cache_key = f"{cfg.llm.provider}:{cfg.llm.model}"
        assert cache_key in _subagent_cache

    def test_returns_same_instance_on_second_call(self, cfg: Config):
        sub1 = _get_or_create_subagent(cfg)
        sub2 = _get_or_create_subagent(cfg)
        assert sub1 is sub2

    def test_rebuilds_on_model_change(self):
        cfg1 = Config(llm={"api_key": "test-key", "model": "model-a"})
        cfg2 = Config(llm={"api_key": "test-key", "model": "model-b"})
        sub1 = _get_or_create_subagent(cfg1)
        sub2 = _get_or_create_subagent(cfg2)
        assert sub1 is not sub2
        assert len(_subagent_cache) == 2


class TestSubagentTools:
    def test_subagent_has_expected_tools(self, cfg: Config):
        sub = _get_or_create_subagent(cfg)
        tool_names: set[str] = set()
        for ts in sub.toolsets:
            tools = getattr(ts, "tools", None)
            if isinstance(tools, dict):
                tool_names.update(tools.keys())

        # External tools
        assert "web_search" in tool_names
        assert "run_python_script" in tool_names
        # Memory tools
        assert "archival_memory_search" in tool_names
        assert "archival_memory_insert" in tool_names
        assert "core_memory_read" in tool_names
        # Skill tools
        assert "skill_invoke" in tool_names
        assert "skill_list" in tool_names

    def test_subagent_does_not_have_parent_only_tools(self, cfg: Config):
        sub = _get_or_create_subagent(cfg)
        tool_names: set[str] = set()
        for ts in sub.toolsets:
            tools = getattr(ts, "tools", None)
            if isinstance(tools, dict):
                tool_names.update(tools.keys())

        # These are parent-only control-flow tools and must NOT be on the sub-agent
        assert "end_turn" not in tool_names
        assert "respond_to_user" not in tool_names


class TestParentAgentHasSubagentTools:
    def test_parent_agent_registers_subagent_tools(self, cfg: Config):
        agent = create_agent(cfg)
        tool_names: set[str] = set()
        for ts in agent.toolsets:
            tools = getattr(ts, "tools", None)
            if isinstance(tools, dict):
                tool_names.update(tools.keys())

        assert "run_subagent" in tool_names
        assert "run_parallel_subagents" in tool_names


class TestParallelSubagentsValidation:
    """Test input validation for run_parallel_subagents without LLM calls."""

    @pytest.mark.asyncio
    async def test_too_many_tasks_returns_error(self, cfg: Config):
        """run_parallel_subagents should reject more than 5 tasks."""
        memory = MagicMock()
        bus = EventBus()
        deps = AgentDeps(memory=memory, cfg=cfg, bus=bus)

        # Import the tool function directly to test its validation logic
        # by calling the inner function via the registered agent.
        agent = create_agent(cfg)

        # Find the tool handler on the agent's toolsets
        handler = None
        for ts in agent.toolsets:
            tools = getattr(ts, "tools", None)
            if isinstance(tools, dict) and "run_parallel_subagents" in tools:
                handler = tools["run_parallel_subagents"]
                break

        assert handler is not None, "run_parallel_subagents not found on parent agent"

        ctx = MagicMock()
        ctx.deps = deps

        result = await handler.function(ctx, tasks=["t1", "t2", "t3", "t4", "t5", "t6"])
        assert "Too many tasks" in result
        assert "6" in result

    @pytest.mark.asyncio
    async def test_empty_tasks_returns_error(self, cfg: Config):
        """run_parallel_subagents should handle empty task list."""
        memory = MagicMock()
        bus = EventBus()
        deps = AgentDeps(memory=memory, cfg=cfg, bus=bus)

        agent = create_agent(cfg)
        handler = None
        for ts in agent.toolsets:
            tools = getattr(ts, "tools", None)
            if isinstance(tools, dict) and "run_parallel_subagents" in tools:
                handler = tools["run_parallel_subagents"]
                break

        assert handler is not None

        ctx = MagicMock()
        ctx.deps = deps

        result = await handler.function(ctx, tasks=[])
        assert "No tasks provided" in result
