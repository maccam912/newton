"""Unit tests for context assembly and cache-friendly prompt shape."""

from __future__ import annotations

import pytest

from newton.config import Config
from newton.context import build_system_prefix, build_system_prompt, build_turn_context_suffix


pytestmark = pytest.mark.unit


class DummyMemory:
    async def context_collapse_get_note(self):
        return "Working on quarterly plan; next step is budget review."

    async def get_core_blocks(self):
        return {"notebook": "remember this"}

    async def get_all_users_summary(self):
        return [{"key": "u1", "name": "Ada", "relationship": "friend"}]

    async def skill_list(self):
        return [{"name": "skill-a", "description": "does things"}]

    async def session_summaries_recent(self, n: int):
        return [{"created": "2026-01-02T03:04:05", "msg_count": 2, "channels": "local", "summary": "talked"}]

    async def archival_search(self, incoming: str, k: int):
        return [f"fact about {incoming}"]

    async def recall_recent(self, n: int):
        return [{"role": "user", "content": "hello", "handled": "1"}]


async def test_build_system_prefix_contains_stable_sections():
    cfg = Config()
    memory = DummyMemory()
    prefix = await build_system_prefix(cfg, memory)
    assert "--- CONTEXT COLLAPSE NOTE ---" in prefix
    assert "quarterly plan" in prefix
    assert "--- CORE MEMORY ---" in prefix
    assert "--- KNOWN USERS ---" in prefix
    assert "--- AVAILABLE SKILLS ---" in prefix
    assert "--- PREVIOUS SESSIONS ---" in prefix


async def test_build_turn_context_suffix_contains_variable_sections():
    cfg = Config()
    memory = DummyMemory()
    suffix = await build_turn_context_suffix(cfg, memory, "meeting")
    assert "--- RELEVANT MEMORIES ---" in suffix
    assert "fact about meeting" in suffix
    assert "--- RECENT CONVERSATION ---" in suffix


async def test_build_system_prompt_uses_cached_prefix():
    cfg = Config()
    memory = DummyMemory()
    cached = "STATIC PREFIX"
    prompt = await build_system_prompt(cfg, memory, "meeting", cached_prefix=cached)
    assert prompt.startswith("STATIC PREFIX")
    assert "fact about meeting" in prompt
