"""Integration tests for Newton's memory + agent pipeline.

Each test gets a fresh SQLite database. Messages are processed one at a time
through the full pipeline: memory → context → agent → response.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv

# Load .env so OPENROUTER_API_KEY (and other secrets) are available
load_dotenv()

from newton.config import Config, MemoryConfig
from newton.memory import MemoryStore
from newton.context import build_system_prompt
from newton.agent import create_agent, AgentDeps


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def memory(tmp_path: Path) -> MemoryStore:
    """Provide a fresh MemoryStore backed by a temp database."""
    cfg = Config(
        memory=MemoryConfig(db_path=str(tmp_path / "test.db")),
    )
    store = MemoryStore(cfg)
    await store.init()
    yield store
    await store.close()


@pytest.fixture
def cfg() -> Config:
    """Provide a test config (loads defaults + env vars)."""
    return Config()


async def send_message(
    cfg: Config,
    memory: MemoryStore,
    message: str,
    source: str = "test",
) -> str:
    """Process a single message through the full agent pipeline.
    Returns the agent's response text.
    """
    from newton.events import EventBus

    # Save inbound to recall
    await memory.recall_save("user", message, channel=source)

    # Create a throwaway bus for test runs
    bus = EventBus()
    bus.register_channel(source)

    # Run the agent
    agent = create_agent(cfg)
    deps = AgentDeps(memory=memory, cfg=cfg, bus=bus)
    deps.current_message = message

    result = await agent.run(message, deps=deps)

    # Save response to recall
    await memory.recall_save("assistant", result.output, channel=source)

    return result.output


# ---------------------------------------------------------------------------
# Core Memory Tests
# ---------------------------------------------------------------------------

class TestCoreMemory:
    @pytest.mark.asyncio
    async def test_default_blocks_seeded(self, memory: MemoryStore):
        """On first init, persona and directives blocks should exist."""
        blocks = await memory.get_core_blocks()
        assert "persona" in blocks
        assert "directives" in blocks

    @pytest.mark.asyncio
    async def test_update_and_read_block(self, memory: MemoryStore):
        """Updating a core block persists the new value."""
        await memory.update_core_block("persona", "I am a test bot.")
        content = await memory.get_core_block("persona")
        assert content == "I am a test bot."

    @pytest.mark.asyncio
    async def test_create_new_block(self, memory: MemoryStore):
        """Can create a brand-new core memory block."""
        await memory.update_core_block("custom_block", "hello world")
        content = await memory.get_core_block("custom_block")
        assert content == "hello world"

    @pytest.mark.asyncio
    async def test_nonexistent_block_returns_none(self, memory: MemoryStore):
        result = await memory.get_core_block("does_not_exist")
        assert result is None


# ---------------------------------------------------------------------------
# User Registry Tests
# ---------------------------------------------------------------------------

class TestUserRegistry:
    @pytest.mark.asyncio
    async def test_upsert_and_summary(self, memory: MemoryStore):
        """Inserting a user makes them appear in the summary."""
        await memory.upsert_user("macca", name="Mac", relationship="creator")
        users = await memory.get_all_users_summary()
        assert len(users) == 1
        assert users[0]["key"] == "macca"
        assert users[0]["name"] == "Mac"
        assert users[0]["relationship"] == "creator"

    @pytest.mark.asyncio
    async def test_upsert_preserves_existing_fields(self, memory: MemoryStore):
        """Updating with empty fields doesn't overwrite existing data."""
        await memory.upsert_user("macca", name="Mac", relationship="creator")
        await memory.upsert_user("macca", details="Loves Python")
        users = await memory.get_all_users_summary()
        assert users[0]["name"] == "Mac"
        assert users[0]["relationship"] == "creator"

    @pytest.mark.asyncio
    async def test_user_details(self, memory: MemoryStore):
        await memory.upsert_user("macca", details="Created Newton in Feb 2026")
        details = await memory.get_user_details("macca")
        assert details == "Created Newton in Feb 2026"

    @pytest.mark.asyncio
    async def test_unknown_user_details_returns_none(self, memory: MemoryStore):
        details = await memory.get_user_details("nobody")
        assert details is None

    @pytest.mark.asyncio
    async def test_multiple_users(self, memory: MemoryStore):
        await memory.upsert_user("macca", name="Mac", relationship="creator")
        await memory.upsert_user("sarah", name="Sarah", relationship="family")
        users = await memory.get_all_users_summary()
        assert len(users) == 2
        keys = {u["key"] for u in users}
        assert keys == {"macca", "sarah"}


# ---------------------------------------------------------------------------
# Recall Memory Tests
# ---------------------------------------------------------------------------

class TestRecallMemory:
    @pytest.mark.asyncio
    async def test_save_and_retrieve(self, memory: MemoryStore):
        await memory.recall_save("user", "Hello!")
        await memory.recall_save("assistant", "Hi there!")
        history = await memory.recall_recent(n=10)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello!"
        assert history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_recall_window_limits(self, memory: MemoryStore):
        """Only the last N messages are returned."""
        for i in range(20):
            await memory.recall_save("user", f"msg {i}")
        history = await memory.recall_recent(n=5)
        assert len(history) == 5
        assert history[0]["content"] == "msg 15"
        assert history[4]["content"] == "msg 19"

    @pytest.mark.asyncio
    async def test_channel_is_stored(self, memory: MemoryStore):
        await memory.recall_save("user", "Hello", channel="telegram")
        history = await memory.recall_recent(n=1)
        assert history[0]["channel"] == "telegram"


# ---------------------------------------------------------------------------
# Context Assembly Tests
# ---------------------------------------------------------------------------

class TestContext:
    @pytest.mark.asyncio
    async def test_base_instructions_present(self, memory: MemoryStore, cfg: Config):
        prompt = await build_system_prompt(cfg, memory, "")
        assert cfg.llm.system_prompt in prompt

    @pytest.mark.asyncio
    async def test_core_blocks_present(self, memory: MemoryStore, cfg: Config):
        await memory.update_core_block("persona", "I am TestBot.")
        prompt = await build_system_prompt(cfg, memory, "")
        assert "I am TestBot." in prompt
        assert "CORE MEMORY" in prompt

    @pytest.mark.asyncio
    async def test_users_present(self, memory: MemoryStore, cfg: Config):
        await memory.upsert_user("macca", name="Mac", relationship="creator")
        prompt = await build_system_prompt(cfg, memory, "")
        assert "KNOWN USERS" in prompt
        assert "macca" in prompt
        assert "Mac" in prompt
        assert "creator" in prompt

    @pytest.mark.asyncio
    async def test_no_users_message(self, memory: MemoryStore, cfg: Config):
        prompt = await build_system_prompt(cfg, memory, "")
        assert "No known users yet" in prompt

    @pytest.mark.asyncio
    async def test_recall_history_in_prompt(self, memory: MemoryStore, cfg: Config):
        await memory.recall_save("user", "What is 2+2?")
        await memory.recall_save("assistant", "4")
        prompt = await build_system_prompt(cfg, memory, "next question")
        assert "RECENT CONVERSATION" in prompt
        assert "What is 2+2?" in prompt


# ---------------------------------------------------------------------------
# Archival Memory Tests
# ---------------------------------------------------------------------------

class TestArchivalMemory:
    @pytest.mark.asyncio
    async def test_insert_and_search(self, memory: MemoryStore):
        await memory.archival_insert("Python was created by Guido van Rossum")
        await memory.archival_insert("The capital of France is Paris")
        results = await memory.archival_search("Who made Python?", k=1)
        assert len(results) == 1
        assert "Guido" in results[0]

    @pytest.mark.asyncio
    async def test_access_count_bumped(self, memory: MemoryStore):
        """Searching should bump access_count on matched rows."""
        await memory.archival_insert("Newton is an AI agent framework")
        await memory.archival_search("What is Newton?", k=1)
        await memory.archival_search("Tell me about Newton", k=1)

        assert memory._db
        cursor = await memory._db.execute(
            "SELECT access_count FROM archival_memory WHERE id = 1"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 2

    @pytest.mark.asyncio
    async def test_empty_search(self, memory: MemoryStore):
        """Search with no data returns empty list."""
        results = await memory.archival_search("anything", k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Full Pipeline: Diamond Jack
# ---------------------------------------------------------------------------

class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_agent_responds(self, memory: MemoryStore, cfg: Config):
        """Send a message through the full pipeline and get a response."""
        response = await send_message(cfg, memory, "Say exactly: PONG")
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_diamond_jack_in_context(self, memory: MemoryStore, cfg: Config):
        """After telling the agent a name, the next prompt's context
        should include that name in recall history."""
        # First message — introduce ourselves
        await send_message(cfg, memory, "Hey I'm Diamond Jack")

        # Now build the context for a *second* turn
        prompt = await build_system_prompt(cfg, memory, "What's my name?")

        # The recall history should contain "Diamond Jack"
        assert "Diamond Jack" in prompt

    @pytest.mark.asyncio
    async def test_recall_saved_after_exchange(self, memory: MemoryStore, cfg: Config):
        """After processing a message, both user + assistant are in recall."""
        await send_message(cfg, memory, "Hello Newton")
        history = await memory.recall_recent(n=10)
        roles = [m["role"] for m in history]
        assert "user" in roles
        assert "assistant" in roles
