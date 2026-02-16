"""Memory — three-tier store inspired by MemGPT/Letta.

Core Memory    → key/value blocks, always in the system prompt.
Archival Memory → long-term text with embeddings, semantic search.
Recall Memory   → conversation history, most-recent-N in prompt.
"""

from __future__ import annotations

import json
import struct
from datetime import datetime, timezone

import aiosqlite
import sqlite_vec
import os

from openai import AsyncOpenAI

from newton.config import Config
from newton.tracing import get_tracer

tracer = get_tracer("newton.memory")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_embedding(vec: list[float]) -> bytes:
    """Pack a float list into a compact binary blob for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


_openai_client: AsyncOpenAI | None = None


def _get_openai_client() -> AsyncOpenAI:
    """Lazy-init an AsyncOpenAI client pointed at OpenRouter."""
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    return _openai_client


async def _embed(text: str, model: str) -> list[float]:
    """Get an embedding vector via OpenRouter (OpenAI-compatible API)."""
    with tracer.start_as_current_span("embed", attributes={"model": model, "text_len": len(text)}):
        client = _get_openai_client()
        resp = await client.embeddings.create(model=model, input=[text])
        return list(resp.data[0].embedding)


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

# Default core memory blocks seeded on first run.
_DEFAULT_CORE_BLOCKS: dict[str, str] = {
    "persona": "I am Newton, a helpful AI assistant.",
    "directives": "Be concise.  Ask clarifying questions when unsure.",
    "notebook": "",
}


class MemoryStore:
    """Unified memory store backed by a single SQLite database."""

    def __init__(self, cfg: Config) -> None:
        self.db_path = cfg.memory.db_path
        self.embedding_model = cfg.memory.embedding_model
        self._db: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """Open the database and create tables."""
        self._db = await aiosqlite.connect(self.db_path, check_same_thread=False)
        # Load the sqlite-vec extension
        self._db._connection.enable_load_extension(True)
        sqlite_vec.load(self._db._connection)  # type: ignore[arg-type]
        self._db._connection.enable_load_extension(False)

        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS core_memory (
                block   TEXT PRIMARY KEY,
                content TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS users (
                key          TEXT PRIMARY KEY,
                name         TEXT NOT NULL DEFAULT '',
                relationship TEXT NOT NULL DEFAULT '',
                details      TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS archival_memory (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                content       TEXT NOT NULL,
                embedding     BLOB NOT NULL,
                created       TEXT NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0,
                last_accessed TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS recall_memory (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                role      TEXT NOT NULL,
                channel   TEXT NOT NULL DEFAULT '',
                content   TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                handled   INTEGER NOT NULL DEFAULT 0
            );
        """)
        await self._db.commit()

        # Create the vec0 virtual table for archival search (if missing).
        # We detect the embedding dimension from a test call on first run.
        try:
            await self._db.execute("SELECT * FROM archival_vec LIMIT 0")
        except Exception:
            test_vec = await _embed("hello", self.embedding_model)
            dim = len(test_vec)
            await self._db.execute(
                f"CREATE VIRTUAL TABLE archival_vec USING vec0("
                f"  id INTEGER PRIMARY KEY,"
                f"  embedding float[{dim}]"
                f")"
            )
            await self._db.commit()

        # Seed any missing default core blocks
        cursor = await self._db.execute("SELECT block FROM core_memory")
        existing = {row[0] for row in await cursor.fetchall()}
        for block, content in _DEFAULT_CORE_BLOCKS.items():
            if block not in existing:
                await self._db.execute(
                    "INSERT INTO core_memory (block, content) VALUES (?, ?)",
                    (block, content),
                )
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    # ------------------------------------------------------------------
    # Core Memory
    # ------------------------------------------------------------------

    async def get_core_blocks(self) -> dict[str, str]:
        """Return all core memory blocks as {block: content}."""
        with tracer.start_as_current_span("memory.get_core_blocks"):
            assert self._db
            cursor = await self._db.execute("SELECT block, content FROM core_memory")
            rows = await cursor.fetchall()
            return {row[0]: row[1] for row in rows}

    async def get_core_block(self, block: str) -> str | None:
        """Read a single core block."""
        with tracer.start_as_current_span("memory.get_core_block", attributes={"block": block}):
            assert self._db
            cursor = await self._db.execute(
                "SELECT content FROM core_memory WHERE block = ?", (block,)
            )
            row = await cursor.fetchone()
            return row[0] if row else None

    async def update_core_block(self, block: str, content: str) -> None:
        """Create or overwrite a core memory block."""
        with tracer.start_as_current_span("memory.update_core_block", attributes={"block": block}):
            assert self._db
            await self._db.execute(
                "INSERT OR REPLACE INTO core_memory (block, content) VALUES (?, ?)",
                (block, content),
            )
            await self._db.commit()

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    async def get_all_users_summary(self) -> list[dict[str, str]]:
        """Return all users with key, name, relationship (no details)."""
        with tracer.start_as_current_span("memory.get_all_users_summary"):
            assert self._db
            cursor = await self._db.execute(
                "SELECT key, name, relationship FROM users"
            )
            rows = await cursor.fetchall()
            return [
                {"key": row[0], "name": row[1], "relationship": row[2]}
                for row in rows
            ]

    async def get_user_details(self, key: str) -> str | None:
        """Return the detailed info for a specific user."""
        with tracer.start_as_current_span("memory.get_user_details", attributes={"user_key": key}):
            assert self._db
            cursor = await self._db.execute(
                "SELECT details FROM users WHERE key = ?", (key,)
            )
            row = await cursor.fetchone()
            return row[0] if row else None

    async def upsert_user(
        self,
        key: str,
        name: str = "",
        relationship: str = "",
        details: str = "",
    ) -> None:
        """Create or update a user record."""
        with tracer.start_as_current_span("memory.upsert_user", attributes={"user_key": key, "name": name}):
            assert self._db
            await self._db.execute(
                "INSERT INTO users (key, name, relationship, details) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "  name = COALESCE(NULLIF(excluded.name, ''), name), "
                "  relationship = COALESCE(NULLIF(excluded.relationship, ''), relationship), "
                "  details = COALESCE(NULLIF(excluded.details, ''), details)",
                (key, name, relationship, details),
            )
            await self._db.commit()

    # ------------------------------------------------------------------
    # Archival Memory
    # ------------------------------------------------------------------

    async def archival_insert(self, content: str) -> int:
        """Store a piece of knowledge with its embedding. Returns the id."""
        with tracer.start_as_current_span("memory.archival_insert", attributes={"content_len": len(content)}) as span:
            assert self._db
            vec = await _embed(content, self.embedding_model)
            blob = _serialize_embedding(vec)
            now = datetime.now(timezone.utc).isoformat()

            cursor = await self._db.execute(
                "INSERT INTO archival_memory (content, embedding, created) "
                "VALUES (?, ?, ?)",
                (content, blob, now),
            )
            row_id = cursor.lastrowid
            assert row_id is not None

            await self._db.execute(
                "INSERT INTO archival_vec (id, embedding) VALUES (?, ?)",
                (row_id, blob),
            )
            await self._db.commit()
            span.set_attribute("row_id", row_id)
            return row_id

    async def archival_search(self, query: str, k: int = 5) -> list[str]:
        """Semantic search over archival memory.  Returns top-k content strings."""
        with tracer.start_as_current_span("memory.archival_search", attributes={"query": query[:80], "k": k}) as span:
            assert self._db
            vec = await _embed(query, self.embedding_model)
            blob = _serialize_embedding(vec)

            cursor = await self._db.execute(
                """
                SELECT a.id, a.content
                FROM archival_vec AS v
                JOIN archival_memory AS a ON a.id = v.id
                WHERE v.embedding MATCH ?
                  AND k = ?
                ORDER BY v.distance
                """,
                (blob, k),
            )
            rows = await cursor.fetchall()
            span.set_attribute("result_count", len(rows))

            if rows:
                now = datetime.now(timezone.utc).isoformat()
                for row in rows:
                    await self._db.execute(
                        "UPDATE archival_memory "
                        "SET access_count = access_count + 1, last_accessed = ? "
                        "WHERE id = ?",
                        (now, row[0]),
                    )
                await self._db.commit()

            return [row[1] for row in rows]

    # ------------------------------------------------------------------
    # Recall Memory
    # ------------------------------------------------------------------

    async def recall_save(
        self, role: str, content: str, channel: str = "", handled: bool = False
    ) -> int:
        """Append a message to the conversation log. Returns the row id."""
        with tracer.start_as_current_span("memory.recall_save", attributes={"role": role, "channel": channel}) as span:
            assert self._db
            now = datetime.now(timezone.utc).isoformat()
            cursor = await self._db.execute(
                "INSERT INTO recall_memory (role, channel, content, timestamp, handled) "
                "VALUES (?, ?, ?, ?, ?)",
                (role, channel, content, now, int(handled)),
            )
            await self._db.commit()
            row_id = cursor.lastrowid or 0
            span.set_attribute("row_id", row_id)
            return row_id

    async def recall_mark_handled(self) -> None:
        """Mark all unhandled user messages as handled."""
        with tracer.start_as_current_span("memory.recall_mark_handled"):
            assert self._db
            await self._db.execute(
                "UPDATE recall_memory SET handled = 1 WHERE handled = 0"
            )
            await self._db.commit()

    async def recall_recent(self, n: int = 10) -> list[dict[str, str]]:
        """Return the last N messages ordered by timestamp."""
        with tracer.start_as_current_span("memory.recall_recent", attributes={"n": n}) as span:
            assert self._db
            cursor = await self._db.execute(
                "SELECT role, content, channel, timestamp, handled "
                "FROM recall_memory "
                "ORDER BY timestamp DESC, id DESC LIMIT ?",
                (n,),
            )
            rows = await cursor.fetchall()
            result = [
                {
                    "role": row[0],
                    "content": row[1],
                    "channel": row[2],
                    "timestamp": row[3],
                    "handled": str(row[4]),
                }
                for row in reversed(rows)
            ]
            span.set_attribute("result_count", len(result))
            return result
