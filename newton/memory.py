"""Memory — three-tier store inspired by MemGPT/Letta.

Core Memory    → key/value blocks, always in the system prompt.
Archival Memory → long-term text with embeddings, semantic search.
Recall Memory   → conversation history, most-recent-N in prompt.
"""

from __future__ import annotations

import json
import struct
from datetime import datetime, timedelta, timezone

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


def _make_openai_client(base_url: str = "") -> AsyncOpenAI:
    """Create an AsyncOpenAI-compatible embeddings client.

    Resolution order for API key + base URL:
      1. If base_url is explicitly set, use it with OPENROUTER_API_KEY (or OPENAI_API_KEY).
      2. If OPENAI_API_KEY is set, use OpenAI directly (no custom base_url).
      3. Fall back to OpenRouter using OPENROUTER_API_KEY.
    """
    if base_url:
        # Explicit base URL — pick whichever key is available
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ["OPENROUTER_API_KEY"]
        return AsyncOpenAI(base_url=base_url, api_key=api_key)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        # Use OpenAI directly
        return AsyncOpenAI(api_key=openai_key)

    # Fall back to OpenRouter
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


async def _embed(text: str, model: str, client: AsyncOpenAI) -> list[float]:
    """Get an embedding vector via an OpenAI-compatible API."""
    with tracer.start_as_current_span("embed", attributes={"model": model, "text_len": len(text)}):
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
        self._embed_client: AsyncOpenAI = _make_openai_client(cfg.memory.embedding_base_url)

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

            CREATE TABLE IF NOT EXISTS reminders (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                message          TEXT NOT NULL,
                channel          TEXT NOT NULL DEFAULT '',
                metadata         TEXT NOT NULL DEFAULT '{}',
                fire_at          TEXT NOT NULL,
                interval_minutes INTEGER,
                end_at           TEXT,
                active           INTEGER NOT NULL DEFAULT 1,
                created          TEXT NOT NULL,
                last_fired       TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_reminders_active_fire
                ON reminders (active, fire_at);

            CREATE TABLE IF NOT EXISTS skills (
                name        TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                full_prompt TEXT NOT NULL,
                created     TEXT NOT NULL,
                updated     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_summaries (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                summary   TEXT NOT NULL,
                channels  TEXT NOT NULL DEFAULT '',
                msg_count INTEGER NOT NULL DEFAULT 0,
                created   TEXT NOT NULL
            );
        """)
        await self._db.commit()

        # Create the vec0 virtual table for archival search (if missing).
        # We detect the embedding dimension from a test call on first run.
        try:
            await self._db.execute("SELECT * FROM archival_vec LIMIT 0")
        except Exception:
            test_vec = await _embed("hello", self.embedding_model, self._embed_client)
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
            vec = await _embed(content, self.embedding_model, self._embed_client)
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
            vec = await _embed(query, self.embedding_model, self._embed_client)
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

    async def recall_get_all(self) -> list[dict[str, str]]:
        """Return all recall rows chronologically (for session summarization)."""
        with tracer.start_as_current_span("memory.recall_get_all") as span:
            assert self._db
            cursor = await self._db.execute(
                "SELECT role, content, channel, timestamp, handled "
                "FROM recall_memory ORDER BY timestamp ASC, id ASC"
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
                for row in rows
            ]
            span.set_attribute("result_count", len(result))
            return result

    async def recall_clear(self) -> int:
        """Delete all recall rows. Returns count of deleted rows."""
        with tracer.start_as_current_span("memory.recall_clear") as span:
            assert self._db
            cursor = await self._db.execute("SELECT COUNT(*) FROM recall_memory")
            row = await cursor.fetchone()
            count = row[0] if row else 0
            await self._db.execute("DELETE FROM recall_memory")
            await self._db.commit()
            span.set_attribute("deleted_count", count)
            return count

    async def recall_count(self) -> int:
        """Return the number of recall rows."""
        with tracer.start_as_current_span("memory.recall_count") as span:
            assert self._db
            cursor = await self._db.execute("SELECT COUNT(*) FROM recall_memory")
            row = await cursor.fetchone()
            count = row[0] if row else 0
            span.set_attribute("count", count)
            return count

    async def session_summary_save(self, summary: str, channels: str, msg_count: int) -> int:
        """Insert a session summary row. Returns the row id."""
        with tracer.start_as_current_span("memory.session_summary_save") as span:
            assert self._db
            now = datetime.now(timezone.utc).isoformat()
            cursor = await self._db.execute(
                "INSERT INTO session_summaries (summary, channels, msg_count, created) "
                "VALUES (?, ?, ?, ?)",
                (summary, channels, msg_count, now),
            )
            await self._db.commit()
            row_id = cursor.lastrowid or 0
            span.set_attribute("row_id", row_id)
            return row_id

    async def session_summaries_recent(self, n: int = 3) -> list[dict]:
        """Return the last N session summaries, oldest first."""
        with tracer.start_as_current_span("memory.session_summaries_recent", attributes={"n": n}) as span:
            assert self._db
            cursor = await self._db.execute(
                "SELECT id, summary, channels, msg_count, created "
                "FROM session_summaries ORDER BY id DESC LIMIT ?",
                (n,),
            )
            rows = await cursor.fetchall()
            result = [
                {
                    "id": row[0],
                    "summary": row[1],
                    "channels": row[2],
                    "msg_count": row[3],
                    "created": row[4],
                }
                for row in reversed(rows)  # oldest first
            ]
            span.set_attribute("result_count", len(result))
            return result

    # ------------------------------------------------------------------
    # Reminders
    # ------------------------------------------------------------------

    async def create_reminder(
        self,
        message: str,
        fire_at: datetime,
        channel: str = "",
        metadata: dict[str, str] | None = None,
        interval_minutes: int | None = None,
        end_at: datetime | None = None,
    ) -> int:
        """Create a new scheduled reminder. Returns the row id."""
        with tracer.start_as_current_span("memory.create_reminder") as span:
            assert self._db
            now = datetime.now(timezone.utc).isoformat()
            meta_json = json.dumps(metadata or {})
            cursor = await self._db.execute(
                "INSERT INTO reminders "
                "(message, channel, metadata, fire_at, interval_minutes, end_at, active, created) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                (
                    message,
                    channel,
                    meta_json,
                    fire_at.isoformat(),
                    interval_minutes,
                    end_at.isoformat() if end_at else None,
                    now,
                ),
            )
            await self._db.commit()
            row_id = cursor.lastrowid or 0
            span.set_attribute("row_id", row_id)
            return row_id

    async def get_due_reminders(self) -> list[dict]:
        """Return all active reminders whose fire_at <= now (UTC)."""
        assert self._db
        now = datetime.now(timezone.utc).isoformat()
        cursor = await self._db.execute(
            "SELECT id, message, channel, metadata, fire_at, interval_minutes, end_at "
            "FROM reminders WHERE active = 1 AND fire_at <= ?",
            (now,),
        )
        rows = await cursor.fetchall()
        result = [
            {
                "id": row[0],
                "message": row[1],
                "channel": row[2],
                "metadata": row[3],
                "fire_at": row[4],
                "interval_minutes": row[5],
                "end_at": row[6],
            }
            for row in rows
        ]
        if result:
            with tracer.start_as_current_span("memory.get_due_reminders") as span:
                span.set_attribute("result_count", len(result))
        return result

    async def advance_or_deactivate_reminder(self, reminder_id: int) -> None:
        """After firing: advance recurring reminders or deactivate one-time ones."""
        with tracer.start_as_current_span(
            "memory.advance_reminder", attributes={"reminder_id": reminder_id}
        ):
            assert self._db
            now = datetime.now(timezone.utc)
            cursor = await self._db.execute(
                "SELECT interval_minutes, end_at, fire_at FROM reminders WHERE id = ?",
                (reminder_id,),
            )
            row = await cursor.fetchone()
            if not row:
                return

            interval, end_at_str, fire_at_str = row
            fire_at = datetime.fromisoformat(fire_at_str)

            if interval is None:
                # One-time: deactivate
                await self._db.execute(
                    "UPDATE reminders SET active = 0, last_fired = ? WHERE id = ?",
                    (now.isoformat(), reminder_id),
                )
            else:
                # Recurring: skip forward past now
                delta = timedelta(minutes=interval)
                next_fire = fire_at
                while next_fire <= now:
                    next_fire += delta

                if end_at_str and next_fire > datetime.fromisoformat(end_at_str):
                    await self._db.execute(
                        "UPDATE reminders SET active = 0, last_fired = ? WHERE id = ?",
                        (now.isoformat(), reminder_id),
                    )
                else:
                    await self._db.execute(
                        "UPDATE reminders SET fire_at = ?, last_fired = ? WHERE id = ?",
                        (next_fire.isoformat(), now.isoformat(), reminder_id),
                    )
            await self._db.commit()

    async def list_active_reminders(self) -> list[dict]:
        """Return all active reminders."""
        with tracer.start_as_current_span("memory.list_active_reminders") as span:
            assert self._db
            cursor = await self._db.execute(
                "SELECT id, message, channel, fire_at, interval_minutes, end_at "
                "FROM reminders WHERE active = 1 ORDER BY fire_at"
            )
            rows = await cursor.fetchall()
            result = [
                {
                    "id": row[0],
                    "message": row[1],
                    "channel": row[2],
                    "fire_at": row[3],
                    "interval_minutes": row[4],
                    "end_at": row[5],
                }
                for row in rows
            ]
            span.set_attribute("result_count", len(result))
            return result

    async def cancel_reminder(self, reminder_id: int) -> bool:
        """Deactivate a reminder by ID. Returns True if found and cancelled."""
        with tracer.start_as_current_span(
            "memory.cancel_reminder", attributes={"reminder_id": reminder_id}
        ):
            assert self._db
            cursor = await self._db.execute(
                "UPDATE reminders SET active = 0 WHERE id = ? AND active = 1",
                (reminder_id,),
            )
            await self._db.commit()
            return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    async def skill_create(self, name: str, description: str, full_prompt: str) -> None:
        """Create a new skill. Raises ValueError if name already exists."""
        with tracer.start_as_current_span("memory.skill_create", attributes={"name": name}):
            assert self._db
            now = datetime.now(timezone.utc).isoformat()
            try:
                await self._db.execute(
                    "INSERT INTO skills (name, description, full_prompt, created, updated) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (name, description, full_prompt, now, now),
                )
                await self._db.commit()
            except aiosqlite.IntegrityError:
                raise ValueError(f"Skill '{name}' already exists")

    async def skill_update(
        self, name: str, description: str | None = None, full_prompt: str | None = None
    ) -> bool:
        """Update an existing skill. Returns True if found and updated."""
        with tracer.start_as_current_span("memory.skill_update", attributes={"name": name}):
            assert self._db
            now = datetime.now(timezone.utc).isoformat()
            sets: list[str] = ["updated = ?"]
            params: list[str] = [now]
            if description is not None:
                sets.append("description = ?")
                params.append(description)
            if full_prompt is not None:
                sets.append("full_prompt = ?")
                params.append(full_prompt)
            params.append(name)
            cursor = await self._db.execute(
                f"UPDATE skills SET {', '.join(sets)} WHERE name = ?",
                params,
            )
            await self._db.commit()
            return cursor.rowcount > 0

    async def skill_delete(self, name: str) -> bool:
        """Delete a skill by name. Returns True if found and deleted."""
        with tracer.start_as_current_span("memory.skill_delete", attributes={"name": name}):
            assert self._db
            cursor = await self._db.execute(
                "DELETE FROM skills WHERE name = ?", (name,)
            )
            await self._db.commit()
            return cursor.rowcount > 0

    async def skill_list(self) -> list[dict[str, str]]:
        """Return all skills as [{name, description}] — no full_prompt."""
        with tracer.start_as_current_span("memory.skill_list") as span:
            assert self._db
            cursor = await self._db.execute(
                "SELECT name, description FROM skills ORDER BY name"
            )
            rows = await cursor.fetchall()
            result = [{"name": row[0], "description": row[1]} for row in rows]
            span.set_attribute("result_count", len(result))
            return result

    async def skill_get(self, name: str) -> dict[str, str] | None:
        """Return the full skill record, or None if not found."""
        with tracer.start_as_current_span("memory.skill_get", attributes={"name": name}):
            assert self._db
            cursor = await self._db.execute(
                "SELECT name, description, full_prompt FROM skills WHERE name = ?",
                (name,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return {"name": row[0], "description": row[1], "full_prompt": row[2]}
