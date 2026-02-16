"""Event model and EventBus — the central nervous system of Newton."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum

from opentelemetry import trace
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Colored log formatter
# ---------------------------------------------------------------------------

_COLORS = {
    "DEBUG":    "\033[36m",    # cyan
    "INFO":     "\033[34m",    # blue
    "WARNING":  "\033[33m",    # yellow
    "ERROR":    "\033[31m",    # red
    "CRITICAL": "\033[1;31m",  # bold red
}
_RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    """Compact colored formatter for terminal output."""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        ts = self.formatTime(record, "%H:%M:%S")
        return f"{color}{ts} [{record.name}] {record.getMessage()}{_RESET}"


def _make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger


log = _make_logger("newton.bus")


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------

class EventKind(str, Enum):
    """The flavour of event flowing through the bus."""
    MESSAGE = "message"        # a chat message arrived
    HEARTBEAT = "heartbeat"    # periodic pulse
    RESPONSE = "response"      # agent is replying
    REMINDER = "reminder"      # scheduled reminder fired


class Event(BaseModel):
    """A single thing that happened."""
    source: str                         # e.g. "telegram", "local", "scheduler"
    kind: EventKind
    payload: str = ""                   # free-form content
    reply_to: str = ""                  # channel to route the response back to
    metadata: dict[str, str] = {}       # channel-specific context (e.g. chat_id)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EventBus:
    """Shared inbox + per-channel outbox routing."""

    def __init__(self) -> None:
        self.inbox: asyncio.Queue[Event] = asyncio.Queue()
        self._outboxes: dict[str, asyncio.Queue[Event]] = {}

    def register_channel(self, name: str) -> asyncio.Queue[Event]:
        """Register an outbox queue for a named channel (e.g. 'telegram', 'local')."""
        queue: asyncio.Queue[Event] = asyncio.Queue()
        self._outboxes[name] = queue
        log.info("channel registered: %s", name)
        return queue

    async def put_inbox(self, event: Event) -> None:
        """Put an event onto the inbox (with logging + tracing)."""
        log.info(
            "→ inbox  | %s.%s  %s",
            event.source,
            event.kind.value,
            _truncate(event.payload),
        )
        current_span = trace.get_current_span()
        current_span.add_event("bus.inbox", attributes={
            "source": event.source, "kind": event.kind.value,
            "payload_len": len(event.payload),
        })
        await self.inbox.put(event)

    async def put_outbox(self, event: Event) -> None:
        """Route a response event to the correct channel's outbox."""
        queue = self._outboxes.get(event.reply_to)
        if queue:
            log.info(
                "← outbox | %s → %s  %s",
                event.source,
                event.reply_to,
                _truncate(event.payload),
            )
            current_span = trace.get_current_span()
            current_span.add_event("bus.outbox", attributes={
                "source": event.source, "reply_to": event.reply_to,
                "payload_len": len(event.payload),
            })
            await queue.put(event)
        else:
            log.warning("no outbox for channel %r, dropping event", event.reply_to)


def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate long payloads for readable log lines."""
    text = text.replace("\n", " ")
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text
