"""Scheduler — heartbeat timer and reminder checker."""

from __future__ import annotations

import json

from apscheduler import AsyncScheduler
from apscheduler.triggers.interval import IntervalTrigger

from newton.config import Config
from newton.events import Event, EventBus, EventKind
from newton.memory import MemoryStore


async def _heartbeat(bus: EventBus) -> None:
    await bus.put_inbox(
        Event(source="scheduler", kind=EventKind.HEARTBEAT, payload="ping")
    )


async def _check_reminders(bus: EventBus, memory: MemoryStore) -> None:
    """Find due reminders, fire them as REMINDER events, advance/deactivate."""
    due = await memory.get_due_reminders()
    for r in due:
        metadata = json.loads(r["metadata"]) if r["metadata"] else {}
        await bus.put_inbox(
            Event(
                source="scheduler",
                kind=EventKind.REMINDER,
                payload=r["message"],
                reply_to=r["channel"],
                metadata=metadata,
            )
        )
        await memory.advance_or_deactivate_reminder(r["id"])


async def start_scheduler(bus: EventBus, cfg: Config, memory: MemoryStore) -> None:
    """Fire up APScheduler with a heartbeat interval and reminder checker."""
    async with AsyncScheduler() as scheduler:
        await scheduler.add_schedule(
            _heartbeat,
            IntervalTrigger(minutes=cfg.scheduler.heartbeat_minutes),
            args=[bus],
        )
        await scheduler.add_schedule(
            _check_reminders,
            IntervalTrigger(seconds=cfg.scheduler.reminder_check_seconds),
            args=[bus, memory],
        )
        await scheduler.run_until_stopped()
