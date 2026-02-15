"""Scheduler — heartbeat timer that pushes events onto the inbox."""

from __future__ import annotations

from apscheduler import AsyncScheduler
from apscheduler.triggers.interval import IntervalTrigger

from newton.config import Config
from newton.events import Event, EventBus, EventKind


async def _heartbeat(bus: EventBus) -> None:
    await bus.put_inbox(
        Event(source="scheduler", kind=EventKind.HEARTBEAT, payload="ping")
    )


async def start_scheduler(bus: EventBus, cfg: Config) -> None:
    """Fire up APScheduler with a heartbeat interval."""
    async with AsyncScheduler() as scheduler:
        await scheduler.add_schedule(
            _heartbeat,
            IntervalTrigger(minutes=cfg.scheduler.heartbeat_minutes),
            args=[bus],
        )
        await scheduler.run_until_stopped()
