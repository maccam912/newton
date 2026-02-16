"""Unit tests for newton.events."""

from __future__ import annotations

import asyncio

import pytest

from newton.events import Event, EventBus, EventKind


pytestmark = pytest.mark.unit


class TestEvent:
    def test_create_message(self):
        e = Event(source="local", kind=EventKind.MESSAGE, payload="hello")
        assert e.source == "local"
        assert e.kind == EventKind.MESSAGE
        assert e.payload == "hello"
        assert e.timestamp is not None

    def test_defaults(self):
        e = Event(source="test", kind=EventKind.HEARTBEAT)
        assert e.payload == ""
        assert e.reply_to == ""
        assert e.metadata == {}


class TestEventBus:
    async def test_inbox(self):
        bus = EventBus()
        event = Event(source="test", kind=EventKind.MESSAGE, payload="hi")
        await bus.put_inbox(event)
        got = await asyncio.wait_for(bus.inbox.get(), timeout=1)
        assert got.payload == "hi"

    async def test_outbox_routing(self):
        bus = EventBus()
        q = bus.register_channel("telegram")
        event = Event(
            source="agent", kind=EventKind.RESPONSE,
            payload="reply", reply_to="telegram",
        )
        await bus.put_outbox(event)
        got = await asyncio.wait_for(q.get(), timeout=1)
        assert got.payload == "reply"

    async def test_outbox_unknown_channel(self):
        bus = EventBus()
        event = Event(
            source="agent", kind=EventKind.RESPONSE,
            payload="drop me", reply_to="nonexistent",
        )
        # Should not raise, just logs a warning
        await bus.put_outbox(event)
