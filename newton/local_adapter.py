"""Local CLI adapter — type in the terminal, see agent replies."""

from __future__ import annotations

import asyncio

from newton.events import Event, EventBus, EventKind

CHANNEL = "local"


async def read_stdin(bus: EventBus) -> None:
    """Read lines from stdin and push them onto the inbox."""
    loop = asyncio.get_running_loop()
    print("[local] Type a message and press Enter. Ctrl+C to quit.\n")

    while True:
        try:
            line = await loop.run_in_executor(None, input, "you> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not line.strip():
            continue
        await bus.put_inbox(
            Event(source=CHANNEL, kind=EventKind.MESSAGE, payload=line)
        )


async def print_replies(bus: EventBus) -> None:
    """Pull response events from this channel's outbox and print them."""
    outbox = bus.register_channel(CHANNEL)
    while True:
        event = await outbox.get()
        print(f"\nnewton> {event.payload}\n")
