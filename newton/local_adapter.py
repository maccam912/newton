"""Local CLI adapter — type in the terminal, see agent replies."""

from __future__ import annotations

import asyncio
import sys

from newton.events import Event, EventBus, EventKind

CHANNEL = "local"


def _has_stdin() -> bool:
    """Return True if stdin is usable (False under nohup / daemonised)."""
    try:
        return sys.stdin is not None and sys.stdin.fileno() >= 0
    except Exception:
        return False


async def read_stdin(bus: EventBus) -> None:
    """Read lines from stdin and push them onto the inbox."""
    if not _has_stdin():
        print("[local] No stdin available — local input disabled.")
        return

    loop = asyncio.get_running_loop()
    print("[local] Type a message and press Enter. Ctrl+C to quit.\n")

    while True:
        try:
            line = await loop.run_in_executor(None, input, "you> ")
        except (EOFError, KeyboardInterrupt, OSError):
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
        try:
            print(f"\nnewton> {event.payload}\n")
        except OSError:
            pass
