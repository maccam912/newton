"""Newton — entry point.  Wires the event bus, agent, memory, and adapters."""

import asyncio

import litellm
import logfire

from newton.config import load_config
from newton.events import EventBus
from newton.memory import MemoryStore
from newton.agent import run_agent_loop
from newton.telegram_adapter import start_telegram, send_replies
from newton.local_adapter import read_stdin, print_replies
from newton.scheduler import start_scheduler

# Suppress litellm noise
litellm.suppress_debug_info = True

# Logfire — sets up OTEL TracerProvider + sends traces if token present
logfire.configure(send_to_logfire="if-token-present", console=False)
logfire.instrument_pydantic_ai()


async def main() -> None:
    cfg = load_config()
    bus = EventBus()

    memory = MemoryStore(cfg)
    await memory.init()

    print("[newton] Starting up...")

    async with asyncio.TaskGroup() as tg:
        tg.create_task(run_agent_loop(bus, cfg, memory))
        tg.create_task(start_telegram(bus, cfg))
        tg.create_task(send_replies(bus, cfg))
        tg.create_task(read_stdin(bus))
        tg.create_task(print_replies(bus))
        tg.create_task(start_scheduler(bus, cfg))


if __name__ == "__main__":
    asyncio.run(main())
