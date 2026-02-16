"""Agent — agentic loop with step tracking, memory tools, and channel I/O."""

from __future__ import annotations

import logging

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings

from newton.config import Config
from datetime import datetime

from newton.context import build_heartbeat_prompt, build_system_prompt
from newton.events import Event, EventBus, EventKind, ColorFormatter
from newton.memory import MemoryStore
from newton.tools.browser import create_browser_server
from opentelemetry import trace
from newton.tracing import get_tracer

atracer = get_tracer("newton.agent")


def _make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger


log = _make_logger("newton.agent")


# ---------------------------------------------------------------------------
# Agent deps — passed to every tool call via RunContext
# ---------------------------------------------------------------------------

class AgentDeps:
    """Runtime dependencies injected into every tool and system prompt call."""

    def __init__(
        self,
        memory: MemoryStore,
        cfg: Config,
        bus: EventBus,
    ) -> None:
        self.memory = memory
        self.cfg = cfg
        self.bus = bus
        self.current_message: str = ""
        # Per-turn state (reset each turn)
        self.step: int = 0
        self.max_steps: int = cfg.agent.max_steps
        self.event_metadata: dict[str, str] = {}
        self.event_source: str = ""
        self.turn_ended: bool = False
        self.response_count: int = 0
        self.is_heartbeat: bool = False


def _step_tag(deps: AgentDeps) -> str:
    """Increment the step counter and return a tag the agent can see."""
    deps.step += 1
    if deps.step >= deps.max_steps:
        log.warning("step %d/%d — LIMIT REACHED", deps.step, deps.max_steps)
        return (
            f"\n⚠ [step {deps.step}/{deps.max_steps} — LIMIT REACHED. "
            f"You MUST call end_turn now.]"
        )
    return f"\n[step {deps.step}/{deps.max_steps}]"


# ---------------------------------------------------------------------------
# Build agent + register tools
# ---------------------------------------------------------------------------

def create_agent(cfg: Config) -> Agent[AgentDeps, str]:
    """Build the agent with memory, communication, and control-flow tools."""
    model = OpenRouterModel(cfg.llm.model)
    browser_server = create_browser_server(cfg)
    model_settings = OpenRouterModelSettings(
        openrouter_reasoning={"effort": cfg.llm.reasoning_effort},
    )
    agent: Agent[AgentDeps, str] = Agent(
        model,
        deps_type=AgentDeps,
        toolsets=[browser_server],
        model_settings=model_settings,
    )

    # -- Dynamic system prompt built from memory each turn -----------------

    @agent.system_prompt
    async def dynamic_prompt(ctx: RunContext[AgentDeps]) -> str:
        if ctx.deps.is_heartbeat:
            prompt = await build_heartbeat_prompt(ctx.deps.cfg, ctx.deps.memory)
            prompt += (
                f"\n\n--- TURN INFO ---\n"
                f"Step budget: {ctx.deps.max_steps} tool calls per turn.\n"
                f"Call respond_to_user to send messages. "
                f"Call end_turn when you are finished."
            )
            return prompt

        prompt = await build_system_prompt(
            ctx.deps.cfg, ctx.deps.memory, ctx.deps.current_message
        )
        prompt += (
            f"\n\n--- TURN INFO ---\n"
            f"This message arrived on channel: {ctx.deps.event_source}\n"
            f"Step budget: {ctx.deps.max_steps} tool calls per turn.\n"
            f"Call respond_to_user to send messages. "
            f"Call end_turn when you are finished.\n\n"
            f"You have a 'notebook' core memory block that is yours to maintain. "
            f"Use core_memory_update(block='notebook', content=...) to write to it. "
            f"Anything you put there will be included in your system prompt on "
            f"every future turn — use it for things you want to always have in "
            f"mind: important patterns, ongoing tasks, reminders, or anything "
            f"you find essential enough to keep front-and-center."
        )
        return prompt

    # -- Register external tools -------------------------------------------
    import newton.tools.searxng
    import newton.tools.scripts

    newton.tools.searxng.register(agent)
    newton.tools.scripts.register(agent)

    import newton.tools.bash
    newton.tools.bash.register(agent)

    # == Control-flow tools ================================================

    @agent.tool
    async def end_turn(ctx: RunContext[AgentDeps]) -> str:
        """Signal that you are done processing this turn.
        Call this when you have nothing more to do."""
        ctx.deps.turn_ended = True

        # Cancel typing indicator for the originating chat
        from newton.telegram_adapter import _stop_typing
        chat_id = ctx.deps.event_metadata.get("chat_id")
        if chat_id:
            _stop_typing(chat_id)

        log.info("🛑 end_turn called  (responses_sent=%d)", ctx.deps.response_count)
        return "Turn ended. Produce a brief internal summary of what you did."

    @agent.tool
    async def respond_to_user(
        ctx: RunContext[AgentDeps], channel: str, message: str
    ) -> str:
        """Send a visible message to a channel (e.g. 'local', 'telegram').
        You can respond to any channel, not just the one that sent the message."""
        ctx.deps.response_count += 1
        log.info("💬 respond_to_user → %s: %s", channel, message[:120])
        await ctx.deps.bus.put_outbox(
            Event(
                source="agent",
                kind=EventKind.RESPONSE,
                payload=message,
                reply_to=channel,
                metadata=ctx.deps.event_metadata,
            )
        )
        return f"Message sent to '{channel}'." + _step_tag(ctx.deps)

    # == Memory tools ======================================================

    @agent.tool
    async def core_memory_read(ctx: RunContext[AgentDeps], block: str) -> str:
        """Read a core memory block (e.g. 'persona', 'directives')."""
        log.debug("🧠 core_memory_read(%s)", block)
        content = await ctx.deps.memory.get_core_block(block)
        result = content or f"[block '{block}' not found]"
        return result + _step_tag(ctx.deps)

    @agent.tool
    async def core_memory_update(
        ctx: RunContext[AgentDeps], block: str, content: str
    ) -> str:
        """Overwrite a core memory block with new content."""
        log.info("🧠 core_memory_update(%s) → %s", block, content[:80])
        await ctx.deps.memory.update_core_block(block, content)
        return f"Core memory block '{block}' updated." + _step_tag(ctx.deps)

    @agent.tool
    async def archival_memory_insert(
        ctx: RunContext[AgentDeps], content: str
    ) -> str:
        """Store a piece of knowledge in long-term archival memory."""
        log.info("📦 archival_insert: %s", content[:80])
        row_id = await ctx.deps.memory.archival_insert(content)
        return f"Stored in archival memory (id={row_id})." + _step_tag(ctx.deps)

    @agent.tool
    async def archival_memory_search(
        ctx: RunContext[AgentDeps], query: str, k: int = 5
    ) -> str:
        """Search archival memory semantically.  Returns the top-k matches."""
        log.debug("🔍 archival_search(%s, k=%d)", query[:60], k)
        results = await ctx.deps.memory.archival_search(query, k)
        if not results:
            return "No archival memories found." + _step_tag(ctx.deps)
        body = "\n".join(f"- {r}" for r in results)
        return body + _step_tag(ctx.deps)

    # == User management tools =============================================

    @agent.tool
    async def user_upsert(
        ctx: RunContext[AgentDeps],
        key: str,
        name: str = "",
        relationship: str = "",
        details: str = "",
    ) -> str:
        """Create or update a user. Key is how they identify themselves
        (username, user ID, etc.). Only non-empty fields are updated."""
        log.info("👤 user_upsert(%s, name=%s, rel=%s)", key, name, relationship)
        await ctx.deps.memory.upsert_user(key, name, relationship, details)
        return f"User '{key}' saved." + _step_tag(ctx.deps)

    @agent.tool
    async def user_details(ctx: RunContext[AgentDeps], key: str) -> str:
        """Read the detailed info for a specific user."""
        log.debug("👤 user_details(%s)", key)
        details = await ctx.deps.memory.get_user_details(key)
        if not details:
            return f"No detailed info stored for '{key}'." + _step_tag(ctx.deps)
        return details + _step_tag(ctx.deps)

    # == Reminder tools ====================================================

    @agent.tool
    async def create_reminder(
        ctx: RunContext[AgentDeps],
        message: str,
        fire_at: str,
        channel: str = "",
        interval_minutes: int | None = None,
        end_at: str | None = None,
    ) -> str:
        """Create a scheduled reminder.

        Args:
            message: What to remind about.
            fire_at: When to fire (ISO 8601 UTC, e.g. '2026-02-16T15:00:00+00:00').
            channel: Channel to notify (e.g. 'telegram', 'local'). Empty for internal.
            interval_minutes: If set, reminder recurs every N minutes.
            end_at: If set with interval_minutes, stop recurring after this datetime.
        """
        fire_dt = datetime.fromisoformat(fire_at)
        end_dt = datetime.fromisoformat(end_at) if end_at else None
        metadata = dict(ctx.deps.event_metadata) if ctx.deps.event_metadata else {}
        row_id = await ctx.deps.memory.create_reminder(
            message=message,
            fire_at=fire_dt,
            channel=channel,
            metadata=metadata,
            interval_minutes=interval_minutes,
            end_at=end_dt,
        )
        kind = "one-time"
        if interval_minutes:
            kind = f"recurring every {interval_minutes}min"
            if end_dt:
                kind += f" until {end_at}"
        log.info("⏰ create_reminder #%d (%s): %s at %s", row_id, kind, message[:60], fire_at)
        return f"Reminder #{row_id} created ({kind}): '{message}' at {fire_at}" + _step_tag(ctx.deps)

    @agent.tool
    async def list_reminders(ctx: RunContext[AgentDeps]) -> str:
        """List all active reminders."""
        log.debug("⏰ list_reminders")
        reminders = await ctx.deps.memory.list_active_reminders()
        if not reminders:
            return "No active reminders." + _step_tag(ctx.deps)
        lines = []
        for r in reminders:
            line = f"#{r['id']}: '{r['message']}' | next: {r['fire_at']}"
            if r.get("interval_minutes"):
                line += f" | every {r['interval_minutes']}min"
            if r.get("end_at"):
                line += f" | until {r['end_at']}"
            if r.get("channel"):
                line += f" | -> {r['channel']}"
            lines.append(line)
        return "\n".join(lines) + _step_tag(ctx.deps)

    @agent.tool
    async def cancel_reminder(ctx: RunContext[AgentDeps], reminder_id: int) -> str:
        """Cancel (deactivate) a reminder by its ID."""
        log.info("⏰ cancel_reminder(%d)", reminder_id)
        success = await ctx.deps.memory.cancel_reminder(reminder_id)
        if success:
            return f"Reminder #{reminder_id} cancelled." + _step_tag(ctx.deps)
        return f"Reminder #{reminder_id} not found or already inactive." + _step_tag(ctx.deps)

    return agent


# ---------------------------------------------------------------------------
# Reflection Agent — decides what to archive
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field

class ArchivalDecision(BaseModel):
    """Decision on whether to archive information from the conversation."""
    should_archive: bool = Field(
        description="True if the conversation contains long-term knowledge worth saving."
    )
    content: str | None = Field(
        default=None,
        description="The standalone fact or knowledge to be archived. Required if should_archive is True."
    )

_archival_agent: Agent[AgentDeps, ArchivalDecision] | None = None
_archival_agent_model: str | None = None


def _get_archival_agent(cfg: Config) -> Agent[AgentDeps, ArchivalDecision]:
    """Lazy-init the reflection agent (avoids requiring API key at import)."""
    global _archival_agent, _archival_agent_model
    if _archival_agent is None or _archival_agent_model != cfg.llm.model:
        _archival_agent_model = cfg.llm.model
        _archival_agent = Agent(
            OpenRouterModel(cfg.llm.model),
            output_type=ArchivalDecision,
            deps_type=AgentDeps,
            system_prompt=(
                "You are a memory manager. Analyze the conversation turn. "
                "Extract any PERMANENT facts, user preferences, or important context that "
                "should be stored in long-term archival memory. "
                "Ignore usage instructions, casual greetings, or temporary state. "
                "If nothing is worth saving, set should_archive=False."
            ),
        )
    return _archival_agent


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def process_turn(
    event: Event,
    memory: MemoryStore,
    bus: EventBus,
    agent: Agent[AgentDeps, str],
    cfg: Config,
) -> None:
    """Process a single turn of the conversation."""
    deps = AgentDeps(memory=memory, cfg=cfg, bus=bus)

    with atracer.start_as_current_span(
        "agent.turn",
        attributes={
            "source": event.source,
            "message_len": len(event.payload),
        },
    ) as turn_span:
        log.info("▶ turn start  src=%s  msg=%s", event.source, event.payload[:100])

        # Save inbound message to recall memory (unhandled)
        await memory.recall_save("user", event.payload, channel=event.source)

        # Reset per-turn state
        deps.current_message = event.payload
        deps.step = 0
        deps.turn_ended = False
        deps.response_count = 0
        deps.event_source = event.source
        deps.event_metadata = event.metadata

        # Mark all pending messages as handled now that we're responding
        await memory.recall_mark_handled()

        # Run the agent — pydantic-ai loops internally on tool calls.
        result = None
        try:
            with atracer.start_as_current_span("agent.run"):
                result = await agent.run(event.payload, deps=deps)
        except Exception:
            log.exception("agent.run failed — will attempt last-chance reply")

        if result is not None:
            turn_span.set_attribute("steps", deps.step)
            turn_span.set_attribute("turn_ended", deps.turn_ended)
            turn_span.set_attribute("output_len", len(result.output))

            log.info(
                "◀ turn end    steps=%d  ended=%s  output=%s",
                deps.step, deps.turn_ended, result.output[:120],
            )

            # Save the agent's internal summary to recall
            await memory.recall_save(
                "assistant", result.output, channel=event.source, handled=True
            )

        # -- LAST-CHANCE RESPONSE -------------------------------------------
        # If the agent finished without sending any response to the user
        # (or crashed entirely), give it one more shot.
        if deps.response_count == 0:
            log.warning(
                "agent sent 0 responses — running last-chance pass for %s",
                event.source,
            )
            # Stop typing if still going (agent may not have called end_turn)
            from newton.telegram_adapter import _stop_typing
            chat_id = event.metadata.get("chat_id")
            if chat_id:
                _stop_typing(chat_id)

            deps.turn_ended = False
            last_chance_prompt = (
                f"You received a message from the user but have NOT sent any "
                f"reply yet. This is your last chance to respond. You MUST call "
                f"respond_to_user(channel='{event.source}', message=...) now to "
                f"reply, then call end_turn.\n\n"
                f"Original message: {event.payload}"
            )
            try:
                with atracer.start_as_current_span("agent.last_chance"):
                    result = await agent.run(last_chance_prompt, deps=deps)

                log.info(
                    "last-chance pass done  responses=%d  output=%s",
                    deps.response_count, result.output[:120],
                )
            except Exception:
                log.exception("last-chance pass also failed")

        # -- ARCHIVAL REFLECTION LOOP ------------------------------------
        # The agent MUST decide whether to archive anything from this turn.
        with atracer.start_as_current_span("agent.reflection"):
            # Prepare transcript for the reflection agent
            transcript = (
                f"User: {event.payload}\n"
                f"Assistant: {result.output}\n"
            )

            try:
                reflection = await _get_archival_agent(cfg).run(transcript, deps=deps)
                decision = reflection.output

                if decision.should_archive and decision.content:
                    log.info("🧠 reflection: archiving -> %s", decision.content[:60])
                    row_id = await memory.archival_insert(decision.content)
                    trace.get_current_span().set_attribute("archived_id", row_id)
                else:
                    log.info("🧠 reflection: nothing to archive")

            except Exception as e:
                log.error("Reflection failed", exc_info=e)
                trace.get_current_span().record_exception(e)


async def process_heartbeat(
    event: Event,
    memory: MemoryStore,
    bus: EventBus,
    agent: Agent[AgentDeps, str],
    cfg: Config,
) -> None:
    """Process a heartbeat with lightweight context and tight step budget."""
    deps = AgentDeps(memory=memory, cfg=cfg, bus=bus)
    deps.current_message = ""
    deps.step = 0
    deps.turn_ended = False
    deps.response_count = 0
    deps.event_source = "scheduler"
    deps.event_metadata = {}
    deps.is_heartbeat = True
    deps.max_steps = min(cfg.agent.max_steps, 5)

    with atracer.start_as_current_span("agent.heartbeat"):
        log.info("💓 heartbeat processing")
        try:
            result = await agent.run("heartbeat", deps=deps)
            if result:
                await memory.recall_save(
                    "assistant", f"[heartbeat] {result.output}",
                    channel="scheduler", handled=True,
                )
                log.info("💓 heartbeat done  steps=%d  output=%s", deps.step, result.output[:80])
        except Exception:
            log.exception("heartbeat agent.run failed")


async def process_reminder(
    event: Event,
    memory: MemoryStore,
    bus: EventBus,
    agent: Agent[AgentDeps, str],
    cfg: Config,
) -> None:
    """Process a fired reminder — the agent sees it and can act on it."""
    deps = AgentDeps(memory=memory, cfg=cfg, bus=bus)
    deps.current_message = event.payload
    deps.step = 0
    deps.turn_ended = False
    deps.response_count = 0
    deps.event_source = "scheduler"
    deps.event_metadata = event.metadata
    deps.max_steps = min(cfg.agent.max_steps, 8)

    prompt = f"A scheduled reminder has fired.\nReminder message: {event.payload}\n"
    if event.reply_to:
        prompt += (
            f"Target channel: {event.reply_to}\n"
            f"You should send a message to the '{event.reply_to}' channel "
            f"about this reminder using respond_to_user, then call end_turn.\n"
        )
    else:
        prompt += "This is an internal reminder. Take any appropriate action, then call end_turn.\n"

    with atracer.start_as_current_span(
        "agent.reminder",
        attributes={"reminder_message": event.payload[:100]},
    ):
        log.info("⏰ reminder processing: %s", event.payload[:100])
        try:
            result = await agent.run(prompt, deps=deps)
            if result:
                await memory.recall_save(
                    "assistant", f"[reminder] {result.output}",
                    channel="scheduler", handled=True,
                )
                log.info("⏰ reminder done  steps=%d  output=%s", deps.step, result.output[:80])
        except Exception:
            log.exception("reminder agent.run failed")


def _batch_key(event: Event) -> str:
    """Group key: source + chat_id (so telegram chats batch separately)."""
    chat_id = event.metadata.get("chat_id", "")
    return f"{event.source}:{chat_id}"


def _merge_events(events: list[Event]) -> Event:
    """Combine multiple events from the same chat into one.

    Payloads are joined with newlines; metadata comes from the latest event.
    """
    if len(events) == 1:
        return events[0]

    combined_payload = "\n".join(e.payload for e in events if e.payload)
    latest = events[-1]
    log.info(
        "batched %d messages from %s into one turn (%d chars)",
        len(events), _batch_key(latest), len(combined_payload),
    )
    return Event(
        source=latest.source,
        kind=latest.kind,
        payload=combined_payload,
        reply_to=latest.reply_to,
        metadata=latest.metadata,
        timestamp=latest.timestamp,
    )


async def run_agent_loop(
    bus: EventBus, cfg: Config, memory: MemoryStore
) -> None:
    """Pull events from the inbox, run the agent with step tracking."""
    agent = create_agent(cfg)

    async with agent:  # opens MCP server connections (e.g. Playwright)
        while True:
            # Block until at least one event arrives
            event = await bus.inbox.get()

            # --- HEARTBEAT: lightweight processing, no batching ---
            if event.kind == EventKind.HEARTBEAT:
                try:
                    await process_heartbeat(event, memory, bus, agent, cfg)
                except Exception:
                    log.exception("heartbeat processing failed")
                continue

            # --- REMINDER: process individually, no batching ---
            if event.kind == EventKind.REMINDER:
                try:
                    await process_reminder(event, memory, bus, agent, cfg)
                except Exception:
                    log.exception("reminder processing failed")
                continue

            # --- Skip non-actionable events ---
            if event.kind != EventKind.MESSAGE:
                continue

            # Drain any additional queued events that arrived while we were busy
            pending: list[Event] = [event]
            requeue: list[Event] = []
            while not bus.inbox.empty():
                extra = bus.inbox.get_nowait()
                if extra.kind == EventKind.MESSAGE:
                    pending.append(extra)
                else:
                    # Re-queue heartbeats/reminders so they aren't lost
                    requeue.append(extra)
            for ev in requeue:
                await bus.inbox.put(ev)

            # Group by source+chat_id, merge each group, process each turn
            groups: dict[str, list[Event]] = {}
            for ev in pending:
                groups.setdefault(_batch_key(ev), []).append(ev)

            for batch in groups.values():
                merged = _merge_events(batch)
                try:
                    await process_turn(merged, memory, bus, agent, cfg)
                except Exception:
                    log.exception(
                        "process_turn failed for src=%s — skipping",
                        merged.source,
                    )
                    # Clean up typing indicator so it doesn't stick
                    from newton.telegram_adapter import _stop_typing
                    chat_id = merged.metadata.get("chat_id")
                    if chat_id:
                        _stop_typing(chat_id)

