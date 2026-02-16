"""Agent — agentic loop with step tracking, memory tools, and channel I/O."""

from __future__ import annotations

import logging

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings

from newton.config import Config
from newton.context import build_system_prompt
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
        self.turn_ended: bool = False


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
        prompt = await build_system_prompt(
            ctx.deps.cfg, ctx.deps.memory, ctx.deps.current_message
        )
        prompt += (
            f"\n\n--- TURN INFO ---\n"
            f"Step budget: {ctx.deps.max_steps} tool calls per turn.\n"
            f"Call respond_to_user to send messages. "
            f"Call end_turn when you are finished."
        )
        return prompt

    # -- Register external tools -------------------------------------------
    import newton.tools.searxng
    import newton.tools.scripts

    newton.tools.searxng.register(agent)
    newton.tools.scripts.register(agent)

    # == Control-flow tools ================================================

    @agent.tool
    async def end_turn(ctx: RunContext[AgentDeps]) -> str:
        """Signal that you are done processing this turn.
        Call this when you have nothing more to do."""
        ctx.deps.turn_ended = True
        log.info("🛑 end_turn called")
        return "Turn ended. Produce a brief internal summary of what you did."

    @agent.tool
    async def respond_to_user(
        ctx: RunContext[AgentDeps], channel: str, message: str
    ) -> str:
        """Send a visible message to a channel (e.g. 'local', 'telegram').
        You can respond to any channel, not just the one that sent the message."""
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

    return agent


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


def _get_archival_agent() -> Agent[AgentDeps, ArchivalDecision]:
    """Lazy-init the reflection agent (avoids requiring API key at import)."""
    global _archival_agent
    if _archival_agent is None:
        _archival_agent = Agent(
            OpenRouterModel("stepfun/step-3.5-flash:free"),
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
        deps.event_metadata = event.metadata

        # Mark all pending messages as handled now that we're responding
        await memory.recall_mark_handled()

        # Run the agent — pydantic-ai loops internally on tool calls.
        with atracer.start_as_current_span("agent.run"):
            result = await agent.run(event.payload, deps=deps)

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

        # If the agent produced a final text but never called respond_to_user,
        # send it back to the originating channel as a fallback.
        if not deps.turn_ended and deps.step == 0:
            log.info("↩ fallback reply → %s", event.source)
            await bus.put_outbox(
                Event(
                    source="agent",
                    kind=EventKind.RESPONSE,
                    payload=result.output,
                    reply_to=event.source,
                    metadata=event.metadata,
                )
            )

        # -- ARCHIVAL REFLECTION LOOP ------------------------------------
        # The agent MUST decide whether to archive anything from this turn.
        with atracer.start_as_current_span("agent.reflection"):
            # Prepare transcript for the reflection agent
            transcript = (
                f"User: {event.payload}\n"
                f"Assistant: {result.output}\n"
            )

            try:
                reflection = await _get_archival_agent().run(transcript, deps=deps)
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


async def run_agent_loop(
    bus: EventBus, cfg: Config, memory: MemoryStore
) -> None:
    """Pull events from the inbox, run the agent with step tracking."""
    agent = create_agent(cfg)

    async with agent:  # opens MCP server connections (e.g. Playwright)
        while True:
            event = await bus.inbox.get()

            # Skip non-message events (e.g. heartbeats)
            if event.kind != EventKind.MESSAGE:
                continue

            await process_turn(event, memory, bus, agent, cfg)

