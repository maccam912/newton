"""Agent — agentic loop with step tracking, memory tools, and channel I/O."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, UTC

from pydantic_ai import Agent, RunContext, UserContent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.providers.openai import OpenAIProvider

from newton.config import Config

from newton.context import build_heartbeat_prompt, build_system_prefix, build_system_prompt
from newton.events import Event, EventBus, EventKind, ColorFormatter
from newton.memory import MemoryStore
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
        self.system_prefix_cache: str = ""
        self.system_prefix_cached_at: datetime | None = None


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


def _estimate_tokens(text: str) -> int:
    """Very rough token estimator for budget checks (about 4 chars/token)."""
    trimmed = text.strip()
    if not trimmed:
        return 0
    return max(1, len(trimmed) // 4)



# ---------------------------------------------------------------------------
# Build agent + register tools
# ---------------------------------------------------------------------------


def _build_model(cfg: Config):
    """Create an LLM model from provider config."""
    provider = cfg.llm.provider.strip().lower()
    if provider == "openrouter":
        if cfg.llm.api_key:
            openrouter_provider = OpenRouterProvider(api_key=cfg.llm.api_key)
            return OpenRouterModel(cfg.llm.model, provider=openrouter_provider)
        return OpenRouterModel(cfg.llm.model)
    if provider == "zai":
        base_url = cfg.llm.base_url or "https://api.z.ai/api/paas/v4"
        zai_provider = OpenAIProvider(base_url=base_url, api_key=cfg.llm.api_key or None)
        return OpenAIModel(cfg.llm.model, provider=zai_provider)

    raise ValueError(f"Unsupported llm.provider '{cfg.llm.provider}'. Use 'openrouter' or 'zai'.")


def _build_model_settings(cfg: Config):
    """Provider-specific model settings."""
    if cfg.llm.provider.strip().lower() == "openrouter":
        return OpenRouterModelSettings(
            openrouter_reasoning={"effort": cfg.llm.reasoning_effort},
        )
    return None


def create_agent(cfg: Config) -> Agent[AgentDeps, str]:
    """Build the agent with memory, communication, and control-flow tools."""
    model = _build_model(cfg)
    model_settings = _build_model_settings(cfg)
    agent: Agent[AgentDeps, str] = Agent(
        model,
        deps_type=AgentDeps,
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

        ttl = max(ctx.deps.cfg.llm.prompt_prefix_cache_ttl_seconds, 0)
        now = datetime.now(UTC)
        cache_valid = (
            ttl > 0
            and ctx.deps.system_prefix_cache
            and ctx.deps.system_prefix_cached_at is not None
            and (now - ctx.deps.system_prefix_cached_at).total_seconds() < ttl
        )
        if not cache_valid:
            ctx.deps.system_prefix_cache = await build_system_prefix(
                ctx.deps.cfg, ctx.deps.memory
            )
            ctx.deps.system_prefix_cached_at = now

        prompt = await build_system_prompt(
            ctx.deps.cfg,
            ctx.deps.memory,
            ctx.deps.current_message,
            cached_prefix=ctx.deps.system_prefix_cache or None,
        )
        prompt += (
            f"\n\n--- TURN INFO ---\n"
            f"Channel: {ctx.deps.event_source} | "
            f"Budget: {ctx.deps.max_steps} steps | "
            f"respond_to_user to reply, end_turn when done.\n"
            f"'notebook' core memory block persists across turns — "
            f"update it with core_memory_update. "
            f"run_subagent/run_parallel_subagents for isolated multi-step tasks."
        )
        return prompt

    # -- Register external tools -------------------------------------------
    import newton.tools.searxng
    import newton.tools.scripts

    newton.tools.searxng.register(agent)
    newton.tools.scripts.register(agent)

    import newton.tools.bash
    newton.tools.bash.register(agent)

    import newton.tools.vikunja
    newton.tools.vikunja.register(agent)

    import newton.tools.subagent
    newton.tools.subagent.register(agent)

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
        # Save the actual response to recall so context shows what was said
        await ctx.deps.memory.recall_save(
            "assistant", message, channel=channel, handled=True
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

    # == Skill tools ====================================================

    @agent.tool
    async def skill_invoke(ctx: RunContext[AgentDeps], name: str) -> str:
        """Load the full instructions for a skill by name.
        Use this when you need to follow a skill's detailed guidance."""
        log.info("🎯 skill_invoke(%s)", name)
        skill = await ctx.deps.memory.skill_get(name)
        if skill is None:
            return f"Skill '{name}' not found." + _step_tag(ctx.deps)
        return (
            f"=== SKILL: {skill['name']} ===\n"
            f"{skill['full_prompt']}"
        ) + _step_tag(ctx.deps)

    @agent.tool
    async def skill_create(
        ctx: RunContext[AgentDeps], name: str, description: str, full_prompt: str
    ) -> str:
        """Create a new skill with a name, short description, and full prompt.

        Args:
            name: Unique skill identifier (lowercase, no spaces — use hyphens).
            description: One-line summary shown in the skills menu.
            full_prompt: Detailed instructions loaded when the skill is invoked.
        """
        log.info("🎯 skill_create(%s)", name)
        core_blocks = await ctx.deps.memory.get_core_blocks()
        if name in core_blocks:
            return (
                f"Cannot create skill '{name}' — conflicts with a core memory block name."
            ) + _step_tag(ctx.deps)
        try:
            await ctx.deps.memory.skill_create(name, description, full_prompt)
        except ValueError as e:
            return str(e) + _step_tag(ctx.deps)
        return f"Skill '{name}' created." + _step_tag(ctx.deps)

    @agent.tool
    async def skill_update(
        ctx: RunContext[AgentDeps],
        name: str,
        description: str = "",
        full_prompt: str = "",
    ) -> str:
        """Update an existing skill's description and/or full_prompt.
        Only non-empty fields are updated."""
        log.info("🎯 skill_update(%s)", name)
        updated = await ctx.deps.memory.skill_update(
            name,
            description=description or None,
            full_prompt=full_prompt or None,
        )
        if not updated:
            return f"Skill '{name}' not found." + _step_tag(ctx.deps)
        return f"Skill '{name}' updated." + _step_tag(ctx.deps)

    @agent.tool
    async def skill_delete(ctx: RunContext[AgentDeps], name: str) -> str:
        """Delete a skill by name."""
        log.info("🎯 skill_delete(%s)", name)
        deleted = await ctx.deps.memory.skill_delete(name)
        if not deleted:
            return f"Skill '{name}' not found." + _step_tag(ctx.deps)
        return f"Skill '{name}' deleted." + _step_tag(ctx.deps)

    @agent.tool
    async def skill_list_tool(ctx: RunContext[AgentDeps]) -> str:
        """List all available skills with their descriptions."""
        log.debug("🎯 skill_list")
        skills = await ctx.deps.memory.skill_list()
        if not skills:
            return "No skills defined yet." + _step_tag(ctx.deps)
        lines = [f"- {s['name']}: {s['description']}" for s in skills]
        return "\n".join(lines) + _step_tag(ctx.deps)

    return agent


# ---------------------------------------------------------------------------
# Reflection Agent — decides what to archive
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field

class SessionArchivalResult(BaseModel):
    """Result of session archival: a summary plus extractable facts."""
    summary: str = Field(
        description="2-5 sentence summary of the session for continuity."
    )
    archival_facts: list[str] = Field(
        default_factory=list,
        description="Standalone facts worth saving in long-term archival memory. "
        "Each should be self-contained. Empty list if nothing worth saving."
    )

_session_archival_agent: Agent[AgentDeps, SessionArchivalResult] | None = None
_session_archival_agent_model: str | None = None


def _get_session_archival_agent(cfg: Config) -> Agent[AgentDeps, SessionArchivalResult]:
    """Lazy-init the session archival agent."""
    global _session_archival_agent, _session_archival_agent_model
    if _session_archival_agent is None or _session_archival_agent_model != cfg.llm.model:
        _session_archival_agent_model = cfg.llm.model
        _session_archival_agent = Agent(
            _build_model(cfg),
            output_type=SessionArchivalResult,
            deps_type=AgentDeps,
            system_prompt=(
                "You are a memory manager. You are given a full session transcript. "
                "Produce TWO things:\n"
                "1. A 2-5 sentence SUMMARY of the session — what was discussed, "
                "what was accomplished, any open threads.\n"
                "2. A list of STANDALONE FACTS worth saving permanently in long-term "
                "archival memory (user preferences, important decisions, learned info). "
                "Each fact must be self-contained and understandable without the conversation. "
                "If nothing is worth archiving, return an empty list for archival_facts."
            ),
        )
    return _session_archival_agent


async def _run_session_archival(
    memory: MemoryStore, cfg: Config, bus: EventBus
) -> None:
    """Summarize the current session, archive facts, save summary, clear recall."""
    with atracer.start_as_current_span("agent.session_archival") as span:
        messages = await memory.recall_get_all()
        if not messages:
            return

        msg_count = len(messages)
        channels = ", ".join(sorted({m["channel"] for m in messages if m["channel"]})) or "unknown"

        # Build transcript
        lines: list[str] = []
        for m in messages:
            role = "User" if m["role"] == "user" else "Newton"
            ts = m.get("timestamp", "")[:19].replace("T", " ")
            lines.append(f"[{ts}] {role}: {m['content']}")
        transcript = "\n".join(lines)

        log.info("session archival: %d messages, channels=%s", msg_count, channels)
        span.set_attribute("msg_count", msg_count)

        deps = AgentDeps(memory=memory, cfg=cfg, bus=bus)
        try:
            result = await _get_session_archival_agent(cfg).run(transcript, deps=deps)
            decision = result.output

            # Archive each fact
            for fact in decision.archival_facts:
                row_id = await memory.archival_insert(fact)
                log.info("session archival: stored fact id=%d -> %s", row_id, fact[:60])

            # Save session summary
            summary_id = await memory.session_summary_save(
                summary=decision.summary,
                channels=channels,
                msg_count=msg_count,
            )
            log.info("session archival: summary id=%d", summary_id)
            span.set_attribute("facts_count", len(decision.archival_facts))
            span.set_attribute("summary_id", summary_id)

        except Exception as e:
            log.error("session archival agent failed", exc_info=e)
            span.record_exception(e)
            return

        # Clear recall only after successful archival
        cleared = await memory.recall_clear()
        log.info("session archival: cleared %d recall rows", cleared)


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
    run_input, run_prompt = _build_run_input(event)
    response_metadata = dict(event.metadata)
    response_metadata.pop("image_url", None)

    with atracer.start_as_current_span(
        "agent.turn",
        attributes={
            "source": event.source,
            "message_len": len(run_prompt),
        },
    ) as turn_span:
        log.info("▶ turn start  src=%s  msg=%s", event.source, run_prompt[:100])

        # Save inbound message to recall memory (unhandled)
        await memory.recall_save("user", event.payload, channel=event.source)

        # Reset per-turn state
        deps.current_message = run_prompt
        deps.step = 0
        deps.turn_ended = False
        deps.response_count = 0
        deps.event_source = event.source
        deps.event_metadata = response_metadata

        # Mark all pending messages as handled now that we're responding
        await memory.recall_mark_handled()

        # Run the agent — pydantic-ai loops internally on tool calls.
        result = None
        try:
            with atracer.start_as_current_span("agent.run"):
                result = await agent.run(run_input, deps=deps)
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
            last_chance_prompt = _build_last_chance_prompt(event, run_prompt)
            try:
                with atracer.start_as_current_span("agent.last_chance"):
                    result = await agent.run(last_chance_prompt, deps=deps)

                log.info(
                    "last-chance pass done  responses=%d  output=%s",
                    deps.response_count, result.output[:120],
                )
            except Exception:
                log.exception("last-chance pass also failed")

        # -- AUTO-ARCHIVAL on context size ------------------------------------
        await _maybe_auto_archive(memory, cfg, bus)


async def _maybe_auto_archive(
    memory: MemoryStore, cfg: Config, bus: EventBus
) -> None:
    """Archive and reset context if recall memory exceeds the token budget."""
    messages = await memory.recall_get_all()
    if not messages:
        return
    total_text = "\n".join(m["content"] for m in messages if m.get("content"))
    token_est = _estimate_tokens(total_text)
    if token_est > cfg.memory.max_recall_tokens:
        log.info(
            "auto-archive: recall tokens ~%d > limit %d — archiving",
            token_est, cfg.memory.max_recall_tokens,
        )
        try:
            await _run_session_archival(memory, cfg, bus)
        except Exception:
            log.exception("auto-archive session archival failed")


def _build_run_input(event: Event) -> tuple[str | Sequence[UserContent], str]:
    """Build model input for this turn plus a text summary for logging/context."""
    image_url = event.metadata.get("image_url", "").strip()
    if not image_url:
        return event.payload, event.payload

    prompt_text = event.payload.strip() or "[User sent an image.]"
    parts: list[UserContent] = [prompt_text, image_url]
    return parts, prompt_text


def _build_last_chance_prompt(event: Event, run_prompt: str) -> str | Sequence[UserContent]:
    """Build last-chance input, preserving image context when available."""
    instruction = (
        f"You received a message from the user but have NOT sent any "
        f"reply yet. This is your last chance to respond. You MUST call "
        f"respond_to_user(channel='{event.source}', message=...) now to "
        f"reply, then call end_turn.\n\n"
        f"Original message: {run_prompt}"
    )

    image_url = event.metadata.get("image_url", "").strip()
    if not image_url:
        return instruction
    return [instruction, image_url]


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
    import asyncio
    agent = create_agent(cfg)
    idle_seconds = cfg.memory.idle_archival_seconds

    async with agent:  # opens MCP server connections (e.g. Playwright)
        while True:
            # Block until an event arrives, or idle timeout triggers archival
            try:
                event = await asyncio.wait_for(bus.inbox.get(), timeout=idle_seconds)
            except asyncio.TimeoutError:
                if await memory.recall_count() > 0:
                    log.info("idle timeout (%ds) — running session archival", idle_seconds)
                    try:
                        await _run_session_archival(memory, cfg, bus)
                    except Exception:
                        log.exception("session archival failed")
                continue

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
