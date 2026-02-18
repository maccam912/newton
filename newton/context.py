"""Context assembly — builds system prompts with a cache-friendly static prefix."""

from __future__ import annotations

from newton.config import Config
from newton.memory import MemoryStore
from newton.tracing import get_tracer

tracer = get_tracer("newton.context")


async def build_system_prefix(cfg: Config, memory: MemoryStore) -> str:
    """Build the large, mostly-stable system prefix shared across turns."""
    parts: list[str] = [cfg.llm.system_prompt]

    with tracer.start_as_current_span("context.core_memory"):
        blocks = await memory.get_core_blocks()
        if blocks:
            parts.append("\n--- CORE MEMORY ---")
            for block, content in blocks.items():
                parts.append(f"[{block}]\n{content}")

    with tracer.start_as_current_span("context.users"):
        users = await memory.get_all_users_summary()
        if users:
            parts.append("\n--- KNOWN USERS ---")
            parts.append("Use the user_details tool to read detailed info about a user.")
            for u in users:
                line = f"- {u['key']}"
                if u["name"]:
                    line += f" ({u['name']})"
                if u["relationship"]:
                    line += f" — {u['relationship']}"
                parts.append(line)
        else:
            parts.append("\n--- KNOWN USERS ---\nNo known users yet. Use user_upsert to remember new users.")

    with tracer.start_as_current_span("context.skills"):
        skills = await memory.skill_list()
        if skills:
            parts.append("\n--- AVAILABLE SKILLS ---")
            parts.append(
                "Use skill_invoke(name) to load full instructions for a skill. "
                "Only invoke a skill when the user's request clearly matches it."
            )
            for s in skills:
                parts.append(f"- {s['name']}: {s['description']}")

    with tracer.start_as_current_span("context.session_summaries"):
        summaries = await memory.session_summaries_recent(n=3)
        if summaries:
            parts.append("\n--- PREVIOUS SESSIONS ---")
            for s in summaries:
                ts = s["created"][:19].replace("T", " ")
                parts.append(
                    f"[{ts}] ({s['msg_count']} messages, channels: {s['channels']})\n"
                    f"{s['summary']}"
                )

    return "\n\n".join(parts)


async def build_turn_context_suffix(
    cfg: Config,
    memory: MemoryStore,
    incoming_message: str,
) -> str:
    """Build the turn-variant context that changes frequently."""
    parts: list[str] = []

    if incoming_message.strip():
        with tracer.start_as_current_span("context.archival_search"):
            results = await memory.archival_search(
                incoming_message, k=cfg.memory.archival_search_k
            )
            if results:
                parts.append("\n--- RELEVANT MEMORIES ---")
                for i, text in enumerate(results, 1):
                    parts.append(f"{i}. {text}")

    with tracer.start_as_current_span("context.recall"):
        history = await memory.recall_recent(n=cfg.memory.recall_window)
        if history:
            parts.append("\n--- RECENT CONVERSATION ---")
            for msg in history:
                role = "User" if msg["role"] == "user" else "Newton"
                pending = ""
                if msg["role"] == "user" and msg.get("handled") == "0":
                    pending = " [pending]"
                parts.append(f"{role}{pending}: {msg['content']}")

    return "\n\n".join(parts)


async def build_system_prompt(
    cfg: Config,
    memory: MemoryStore,
    incoming_message: str,
    *,
    cached_prefix: str | None = None,
) -> str:
    """Assemble a dynamic system prompt with a stable shared prefix."""
    with tracer.start_as_current_span(
        "build_system_prompt",
        attributes={"incoming_message_len": len(incoming_message)},
    ) as span:
        prefix = cached_prefix or await build_system_prefix(cfg, memory)
        suffix = await build_turn_context_suffix(cfg, memory, incoming_message)

        prompt = prefix if not suffix else f"{prefix}\n\n{suffix}"
        span.set_attribute("prompt_len", len(prompt))
        span.set_attribute("used_cached_prefix", cached_prefix is not None)
        return prompt


async def build_heartbeat_prompt(
    cfg: Config,
    memory: MemoryStore,
) -> str:
    """Build a lightweight system prompt for heartbeat processing."""

    with tracer.start_as_current_span("build_heartbeat_prompt") as span:
        parts: list[str] = []

        parts.append(
            f"{cfg.llm.system_prompt}\n\n"
            "This is a HEARTBEAT event — time has passed since your last activity. "
            "You are NOT responding to a user message. Use this opportunity to:\n"
            "- Check if you have any pending tasks or proactive things to do\n"
            "- Act on anything relevant from your memories\n"
            "- If there is nothing to do, just call end_turn\n"
            "Keep actions minimal. Do NOT send messages to users unless you have "
            "a specific reason (e.g., a proactive notification you planned)."
        )

        blocks = await memory.get_core_blocks()
        if blocks:
            parts.append("\n--- CORE MEMORY ---")
            for block, content in blocks.items():
                parts.append(f"[{block}]\n{content}")

        skills = await memory.skill_list()
        if skills:
            parts.append("\n--- AVAILABLE SKILLS ---")
            parts.append(
                "Use skill_invoke(name) to load full instructions for a skill. "
                "Only invoke a skill when the user's request clearly matches it."
            )
            for s in skills:
                parts.append(f"- {s['name']}: {s['description']}")

        for query in ("heartbeat reminder", "remember to do"):
            results = await memory.archival_search(query, k=3)
            if results:
                parts.append(f"\n--- MEMORIES matching '{query}' ---")
                for i, text in enumerate(results, 1):
                    parts.append(f"{i}. {text}")

        active = await memory.list_active_reminders()
        if active:
            parts.append("\n--- ACTIVE REMINDERS ---")
            for r in active:
                line = f"#{r['id']}: '{r['message']}' next: {r['fire_at']}"
                if r.get("channel"):
                    line += f" -> {r['channel']}"
                parts.append(line)

        summaries = await memory.session_summaries_recent(n=2)
        if summaries:
            parts.append("\n--- PREVIOUS SESSIONS ---")
            for s in summaries:
                ts = s["created"][:19].replace("T", " ")
                parts.append(
                    f"[{ts}] ({s['msg_count']} messages, channels: {s['channels']})\n"
                    f"{s['summary']}"
                )

        history = await memory.recall_recent(n=3)
        if history:
            parts.append("\n--- RECENT ACTIVITY (last 3) ---")
            for msg in history:
                role = "User" if msg["role"] == "user" else "Newton"
                parts.append(f"{role}: {msg['content'][:200]}")

        prompt = "\n\n".join(parts)
        span.set_attribute("prompt_len", len(prompt))
        return prompt
