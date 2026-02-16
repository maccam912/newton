"""Context assembly — builds the full system prompt before each agent call.

Layers (in order):
  1. Base instructions (from config)
  2. Core memory blocks (always present)
  3. Archival search results (semantic match on the incoming message)
  4. Recent conversation history (recall memory)
"""

from __future__ import annotations

from newton.config import Config
from newton.memory import MemoryStore
from newton.tracing import get_tracer

tracer = get_tracer("newton.context")


async def build_system_prompt(
    cfg: Config,
    memory: MemoryStore,
    incoming_message: str,
) -> str:
    """Assemble a dynamic system prompt for this turn."""

    with tracer.start_as_current_span(
        "build_system_prompt",
        attributes={"incoming_message_len": len(incoming_message)},
    ) as span:
        parts: list[str] = []

        # 1. Base instructions
        parts.append(cfg.llm.system_prompt)

        # 2. Core memory — always present
        with tracer.start_as_current_span("context.core_memory"):
            blocks = await memory.get_core_blocks()
            if blocks:
                parts.append("\n--- CORE MEMORY ---")
                for block, content in blocks.items():
                    parts.append(f"[{block}]\n{content}")

        # 3. Known users — summary always present
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

        # 4. Available skills — name + description only (progressive disclosure)
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

        # 5. Archival search — only if there's something to search for
        if incoming_message.strip():
            with tracer.start_as_current_span("context.archival_search"):
                results = await memory.archival_search(
                    incoming_message, k=cfg.memory.archival_search_k
                )
                if results:
                    parts.append("\n--- RELEVANT MEMORIES ---")
                    for i, text in enumerate(results, 1):
                        parts.append(f"{i}. {text}")

        # 6. Recent conversation history (ordered by timestamp)
        with tracer.start_as_current_span("context.recall"):
            history = await memory.recall_recent(n=cfg.memory.recall_window)
            if history:
                parts.append("\n--- RECENT CONVERSATION ---")
                for msg in history:
                    role = "User" if msg["role"] == "user" else "Newton"
                    ts = msg.get("timestamp", "")[:19].replace("T", " ")
                    pending = ""
                    if msg["role"] == "user" and msg.get("handled") == "0":
                        pending = " [pending]"
                    parts.append(f"[{ts}] {role}{pending}: {msg['content']}")

        prompt = "\n\n".join(parts)
        span.set_attribute("prompt_len", len(prompt))
        return prompt


async def build_heartbeat_prompt(
    cfg: Config,
    memory: MemoryStore,
) -> str:
    """Build a lightweight system prompt for heartbeat processing.

    Compared to the full prompt this skips the archival search on incoming
    message (there is none) and uses a shorter recall window.  Instead it
    does targeted searches for heartbeat/remember-related memories and
    includes an active-reminders summary.
    """

    with tracer.start_as_current_span("build_heartbeat_prompt") as span:
        parts: list[str] = []

        # 1. Base instructions + heartbeat guidance
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

        # 2. Core memory (always present)
        blocks = await memory.get_core_blocks()
        if blocks:
            parts.append("\n--- CORE MEMORY ---")
            for block, content in blocks.items():
                parts.append(f"[{block}]\n{content}")

        # 3. Available skills (progressive disclosure — names only)
        skills = await memory.skill_list()
        if skills:
            parts.append("\n--- AVAILABLE SKILLS ---")
            parts.append(
                "Use skill_invoke(name) to load full instructions for a skill. "
                "Only invoke a skill when the user's request clearly matches it."
            )
            for s in skills:
                parts.append(f"- {s['name']}: {s['description']}")

        # 4. Targeted archival search for heartbeat-relevant memories
        for query in ("heartbeat reminder", "remember to do"):
            results = await memory.archival_search(query, k=3)
            if results:
                parts.append(f"\n--- MEMORIES matching '{query}' ---")
                for i, text in enumerate(results, 1):
                    parts.append(f"{i}. {text}")

        # 5. Active reminders summary
        active = await memory.list_active_reminders()
        if active:
            parts.append("\n--- ACTIVE REMINDERS ---")
            for r in active:
                line = f"#{r['id']}: '{r['message']}' next: {r['fire_at']}"
                if r.get("channel"):
                    line += f" -> {r['channel']}"
                parts.append(line)

        # 6. Short recall (last 3 messages only)
        history = await memory.recall_recent(n=3)
        if history:
            parts.append("\n--- RECENT ACTIVITY (last 3) ---")
            for msg in history:
                role = "User" if msg["role"] == "user" else "Newton"
                ts = msg.get("timestamp", "")[:19].replace("T", " ")
                parts.append(f"[{ts}] {role}: {msg['content'][:200]}")

        prompt = "\n\n".join(parts)
        span.set_attribute("prompt_len", len(prompt))
        return prompt
