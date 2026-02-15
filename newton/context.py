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

        # 4. Archival search — only if there's something to search for
        if incoming_message.strip():
            with tracer.start_as_current_span("context.archival_search"):
                results = await memory.archival_search(
                    incoming_message, k=cfg.memory.archival_search_k
                )
                if results:
                    parts.append("\n--- RELEVANT MEMORIES ---")
                    for i, text in enumerate(results, 1):
                        parts.append(f"{i}. {text}")

        # 5. Recent conversation history (ordered by timestamp)
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
