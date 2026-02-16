"""Telegram adapter — inbound messages and outbound replies."""

from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters

from newton.config import Config
from newton.events import Event, EventBus, EventKind, ColorFormatter

CHANNEL = "telegram"

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def _make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger

log = _make_logger("newton.telegram")

# ---------------------------------------------------------------------------
# Typing-indicator bookkeeping
# ---------------------------------------------------------------------------

# chat_ids currently waiting for a reply — typing action is sent in a loop
_typing_chats: dict[str, asyncio.Task] = {}


async def _typing_loop(bot, chat_id: str) -> None:
    """Send 'typing' action every 4s until cancelled."""
    try:
        while True:
            await bot.send_chat_action(chat_id=int(chat_id), action="typing")
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


def _start_typing(bot, chat_id: str) -> None:
    """Begin showing 'typing…' for a chat (idempotent)."""
    if chat_id in _typing_chats:
        return  # already typing
    _typing_chats[chat_id] = asyncio.create_task(_typing_loop(bot, chat_id))
    log.debug("typing indicator ON for chat %s", chat_id)


def _stop_typing(chat_id: str) -> None:
    """Cancel the typing indicator for a chat."""
    task = _typing_chats.pop(chat_id, None)
    if task:
        task.cancel()
        log.debug("typing indicator OFF for chat %s", chat_id)


# ---------------------------------------------------------------------------
# Inbound — polling
# ---------------------------------------------------------------------------

async def start_telegram(bus: EventBus, cfg: Config) -> None:
    """Start the Telegram bot and funnel messages into the inbox."""
    if not cfg.telegram.bot_token:
        log.warning("No bot token configured — skipping Telegram adapter.")
        return

    token_preview = cfg.telegram.bot_token[:8] + "…"
    log.info("Building Telegram app (token=%s)", token_preview)

    # Track every chat we've seen so the agent can browse them later
    known_chats: dict[str, str] = {}   # chat_id -> chat title/name

    app = ApplicationBuilder().token(cfg.telegram.bot_token).build()

    # Fetch bot identity so we know the connection works
    log.info("Connecting to Telegram API…")
    try:
        bot_info = await app.bot.get_me()
        log.info(
            "Connected as @%s  (id=%s, name=%s)",
            bot_info.username, bot_info.id, bot_info.full_name,
        )
    except Exception as exc:
        log.error("Failed to connect to Telegram API: %s", exc)
        return

    async def on_message(update: Update, _context) -> None:
        if not update.message:
            return
        chat = update.message.chat
        chat_id = str(chat.id)
        user = update.message.from_user

        # Remember this chat
        chat_label = chat.title or chat.full_name or chat_id
        known_chats[chat_id] = chat_label

        # --- detailed inbound log ---
        log.info(
            "MSG IN  chat_id=%-14s  type=%-12s  title=%s",
            chat_id, chat.type, chat_label,
        )
        if user:
            log.info(
                "        from user  id=%-14s  username=%-16s  name=%s",
                user.id, user.username or "(none)", user.full_name,
            )
        log.info("        text: %s", (update.message.text or "")[:200])

        # Start typing indicator while the agent works
        _start_typing(app.bot, chat_id)

        await bus.put_inbox(
            Event(
                source=CHANNEL,
                kind=EventKind.MESSAGE,
                payload=update.message.text or "",
                metadata={"chat_id": chat_id},
            )
        )

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    # Use the low-level async lifecycle instead of run_polling(), which tries
    # to manage its own event loop and conflicts with the already-running one.
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    log.info("Polling loop started — listening for messages…")

    # Block forever (until the task is cancelled on shutdown)
    try:
        await asyncio.Event().wait()
    finally:
        log.info("Shutting down Telegram polling…")
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


# ---------------------------------------------------------------------------
# Outbound — send replies
# ---------------------------------------------------------------------------

async def send_replies(bus: EventBus, cfg: Config) -> None:
    """Pull response events from this channel's outbox and send on Telegram."""
    if not cfg.telegram.bot_token:
        log.warning("No bot token — send_replies skipped.")
        return

    from telegram import Bot
    bot = Bot(token=cfg.telegram.bot_token)

    outbox = bus.register_channel(CHANNEL)
    log.info("Reply sender ready — waiting for outbox events…")
    while True:
        event = await outbox.get()
        chat_id = event.metadata.get("chat_id")
        if not chat_id:
            log.warning("Outbox event has no chat_id, dropping: %s", event.payload[:80])
            continue
        # Stop typing now that we have a reply
        _stop_typing(chat_id)
        log.info("MSG OUT chat_id=%-14s  len=%d", chat_id, len(event.payload))
        try:
            await bot.send_message(chat_id=chat_id, text=event.payload)
        except Exception as exc:
            log.error("Failed to send to chat %s: %s", chat_id, exc)
