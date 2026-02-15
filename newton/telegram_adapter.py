"""Telegram adapter — inbound messages and outbound replies."""

from __future__ import annotations

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters

from newton.config import Config
from newton.events import Event, EventBus, EventKind

CHANNEL = "telegram"


async def start_telegram(bus: EventBus, cfg: Config) -> None:
    """Start the Telegram bot and funnel messages into the inbox."""
    if not cfg.telegram.bot_token:
        print("[telegram] No bot token configured — skipping.")
        return

    # Track every chat we've seen so the agent can browse them later
    known_chats: dict[str, str] = {}   # chat_id -> chat title/name

    app = ApplicationBuilder().token(cfg.telegram.bot_token).build()

    async def on_message(update: Update, _context) -> None:
        if not update.message:
            return
        chat = update.message.chat
        chat_id = str(chat.id)

        # Remember this chat
        known_chats[chat_id] = chat.title or chat.full_name or chat_id

        await bus.put_inbox(
            Event(
                source=CHANNEL,
                kind=EventKind.MESSAGE,
                payload=update.message.text or "",
                metadata={"chat_id": chat_id},
            )
        )

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    await app.run_polling()


async def send_replies(bus: EventBus, cfg: Config) -> None:
    """Pull response events from this channel's outbox and send on Telegram."""
    if not cfg.telegram.bot_token:
        return

    from telegram import Bot
    bot = Bot(token=cfg.telegram.bot_token)

    outbox = bus.register_channel(CHANNEL)
    while True:
        event = await outbox.get()
        chat_id = event.metadata.get("chat_id")
        if chat_id:
            await bot.send_message(chat_id=chat_id, text=event.payload)
