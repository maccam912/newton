"""Configuration — loads settings from config.toml + secrets from .env."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

import tomllib
from dotenv import load_dotenv

CONFIG_PATH = Path("config.toml")


class LLMConfig(BaseModel):
    provider: str = "anthropic"        # "anthropic" or "openrouter"
    model: str = "claude-sonnet-4-6"
    system_prompt: str = "You are Newton, a helpful assistant."
    use_cache: bool = True             # enable Anthropic prompt caching (ignored for openrouter)
    # OpenRouter-only settings (ignored when provider = "anthropic")
    reasoning_effort: str = "low"     # "low", "medium", or "high"


class TelegramConfig(BaseModel):
    bot_token: str = ""


class SchedulerConfig(BaseModel):
    heartbeat_minutes: int = 60
    reminder_check_seconds: int = 60  # how often to poll for due reminders


class MemoryConfig(BaseModel):
    db_path: str = "memory.db"
    # Embedding model name — use OpenAI naming (e.g. "text-embedding-3-small")
    # or OpenRouter naming (e.g. "openai/text-embedding-3-small") depending on provider.
    embedding_model: str = "text-embedding-3-small"
    # Base URL for the embeddings API.  Leave empty to use OpenAI directly.
    # Set to "https://openrouter.ai/api/v1" to route through OpenRouter.
    embedding_base_url: str = ""
    recall_window: int = 10        # recent messages to include in context
    archival_search_k: int = 5     # archival results per query
    idle_archival_seconds: int = 300  # 5 minutes


class AgentConfig(BaseModel):
    max_steps: int = 15            # max tool calls per turn before forced stop


class ToolsConfig(BaseModel):
    """Configuration for external tools."""
    searxng: dict[str, str | int] = {}  # e.g. {"base_url": "...", "max_results": 5}
    browser: dict[str, str | bool] = {}  # e.g. {"browser": "chromium", "headless": true}
    scripts: dict[str, str | int] = {}   # e.g. {"max_timeout": 300, "max_output_chars": 10000}
    bash: dict[str, str | int] = {}      # e.g. {"max_timeout": 120, "max_output_chars": 10000}
    vikunja: dict[str, str | int] = {}   # e.g. {"request_timeout": 15}


class Config(BaseModel):
    llm: LLMConfig = LLMConfig()
    telegram: TelegramConfig = TelegramConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    memory: MemoryConfig = MemoryConfig()
    agent: AgentConfig = AgentConfig()
    tools: ToolsConfig = ToolsConfig()


# Map of ENV_VAR -> (config section, field)
# Secrets are always read from env vars so they're never committed to config.toml.
_ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "TELEGRAM_BOT_TOKEN": ("telegram", "bot_token"),
}


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Load .env for secrets, then config.toml for everything else.

    Env vars always win for secret fields so you never commit tokens.
    """
    load_dotenv()  # loads .env into os.environ

    # Start with TOML (or defaults)
    if path.exists():
        data = tomllib.loads(path.read_text())
    else:
        data = {}

    # Layer env-var overrides onto the TOML data
    for env_var, (section, field) in _ENV_OVERRIDES.items():
        value = os.getenv(env_var)
        if value:
            data.setdefault(section, {})[field] = value

    return Config(**data)
