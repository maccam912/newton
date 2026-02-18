"""Configuration — loads settings from config.toml + secrets from .env."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

import tomllib
from dotenv import load_dotenv

CONFIG_PATH = Path("config.toml")


class LLMConfig(BaseModel):
    provider: str = "zai"  # "openrouter" or "zai"
    model: str = "glm-5"
    api_key: str = ""
    base_url: str = ""
    system_prompt: str = "You are Newton, a helpful assistant."
    reasoning_effort: str = "low"  # "low", "medium", or "high"
    prompt_prefix_cache_ttl_seconds: int = 300


class TelegramConfig(BaseModel):
    bot_token: str = ""


class SchedulerConfig(BaseModel):
    heartbeat_minutes: int = 60
    reminder_check_seconds: int = 60  # how often to poll for due reminders


class MemoryConfig(BaseModel):
    db_path: str = "memory.db"
    embedding_model: str = "openai/text-embedding-3-small"
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
_ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "TELEGRAM_BOT_TOKEN": ("telegram", "bot_token"),
    "OPENROUTER_API_KEY": ("llm", "openrouter_api_key"),
    "ZAI_API_KEY": ("llm", "zai_api_key"),
    "ZAI_BASE_URL": ("llm", "base_url"),
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

    # Normalize provider-specific API key aliases to llm.api_key.
    llm_data = data.setdefault("llm", {})
    provider = str(llm_data.get("provider", "zai")).lower()
    if not llm_data.get("api_key"):
        if provider == "zai" and llm_data.get("zai_api_key"):
            llm_data["api_key"] = llm_data["zai_api_key"]
        elif provider == "openrouter" and llm_data.get("openrouter_api_key"):
            llm_data["api_key"] = llm_data["openrouter_api_key"]

    llm_data.pop("openrouter_api_key", None)
    llm_data.pop("zai_api_key", None)

    return Config(**data)
