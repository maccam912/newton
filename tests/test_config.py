"""Unit tests for newton.config."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from newton.config import Config, ToolsConfig, load_config


pytestmark = pytest.mark.unit


class TestToolsConfig:
    def test_defaults(self):
        tc = ToolsConfig()
        assert tc.searxng == {}
        assert tc.scripts == {}

    def test_scripts_fields(self):
        tc = ToolsConfig(scripts={"max_timeout": 120, "max_output_chars": 5000})
        assert tc.scripts["max_timeout"] == 120


class TestConfig:
    def test_default_config(self):
        cfg = Config()
        assert cfg.llm.provider == "zai"
        assert cfg.llm.prompt_prefix_cache_ttl_seconds == 300
        assert cfg.tools.scripts == {}
        assert cfg.agent.max_steps == 15

    def test_load_config_missing_file(self, tmp_path: Path):
        """load_config with a non-existent file returns defaults."""
        cfg = load_config(tmp_path / "nope.toml")
        assert cfg.llm.model == "glm-5"

    def test_load_config_with_toml(self, tmp_path: Path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(textwrap.dedent("""\
            [tools.scripts]
            max_timeout = 60
        """))
        cfg = load_config(toml_file)
        assert cfg.tools.scripts["max_timeout"] == 60

    def test_provider_specific_api_key_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(textwrap.dedent("""\
            [llm]
            provider = "zai"
            model = "glm-5"
        """))
        monkeypatch.setenv("ZAI_API_KEY", "zai-secret")
        cfg = load_config(toml_file)
        assert cfg.llm.api_key == "zai-secret"

    def test_openrouter_api_key_not_overridden_by_zai_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(textwrap.dedent("""\
            [llm]
            provider = "openrouter"
        """))
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("ZAI_API_KEY", "zai-key")
        cfg = load_config(toml_file)
        assert cfg.llm.api_key == "or-key"
