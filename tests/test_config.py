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
        assert tc.browser == {}
        assert tc.scripts == {}

    def test_browser_fields(self):
        tc = ToolsConfig(browser={"browser": "firefox", "headless": False})
        assert tc.browser["browser"] == "firefox"
        assert tc.browser["headless"] is False

    def test_scripts_fields(self):
        tc = ToolsConfig(scripts={"max_timeout": 120, "max_output_chars": 5000})
        assert tc.scripts["max_timeout"] == 120


class TestConfig:
    def test_default_config(self):
        cfg = Config()
        assert cfg.tools.browser == {}
        assert cfg.tools.scripts == {}
        assert cfg.agent.max_steps == 15

    def test_load_config_missing_file(self, tmp_path: Path):
        """load_config with a non-existent file returns defaults."""
        cfg = load_config(tmp_path / "nope.toml")
        assert cfg.llm.model == "stepfun/step-3.5-flash:free"

    def test_load_config_with_toml(self, tmp_path: Path):
        toml_file = tmp_path / "test.toml"
        toml_file.write_text(textwrap.dedent("""\
            [tools.browser]
            browser = "firefox"
            headless = false

            [tools.scripts]
            max_timeout = 60
        """))
        cfg = load_config(toml_file)
        assert cfg.tools.browser["browser"] == "firefox"
        assert cfg.tools.browser["headless"] is False
        assert cfg.tools.scripts["max_timeout"] == 60
