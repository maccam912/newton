"""Unit tests for newton.tools.browser."""

from __future__ import annotations

import pytest

from newton.config import Config
from newton.tools.browser import create_browser_server


pytestmark = pytest.mark.unit


class TestCreateBrowserServer:
    def test_default_args(self, cfg: Config):
        server = create_browser_server(cfg)
        # defaults: chromium, headless
        assert server.command == "npx"
        assert "-y" in server.args
        assert "@playwright/mcp@latest" in server.args
        assert "--headless" in server.args
        assert "--browser" in server.args
        assert "chromium" in server.args

    def test_custom_browser(self):
        cfg = Config(tools={"browser": {"browser": "firefox", "headless": True}})
        server = create_browser_server(cfg)
        assert "firefox" in server.args
        assert "--headless" in server.args

    def test_headless_false(self):
        cfg = Config(tools={"browser": {"browser": "chromium", "headless": False}})
        server = create_browser_server(cfg)
        assert "--headless" not in server.args
        assert "chromium" in server.args

    def test_empty_config_uses_defaults(self):
        cfg = Config()
        server = create_browser_server(cfg)
        assert "chromium" in server.args
        assert "--headless" in server.args
