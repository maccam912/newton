"""Browser tool — Playwright MCP server integration."""

from __future__ import annotations

from pydantic_ai.mcp import MCPServerStdio


def create_browser_server(cfg) -> MCPServerStdio:
    """Build an MCPServerStdio pointing at @playwright/mcp."""
    browser_cfg = cfg.tools.browser
    browser = browser_cfg.get("browser", "chromium")
    headless = browser_cfg.get("headless", True)

    args = ["-y", "@playwright/mcp@latest"]
    if headless:
        args.append("--headless")
    args.extend(["--browser", browser])

    return MCPServerStdio("npx", args=args)
