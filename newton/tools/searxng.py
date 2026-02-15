"""SearXNG web search tool — lets the agent search the web.

Requires a SearXNG instance.  Set SEARXNG_BASE_URL in .env (or the environment).
Max results can be tuned in config.toml under [tools.searxng].
"""

from __future__ import annotations

import os

import aiohttp
from pydantic_ai import Agent, RunContext

from newton.tracing import get_tracer

tracer = get_tracer("newton.tools.searxng")

# We import AgentDeps at registration time to avoid circular imports.
# The register function receives the agent and wires things up.

TYPE_CHECKING = False
if TYPE_CHECKING:
    from newton.agent import AgentDeps


def register(agent: Agent) -> None:
    """Register the SearXNG search tool on the given agent."""
    # Import AgentDeps into module globals so get_type_hints() can resolve
    # the RunContext[AgentDeps] annotation at runtime.
    from newton.agent import AgentDeps
    globals()["AgentDeps"] = AgentDeps
    agent.tool(web_search)


async def web_search(
    ctx: RunContext[AgentDeps], query: str, max_results: int = 5
) -> str:
    """Search the web using SearXNG.  Returns a summary of results.

    Args:
        query: The search query.
        max_results: Maximum number of results to return (default 5).
    """
    with tracer.start_as_current_span(
        "tools.web_search", attributes={"query": query, "max_results": max_results}
    ) as span:
        # Config is a Pydantic model, access fields directly
        searxng_cfg = ctx.deps.cfg.tools.searxng
        base_url = os.getenv("SEARXNG_BASE_URL") or searxng_cfg.get("base_url", "http://localhost:8080")
        configured_max = searxng_cfg.get("max_results", 5)
        
        k = min(max_results, int(configured_max))

        params = {
            "q": query,
            "format": "json",
            "categories": "general",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{base_url}/search", params=params, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        except Exception as e:
            span.set_attribute("error", str(e))
            return f"Search failed: {e}"

        results = data.get("results", [])[:k]
        span.set_attribute("result_count", len(results))

        if not results:
            return "No results found."

        lines: list[str] = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("content", "")[:200]
            lines.append(f"{i}. **{title}**\n   {url}\n   {snippet}")

        # Import here to avoid circular dep at module level
        from newton.agent import _step_tag
        return "\n\n".join(lines) + _step_tag(ctx.deps)
