"""Sub-agent tools — let the parent agent delegate multi-step tasks to isolated agents.

A sub-agent is a fresh, context-free agent that receives exactly one task description
from the parent LLM.  It has no access to the parent's conversation history; its only
instruction is what the parent writes.  When it finishes, only its final answer is
returned — all intermediate tool calls stay invisible to the parent.

Typical use cases
-----------------
- Multi-step web research (search → read → synthesise) where only the answer matters.
- Running a script or bash command and interpreting the output without exposing the
  raw output in the parent context.
- Checking Vikunja project / task status across multiple API calls.
- Looking up archival memory across several semantic queries to produce a single
  synthesised fact.
- Parallel independent lookups: e.g. "What is the latest version of package A?",
  "What is the latest version of package B?" — run simultaneously and get both
  answers back in one step.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic_ai import Agent, RunContext

from newton.config import Config
from newton.tracing import get_tracer

log = logging.getLogger(__name__)
tracer = get_tracer("newton.tools.subagent")

TYPE_CHECKING = False
if TYPE_CHECKING:
    from newton.agent import AgentDeps


# ---------------------------------------------------------------------------
# Sub-agent system prompt
# ---------------------------------------------------------------------------

_SUBAGENT_SYSTEM_PROMPT = """\
You are a focused sub-agent.  You have been given a single task by your parent agent.

Your job:
1. Use available tools to complete the task as thoroughly and accurately as possible.
2. Do NOT engage in conversation, ask for clarification, or produce preamble.
   Work with the information given; your task description is complete.
3. When you are done, write a clear, self-contained answer.
   Your final text response IS what gets returned to the parent — make it useful.

You have access to: web search, bash commands, Python scripts, task management
(Vikunja), and archival memory search.  Use whichever tools are necessary.

Do not call respond_to_user or end_turn — those do not exist here.
Simply complete your work and produce your final answer as plain text.
"""

# ---------------------------------------------------------------------------
# Lazy sub-agent cache
# ---------------------------------------------------------------------------

_subagent_cache: dict[str, Agent] = {}  # keyed by "{provider}:{model}"


def _get_or_create_subagent(cfg: Config) -> Agent:
    """Return a cached sub-agent for the current model, building it if needed."""
    from newton.agent import AgentDeps, _build_model  # deferred — avoids circular import

    cache_key = f"{cfg.llm.provider}:{cfg.llm.model}"
    if cache_key in _subagent_cache:
        return _subagent_cache[cache_key]

    model = _build_model(cfg)
    sub: Agent = Agent(
        model,
        deps_type=AgentDeps,
        system_prompt=_SUBAGENT_SYSTEM_PROMPT,
    )

    # -- External tools (same modules as the parent agent) -------------------
    import newton.tools.searxng
    import newton.tools.bash
    import newton.tools.scripts
    import newton.tools.vikunja

    newton.tools.searxng.register(sub)
    newton.tools.bash.register(sub)
    newton.tools.scripts.register(sub)
    newton.tools.vikunja.register(sub)

    # -- Memory tools (read + optional write) --------------------------------
    # Imported at call time to avoid circular dependency.

    @sub.tool
    async def archival_memory_search(
        ctx: RunContext[AgentDeps], query: str, k: int = 5
    ) -> str:
        """Search long-term archival memory semantically.  Returns top-k matches.

        Args:
            query: Natural-language search query.
            k: Number of results to return (default 5).
        """
        from newton.agent import _step_tag
        log.debug("🔍 [sub] archival_search(%s, k=%d)", query[:60], k)
        results = await ctx.deps.memory.archival_search(query, k)
        if not results:
            return "No archival memories found." + _step_tag(ctx.deps)
        body = "\n".join(f"- {r}" for r in results)
        return body + _step_tag(ctx.deps)

    @sub.tool
    async def archival_memory_insert(ctx: RunContext[AgentDeps], content: str) -> str:
        """Store a fact discovered during this task in long-term archival memory.

        Only use this for genuinely reusable facts, not intermediate working notes.

        Args:
            content: The fact to store (self-contained, context-free sentence).
        """
        from newton.agent import _step_tag
        log.info("📦 [sub] archival_insert: %s", content[:80])
        row_id = await ctx.deps.memory.archival_insert(content)
        return f"Stored in archival memory (id={row_id})." + _step_tag(ctx.deps)

    @sub.tool
    async def core_memory_read(ctx: RunContext[AgentDeps], block: str) -> str:
        """Read a core memory block (e.g. 'persona', 'directives', 'notebook').

        Args:
            block: Name of the block to read.
        """
        from newton.agent import _step_tag
        log.debug("🧠 [sub] core_memory_read(%s)", block)
        content = await ctx.deps.memory.get_core_block(block)
        result = content or f"[block '{block}' not found]"
        return result + _step_tag(ctx.deps)

    # -- Skill tools ---------------------------------------------------------

    @sub.tool
    async def skill_invoke(ctx: RunContext[AgentDeps], name: str) -> str:
        """Load the full instructions for a skill by name.

        Args:
            name: Skill identifier as shown by skill_list.
        """
        from newton.agent import _step_tag
        log.info("🎯 [sub] skill_invoke(%s)", name)
        skill = await ctx.deps.memory.skill_get(name)
        if skill is None:
            return f"Skill '{name}' not found." + _step_tag(ctx.deps)
        return (
            f"=== SKILL: {skill['name']} ===\n{skill['full_prompt']}"
        ) + _step_tag(ctx.deps)

    @sub.tool
    async def skill_list(ctx: RunContext[AgentDeps]) -> str:
        """List all available skills with their one-line descriptions."""
        from newton.agent import _step_tag
        log.debug("🎯 [sub] skill_list")
        skills = await ctx.deps.memory.skill_list()
        if not skills:
            return "No skills defined yet." + _step_tag(ctx.deps)
        lines = [f"- {s['name']}: {s['description']}" for s in skills]
        return "\n".join(lines) + _step_tag(ctx.deps)

    _subagent_cache[cache_key] = sub
    log.info("sub-agent built and cached (key=%s)", cache_key)
    return sub


# ---------------------------------------------------------------------------
# Register on parent agent
# ---------------------------------------------------------------------------


def register(agent: Agent) -> None:
    """Register run_subagent and run_parallel_subagents on the parent agent."""
    from newton.agent import AgentDeps, _step_tag  # deferred import

    globals()["AgentDeps"] = AgentDeps

    @agent.tool
    async def run_subagent(ctx: RunContext[AgentDeps], task: str) -> str:
        """Delegate a self-contained multi-step task to a focused sub-agent.

        The sub-agent has a FRESH context — it does NOT see this conversation.
        Its only instruction is the task string you provide here.  It has access
        to the same tools (web search, bash, Python scripts, Vikunja, archival
        memory, skills) and will work autonomously until it produces an answer.
        Only the final answer is returned here; all intermediate tool calls are
        invisible to you.

        Use this when:
        - A task requires several tool calls but you only care about the result,
          not the journey (e.g. "find the current stable Python version").
        - Intermediate outputs would add noise to your context (e.g. raw JSON
          from a Vikunja API call before extracting the count you need).
        - You need to research a topic across multiple web searches and want a
          synthesised summary without the search result dumps in your context.
        - A bash/script command needs to be run, output parsed, and only the
          relevant figure returned.
        - You want to check Vikunja project status by aggregating several calls.

        Write the task as if briefing a competent assistant who knows nothing
        about the current conversation.  Be specific and complete.

        Args:
            task: Full description of the task the sub-agent should complete.
                  Should be self-contained — no references to "the above" or
                  "the user said"; include all context the agent needs.

        Returns:
            The sub-agent's final answer as a plain-text string.

        Examples:
            run_subagent("Find the latest stable release version of pydantic-ai
                          from PyPI and return just the version number.")

            run_subagent("Search for the top 3 most-cited papers on RAG
                          (Retrieval-Augmented Generation) published in 2024.
                          Return each with title, authors, and a one-sentence
                          summary.")

            run_subagent("Run `df -h` on the local machine and return a summary
                          of disk usage, noting any filesystems over 80% full.")

            run_subagent("Look through archival memory for anything stored about
                          project deadlines or milestone dates and return all
                          relevant entries in a clear list.")
        """
        with tracer.start_as_current_span(
            "subagent.run", attributes={"task_len": len(task)}
        ) as span:
            log.info("🤖 run_subagent → task=%s", task[:120])
            sub = _get_or_create_subagent(ctx.deps.cfg)

            # Fresh deps — isolated step counter, no parent event state
            from newton.agent import AgentDeps as _AgentDeps
            fresh_deps = _AgentDeps(
                memory=ctx.deps.memory,
                cfg=ctx.deps.cfg,
                bus=ctx.deps.bus,
            )

            try:
                result = await sub.run(task, deps=fresh_deps)
                answer = result.output
                log.info("🤖 run_subagent ← steps=%d  answer=%s", fresh_deps.step, answer[:120])
                span.set_attribute("steps", fresh_deps.step)
                span.set_attribute("answer_len", len(answer))
            except Exception as e:
                log.error("run_subagent failed", exc_info=e)
                span.record_exception(e)
                answer = f"Sub-agent failed: {e}"

            return answer + _step_tag(ctx.deps)

    @agent.tool
    async def run_parallel_subagents(
        ctx: RunContext[AgentDeps], tasks: list[str]
    ) -> str:
        """Run up to 5 sub-agents in parallel, each with its own isolated context.

        Each sub-agent is independent — they share no state and run concurrently.
        All results are collected and returned together when every sub-agent
        completes (or fails).  Failed tasks produce an error note in their slot
        rather than aborting the whole batch.

        Use this when:
        - You have multiple independent lookups or research questions.
        - Each task is self-contained and doesn't depend on the others.
        - You want to save time by running tasks concurrently rather than
          sequentially.

        Examples:
            run_parallel_subagents([
                "Find the latest stable version of pydantic-ai on PyPI.",
                "Find the latest stable version of aiohttp on PyPI.",
                "Find the latest stable version of python-telegram-bot on PyPI.",
            ])

            run_parallel_subagents([
                "Search the web for the current EUR/USD exchange rate.",
                "Search the web for the current BTC/USD price.",
            ])

            run_parallel_subagents([
                "List all open Vikunja tasks assigned to project 'Alpha'.",
                "List all overdue Vikunja tasks across all projects.",
                "Check if there are any Vikunja tasks due today.",
            ])

        Args:
            tasks: List of task descriptions (1–5 items).  Each is run in a
                   separate sub-agent with no shared context.

        Returns:
            Numbered results block with one section per task, in the same
            order as the input list.
        """
        if not tasks:
            return "No tasks provided." + _step_tag(ctx.deps)

        MAX_PARALLEL = 5
        if len(tasks) > MAX_PARALLEL:
            return (
                f"Too many tasks: {len(tasks)} provided, maximum is {MAX_PARALLEL}. "
                f"Split into multiple calls or reduce the task list."
            ) + _step_tag(ctx.deps)

        with tracer.start_as_current_span(
            "subagent.run_parallel",
            attributes={"task_count": len(tasks)},
        ) as span:
            log.info("🤖 run_parallel_subagents → %d tasks", len(tasks))
            sub = _get_or_create_subagent(ctx.deps.cfg)

            # Build one fresh AgentDeps per task
            from newton.agent import AgentDeps as _AgentDeps

            async def _run_one(task: str, idx: int) -> str:
                fresh_deps = _AgentDeps(
                    memory=ctx.deps.memory,
                    cfg=ctx.deps.cfg,
                    bus=ctx.deps.bus,
                )
                try:
                    result = await sub.run(task, deps=fresh_deps)
                    log.info(
                        "🤖 parallel sub %d/%d done  steps=%d",
                        idx + 1, len(tasks), fresh_deps.step,
                    )
                    return result.output
                except Exception as e:
                    log.error("parallel sub-agent %d failed: %s", idx + 1, e)
                    return f"[Error: {e}]"

            answers = await asyncio.gather(
                *[_run_one(t, i) for i, t in enumerate(tasks)]
            )

            span.set_attribute("completed", len(answers))

            blocks: list[str] = []
            for i, (task, answer) in enumerate(zip(tasks, answers), 1):
                blocks.append(f"**Task {i}:** {task}\n\n**Result {i}:**\n{answer}")

            return "\n\n---\n\n".join(blocks) + _step_tag(ctx.deps)
