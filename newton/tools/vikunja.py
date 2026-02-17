"""Vikunja task/project management tool — gives the agent near-complete
access to a Vikunja instance via its REST API.

Set VIKUNJA_BASE_URL and VIKUNJA_API_TOKEN in .env (or the environment).
Optional tuning in config.toml under [tools.vikunja].

NOTE: Vikunja's API uses PUT for *creation* and POST for *updates*, which
is the reverse of most REST APIs.  The helpers here handle that internally.
"""

from __future__ import annotations

import json
import os
from typing import Any

import aiohttp
from pydantic_ai import Agent, RunContext

from newton.tracing import get_tracer

tracer = get_tracer("newton.tools.vikunja")

TYPE_CHECKING = False
if TYPE_CHECKING:
    from newton.agent import AgentDeps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vikunja_cfg(ctx: RunContext[AgentDeps]) -> dict[str, Any]:
    return ctx.deps.cfg.tools.vikunja


def _base_url(ctx: RunContext[AgentDeps]) -> str:
    url = os.getenv("VIKUNJA_BASE_URL") or _vikunja_cfg(ctx).get("base_url", "")
    return url.rstrip("/")


def _token(ctx: RunContext[AgentDeps]) -> str:
    return os.getenv("VIKUNJA_API_TOKEN") or _vikunja_cfg(ctx).get("api_token", "")


def _timeout(ctx: RunContext[AgentDeps]) -> aiohttp.ClientTimeout:
    secs = int(_vikunja_cfg(ctx).get("request_timeout", 15))
    return aiohttp.ClientTimeout(total=secs)


async def _request(
    ctx: RunContext[AgentDeps],
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    params: dict[str, str] | None = None,
) -> dict[str, Any] | list[Any] | str:
    """Issue a request to the Vikunja API and return the parsed response."""
    base = _base_url(ctx)
    if not base:
        return "Error: VIKUNJA_BASE_URL is not configured."
    token = _token(ctx)
    if not token:
        return "Error: VIKUNJA_API_TOKEN is not configured."

    url = f"{base}/api/v1{path}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method, url, headers=headers, json=body, params=params,
                timeout=_timeout(ctx),
            ) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    return f"HTTP {resp.status}: {text[:500]}"
                if not text or text.strip() == "":
                    return {"status": "ok"}
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text[:1000]
    except Exception as e:
        return f"Request failed: {e}"


def _fmt(data: Any, *, indent: bool = False) -> str:
    """Format API response data for the agent."""
    if isinstance(data, str):
        return data
    return json.dumps(data, indent=2 if indent else None, default=str)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(agent: Agent) -> None:
    """Register all Vikunja tools on the given agent."""
    from newton.agent import AgentDeps
    globals()["AgentDeps"] = AgentDeps

    agent.tool(vikunja_list_projects)
    agent.tool(vikunja_get_project)
    agent.tool(vikunja_create_project)
    agent.tool(vikunja_update_project)
    agent.tool(vikunja_delete_project)
    agent.tool(vikunja_list_tasks)
    agent.tool(vikunja_get_task)
    agent.tool(vikunja_create_task)
    agent.tool(vikunja_update_task)
    agent.tool(vikunja_delete_task)
    agent.tool(vikunja_list_labels)
    agent.tool(vikunja_create_label)
    agent.tool(vikunja_update_label)
    agent.tool(vikunja_delete_label)
    agent.tool(vikunja_add_task_label)
    agent.tool(vikunja_remove_task_label)
    agent.tool(vikunja_assign_user_to_task)
    agent.tool(vikunja_remove_user_from_task)
    agent.tool(vikunja_list_task_comments)
    agent.tool(vikunja_add_task_comment)
    agent.tool(vikunja_api)


# ---------------------------------------------------------------------------
# Project tools
# ---------------------------------------------------------------------------

async def vikunja_list_projects(
    ctx: RunContext[AgentDeps], page: int = 1, search: str = "",
) -> str:
    """List all Vikunja projects. Supports pagination and search.

    Args:
        page: Page number (default 1).
        search: Optional search string to filter projects by title.
    """
    with tracer.start_as_current_span("tools.vikunja_list_projects"):
        params: dict[str, str] = {"page": str(page)}
        if search:
            params["s"] = search
        data = await _request(ctx, "GET", "/projects", params=params)
        from newton.agent import _step_tag
        return _fmt(data, indent=True) + _step_tag(ctx.deps)


async def vikunja_get_project(
    ctx: RunContext[AgentDeps], project_id: int,
) -> str:
    """Get details of a single Vikunja project.

    Args:
        project_id: The project ID.
    """
    with tracer.start_as_current_span("tools.vikunja_get_project"):
        data = await _request(ctx, "GET", f"/projects/{project_id}")
        from newton.agent import _step_tag
        return _fmt(data, indent=True) + _step_tag(ctx.deps)


async def vikunja_create_project(
    ctx: RunContext[AgentDeps],
    title: str,
    description: str = "",
    parent_project_id: int = 0,
    hex_color: str = "",
) -> str:
    """Create a new Vikunja project.

    Args:
        title: Project title (required).
        description: Optional project description.
        parent_project_id: Optional parent project ID for nesting.
        hex_color: Optional hex color (e.g. '#ff0000').
    """
    with tracer.start_as_current_span("tools.vikunja_create_project"):
        body: dict[str, Any] = {"title": title}
        if description:
            body["description"] = description
        if parent_project_id:
            body["parent_project_id"] = parent_project_id
        if hex_color:
            body["hex_color"] = hex_color
        data = await _request(ctx, "PUT", "/projects", body=body)
        from newton.agent import _step_tag
        return _fmt(data, indent=True) + _step_tag(ctx.deps)


async def vikunja_update_project(
    ctx: RunContext[AgentDeps],
    project_id: int,
    title: str = "",
    description: str = "",
    is_archived: bool = False,
    hex_color: str = "",
) -> str:
    """Update an existing Vikunja project.  Only non-empty fields are sent.

    Args:
        project_id: The project ID to update.
        title: New title (leave empty to keep current).
        description: New description.
        is_archived: Set True to archive the project.
        hex_color: New hex color.
    """
    with tracer.start_as_current_span("tools.vikunja_update_project"):
        body: dict[str, Any] = {}
        if title:
            body["title"] = title
        if description:
            body["description"] = description
        if is_archived:
            body["is_archived"] = True
        if hex_color:
            body["hex_color"] = hex_color
        data = await _request(ctx, "POST", f"/projects/{project_id}", body=body)
        from newton.agent import _step_tag
        return _fmt(data, indent=True) + _step_tag(ctx.deps)


async def vikunja_delete_project(
    ctx: RunContext[AgentDeps], project_id: int,
) -> str:
    """Delete a Vikunja project and all its tasks.

    Args:
        project_id: The project ID to delete.
    """
    with tracer.start_as_current_span("tools.vikunja_delete_project"):
        data = await _request(ctx, "DELETE", f"/projects/{project_id}")
        from newton.agent import _step_tag
        return _fmt(data) + _step_tag(ctx.deps)


# ---------------------------------------------------------------------------
# Task tools
# ---------------------------------------------------------------------------

async def vikunja_list_tasks(
    ctx: RunContext[AgentDeps],
    page: int = 1,
    search: str = "",
    sort_by: str = "",
    filter_str: str = "",
    project_id: int = 0,
) -> str:
    """List tasks.  Without project_id, returns all tasks for the current user.
    With project_id, returns tasks in that specific project.

    Args:
        page: Page number (default 1).
        search: Optional search string to filter tasks by title.
        sort_by: Optional sort field (e.g. 'due_date', 'priority', 'done', 'created').
        filter_str: Optional filter string (Vikunja filter syntax, e.g. 'done = false').
        project_id: If set, list tasks for this project only.
    """
    with tracer.start_as_current_span("tools.vikunja_list_tasks"):
        params: dict[str, str] = {"page": str(page)}
        if search:
            params["s"] = search
        if sort_by:
            params["sort_by"] = sort_by
        if filter_str:
            params["filter"] = filter_str
        if project_id:
            path = f"/projects/{project_id}/tasks"
        else:
            path = "/tasks"
        data = await _request(ctx, "GET", path, params=params)
        from newton.agent import _step_tag
        return _fmt(data, indent=True) + _step_tag(ctx.deps)


async def vikunja_get_task(
    ctx: RunContext[AgentDeps], task_id: int,
) -> str:
    """Get full details of a single task, including labels, assignees, and relations.

    Args:
        task_id: The task ID.
    """
    with tracer.start_as_current_span("tools.vikunja_get_task"):
        data = await _request(ctx, "GET", f"/tasks/{task_id}")
        from newton.agent import _step_tag
        return _fmt(data, indent=True) + _step_tag(ctx.deps)


async def vikunja_create_task(
    ctx: RunContext[AgentDeps],
    project_id: int,
    title: str,
    description: str = "",
    done: bool = False,
    priority: int = 0,
    due_date: str = "",
    start_date: str = "",
    end_date: str = "",
    hex_color: str = "",
    repeat_after: int = 0,
    percent_done: float = 0.0,
    labels_json: str = "",
) -> str:
    """Create a new task in a Vikunja project.

    Args:
        project_id: The project to create the task in (required).
        title: Task title (required).
        description: Optional task description (supports markdown).
        done: Whether the task is already done (default false).
        priority: Priority level (0=unset, 1=low, 2=medium, 3=high, 4=urgent).
        due_date: Optional due date in ISO 8601 format (e.g. '2026-03-01T12:00:00Z').
        start_date: Optional start date in ISO 8601 format.
        end_date: Optional end date in ISO 8601 format.
        hex_color: Optional hex color (e.g. '#ff0000').
        repeat_after: Optional repeat interval in seconds (0=no repeat).
        percent_done: Completion percentage as a float (0.0 to 1.0).
        labels_json: Optional JSON array of label objects, e.g. '[{"id": 1}]'.
    """
    with tracer.start_as_current_span("tools.vikunja_create_task"):
        body: dict[str, Any] = {"title": title}
        if description:
            body["description"] = description
        if done:
            body["done"] = True
        if priority:
            body["priority"] = priority
        if due_date:
            body["due_date"] = due_date
        if start_date:
            body["start_date"] = start_date
        if end_date:
            body["end_date"] = end_date
        if hex_color:
            body["hex_color"] = hex_color
        if repeat_after:
            body["repeat_after"] = repeat_after
        if percent_done:
            body["percent_done"] = percent_done
        if labels_json:
            body["labels"] = json.loads(labels_json)
        data = await _request(ctx, "PUT", f"/projects/{project_id}/tasks", body=body)
        from newton.agent import _step_tag
        return _fmt(data, indent=True) + _step_tag(ctx.deps)


async def vikunja_update_task(
    ctx: RunContext[AgentDeps],
    task_id: int,
    title: str = "",
    description: str = "",
    done: bool = False,
    priority: int = -1,
    due_date: str = "",
    hex_color: str = "",
    percent_done: float = -1.0,
) -> str:
    """Update an existing Vikunja task.  Only explicitly set fields are sent.

    Args:
        task_id: The task ID to update.
        title: New title (empty = keep current).
        description: New description (empty = keep current).
        done: Set True to mark the task as done.
        priority: New priority (-1 = keep current; 0=unset, 1=low, 2=medium, 3=high, 4=urgent).
        due_date: New due date in ISO 8601 format (empty = keep current).
        hex_color: New hex color (empty = keep current).
        percent_done: New completion percentage (-1 = keep current; 0.0 to 1.0).
    """
    with tracer.start_as_current_span("tools.vikunja_update_task"):
        body: dict[str, Any] = {}
        if title:
            body["title"] = title
        if description:
            body["description"] = description
        if done:
            body["done"] = True
        if priority >= 0:
            body["priority"] = priority
        if due_date:
            body["due_date"] = due_date
        if hex_color:
            body["hex_color"] = hex_color
        if percent_done >= 0:
            body["percent_done"] = percent_done
        data = await _request(ctx, "POST", f"/tasks/{task_id}", body=body)
        from newton.agent import _step_tag
        return _fmt(data, indent=True) + _step_tag(ctx.deps)


async def vikunja_delete_task(
    ctx: RunContext[AgentDeps], task_id: int,
) -> str:
    """Delete a Vikunja task.

    Args:
        task_id: The task ID to delete.
    """
    with tracer.start_as_current_span("tools.vikunja_delete_task"):
        data = await _request(ctx, "DELETE", f"/tasks/{task_id}")
        from newton.agent import _step_tag
        return _fmt(data) + _step_tag(ctx.deps)


# ---------------------------------------------------------------------------
# Label tools
# ---------------------------------------------------------------------------

async def vikunja_list_labels(
    ctx: RunContext[AgentDeps], page: int = 1, search: str = "",
) -> str:
    """List all labels available to the current user.

    Args:
        page: Page number (default 1).
        search: Optional search string to filter by title.
    """
    with tracer.start_as_current_span("tools.vikunja_list_labels"):
        params: dict[str, str] = {"page": str(page)}
        if search:
            params["s"] = search
        data = await _request(ctx, "GET", "/labels", params=params)
        from newton.agent import _step_tag
        return _fmt(data, indent=True) + _step_tag(ctx.deps)


async def vikunja_create_label(
    ctx: RunContext[AgentDeps],
    title: str,
    hex_color: str = "",
    description: str = "",
) -> str:
    """Create a new label.

    Args:
        title: Label title (required).
        hex_color: Optional hex color (e.g. '#00ff00').
        description: Optional label description.
    """
    with tracer.start_as_current_span("tools.vikunja_create_label"):
        body: dict[str, Any] = {"title": title}
        if hex_color:
            body["hex_color"] = hex_color
        if description:
            body["description"] = description
        data = await _request(ctx, "PUT", "/labels", body=body)
        from newton.agent import _step_tag
        return _fmt(data, indent=True) + _step_tag(ctx.deps)


async def vikunja_update_label(
    ctx: RunContext[AgentDeps],
    label_id: int,
    title: str = "",
    hex_color: str = "",
    description: str = "",
) -> str:
    """Update an existing label.

    Args:
        label_id: The label ID to update.
        title: New title (empty = keep current).
        hex_color: New hex color (empty = keep current).
        description: New description (empty = keep current).
    """
    with tracer.start_as_current_span("tools.vikunja_update_label"):
        body: dict[str, Any] = {}
        if title:
            body["title"] = title
        if hex_color:
            body["hex_color"] = hex_color
        if description:
            body["description"] = description
        data = await _request(ctx, "POST", f"/labels/{label_id}", body=body)
        from newton.agent import _step_tag
        return _fmt(data, indent=True) + _step_tag(ctx.deps)


async def vikunja_delete_label(
    ctx: RunContext[AgentDeps], label_id: int,
) -> str:
    """Delete a label.

    Args:
        label_id: The label ID to delete.
    """
    with tracer.start_as_current_span("tools.vikunja_delete_label"):
        data = await _request(ctx, "DELETE", f"/labels/{label_id}")
        from newton.agent import _step_tag
        return _fmt(data) + _step_tag(ctx.deps)


# ---------------------------------------------------------------------------
# Task label tools
# ---------------------------------------------------------------------------

async def vikunja_add_task_label(
    ctx: RunContext[AgentDeps], task_id: int, label_id: int,
) -> str:
    """Add a label to a task.

    Args:
        task_id: The task ID.
        label_id: The label ID to add.
    """
    with tracer.start_as_current_span("tools.vikunja_add_task_label"):
        body = {"label_id": label_id}
        data = await _request(ctx, "PUT", f"/tasks/{task_id}/labels", body=body)
        from newton.agent import _step_tag
        return _fmt(data) + _step_tag(ctx.deps)


async def vikunja_remove_task_label(
    ctx: RunContext[AgentDeps], task_id: int, label_id: int,
) -> str:
    """Remove a label from a task.

    Args:
        task_id: The task ID.
        label_id: The label ID to remove.
    """
    with tracer.start_as_current_span("tools.vikunja_remove_task_label"):
        data = await _request(ctx, "DELETE", f"/tasks/{task_id}/labels/{label_id}")
        from newton.agent import _step_tag
        return _fmt(data) + _step_tag(ctx.deps)


# ---------------------------------------------------------------------------
# Task assignee tools
# ---------------------------------------------------------------------------

async def vikunja_assign_user_to_task(
    ctx: RunContext[AgentDeps], task_id: int, user_id: int,
) -> str:
    """Assign a user to a task.

    Args:
        task_id: The task ID.
        user_id: The user ID to assign.
    """
    with tracer.start_as_current_span("tools.vikunja_assign_user_to_task"):
        body = {"user_id": user_id}
        data = await _request(ctx, "PUT", f"/tasks/{task_id}/assignees", body=body)
        from newton.agent import _step_tag
        return _fmt(data) + _step_tag(ctx.deps)


async def vikunja_remove_user_from_task(
    ctx: RunContext[AgentDeps], task_id: int, user_id: int,
) -> str:
    """Remove an assigned user from a task.

    Args:
        task_id: The task ID.
        user_id: The user ID to remove.
    """
    with tracer.start_as_current_span("tools.vikunja_remove_user_from_task"):
        data = await _request(ctx, "DELETE", f"/tasks/{task_id}/assignees/{user_id}")
        from newton.agent import _step_tag
        return _fmt(data) + _step_tag(ctx.deps)


# ---------------------------------------------------------------------------
# Task comment tools
# ---------------------------------------------------------------------------

async def vikunja_list_task_comments(
    ctx: RunContext[AgentDeps], task_id: int, page: int = 1,
) -> str:
    """List all comments on a task.

    Args:
        task_id: The task ID.
        page: Page number (default 1).
    """
    with tracer.start_as_current_span("tools.vikunja_list_task_comments"):
        params: dict[str, str] = {"page": str(page)}
        data = await _request(ctx, "GET", f"/tasks/{task_id}/comments", params=params)
        from newton.agent import _step_tag
        return _fmt(data, indent=True) + _step_tag(ctx.deps)


async def vikunja_add_task_comment(
    ctx: RunContext[AgentDeps], task_id: int, comment: str,
) -> str:
    """Add a comment to a task.

    Args:
        task_id: The task ID.
        comment: The comment text.
    """
    with tracer.start_as_current_span("tools.vikunja_add_task_comment"):
        body = {"comment": comment}
        data = await _request(ctx, "PUT", f"/tasks/{task_id}/comments", body=body)
        from newton.agent import _step_tag
        return _fmt(data) + _step_tag(ctx.deps)


# ---------------------------------------------------------------------------
# Generic API tool — covers everything else
# ---------------------------------------------------------------------------

async def vikunja_api(
    ctx: RunContext[AgentDeps],
    method: str,
    path: str,
    body_json: str = "",
    query_params_json: str = "",
) -> str:
    """Make an arbitrary Vikunja API call for any endpoint not covered by
    the dedicated tools.  Provides access to teams, buckets, views,
    notifications, webhooks, shares, relations, attachments, filters, and more.

    Args:
        method: HTTP method — GET, PUT (create), POST (update), or DELETE.
        path: API path *after* /api/v1, e.g. '/teams' or '/tasks/42/relations'.
        body_json: Optional JSON string for the request body.
        query_params_json: Optional JSON string of query parameters, e.g. '{"page": "2"}'.
    """
    with tracer.start_as_current_span(
        "tools.vikunja_api", attributes={"method": method, "path": path}
    ):
        body = json.loads(body_json) if body_json else None
        params = json.loads(query_params_json) if query_params_json else None
        data = await _request(ctx, method.upper(), path, body=body, params=params)
        from newton.agent import _step_tag
        return _fmt(data, indent=True) + _step_tag(ctx.deps)
