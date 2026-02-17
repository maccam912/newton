import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from newton.tools.vikunja import (
    register,
    vikunja_list_projects,
    vikunja_get_project,
    vikunja_create_project,
    vikunja_update_project,
    vikunja_delete_project,
    vikunja_list_tasks,
    vikunja_get_task,
    vikunja_create_task,
    vikunja_update_task,
    vikunja_delete_task,
    vikunja_list_labels,
    vikunja_create_label,
    vikunja_update_label,
    vikunja_delete_label,
    vikunja_add_task_label,
    vikunja_remove_task_label,
    vikunja_assign_user_to_task,
    vikunja_remove_user_from_task,
    vikunja_list_task_comments,
    vikunja_add_task_comment,
    vikunja_api,
)
import sys
from newton.config import Config, ToolsConfig


@pytest.fixture
def mock_step_tag():
    """Patch _step_tag in newton.agent."""
    try:
        import newton.agent
    except ImportError:
        mock_mod = MagicMock()
        mock_mod._step_tag = lambda deps: f" [step {deps.step+1}/{deps.max_steps}]"
        with patch.dict(sys.modules, {"newton.agent": mock_mod}):
            yield
        return
    with patch(
        "newton.agent._step_tag",
        side_effect=lambda deps: f" [step {deps.step+1}/{deps.max_steps}]",
    ):
        yield


class MockDeps:
    def __init__(self):
        self.cfg = Config(
            tools=ToolsConfig(vikunja={"request_timeout": 5})
        )
        self.step = 1
        self.max_steps = 10
        self.bus = MagicMock()
        self.memory = MagicMock()
        self.current_message = ""
        self.turn_ended = False
        self.event_metadata = {}


@pytest.fixture
def mock_ctx():
    deps = MockDeps()
    ctx = MagicMock()
    ctx.deps = deps
    return ctx


def _mock_aiohttp(response_data, status=200):
    """Build patched aiohttp.ClientSession returning the given response."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.text = AsyncMock(return_value=json.dumps(response_data))
    mock_resp.raise_for_status = MagicMock()

    mock_session = AsyncMock()
    mock_session.request = MagicMock()

    req_ctx = MagicMock()
    req_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    req_ctx.__aexit__ = AsyncMock(return_value=None)
    mock_session.request.return_value = req_ctx

    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    session_ctx.__aexit__ = AsyncMock(return_value=None)

    return patch("aiohttp.ClientSession", return_value=session_ctx), mock_session


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register():
    agent = MagicMock()
    register(agent)
    assert agent.tool.call_count == 21


# ---------------------------------------------------------------------------
# Missing config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_base_url(mock_ctx, mock_step_tag):
    """Returns error when VIKUNJA_BASE_URL is not set."""
    with patch.dict("os.environ", {}, clear=True):
        result = await vikunja_list_projects(mock_ctx)
    assert "VIKUNJA_BASE_URL is not configured" in result


@pytest.mark.asyncio
async def test_missing_api_token(mock_ctx, mock_step_tag):
    """Returns error when VIKUNJA_API_TOKEN is not set."""
    env = {"VIKUNJA_BASE_URL": "http://vikunja.local"}
    with patch.dict("os.environ", env, clear=True):
        result = await vikunja_list_projects(mock_ctx)
    assert "VIKUNJA_API_TOKEN is not configured" in result


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

ENV = {"VIKUNJA_BASE_URL": "http://vikunja.local", "VIKUNJA_API_TOKEN": "tok123"}


@pytest.mark.asyncio
async def test_list_projects(mock_ctx, mock_step_tag):
    projects = [{"id": 1, "title": "Inbox"}, {"id": 2, "title": "Work"}]
    patcher, mock_session = _mock_aiohttp(projects)
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_list_projects(mock_ctx, search="Work")
    assert "Work" in result
    assert "[step" in result


@pytest.mark.asyncio
async def test_get_project(mock_ctx, mock_step_tag):
    project = {"id": 1, "title": "Inbox", "description": "Default project"}
    patcher, _ = _mock_aiohttp(project)
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_get_project(mock_ctx, project_id=1)
    assert "Inbox" in result


@pytest.mark.asyncio
async def test_create_project(mock_ctx, mock_step_tag):
    created = {"id": 5, "title": "New Project"}
    patcher, mock_session = _mock_aiohttp(created)
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_create_project(mock_ctx, title="New Project")
    assert "New Project" in result
    # Verify PUT method was used (Vikunja convention)
    mock_session.request.assert_called_once()
    call_args = mock_session.request.call_args
    assert call_args[0][0] == "PUT"


@pytest.mark.asyncio
async def test_update_project(mock_ctx, mock_step_tag):
    updated = {"id": 1, "title": "Renamed"}
    patcher, mock_session = _mock_aiohttp(updated)
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_update_project(mock_ctx, project_id=1, title="Renamed")
    assert "Renamed" in result
    call_args = mock_session.request.call_args
    assert call_args[0][0] == "POST"


@pytest.mark.asyncio
async def test_delete_project(mock_ctx, mock_step_tag):
    patcher, mock_session = _mock_aiohttp({"message": "Successfully deleted."})
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_delete_project(mock_ctx, project_id=1)
    call_args = mock_session.request.call_args
    assert call_args[0][0] == "DELETE"


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_tasks_global(mock_ctx, mock_step_tag):
    tasks = [{"id": 10, "title": "Buy milk", "done": False}]
    patcher, mock_session = _mock_aiohttp(tasks)
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_list_tasks(mock_ctx)
    assert "Buy milk" in result
    url = mock_session.request.call_args[0][1]
    assert "/tasks" in url
    assert "/projects/" not in url


@pytest.mark.asyncio
async def test_list_tasks_by_project(mock_ctx, mock_step_tag):
    tasks = [{"id": 11, "title": "Deploy", "done": False}]
    patcher, mock_session = _mock_aiohttp(tasks)
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_list_tasks(mock_ctx, project_id=3)
    url = mock_session.request.call_args[0][1]
    assert "/projects/3/tasks" in url


@pytest.mark.asyncio
async def test_get_task(mock_ctx, mock_step_tag):
    task = {"id": 10, "title": "Buy milk", "description": "2%", "done": False}
    patcher, _ = _mock_aiohttp(task)
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_get_task(mock_ctx, task_id=10)
    assert "Buy milk" in result


@pytest.mark.asyncio
async def test_create_task(mock_ctx, mock_step_tag):
    created = {"id": 20, "title": "New task", "project_id": 1}
    patcher, mock_session = _mock_aiohttp(created)
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_create_task(
            mock_ctx, project_id=1, title="New task", priority=3
        )
    assert "New task" in result
    call_args = mock_session.request.call_args
    assert call_args[0][0] == "PUT"
    body = call_args[1]["json"]
    assert body["title"] == "New task"
    assert body["priority"] == 3


@pytest.mark.asyncio
async def test_update_task(mock_ctx, mock_step_tag):
    updated = {"id": 10, "title": "Buy milk", "done": True}
    patcher, mock_session = _mock_aiohttp(updated)
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_update_task(mock_ctx, task_id=10, done=True)
    call_args = mock_session.request.call_args
    assert call_args[0][0] == "POST"
    body = call_args[1]["json"]
    assert body["done"] is True


@pytest.mark.asyncio
async def test_delete_task(mock_ctx, mock_step_tag):
    patcher, mock_session = _mock_aiohttp({"message": "Successfully deleted."})
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_delete_task(mock_ctx, task_id=10)
    call_args = mock_session.request.call_args
    assert call_args[0][0] == "DELETE"


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_labels(mock_ctx, mock_step_tag):
    labels = [{"id": 1, "title": "Bug", "hex_color": "#ff0000"}]
    patcher, _ = _mock_aiohttp(labels)
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_list_labels(mock_ctx)
    assert "Bug" in result


@pytest.mark.asyncio
async def test_create_label(mock_ctx, mock_step_tag):
    created = {"id": 5, "title": "Feature"}
    patcher, mock_session = _mock_aiohttp(created)
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_create_label(mock_ctx, title="Feature")
    assert "Feature" in result
    assert mock_session.request.call_args[0][0] == "PUT"


@pytest.mark.asyncio
async def test_update_label(mock_ctx, mock_step_tag):
    updated = {"id": 1, "title": "Critical Bug"}
    patcher, _ = _mock_aiohttp(updated)
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_update_label(mock_ctx, label_id=1, title="Critical Bug")
    assert "Critical Bug" in result


@pytest.mark.asyncio
async def test_delete_label(mock_ctx, mock_step_tag):
    patcher, mock_session = _mock_aiohttp({"message": "Successfully deleted."})
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_delete_label(mock_ctx, label_id=1)
    assert mock_session.request.call_args[0][0] == "DELETE"


# ---------------------------------------------------------------------------
# Task labels & assignees
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_task_label(mock_ctx, mock_step_tag):
    resp = {"label_id": 1, "created": "2026-01-01T00:00:00Z"}
    patcher, mock_session = _mock_aiohttp(resp)
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_add_task_label(mock_ctx, task_id=10, label_id=1)
    assert mock_session.request.call_args[0][0] == "PUT"
    assert "/tasks/10/labels" in mock_session.request.call_args[0][1]


@pytest.mark.asyncio
async def test_remove_task_label(mock_ctx, mock_step_tag):
    patcher, mock_session = _mock_aiohttp({"message": "Successfully deleted."})
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_remove_task_label(mock_ctx, task_id=10, label_id=1)
    assert mock_session.request.call_args[0][0] == "DELETE"
    assert "/tasks/10/labels/1" in mock_session.request.call_args[0][1]


@pytest.mark.asyncio
async def test_assign_user_to_task(mock_ctx, mock_step_tag):
    resp = {"user_id": 42}
    patcher, mock_session = _mock_aiohttp(resp)
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_assign_user_to_task(mock_ctx, task_id=10, user_id=42)
    assert mock_session.request.call_args[0][0] == "PUT"
    assert "/tasks/10/assignees" in mock_session.request.call_args[0][1]


@pytest.mark.asyncio
async def test_remove_user_from_task(mock_ctx, mock_step_tag):
    patcher, mock_session = _mock_aiohttp({"message": "Successfully deleted."})
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_remove_user_from_task(mock_ctx, task_id=10, user_id=42)
    assert mock_session.request.call_args[0][0] == "DELETE"
    assert "/tasks/10/assignees/42" in mock_session.request.call_args[0][1]


# ---------------------------------------------------------------------------
# Task comments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_task_comments(mock_ctx, mock_step_tag):
    comments = [{"id": 1, "comment": "Looks good", "author": {"username": "alice"}}]
    patcher, _ = _mock_aiohttp(comments)
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_list_task_comments(mock_ctx, task_id=10)
    assert "Looks good" in result


@pytest.mark.asyncio
async def test_add_task_comment(mock_ctx, mock_step_tag):
    created = {"id": 5, "comment": "Nice work"}
    patcher, mock_session = _mock_aiohttp(created)
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_add_task_comment(mock_ctx, task_id=10, comment="Nice work")
    assert "Nice work" in result
    assert mock_session.request.call_args[0][0] == "PUT"


# ---------------------------------------------------------------------------
# Generic API tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vikunja_api_get(mock_ctx, mock_step_tag):
    teams = [{"id": 1, "name": "Dev team"}]
    patcher, mock_session = _mock_aiohttp(teams)
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_api(mock_ctx, method="GET", path="/teams")
    assert "Dev team" in result
    call_args = mock_session.request.call_args
    assert call_args[0][0] == "GET"
    assert call_args[0][1].endswith("/api/v1/teams")


@pytest.mark.asyncio
async def test_vikunja_api_put_with_body(mock_ctx, mock_step_tag):
    created = {"id": 2, "name": "QA team"}
    patcher, mock_session = _mock_aiohttp(created)
    body = json.dumps({"name": "QA team"})
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_api(
            mock_ctx, method="PUT", path="/teams", body_json=body
        )
    assert "QA team" in result
    call_args = mock_session.request.call_args
    assert call_args[0][0] == "PUT"
    assert call_args[1]["json"] == {"name": "QA team"}


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_error(mock_ctx, mock_step_tag):
    patcher, _ = _mock_aiohttp({"message": "not found"}, status=404)
    with patch.dict("os.environ", ENV, clear=True), patcher:
        result = await vikunja_list_projects(mock_ctx)
    assert "HTTP 404" in result


@pytest.mark.asyncio
async def test_network_error(mock_ctx, mock_step_tag):
    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(side_effect=Exception("Connection refused"))
    session_ctx.__aexit__ = AsyncMock(return_value=None)
    with (
        patch.dict("os.environ", ENV, clear=True),
        patch("aiohttp.ClientSession", return_value=session_ctx),
    ):
        result = await vikunja_list_projects(mock_ctx)
    assert "Connection refused" in result
