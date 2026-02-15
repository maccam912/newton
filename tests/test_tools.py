import pytest
import aiohttp
from unittest.mock import AsyncMock, patch, MagicMock
from newton.tools.searxng import web_search, register
import sys
from newton.config import Config, ToolsConfig

@pytest.fixture
def mock_step_tag():
    """Patch _step_tag in newton.agent."""
    # We try to import it first to ensure we patch the real one if loaded
    try:
        import newton.agent
    except ImportError:
        # If not importable, we mock the module
        mock_mod = MagicMock()
        mock_mod._step_tag = lambda deps: f" [step {deps.step+1}/{deps.max_steps}]"
        with patch.dict(sys.modules, {"newton.agent": mock_mod}):
             yield
        return

    # If importable, patch the function
    with patch("newton.agent._step_tag", side_effect=lambda deps: f" [step {deps.step+1}/{deps.max_steps}]"):
        yield

class MockDeps:
    def __init__(self):
        self.cfg = Config(tools=ToolsConfig(searxng={"base_url": "http://test", "max_results": 3}))
        self.step = 1
        self.max_steps = 10
        self.bus = MagicMock()
        self.memory = MagicMock()
        self.current_message = ""
        self.turn_ended = False
        self.event_metadata = {}

@pytest.fixture
def mock_ctx():
    """Create a mock RunContext with necessary deps."""
    deps = MockDeps()
    ctx = MagicMock()
    ctx.deps = deps
    return ctx

@pytest.mark.asyncio
async def test_searxng_search_success(mock_ctx, mock_step_tag):
    """Test successful search with results."""
    mock_resp = AsyncMock()
    mock_resp.json.return_value = {
        "results": [
            {"title": "Result 1", "url": "http://1.com", "content": "Desc 1"},
            {"title": "Result 2", "url": "http://2.com", "content": "Desc 2"},
        ]
    }
    mock_resp.raise_for_status = MagicMock()

    mock_session = AsyncMock()
    mock_session.get = MagicMock()
    
    get_ctx = MagicMock()
    get_ctx.__aenter__.return_value = mock_resp
    get_ctx.__aexit__.return_value = None
    mock_session.get.return_value = get_ctx

    session_ctx = MagicMock()
    session_ctx.__aenter__.return_value = mock_session
    session_ctx.__aexit__.return_value = None

    with patch("aiohttp.ClientSession", return_value=session_ctx):
        result = await web_search(mock_ctx, "query", max_results=5)

    assert "Result 1" in result
    assert "http://1.com" in result
    assert "Desc 1" in result
    assert "[step 2/10]" in result

@pytest.mark.asyncio
async def test_searxng_search_empty(mock_ctx, mock_step_tag):
    """Test search with no results."""
    mock_resp = AsyncMock()
    mock_resp.json.return_value = {"results": []}
    mock_resp.raise_for_status = MagicMock()
    
    mock_session = AsyncMock()
    mock_session.get = MagicMock()
    
    get_ctx = MagicMock()
    get_ctx.__aenter__.return_value = mock_resp
    get_ctx.__aexit__.return_value = None
    mock_session.get.return_value = get_ctx
    
    session_ctx = MagicMock()
    session_ctx.__aenter__.return_value = mock_session
    session_ctx.__aexit__.return_value = None

    with patch("aiohttp.ClientSession", return_value=session_ctx):
        result = await web_search(mock_ctx, "query")

    assert "No results found" in result

@pytest.mark.asyncio
async def test_searxng_search_error(mock_ctx, mock_step_tag):
    """Test search failure."""
    
    session_ctx = MagicMock()
    session_ctx.__aenter__.side_effect = Exception("Boom")
    
    with patch("aiohttp.ClientSession", return_value=session_ctx):
        result = await web_search(mock_ctx, "query")
    
    assert "Search failed: Boom" in result

def test_register():
    """Verify tool registration."""
    agent = MagicMock()
    register(agent)
    agent.tool.assert_called_once_with(web_search)
