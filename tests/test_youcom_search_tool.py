import io
import json
import urllib.error
from email.message import Message
from unittest import mock

import pytest

from hybrid_agent.tools import youcom_search_tool as yst
from hybrid_agent.tools.executor import build_tool_executor
from hybrid_agent.tools.youcom_search_tool import YouComSearchTool

PAYLOAD = {
    "results": {
        "web": [
            {"url": "https://a.example", "title": "A", "description": "alpha text", "snippets": ["x"]},
            {"url": "https://b.example", "title": "B", "description": "", "snippets": ["beta", "two"]},
            {"url": "https://c.example", "title": "C", "description": "", "snippets": []},
        ]
    }
}


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ok(payload=PAYLOAD):
    return _Resp(json.dumps(payload).encode())


def _http_error(code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError(yst.FREE_URL, code, "err", headers, io.BytesIO(b"{}"))


@pytest.fixture(autouse=True)
def _no_disk_cache(monkeypatch):
    monkeypatch.delenv("HYBRID_SEARCH_CACHE_DIR", raising=False)
    monkeypatch.delenv("YDC_API_KEY", raising=False)


def test_keyless_get_formats_pages():
    tool = YouComSearchTool(max_results=5, result_length=5)
    with mock.patch.object(yst.urllib.request, "urlopen", return_value=_ok()) as urlopen:
        out = tool.execute("hybrid agents")

    assert out == "Page 1: alpha\nPage 2: beta "
    req = urlopen.call_args.args[0]
    assert req.get_method() == "GET"
    assert req.full_url.startswith(yst.FREE_URL + "?")
    assert "query=hybrid+agents" in req.full_url and "count=5" in req.full_url
    assert req.get_header("User-agent") == yst.USER_AGENT
    assert req.get_header("X-api-key") is None
    assert tool.name == "youcom_search" and tool.trigger_tag == "search"


def test_keyed_post(monkeypatch):
    monkeypatch.setenv("YDC_API_KEY", "k-1")
    tool = YouComSearchTool()
    with mock.patch.object(yst.urllib.request, "urlopen", return_value=_ok()) as urlopen:
        tool.execute("q")

    req = urlopen.call_args.args[0]
    assert req.get_method() == "POST"
    assert req.full_url == yst.KEYED_URL
    assert req.get_header("X-api-key") == "k-1"
    assert json.loads(req.data) == {"query": "q", "count": 10}


def test_in_memory_cache():
    tool = YouComSearchTool()
    with mock.patch.object(yst.urllib.request, "urlopen", return_value=_ok()) as urlopen:
        tool.execute("q")
        tool.execute("q")
    assert urlopen.call_count == 1


@pytest.mark.parametrize("code", [401, 402, 403])
def test_auth_and_quota_errors_do_not_retry(code):
    tool = YouComSearchTool(retry_delay=0)
    with mock.patch.object(yst.urllib.request, "urlopen", side_effect=_http_error(code)) as urlopen, \
            mock.patch.object(yst.time, "sleep") as sleep:
        out = tool.execute("q")
    assert out.startswith("Search failed after")
    assert urlopen.call_count == 1
    sleep.assert_not_called()


def test_429_honors_retry_after():
    tool = YouComSearchTool(retry_delay=0)
    with mock.patch.object(yst.urllib.request, "urlopen", side_effect=[_http_error(429, 7), _ok()]), \
            mock.patch.object(yst.time, "sleep") as sleep:
        out = tool.execute("q")
    assert out.startswith("Page 1:")
    sleep.assert_called_once_with(7.0)


def test_no_results():
    tool = YouComSearchTool()
    with mock.patch.object(yst.urllib.request, "urlopen", return_value=_ok({"results": {"web": []}})):
        assert tool.execute("q") == "No search results found."


def test_executor_opt_in(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("HYBRID_SEARCH_PROVIDER", "youcom")
    executor = build_tool_executor()
    assert executor.get_tool("search").name == "youcom_search"


def test_executor_default_still_requires_tavily(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("HYBRID_SEARCH_PROVIDER", raising=False)
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        build_tool_executor()
