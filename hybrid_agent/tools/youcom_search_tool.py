"""Live search tool backed by You.com: same "Page i: ..." observation format as
TavilySearchTool, so it is a drop-in swap for inference, evals and HybridRL rollouts.

Without a key it uses the free tier (GET api.you.com/v1/agents/search, 100 searches/day
per IP), which is enough for demos and small evals. Set YDC_API_KEY for RL-scale volume
(POST ydc-index.io/v1/search). Standard library only.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict

from .base import BaseTool
from . import search_cache as _search_cache

logger = logging.getLogger(__name__)

FREE_URL = "https://api.you.com/v1/agents/search"
KEYED_URL = "https://ydc-index.io/v1/search"
USER_AGENT = "youdotcom-integration/togethercomputer-HybridAgent"
# Cache namespace; keeps You.com responses apart from Tavily's "basic" entries.
_CACHE_DEPTH = "youcom"
_MAX_RETRY_AFTER = 60.0


class YouComSearchTool(BaseTool):

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 10,
        result_length: int = 1000,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 60.0,
    ):
        self._api_key = (api_key if api_key is not None else os.environ.get("YDC_API_KEY", "")).strip()
        self._max_results = max_results
        self._result_length = result_length
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._timeout = timeout
        self._cache: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "youcom_search"

    @property
    def trigger_tag(self) -> str:
        return "search"

    def execute(self, query: str, **kwargs) -> str:
        if not query or not query.strip():
            return "Empty search query."

        query = query.strip()
        if query in self._cache:
            return self._cache[query]

        _cached = _search_cache.get(query, _CACHE_DEPTH, self._max_results)
        if _cached is not None:
            formatted = self._format_results(_cached)
            if formatted:
                self._cache[query] = formatted
                return formatted
            return "No search results found."

        for attempt in range(self._max_retries):
            try:
                response = self._request(query)
                formatted = self._format_results(response)
                if formatted:
                    _search_cache.put(query, _CACHE_DEPTH, self._max_results, response)
                    self._cache[query] = formatted
                    return formatted
                return "No search results found."
            except urllib.error.HTTPError as e:
                delay = self._retry_delay
                if e.code in (401, 403):
                    logger.error(f"You.com search rejected the request ({e.code}); check YDC_API_KEY.")
                    break
                if e.code == 402:
                    logger.error("You.com search quota used up (402). Set YDC_API_KEY or add credits.")
                    break
                if e.code == 429:
                    delay = _retry_after(e, default=self._retry_delay)
                logger.warning(f"You.com search attempt {attempt + 1}/{self._max_retries} failed: HTTP {e.code}")
                if attempt < self._max_retries - 1:
                    time.sleep(delay)
            except Exception as e:
                logger.warning(f"You.com search attempt {attempt + 1}/{self._max_retries} failed: {e}")
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay)

        return f"Search failed after {self._max_retries} attempts for query: {query}"

    def _request(self, query: str) -> dict:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        count = max(1, min(self._max_results, 20))
        if self._api_key:
            headers["X-API-Key"] = self._api_key
            headers["Content-Type"] = "application/json"
            body = json.dumps({"query": query, "count": count}).encode("utf-8")
            req = urllib.request.Request(KEYED_URL, data=body, headers=headers, method="POST")
        else:
            params = urllib.parse.urlencode({"query": query, "count": count})
            req = urllib.request.Request(f"{FREE_URL}?{params}", headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _format_results(self, response: dict) -> str | None:
        web = (response.get("results") or {}).get("web") or []
        if not web:
            return None

        formatted = []
        for item in web[: self._max_results]:
            snippet = (item.get("description") or "").strip()
            if not snippet:
                snippet = " ".join(item.get("snippets") or []).strip()
            if snippet:
                formatted.append(f"Page {len(formatted) + 1}: {snippet[: self._result_length]}")

        return "\n".join(formatted) if formatted else None


def _retry_after(error: urllib.error.HTTPError, default: float) -> float:
    try:
        value = float(error.headers.get("Retry-After", ""))
    except (TypeError, ValueError, AttributeError):
        return default
    return min(max(value, 0.0), _MAX_RETRY_AFTER)
