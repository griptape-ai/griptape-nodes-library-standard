from __future__ import annotations

import json
import logging
from enum import StrEnum
from typing import Any

import httpx
import trafilatura
from ddgs import DDGS
from exa_py import Exa
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from trafilatura.settings import use_config

DEFAULT_RESULTS_COUNT = 5
GOOGLE_API_KEY_SECRET = "GOOGLE_API_KEY"
GOOGLE_SEARCH_ID_SECRET = "GOOGLE_API_SEARCH_ID"
EXA_API_KEY_SECRET = "EXA_API_KEY"


class SearchEngine(StrEnum):
    DUCKDUCKGO = "DuckDuckGo"
    GOOGLE = "Google"
    EXA = "Exa"


def _secret(name: str) -> str:
    value = GriptapeNodes.SecretsManager().get_secret(name, should_error_on_not_found=False)
    if not value:
        msg = f"Secret '{name}' is not set."
        raise KeyError(msg)
    return value


def search_web(
    query: str, engine: str = SearchEngine.DUCKDUCKGO, *, results_count: int = DEFAULT_RESULTS_COUNT
) -> list[dict[str, Any]]:
    """Return `[{"title", "url", "description"}]` results for `query`."""
    match engine:
        case SearchEngine.DUCKDUCKGO:
            results = DDGS().text(query, region="en-us", max_results=results_count)
            return [{"title": r["title"], "url": r["href"], "description": r["body"]} for r in results]
        case SearchEngine.GOOGLE:
            # Key in a header, not the query string, so it stays out of URLs in error text.
            response = httpx.get(
                "https://www.googleapis.com/customsearch/v1",
                headers={"X-Goog-Api-Key": _secret(GOOGLE_API_KEY_SECRET)},
                params={
                    "cx": _secret(GOOGLE_SEARCH_ID_SECRET),
                    "q": query,
                    "start": 0,
                    "lr": "lang_en",
                    "num": results_count,
                    "gl": "us",
                },
                timeout=30,
            )
            if not response.is_success:
                # `raise_for_status` text includes the URL, and this error reaches the model.
                msg = f"Google search failed: HTTP {response.status_code}{_google_error_reason(response)}"
                raise RuntimeError(msg)
            return [
                {"title": r["title"], "url": r["link"], "description": r["snippet"]}
                for r in response.json().get("items", [])
            ]
        case SearchEngine.EXA:
            response = Exa(api_key=_secret(EXA_API_KEY_SECRET)).search_and_contents(
                query, num_results=results_count, text=True, highlights=True
            )
            return [
                {"title": r.title, "url": r.url, "description": " ".join(r.highlights or []), "text": r.text}
                for r in response.results
            ]
        case _:
            msg = f"Unknown search engine: {engine!r}"
            raise ValueError(msg)


def _google_error_reason(response: httpx.Response) -> str:
    try:
        reason = response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return ""
    return f": {reason}"


def format_search_results(results: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(r) for r in results)


def scrape_url(url: str) -> str:
    """Fetch `url` and return its main text content."""
    # Thread-safe extraction: trafilatura's signal-based timeout only works on the main thread.
    config = use_config()
    config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
    logging.getLogger("trafilatura").setLevel(logging.FATAL)
    page = trafilatura.fetch_url(url)
    if page is None:
        msg = f"Can't access URL: {url}"
        raise RuntimeError(msg)
    extracted = trafilatura.extract(page, include_links=True, output_format="json", config=config)
    if not extracted:
        msg = f"Can't extract page content: {url}"
        raise RuntimeError(msg)
    return json.loads(extracted).get("text") or ""
