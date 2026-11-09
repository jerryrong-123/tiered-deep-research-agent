"""Raw-search reuse is partitioned at the actual HTTP request boundary."""
from __future__ import annotations

import importlib
import json
import os
import unittest
from functools import partial
from unittest.mock import patch

import httpx

from frontier_agent.infra.config import get_config
from plugins.tools._search_run_cache import search_run_cache

search = importlib.import_module("plugins.tools.web_search")


class SearchReuseHTTPTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {
            "SERPER_API_KEY": "offline-account-one",
            "SERPER_BASE_URL": "https://serper.invalid",
        })
        environment.start()
        get_config(force_reload=True)
        self.addCleanup(lambda: get_config(force_reload=True))
        self.addCleanup(environment.stop)

    async def test_exact_query_locale_endpoint_and_account_define_reuse(self):
        calls = []

        def respond(request):
            calls.append(request)
            return httpx.Response(200, json={"organic": [{
                "title": "Public navigation", "link": "https://example.com/source",
                "snippet": "This fixture is search navigation, not a verified conclusion.",
            }]})

        client = httpx.AsyncClient
        with patch.object(httpx, "AsyncClient", partial(client, transport=httpx.MockTransport(respond))):
            with search_run_cache(enabled=True):
                await search.raw_web_search("Exact Query")
                await search.raw_web_search("Exact Query")
                self.assertEqual(len(calls), 1)
                await search.raw_web_search("exact query")
                await search.raw_web_search("Exact Query", gl="gb")
                await search.raw_web_search("Exact Query", hl="fr")
                await search.raw_web_search("Exact Query", tbs="qdr:y")
                await search.raw_web_search("Exact Query", num_results=20)
                os.environ["SERPER_BASE_URL"] = "https://other-serper.invalid"
                get_config(force_reload=True)
                await search.raw_web_search("Exact Query")
                os.environ["SERPER_API_KEY"] = "offline-account-two"
                get_config(force_reload=True)
                await search.raw_web_search("Exact Query")
            self.assertEqual(len(calls), 8)
            with search_run_cache(enabled=True):
                await search.raw_web_search("Exact Query")
            self.assertEqual(len(calls), 9)

    async def test_error_then_success_can_recover_and_success_is_copied(self):
        calls = []

        def respond(request):
            calls.append(json.loads(request.content))
            if len(calls) == 1:
                return httpx.Response(400, json={"message": "Query rejected"})
            return httpx.Response(200, json={"organic": [{
                "title": "Recovered source", "link": "https://example.com/recovered",
                "snippet": "Original public result.",
            }]})

        client = httpx.AsyncClient
        with patch.object(httpx, "AsyncClient", partial(client, transport=httpx.MockTransport(respond))):
            with search_run_cache(enabled=True):
                failed = await search.raw_web_search("recoverable query")
                self.assertIn("_search_error", failed)
                successful = await search.raw_web_search("recoverable query")
                successful["organic"][0]["snippet"] = "caller mutation"
                repeated = await search.raw_web_search("recoverable query")
        self.assertEqual(len(calls), 2)
        self.assertEqual(repeated["organic"][0]["snippet"], "Original public result.")

    async def test_success_with_empty_optional_error_is_reusable(self):
        calls = []

        def respond(request):
            calls.append(request)
            return httpx.Response(200, json={"error": None, "organic": [{
                "title": "Valid result", "link": "https://example.com/valid",
            }]})

        client = httpx.AsyncClient
        with patch.object(httpx, "AsyncClient", partial(client, transport=httpx.MockTransport(respond))):
            with search_run_cache(enabled=True):
                await search.raw_web_search("normal query")
                await search.raw_web_search("normal query")
        self.assertEqual(len(calls), 1)
