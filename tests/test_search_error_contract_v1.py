"""Offline HTTP replays through the real raw and formatted search interfaces."""

from __future__ import annotations

import importlib
import json
import os
import unittest
from functools import partial
from unittest.mock import AsyncMock, patch

import httpx

from frontier_agent.infra.config import get_config

search = importlib.import_module("plugins.tools.web_search")


class SearchErrorContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(os.environ, {
            "SERPER_API_KEY": "offline-test-key",
            "SERPER_BASE_URL": "https://serper.invalid",
            "WEB_SEARCH_PROVIDER": "serper",
            "TOOL_CALL_COALESCE": "0",
        })
        self.environment.start()
        get_config(force_reload=True)
        self.addCleanup(lambda: get_config(force_reload=True))
        self.addCleanup(self.environment.stop)

    async def test_free_plan_size_refusal_retries_once_at_the_allowed_size(self) -> None:
        body = b'{"message":"Query pattern not allowed for free accounts.","statusCode":400}'
        sizes = []

        def respond(request):
            size = json.loads(request.content)["num"]
            sizes.append(size)
            if size > 10:
                return httpx.Response(400, content=body)
            return httpx.Response(200, json={"organic": [{"title": "t", "link": "https://x.invalid", "snippet": "s"}]})

        real_client = httpx.AsyncClient
        with patch.object(httpx, "AsyncClient", partial(real_client, transport=httpx.MockTransport(respond))):
            raw = await search.raw_web_search('"quoted phrase" site:x.invalid', num_results=20)

        self.assertEqual(sizes, [20, 10])
        self.assertEqual(len(raw["organic"]), 1)

    async def test_other_serper_400_is_not_retried_at_a_smaller_size(self) -> None:
        sizes = []

        def respond(request):
            sizes.append(json.loads(request.content)["num"])
            return httpx.Response(400, content=b'{"message":"Bad request.","statusCode":400}')

        real_client = httpx.AsyncClient
        with patch.object(httpx, "AsyncClient", partial(real_client, transport=httpx.MockTransport(respond))):
            raw = await search.raw_web_search("another query", num_results=20)

        self.assertEqual(sizes, [20])
        self.assertEqual(raw, {"_search_error": {"reason": "http_error", "http_status": 400}})

    async def test_replayed_serper_400_is_a_safe_error_not_an_empty_search(self) -> None:
        # Preserve the actual public provider failure text at the HTTP seam.
        body = b'{"message":"Query pattern not allowed for free accounts.","statusCode":400}'
        transport = httpx.MockTransport(lambda request: httpx.Response(400, content=body))
        real_client = httpx.AsyncClient
        with patch.object(httpx, "AsyncClient", partial(real_client, transport=transport)):
            raw = await search.raw_web_search("restricted query")
            output = await search.web_search.ainvoke({"q": "restricted query"})

        self.assertEqual(raw, {"_search_error": {"reason": "http_error", "http_status": 400}})
        self.assertIn("Error:", output)
        self.assertIn("http_error", output)
        self.assertIn("HTTP 400", output)
        self.assertNotIn("No results", output)
        self.assertNotIn("Query pattern", output)
        self.assertNotIn("offline-test-key", output)

    async def test_exhausted_retries_remain_safe_errors_with_original_limits(self) -> None:
        scenarios = [
            (429, "rate_limited", [1, 2, 4]),
            (503, "server_error", [1, 2, 4]),
            (None, "timeout", [1, 1]),
        ]
        real_client = httpx.AsyncClient
        for status, reason, expected_waits in scenarios:
            with self.subTest(reason=reason):
                requests = []

                def respond(request):
                    requests.append(request)
                    if status is None:
                        raise httpx.ReadTimeout("PRIVATE timeout body offline-test-key", request=request)
                    return httpx.Response(status, text="PRIVATE service body offline-test-key")

                transport = httpx.MockTransport(respond)
                sleep = AsyncMock()
                with (
                    patch.object(httpx, "AsyncClient", partial(real_client, transport=transport)),
                    patch("asyncio.sleep", sleep),
                    self.assertLogs(search.logger, level="WARNING") as logs,
                ):
                    raw = await search.raw_web_search(f"retry {reason}")
                    output = await search.web_search.ainvoke({"q": f"retry {reason}"})

                error = {"reason": reason}
                if status is not None:
                    error["http_status"] = status
                self.assertEqual(raw, {"_search_error": error})
                self.assertIn(reason, output)
                self.assertNotIn("No results", output)
                self.assertEqual(len(requests), 6)
                self.assertEqual([call.args[0] for call in sleep.await_args_list], expected_waits * 2)
                safe_text = output + repr(raw) + "\n".join(logs.output)
                self.assertNotIn("PRIVATE", safe_text)
                self.assertNotIn("offline-test-key", safe_text)

    async def test_batch_keeps_success_and_separates_failed_and_empty_queries(self) -> None:
        def respond(request):
            query = json.loads(request.content)["q"]
            if query == "blocked":
                return httpx.Response(400, json={"message": "PRIVATE body offline-test-key"})
            if query == "empty":
                return httpx.Response(200, json={"organic": []})
            if query == "empty_dict":
                return httpx.Response(200, json={})
            return httpx.Response(200, json={"organic": [{
                "title": "Retained successful result", "link": "https://example.com/retained",
                "snippet": "Public source text.",
            }]})

        real_client = httpx.AsyncClient
        transport = httpx.MockTransport(respond)
        with patch.object(httpx, "AsyncClient", partial(real_client, transport=transport)):
            output = await search.web_search.ainvoke({
                "q": ["blocked", "empty", "empty_dict", "success"],
            })

        self.assertIn("## Query: blocked\n\nError: search failed (http_error; HTTP 400).", output)
        self.assertIn("## Query: empty\n\nNo results found.", output)
        self.assertIn("## Query: empty_dict\n\nNo results found.", output)
        self.assertIn("Retained successful result", output)
        self.assertIn("https://example.com/retained", output)
        self.assertNotIn("error or empty", output)
        self.assertNotIn("PRIVATE", output)
        self.assertNotIn("offline-test-key", output)

    async def test_successful_and_empty_responses_keep_the_raw_dictionary_shape(self) -> None:
        responses = [
            {}, {"organic": []},
            {"organic": [{"title": "Found", "link": "https://example.com/found", "snippet": "Snippet"}],
             "answerBox": {"title": "Direct", "answer": "An answer"},
             "knowledgeGraph": {"title": "Entity", "description": "Description"}},
        ]
        real_client = httpx.AsyncClient
        for index, expected in enumerate(responses):
            with self.subTest(response=index):
                transport = httpx.MockTransport(lambda request: httpx.Response(200, json=expected))
                with patch.object(httpx, "AsyncClient", partial(real_client, transport=transport)):
                    raw = await search.raw_web_search(f"normal {index}")
                    output = await search.web_search.ainvoke({"q": f"normal {index}"})
                self.assertEqual(raw, expected)
                self.assertNotIn("_search_error", raw)
                self.assertNotIn("Error:", output)
                self.assertIn("Found" if index == 2 else "No results found", output)

    async def test_transport_and_invalid_responses_are_safe_local_failures(self) -> None:
        real_client = httpx.AsyncClient
        for failure, expected_reason in [
            ("transport", "transport_error"), ("unexpected", "unexpected_error"),
            ("invalid_json", "invalid_response"), ("invalid_shape", "invalid_response"),
        ]:
            with self.subTest(failure=failure):
                requests = []

                def respond(request):
                    requests.append(request)
                    if failure == "transport":
                        raise httpx.ConnectError("PRIVATE service body offline-test-key", request=request)
                    if failure == "unexpected":
                        raise RuntimeError("PRIVATE service body offline-test-key")
                    if failure == "invalid_json":
                        return httpx.Response(200, text="PRIVATE service body offline-test-key")
                    return httpx.Response(200, json=["PRIVATE service body offline-test-key"])

                transport = httpx.MockTransport(respond)
                with (
                    patch.object(httpx, "AsyncClient", partial(real_client, transport=transport)),
                    self.assertLogs(search.logger, level="WARNING") as logs,
                ):
                    raw = await search.raw_web_search(failure)
                    output = await search.web_search.ainvoke({"q": failure})
                self.assertEqual(raw, {"_search_error": {"reason": expected_reason}})
                self.assertIn(expected_reason, output)
                self.assertEqual(len(requests), 2)
                safe_text = output + repr(raw) + "\n".join(logs.output)
                self.assertNotIn("PRIVATE", safe_text)
                self.assertNotIn("offline-test-key", safe_text)

    async def test_missing_key_is_configuration_failure_without_http_calls(self) -> None:
        calls = []
        transport = httpx.MockTransport(lambda request: calls.append(request))
        real_client = httpx.AsyncClient
        with (
            patch.dict(os.environ, {"SERPER_API_KEY": ""}),
            patch.object(httpx, "AsyncClient", partial(real_client, transport=transport)),
        ):
            get_config(force_reload=True)
            raw = await search.raw_web_search("missing key")
            output = await search.web_search.ainvoke({"q": "missing key"})
        self.assertEqual(raw, {"_search_error": {"reason": "configuration_error"}})
        self.assertIn("configuration_error", output)
        self.assertEqual(calls, [])

    async def test_provider_error_payload_on_http_200_is_a_safe_failure(self) -> None:
        responses = [
            {"error": "PRIVATE provider rejected request offline-test-key"},
            {"error": {"message": "PRIVATE provider rejected request offline-test-key"}},
            {"statusCode": 400, "message": "PRIVATE provider rejected request offline-test-key"},
        ]
        real_client = httpx.AsyncClient
        for index, payload in enumerate(responses):
            with self.subTest(response=index):
                requests = []

                def respond(request):
                    requests.append(request)
                    return httpx.Response(200, json=payload)

                transport = httpx.MockTransport(respond)
                with patch.object(httpx, "AsyncClient", partial(real_client, transport=transport)):
                    raw = await search.raw_web_search(f"provider failure {index}")
                    output = await search.web_search.ainvoke({"q": f"provider failure {index}"})
                self.assertEqual(raw, {
                    "_search_error": {"reason": "provider_error", "http_status": 200},
                })
                self.assertIn("provider_error", output)
                self.assertNotIn("No results", output)
                self.assertNotIn("PRIVATE", output + repr(raw))
                self.assertNotIn("offline-test-key", output + repr(raw))
                self.assertEqual(len(requests), 2)

    async def test_empty_optional_error_on_http_200_keeps_successful_results(self) -> None:
        real_client = httpx.AsyncClient
        for index, empty_error in enumerate([None, "", {}, [], False]):
            with self.subTest(response=index):
                payload = {"error": empty_error, "organic": [{
                    "title": "Preserved success", "link": "https://example.com/success",
                    "snippet": "Public search text.",
                }]}
                transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
                with patch.object(httpx, "AsyncClient", partial(real_client, transport=transport)):
                    raw = await search.raw_web_search(f"optional empty error {index}")
                    output = await search.web_search.ainvoke({"q": f"optional empty error {index}"})
                self.assertEqual(raw, payload)
                self.assertIn("Preserved success", output)
                self.assertNotIn("Error:", output)


if __name__ == "__main__":
    unittest.main()
