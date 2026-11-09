"""Run-scoped reuse of successful raw search responses, without network calls."""

from __future__ import annotations

import asyncio
import unittest

from plugins.tools._search_run_cache import cached_search_result, search_run_cache


class SearchRunCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_result_is_reused_as_independent_copies(self) -> None:
        response = {"organic": [{"title": "Original", "link": "https://example.com"}]}
        calls = 0

        async def search() -> dict:
            nonlocal calls
            calls += 1
            return response

        with search_run_cache(enabled=True):
            first = await cached_search_result(("serper", "exact query"), search)
            first["organic"][0]["title"] = "Caller mutation"
            response["organic"][0]["title"] = "Upstream mutation"
            second = await cached_search_result(("serper", "exact query"), search)
            second["organic"].clear()
            third = await cached_search_result(("serper", "exact query"), search)

        self.assertEqual(calls, 1)
        self.assertEqual(third, {
            "organic": [{"title": "Original", "link": "https://example.com"}],
        })

    async def test_failures_and_empty_responses_are_retried_by_the_caller(self) -> None:
        for response in (
            {},
            {"organic": []},
            {"organic": [{}]},
            {"answerBox": {"answer": " "}},
            {"searchParameters": {"q": "exact query"}},
            {"_search_error": {"reason": "http_error"}},
            {"_search_error": None, "organic": [{"title": "Partial"}]},
            {"error": "provider failure", "organic": [{"title": "Partial"}]},
        ):
            with self.subTest(response=response):
                calls = 0

                async def search(response: dict = response) -> dict:
                    nonlocal calls
                    calls += 1
                    return response

                with search_run_cache(enabled=True):
                    self.assertEqual(await cached_search_result("same", search), response)
                    self.assertEqual(await cached_search_result("same", search), response)

                self.assertEqual(calls, 2)

    async def test_inherited_child_cannot_repopulate_a_finished_run(self) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def search() -> dict:
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return {"organic": [{"title": "Late response"}]}

        async def child() -> None:
            await cached_search_result("same", search)
            await cached_search_result("same", search)

        with search_run_cache(enabled=True):
            task = asyncio.create_task(child())
            await entered.wait()
        release.set()
        await task

        self.assertEqual(calls, 2)

    async def test_only_explicit_true_enables_reuse(self) -> None:
        calls = 0

        async def search() -> dict:
            nonlocal calls
            calls += 1
            return {"organic": [{"title": "Native response"}]}

        await cached_search_result("same", search)
        await cached_search_result("same", search)
        with search_run_cache():
            await cached_search_result("same", search)
            await cached_search_result("same", search)
        for enabled in (False, None, "true", "false", 1):
            with search_run_cache(enabled=enabled):
                await cached_search_result("same", search)
                await cached_search_result("same", search)

        self.assertEqual(calls, 14)

    async def test_direct_answer_and_other_raw_content_can_be_reused(self) -> None:
        for response in (
            {"answerBox": {"answer": "Provider answer"}},
            {"knowledgeGraph": {"description": "Provider description"}},
            {"peopleAlsoAsk": [{"question": "Related question"}]},
            {"relatedSearches": [{"query": "Related query"}]},
        ):
            with self.subTest(response=response):
                calls = 0

                async def search(response: dict = response) -> dict:
                    nonlocal calls
                    calls += 1
                    return response

                with search_run_cache(enabled=True):
                    self.assertEqual(await cached_search_result("same", search), response)
                    self.assertEqual(await cached_search_result("same", search), response)

                self.assertEqual(calls, 1)

    async def test_exceptions_and_cancellation_propagate_without_being_cached(self) -> None:
        for failure in (RuntimeError("transport failure"), asyncio.CancelledError()):
            with self.subTest(failure=type(failure).__name__):
                calls = 0

                async def search(failure: BaseException = failure) -> dict:
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        raise failure
                    return {"organic": [{"title": "Recovered"}]}

                with search_run_cache(enabled=True):
                    with self.assertRaises(type(failure)):
                        await cached_search_result("same", search)
                    response = await cached_search_result("same", search)
                    again = await cached_search_result("same", search)

                self.assertEqual(calls, 2)
                self.assertEqual(response, {"organic": [{"title": "Recovered"}]})
                self.assertEqual(again, response)

    async def test_structured_failure_does_not_hide_a_later_success(self) -> None:
        responses = iter((
            {"_search_error": {"reason": "timeout"}},
            {"organic": [{"title": "Recovered provider response"}]},
        ))

        async def search() -> dict:
            return next(responses)

        with search_run_cache(enabled=True):
            first = await cached_search_result("same", search)
            second = await cached_search_result("same", search)
            third = await cached_search_result("same", search)

        self.assertEqual(first, {"_search_error": {"reason": "timeout"}})
        self.assertEqual(second, {"organic": [{"title": "Recovered provider response"}]})
        self.assertEqual(third, second)

    async def test_only_the_exact_request_key_is_reused(self) -> None:
        calls = 0

        async def search() -> dict:
            nonlocal calls
            calls += 1
            return {"organic": [{"title": f"Response {calls}"}]}

        keys = (
            ("serper", "query", 10, "us", "en", ""),
            ("serper", "Query", 10, "us", "en", ""),
            ("serper", "query ", 10, "us", "en", ""),
            ("serper", "query", 20, "us", "en", ""),
            ("serper", "query", 10, "cn", "en", ""),
            ("serper", "query", 10, "us", "zh", ""),
            ("serper", "query", 10, "us", "en", "qdr:d"),
            ("parallel", "query", 10, "us", "en", ""),
        )
        with search_run_cache(enabled=True):
            responses = [await cached_search_result(key, search) for key in keys]
            repeated = [await cached_search_result(key, search) for key in keys]

        self.assertEqual(calls, 8)
        self.assertEqual(responses, repeated)
        self.assertEqual(responses[0], {"organic": [{"title": "Response 1"}]})
        self.assertEqual(responses[-1], {"organic": [{"title": "Response 8"}]})

    async def test_nested_run_isolated_and_exception_restores_parent(self) -> None:
        calls = 0

        async def search() -> dict:
            nonlocal calls
            calls += 1
            return {"organic": [{"title": f"Run {calls}"}]}

        with search_run_cache(enabled=True):
            outer = await cached_search_result("same", search)
            with self.assertRaisesRegex(RuntimeError, "nested failure"):
                with search_run_cache(enabled=True):
                    nested = await cached_search_result("same", search)
                    raise RuntimeError("nested failure")
            restored = await cached_search_result("same", search)
            with search_run_cache():
                disabled = await cached_search_result("same", search)
            after_disabled = await cached_search_result("same", search)
        with search_run_cache(enabled=True):
            next_run = await cached_search_result("same", search)
        outside = await cached_search_result("same", search)

        self.assertEqual(outer, {"organic": [{"title": "Run 1"}]})
        self.assertEqual(nested, {"organic": [{"title": "Run 2"}]})
        self.assertEqual(disabled, {"organic": [{"title": "Run 3"}]})
        self.assertEqual(next_run, {"organic": [{"title": "Run 4"}]})
        self.assertEqual(outside, {"organic": [{"title": "Run 5"}]})
        self.assertEqual(restored, outer)
        self.assertEqual(after_disabled, outer)

    async def test_root_cancellation_clears_inherited_results(self) -> None:
        ready = asyncio.Event()
        release = asyncio.Event()
        children = []
        calls = 0

        async def search() -> dict:
            nonlocal calls
            calls += 1
            return {"organic": [{"title": f"Response {calls}"}]}

        async def child() -> dict:
            await release.wait()
            return await cached_search_result("same", search)

        async def root() -> None:
            with search_run_cache(enabled=True):
                await cached_search_result("same", search)
                children.append(asyncio.create_task(child()))
                ready.set()
                await asyncio.Event().wait()

        root_task = asyncio.create_task(root())
        await ready.wait()
        root_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await root_task
        release.set()
        response = await children[0]

        self.assertEqual(calls, 2)
        self.assertEqual(response, {"organic": [{"title": "Response 2"}]})

    async def test_children_inherit_shared_results_but_parallel_roots_isolate(self) -> None:
        async def root(title: str) -> list[dict]:
            calls = 0

            async def search() -> dict:
                nonlocal calls
                calls += 1
                return {"organic": [{"title": title}]}

            with search_run_cache(enabled=True):
                first = await cached_search_result("same", search)
                children = await asyncio.gather(*(
                    cached_search_result("same", search) for _ in range(4)
                ))
            self.assertEqual(calls, 1)
            return [first, *children]

        left, right = await asyncio.gather(root("Left topic"), root("Right topic"))

        self.assertTrue(all(item == {"organic": [{"title": "Left topic"}]} for item in left))
        self.assertTrue(all(item == {"organic": [{"title": "Right topic"}]} for item in right))

    async def test_existing_single_flight_still_owns_concurrent_requests(self) -> None:
        from frontier_agent.infra.usage_meter import (
            ExternalAPIMeter,
            bind_usage_meter,
            record_api_request,
            reset_usage_meter,
        )
        from plugins.tools._single_flight import SingleFlightCoalescer

        coalescer = SingleFlightCoalescer(meter_provider="serper")
        meter = ExternalAPIMeter()
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def search() -> dict:
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            record_api_request("serper")
            return {"organic": [{"title": "One upstream response"}]}

        async def native_search() -> dict:
            return await coalescer.run("same", search)

        def on_hit() -> None:
            record_api_request("serper", requests=0, cache_hits=1)

        token = bind_usage_meter(meter)
        try:
            with search_run_cache(enabled=True):
                leader = asyncio.create_task(cached_search_result(
                    "same", native_search, on_cache_hit=on_hit,
                ))
                await entered.wait()
                follower = asyncio.create_task(cached_search_result(
                    "same", native_search, on_cache_hit=on_hit,
                ))
                await asyncio.sleep(0)
                release.set()
                first, second = await asyncio.gather(leader, follower)
                second["organic"][0]["title"] = "Follower mutation"
                cached = await cached_search_result(
                    "same", native_search, on_cache_hit=on_hit,
                )
        finally:
            reset_usage_meter(token)

        self.assertEqual(calls, 1)
        self.assertEqual(first, {"organic": [{"title": "One upstream response"}]})
        self.assertEqual(cached, first)
        usage = meter.snapshot()["external_apis"]["serper"]
        self.assertEqual(usage["requests"], 1)
        self.assertEqual(usage["cache_hits"], 2)

    async def test_existing_single_flight_follower_can_retry_a_failed_leader(self) -> None:
        from plugins.tools._single_flight import SingleFlightCoalescer

        coalescer = SingleFlightCoalescer()
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def search() -> dict:
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await release.wait()
                raise RuntimeError("leader transport failed")
            return {"organic": [{"title": "Follower recovered"}]}

        async def native_search() -> dict:
            return await coalescer.run("same", search)

        with search_run_cache(enabled=True):
            leader = asyncio.create_task(cached_search_result("same", native_search))
            await entered.wait()
            follower = asyncio.create_task(cached_search_result("same", native_search))
            await asyncio.sleep(0)
            release.set()
            failed, recovered = await asyncio.gather(leader, follower, return_exceptions=True)
            cached = await cached_search_result("same", native_search)

        self.assertIsInstance(failed, RuntimeError)
        self.assertEqual(recovered, {"organic": [{"title": "Follower recovered"}]})
        self.assertEqual(cached, recovered)
        self.assertEqual(calls, 2)


if __name__ == "__main__":
    unittest.main()
