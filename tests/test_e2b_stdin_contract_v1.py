"""Offline stdin compatibility at the installed E2B SDK boundary."""

from __future__ import annotations

import asyncio
import importlib
import json
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, create_autospec

from httpx import ReadError

try:
    from e2b.exceptions import TimeoutException
    from e2b.sandbox.commands.command_handle import CommandExitException, CommandResult
    from e2b.sandbox_sync.commands.command import Commands
    from e2b.sandbox_sync.commands.command_handle import CommandHandle
except ImportError:
    SDK_AVAILABLE = False
else:
    SDK_AVAILABLE = True

from plugins.tools._sandbox import arun_sandbox_cmd, clear_task_sandbox, set_task_sandbox

download_module = importlib.import_module("plugins.tools.download_file")


@unittest.skipUnless(SDK_AVAILABLE, "optional E2B SDK is not installed")
class E2BStdinContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_download_file_pipes_runner_through_actual_sdk_stdin_contract(self) -> None:
        commands = create_autospec(Commands, instance=True, spec_set=True)
        handle = create_autospec(CommandHandle, instance=True, spec_set=True)
        commands.run.return_value = handle
        expected = {"status": "ok", "path": "/workspace/downloads/sample.txt"}
        handle.wait.return_value = CommandResult(
            stdout=json.dumps(expected), stderr="", exit_code=0, error=None,
        )
        lifecycle = Mock()
        lifecycle.attach_mock(commands.run, "run")
        lifecycle.attach_mock(handle.send_stdin, "send_stdin")
        lifecycle.attach_mock(handle.close_stdin, "close_stdin")
        lifecycle.attach_mock(handle.wait, "wait")
        lifecycle.attach_mock(handle.disconnect, "disconnect")
        token = set_task_sandbox(SimpleNamespace(commands=commands))
        try:
            output = await download_module.download_file.ainvoke({
                "url": "https://example.com/sample.txt", "path": "sample.txt",
            })
        finally:
            clear_task_sandbox(token)

        self.assertEqual(json.loads(output), expected)
        command = commands.run.call_args.args[0]
        source = Path(download_module.__file__).with_name("_download_runner.py").read_text()
        self.assertEqual(commands.run.call_args.kwargs, {
            "timeout": 660, "background": True, "stdin": True,
        })
        self.assertNotIn(source, command)
        self.assertIn("python3 - ", command)
        self.assertEqual(lifecycle.mock_calls, [
            call.run(command, timeout=660, background=True, stdin=True),
            call.send_stdin(source), call.close_stdin(), call.wait(), call.disconnect(),
        ])
        handle.kill.assert_not_called()

    async def test_empty_unicode_and_large_stdin_keep_content_and_eof(self) -> None:
        for source in ["", "print('你好🙂')\n", "# 大输入🙂\n" * 25000]:
            with self.subTest(characters=len(source)):
                commands = create_autospec(Commands, instance=True, spec_set=True)
                handle = create_autospec(CommandHandle, instance=True, spec_set=True)
                commands.run.return_value = handle
                expected = CommandResult(stdout="done", stderr="", exit_code=0, error=None)
                handle.wait.return_value = expected
                result = await arun_sandbox_cmd(
                    SimpleNamespace(commands=commands), "python3 -", timeout=37, input=source,
                )
                self.assertIs(result, expected)
                commands.run.assert_called_once_with(
                    "python3 -", timeout=37, background=True, stdin=True,
                )
                handle.send_stdin.assert_called_once_with(source)
                handle.close_stdin.assert_called_once_with()
                handle.wait.assert_called_once_with()
                handle.disconnect.assert_called_once_with()

    async def test_no_stdin_keeps_original_sdk_run_call(self) -> None:
        commands = create_autospec(Commands, instance=True, spec_set=True)
        expected = CommandResult(stdout="ok", stderr="", exit_code=0, error=None)
        commands.run.return_value = expected
        result = await arun_sandbox_cmd(
            SimpleNamespace(commands=commands), "echo ok", timeout=23,
            allow_net=True, env_allow=("PUBLIC_",),
        )
        self.assertIs(result, expected)
        commands.run.assert_called_once_with("echo ok", timeout=23)

    async def test_local_input_parameter_keeps_original_direct_call(self) -> None:
        observed = []
        expected = SimpleNamespace(stdout="local", stderr="", exit_code=0)

        def local_run(command, timeout=60, input=None, allow_net=False, env_allow=()):
            observed.append((command, timeout, input, allow_net, env_allow))
            return expected

        result = await arun_sandbox_cmd(
            SimpleNamespace(commands=SimpleNamespace(run=local_run)),
            "python3 -", timeout=29, input="本地 stdin", allow_net=True, env_allow=("PUBLIC_",),
        )
        self.assertIs(result, expected)
        self.assertEqual(observed, [("python3 -", 29, "本地 stdin", True, ("PUBLIC_",))])

    async def test_sdk_exit_and_timeout_keep_existing_normalization(self) -> None:
        for failure in [
            CommandExitException(stdout="partial", stderr="failed", exit_code=7, error=None),
            TimeoutException("offline SDK timed out"),
        ]:
            with self.subTest(exception=type(failure).__name__):
                commands = create_autospec(Commands, instance=True, spec_set=True)
                handle = create_autospec(CommandHandle, instance=True, spec_set=True)
                commands.run.return_value = handle
                handle.wait.side_effect = failure
                if isinstance(failure, TimeoutException):
                    with self.assertRaisesRegex(TimeoutError, "offline SDK timed out"):
                        await arun_sandbox_cmd(
                            SimpleNamespace(commands=commands), "python3 -", timeout=31, input="source",
                        )
                    handle.kill.assert_called_once_with()
                else:
                    result = await arun_sandbox_cmd(
                        SimpleNamespace(commands=commands), "python3 -", timeout=31, input="source",
                    )
                    self.assertEqual((result.stdout, result.stderr, result.exit_code), ("partial", "failed", 7))
                    handle.kill.assert_not_called()
                handle.disconnect.assert_called_once_with()

    async def test_wait_transport_failure_kills_then_disconnects_and_preserves_error(self) -> None:
        commands = create_autospec(Commands, instance=True, spec_set=True)
        handle = create_autospec(CommandHandle, instance=True, spec_set=True)
        commands.run.return_value = handle
        failure = ReadError("offline transport failure")
        handle.wait.side_effect = failure
        lifecycle = Mock()
        for name, method in [
            ("run", commands.run), ("send_stdin", handle.send_stdin),
            ("close_stdin", handle.close_stdin), ("wait", handle.wait),
            ("kill", handle.kill), ("disconnect", handle.disconnect),
        ]:
            lifecycle.attach_mock(method, name)
        with self.assertRaises(ReadError) as caught:
            await arun_sandbox_cmd(
                SimpleNamespace(commands=commands), "python3 -", timeout=31, input="source",
            )
        self.assertIs(caught.exception, failure)
        handle.kill.assert_called_once_with()
        handle.disconnect.assert_called_once_with()
        self.assertEqual([entry[0] for entry in lifecycle.mock_calls], [
            "run", "send_stdin", "close_stdin", "wait", "kill", "disconnect",
        ])

    async def test_failed_kill_warns_safely_without_hiding_wait_failure(self) -> None:
        commands = create_autospec(Commands, instance=True, spec_set=True)
        handle = create_autospec(CommandHandle, instance=True, spec_set=True)
        commands.run.return_value = handle
        failure = ReadError("PRIVATE transport body")
        handle.wait.side_effect = failure
        handle.kill.side_effect = RuntimeError("PRIVATE cleanup body")
        with self.assertLogs("plugins.tools._sandbox", level="WARNING") as logs:
            with self.assertRaises(ReadError) as caught:
                await arun_sandbox_cmd(
                    SimpleNamespace(commands=commands), "python3 -", timeout=31, input="source",
                )
        self.assertIs(caught.exception, failure)
        handle.kill.assert_called_once_with()
        handle.disconnect.assert_called_once_with()
        self.assertIn("RuntimeError", "\n".join(logs.output))
        self.assertNotIn("PRIVATE", "\n".join(logs.output))

    async def test_stdin_setup_failures_kill_and_disconnect_without_hiding_error(self) -> None:
        for stage in ["send_stdin", "close_stdin"]:
            with self.subTest(stage=stage):
                commands = create_autospec(Commands, instance=True, spec_set=True)
                handle = create_autospec(CommandHandle, instance=True, spec_set=True)
                commands.run.return_value = handle
                getattr(handle, stage).side_effect = RuntimeError(f"offline {stage} failure")
                handle.kill.side_effect = RuntimeError("offline cleanup failure")
                with self.assertRaisesRegex(RuntimeError, f"offline {stage} failure"):
                    await arun_sandbox_cmd(
                        SimpleNamespace(commands=commands), "python3 -", timeout=41, input="source",
                    )
                handle.kill.assert_called_once_with()
                handle.disconnect.assert_called_once_with()
                handle.wait.assert_not_called()

    async def test_cancel_before_handle_creation_kills_late_handle(self) -> None:
        await self._cancel_at_sdk_boundary(before_handle=True)

    async def test_cancel_while_sdk_waits_kills_without_cross_thread_disconnect(self) -> None:
        await self._cancel_at_sdk_boundary(before_handle=False)

    async def _cancel_at_sdk_boundary(self, *, before_handle) -> None:
        commands = create_autospec(Commands, instance=True, spec_set=True)
        handle = create_autospec(CommandHandle, instance=True, spec_set=True)
        entered = threading.Event()
        release = threading.Event()
        killed = threading.Event()
        disconnected = threading.Event()
        thread_ids = {}

        def run(*args, **kwargs):
            thread_ids["run"] = threading.get_ident()
            if before_handle:
                entered.set()
                release.wait(3)
            return handle

        def wait():
            thread_ids["wait"] = threading.get_ident()
            if not before_handle:
                entered.set()
                release.wait(3)
            return CommandResult(stdout="done", stderr="", exit_code=0, error=None)

        def disconnect():
            thread_ids["disconnect"] = threading.get_ident()
            disconnected.set()

        def kill():
            killed.set()
            release.set()

        commands.run.side_effect = run
        handle.wait.side_effect = wait
        handle.kill.side_effect = kill
        handle.disconnect.side_effect = disconnect
        task = asyncio.create_task(arun_sandbox_cmd(
            SimpleNamespace(commands=commands), "python3 -", timeout=43, input="source",
        ))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2), "SDK boundary was not reached")
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            release.set()
            self.assertTrue(await asyncio.to_thread(killed.wait, 2), "remote command was not killed")
            self.assertTrue(await asyncio.to_thread(disconnected.wait, 2), "worker did not release SDK stream")
            handle.kill.assert_called_once_with()
            handle.disconnect.assert_called_once_with()
            self.assertEqual(thread_ids["disconnect"], thread_ids["run"])
            if before_handle:
                handle.send_stdin.assert_not_called()
                handle.wait.assert_not_called()
        finally:
            release.set()
            if not task.done():
                task.cancel()


if __name__ == "__main__":
    unittest.main()
