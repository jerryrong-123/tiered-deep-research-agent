"""Offline checks at the maintained local trial adapter's public boundaries."""
import concurrent.futures
import hashlib
import json
import os
import tempfile
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from local_runner.answer_label_guard import MARKER, install_guard
from local_runner import trial_helpers
from local_runner.sandbox_registry import SandboxRegistry


class GuardChecks(unittest.TestCase):
    def test_native_review_receipt_is_not_an_external_answer_source(self):
        from workflows.agent_team.subagent_runtime import SwarmToolResultPostProcessor

        question = "Which example record has eight separate words from this current question?"
        receipt = json.dumps({"action": "continue", "question": question, "receipt_ids": ["receipt-1"]})
        with tempfile.TemporaryDirectory() as directory:
            _, restore = install_guard(question, directory)
            try:
                processor = SwarmToolResultPostProcessor()
                result = processor.process(SimpleNamespace(
                    name="review_research", result=receipt, is_error=False,
                ))
                self.assertEqual(result, receipt)
                self.assertFalse((Path(directory) / "ANSWER_LABEL_GUARD.jsonl").exists())
                external = processor.process(SimpleNamespace(
                    name="web_fetch", result=receipt, is_error=False,
                ))
                self.assertEqual(external, MARKER)
            finally:
                restore()

    def test_native_dispatch_and_board_receipts_keep_the_control_protocol(self):
        from workflows.agent_team.subagent_runtime import SwarmToolResultPostProcessor

        question = "Which example record has eight separate words from this current question?"
        with tempfile.TemporaryDirectory() as directory:
            _, restore = install_guard(question, directory)
            try:
                processor = SwarmToolResultPostProcessor()
                for name in ("create_subagent", "assign_task", "collect_reports", "stop_subagent",
                             "add_task", "update_task", "finish_planning", "submit_report"):
                    with self.subTest(tool=name):
                        result = processor.process(SimpleNamespace(name=name, result=question, is_error=False))
                        self.assertEqual(result, question)
                for name in ("web_fetch", "read_file", "new_unclassified_tool"):
                    with self.subTest(tool=name):
                        result = processor.process(SimpleNamespace(name=name, result=question, is_error=False))
                        self.assertEqual(result, MARKER)
            finally:
                restore()


class EventChecks(unittest.TestCase):
    def test_threaded_events_survive_exactly_once_on_trial_filesystem(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_RUNNER_TEST_ARTIFACTS")) as directory:
            old = trial_helpers.HERE
            trial_helpers.HERE = Path(directory)
            barrier = threading.Barrier(8)

            def produce(worker):
                barrier.wait()
                for sequence in range(400):
                    trial_helpers.emit("offline_fixture", producer=worker, sequence=sequence,
                                       payload="public_记录" * 400)

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(produce, range(8)))
                rows = [json.loads(line) for line in (Path(directory) / "events.jsonl").read_bytes().splitlines()]
                self.assertEqual(len(rows), 3200)
                observed = {(row["producer"], row["sequence"]) for row in rows}
                self.assertEqual(observed, {(worker, sequence) for worker in range(8) for sequence in range(400)})
            finally:
                trial_helpers.HERE = old


class CleanupChecks(unittest.TestCase):
    def test_bad_log_line_does_not_prevent_other_sandbox_cleanup(self):
        from local_runner import run_native as runner

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "events.jsonl").write_bytes(
                b'{"kind":"sandbox_created","sandbox_id":"offline-a"}\n'
                b'broken json\xff\n'
                b'{"kind":"sandbox_created","sandbox_id":"offline-b"}\n'
            )
            killed = []
            sdk = types.ModuleType("e2b_code_interpreter")
            sdk.Sandbox = SimpleNamespace(kill=lambda ident, **kwargs: killed.append(ident) or True)
            with patch.object(runner, "HERE", path), patch.object(runner.helpers, "HERE", path), \
                    patch.dict("sys.modules", {"e2b_code_interpreter": sdk}), \
                    patch.dict(os.environ, {"E2B_API_KEY": "offline-unused"}):
                summary = runner.cleanup_sandboxes()
            self.assertEqual(killed, ["offline-a", "offline-b"])
            self.assertEqual(summary["invalid_log_lines"][0]["line"], 2)
            self.assertEqual(len(summary["checks"]), 2)

    def test_registry_recovers_created_sandbox_when_event_write_is_lost(self):
        from local_runner import run_native as runner

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            resource = SimpleNamespace(sandbox_id="offline-only-in-registry", kill=lambda: self.fail("registration succeeded"))
            killed = []
            sdk = types.ModuleType("e2b_code_interpreter")
            sdk.Sandbox = SimpleNamespace(kill=lambda ident, **kwargs: killed.append(ident) or True)
            with patch.object(runner, "HERE", path), patch.object(runner.helpers, "HERE", path), \
                    patch.dict("sys.modules", {"e2b_code_interpreter": sdk}), \
                    patch.dict(os.environ, {"E2B_API_KEY": "offline-unused"}):
                with patch.object(runner, "emit", side_effect=OSError("offline event failure")):
                    with self.assertRaises(OSError):
                        runner.register_created_sandbox(resource)
                (path / "events.jsonl").write_bytes(b'corrupt log without any usable sandbox ids\n')
                summary = runner.cleanup_sandboxes()
                second = runner.cleanup_sandboxes()
            self.assertEqual(killed, ["offline-only-in-registry"])
            self.assertEqual(summary["checks"][0]["confirmed"], True)
            self.assertEqual(second["checks"], [])
            self.assertEqual(SandboxRegistry(path).records()[0][resource.sandbox_id]["status"], "cleanup_confirmed")

    def test_creation_hook_persists_registry_before_returning_sdk_sandbox(self):
        from e2b_code_interpreter import Sandbox
        from local_runner import run_native as runner

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            resource = SimpleNamespace(sandbox_id="offline-hook-created")
            with patch.object(runner, "HERE", path), patch.object(runner.helpers, "HERE", path), \
                    patch.object(Sandbox, "create", return_value=resource):
                restore = runner.observation_hooks()
                try:
                    returned = Sandbox.create()
                    self.assertIs(returned, resource)
                    records, invalid = SandboxRegistry(path).records()
                    self.assertEqual(records[resource.sandbox_id]["status"], "active")
                    self.assertEqual(invalid, [])
                finally:
                    restore()

    def test_failed_registry_persistence_closes_new_sandbox_instead_of_exposing_it(self):
        from local_runner import run_native as runner

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "sandbox_registry").write_text("blocks directory creation")
            closed = []
            resource = SimpleNamespace(sandbox_id="offline-persist-failure", kill=lambda: closed.append(True))
            with patch.object(runner, "HERE", path):
                with self.assertRaises(OSError):
                    runner.register_created_sandbox(resource)
            self.assertEqual(closed, [True])

    def test_one_failed_kill_and_bad_registry_record_do_not_skip_other_ids(self):
        from local_runner import run_native as runner

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            registry = SandboxRegistry(path)
            registry.register("offline-a")
            registry.register("offline-b")
            (registry.directory / "bad.json").write_text("broken registry")
            checked = []

            def kill(ident, **kwargs):
                checked.append(ident)
                if ident == "offline-a":
                    raise OSError("offline fake failure")
                return False

            sdk = types.ModuleType("e2b_code_interpreter")
            sdk.Sandbox = SimpleNamespace(kill=kill)
            with patch.object(runner, "HERE", path), patch.object(runner.helpers, "HERE", path), \
                    patch.dict("sys.modules", {"e2b_code_interpreter": sdk}), \
                    patch.dict(os.environ, {"E2B_API_KEY": "offline-unused"}):
                summary = runner.cleanup_sandboxes()
            self.assertEqual(checked, ["offline-a", "offline-b"])
            self.assertFalse(summary["checks"][0]["confirmed"])
            self.assertTrue(summary["checks"][1]["confirmed"])
            self.assertTrue(summary["checks"][1]["already_absent"])
            self.assertEqual(summary["invalid_registry_records"][0]["file"], "bad.json")

    def test_registry_failure_and_immediate_kill_failure_leave_recoverable_fallback_id(self):
        from local_runner import run_native as runner

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "sandbox_registry").write_text("blocks registry creation")

            def failed_immediate_kill():
                raise OSError("offline transient close failure")

            resource = SimpleNamespace(sandbox_id="offline-recovery", kill=failed_immediate_kill)
            killed = []
            sdk = types.ModuleType("e2b_code_interpreter")
            sdk.Sandbox = SimpleNamespace(kill=lambda ident, **kwargs: killed.append(ident) or True)
            with patch.object(runner, "HERE", path), patch.object(runner.helpers, "HERE", path), \
                    patch.dict("sys.modules", {"e2b_code_interpreter": sdk}), \
                    patch.dict(os.environ, {"E2B_API_KEY": "offline-unused"}):
                with self.assertRaises(OSError):
                    runner.register_created_sandbox(resource)
                runner.cleanup_sandboxes()
            self.assertEqual(killed, ["offline-recovery"])


class PrepareChecks(unittest.TestCase):
    def test_preparation_uses_maintained_templates_and_current_source_without_starting(self):
        from local_runner.prepare import HERE, TEMPLATE_FILES, prepare
        from local_runner import run_native as runner

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / "frontier_research"
            source.mkdir()
            (source / "application.py").write_text("offline fixture source\n")
            question = {"id": "offline-case", "question": "A complete offline fixture question?"}
            question["question_sha256"] = hashlib.sha256(question["question"].encode()).hexdigest()
            question_file = path / "question.json"
            question_file.write_text(json.dumps(question))
            profile = path / "explicit.yaml"
            profile.write_text("agent:\n  progressive_research_v1: true\n")
            trial = path / "fresh_trial"
            metadata = prepare(question_file, trial, profile_file=profile, source_dir=source)
            self.assertEqual(json.loads((trial / "QUESTION.json").read_text()), question)
            self.assertEqual(metadata["adapter_version"], "native_local_runner_v1")
            self.assertEqual(metadata["paid_runs_started"], 0)
            self.assertEqual((trial / "PROFILE.yaml").read_bytes(), profile.read_bytes())
            self.assertFalse((trial / "STARTED.json").exists())
            for name in TEMPLATE_FILES:
                if name not in {"PROFILE.yaml", "launch.ps1"}:
                    self.assertEqual((trial / name).read_bytes(), (HERE / name).read_bytes())
            with patch.object(runner, "HERE", trial), patch.object(runner, "UP", source):
                self.assertEqual(runner.check_source(), question)
                (source / "application.py").write_text("changed source\n")
                with self.assertRaisesRegex(RuntimeError, "source_changed"):
                    runner.check_source()
            launch = (trial / "launch.ps1").read_text()
            test_branch = launch.split("if($Mode -eq 'test') {", 1)[1].split("} else {", 1)[0]
            self.assertIn("'--network','none'", test_branch)
            self.assertNotIn("--env-file", test_branch)
            self.assertFalse((trial / "native").exists())
            with self.assertRaises(FileExistsError):
                prepare(question_file, trial, source_dir=source)

    def test_preparation_rejects_extra_question_fields_without_creating_trial(self):
        from local_runner.prepare import prepare

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            question = path / "question.json"
            question.write_text(json.dumps({"id": "offline", "question": "fixture", "question_sha256": "x", "extra": "blocked"}))
            with self.assertRaisesRegex(ValueError, "question_only_three_keys"):
                prepare(question, path / "trial")
            self.assertFalse((path / "trial").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
