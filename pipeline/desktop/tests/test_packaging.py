"""Regressions for native build failures after a successful model smoke."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def load_helper(filename):
    path = Path(__file__).resolve().parents[2] / "packaging" / filename
    spec = importlib.util.spec_from_file_location(filename, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PackagingTests(unittest.TestCase):
    def test_successful_korean_report_survives_windows_cp1252_console(self):
        runner = load_helper("run-check.py")
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            payload = {"ok": True, "stage": "완료", "transcript": "8ml와 9ml가 남았어요"}
            class Process:
                returncode = 0
                def __enter__(self):
                    report.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                    return self
                def __exit__(self, *_args):
                    pass
                def wait(self, **_kwargs):
                    return 0
            buffer = io.BytesIO()
            console = io.TextIOWrapper(buffer, encoding="cp1252")
            with patch("sys.argv", ["run-check", "--executable", "unused", "--report", str(report)]), \
                    patch.object(runner.subprocess, "Popen", return_value=Process()), \
                    patch("sys.stdout", console):
                result = runner.main()
            console.flush()
            self.assertEqual(result, 0)
            self.assertTrue(buffer.getvalue().isascii())
            self.assertIn("남았어요", report.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(report.read_text(encoding="utf-8")), payload)

    def test_dmg_transient_failure_retries_and_only_replaces_on_success(self):
        helper = load_helper("create-dmg.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            output = root / "final.dmg"
            output.write_bytes(b"previous complete image")
            calls = []
            def run(command, **kwargs):
                self.assertIs(kwargs["check"], False)
                self.assertEqual(output.read_bytes(), b"previous complete image")
                self.assertEqual(command[command.index("-srcfolder") + 1], str(source))
                image = Path(command[-1])
                self.assertNotEqual(image, output)
                self.assertFalse(image.exists())  # previous partial attempt was removed
                calls.append(image)
                image.write_bytes(b"complete" if len(calls) == 3 else b"partial")
                return SimpleNamespace(returncode=0 if len(calls) == 3 else 1)
            with patch.object(helper.subprocess, "run", side_effect=run), \
                    patch.object(helper.time, "sleep") as sleep, patch("sys.stdout", io.StringIO()):
                helper.create_dmg(source, output)
            self.assertEqual(len(calls), 3)
            self.assertEqual([c.args for c in sleep.call_args_list], [(3,), (3,)])
            self.assertEqual(output.read_bytes(), b"complete")
            self.assertTrue(all(not p.parent.exists() for p in calls))

    def test_dmg_retries_are_bounded_and_preserve_previous_output(self):
        helper = load_helper("create-dmg.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "final.dmg"
            output.write_bytes(b"previous complete image")
            unrelated = root / "other.txt"
            unrelated.write_text("keep")
            def fail(command, **_kwargs):
                Path(command[-1]).write_bytes(b"partial")
                return SimpleNamespace(returncode=1)
            with patch.object(helper.subprocess, "run", side_effect=fail) as run, \
                    patch.object(helper.time, "sleep") as sleep, patch("sys.stdout", io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "3 attempts"):
                    helper.create_dmg(root, output)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(output.read_bytes(), b"previous complete image")
            self.assertEqual(unrelated.read_text(), "keep")
            self.assertEqual({p.name for p in root.iterdir()}, {"final.dmg", "other.txt"})


if __name__ == "__main__":
    unittest.main()
