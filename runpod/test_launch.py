"""Tests for launch.py's fetch against a local file server (no RunPod calls).

Run: python runpod/test_launch.py
"""

import functools
import http.server
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch  # noqa: E402


class GhostHandler(http.server.SimpleHTTPRequestHandler):
    """Lists one file that cannot be downloaded, like a transfer that keeps timing out."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.endswith("/ghost.jsonl"):
            self.send_error(504)
            return
        super().do_GET()


class TestFetch(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        run = os.path.join(self.root, "runs", "r1")
        os.makedirs(os.path.join(run, "mixup_logs"))
        os.makedirs(os.path.join(run, "ray_logs"))
        self.files = {
            "progress.txt": b"STAGE:DONE exit=0\n",
            "train_log.txt": b"step:1 - a:1\n" * 1000,
            "self_stop.log": b"",
            "mixup_logs/mixup_calls.1.jsonl": b'{"a": 1}\n' * 5000,
            "mixup_logs/mixup_calls.2.jsonl": b'{"a": 2}\n',
            "ray_logs/worker-1.out": b"x",
        }
        for rel, data in self.files.items():
            with open(os.path.join(run, *rel.split("/")), "wb") as f:
                f.write(data)
        handler = functools.partial(GhostHandler, directory=self.root)
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        port = self.srv.server_address[1]
        self._proxy, self._sleep = launch._proxy, launch.RETRY_SLEEP
        launch._proxy = lambda pod, rel: f"http://127.0.0.1:{port}/{rel}"
        launch.RETRY_SLEEP = 0
        self.run_dir = run

    def tearDown(self):
        launch._proxy, launch.RETRY_SLEEP = self._proxy, self._sleep
        self.srv.shutdown()
        self.srv.server_close()

    def test_everything_listed_is_saved_byte_for_byte(self):
        out = os.path.join(self.root, "out")
        saved, failed = launch.fetch_run("pod", "r1", out)
        self.assertEqual(failed, [])
        self.assertEqual(sorted(saved), sorted(self.files))
        for rel, data in self.files.items():
            with open(os.path.join(out, *rel.split("/")), "rb") as f:
                self.assertEqual(f.read(), data, rel)

    def test_a_listed_file_that_will_not_download_is_reported_not_skipped(self):
        with open(os.path.join(self.run_dir, "mixup_logs", "ghost.jsonl"), "wb") as f:
            f.write(b"never served")
        saved, failed = launch.fetch_run("pod", "r1", os.path.join(self.root, "out2"))
        self.assertEqual(failed, ["mixup_logs/ghost.jsonl"])
        self.assertEqual(len(saved), len(self.files))

    def test_wrong_run_name_saves_nothing(self):
        saved, failed = launch.fetch_run("pod", "nope", os.path.join(self.root, "out3"))
        self.assertEqual((saved, failed), ([], []))


if __name__ == "__main__":
    unittest.main()
