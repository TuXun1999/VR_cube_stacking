"""The fake table publisher, end to end through robot_receiver.py over real TCP."""
import contextlib
import io
import json
import socket
import threading
import unittest
from unittest.mock import patch

from cube_stack_layout import parse_table_layout
from fake_table_publisher_path import publisher
import fake_backend
import robot_receiver


class PublisherMapTests(unittest.TestCase):
    def test_default_script_steps_are_valid_v3_maps(self):
        script = json.loads(publisher.DEFAULT_SCRIPT.read_text(encoding="utf-8"))
        for number, step in enumerate(script["steps"], 1):
            with self.subTest(step=step["name"]):
                layout = parse_table_layout(publisher.build_map(script, step["pieces"], number, 0))
                self.assertEqual(sorted(piece.piece_id for piece in layout.pieces),
                                 sorted(piece["id"] for piece in step["pieces"]))

    def test_state_match_needs_exact_ids_revision_and_pose(self):
        script = {"cube_size": 0.08, "steps": []}
        data = publisher.build_map(script, [{"id": "a", "color": "red", "x": 0, "z": 0}], 3, 1)
        state = {"session_id": "s", "revision": 3, "reset_generation": 1,
                 "cubes": [dict(cube) for cube in data["cubes"]]}
        self.assertTrue(publisher.state_matches(state, data, "s"))
        self.assertFalse(publisher.state_matches({**state, "revision": 2}, data, "s"))
        moved = json.loads(json.dumps(state))
        moved["cubes"][0]["object_to_table"][3] += 0.001
        self.assertFalse(publisher.state_matches(moved, data, "s"))
        self.assertFalse(publisher.state_matches({**state, "cubes": []}, data, "s"))


class PublisherRoundTripTests(unittest.TestCase):
    def test_default_script_confirmed_by_launcher_with_fake_backend(self):
        listener = socket.create_server(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        script = json.loads(publisher.DEFAULT_SCRIPT.read_text(encoding="utf-8"))
        messages = []
        table = publisher.FakeTablePublisher(script, listener, step_timeout=20, connect_timeout=20,
                                             report=messages.append)
        fake_backend.STOP.clear()
        self.addCleanup(fake_backend.STOP.clear)  # later tests reuse FakeBackend
        result = {}

        def receive():
            with patch.dict(robot_receiver.BACKENDS, {"fake": "fake_backend"}):
                result["code"] = robot_receiver.main(
                    ["--sim", "fake", "--live", "--port", str(port), "--timeout", "20"])

        with contextlib.redirect_stdout(io.StringIO()) as receiver_output:
            thread = threading.Thread(target=receive, daemon=True)
            thread.start()
            try:
                code = table.run()
            finally:
                fake_backend.STOP.set()
                thread.join(10)
                table.close()
        self.assertEqual(code, 0, "\n".join(messages))
        self.assertIn("complete: every step confirmed", messages)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result.get("code"), 0, receiver_output.getvalue())
        output = receiver_output.getvalue()
        for line in ("started: add r1", "completed: move g1", "completed: delete b1",
                     "completed: reset reset", "completed: add r2"):
            self.assertIn(line, output)


if __name__ == "__main__":
    unittest.main()
