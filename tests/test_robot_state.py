"""Robot acknowledgements describe completed actual poses over the live socket."""
from dataclasses import replace
import json
import socket
import struct
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cube_stack_stream import LiveMapQueue, LiveStreamClient
from vr_json_cube_stacking import ColorCubeDemo


def cube(cube_id, x=0., y=.025, z=0.):
    return {"id": cube_id, "color": "red", "position": [x, y, z],
            "object_to_table": [.05, 0, 0, x, 0, .05, 0, y,
                                0, 0, .05, z, 0, 0, 0, 1]}


def snapshot(*cubes, revision=1, generation=0):
    return {"format": "openup.cube_stack.map", "version": 2,
            "coordinate_space": "table_local", "units": "meters", "matrix_layout": "row_major",
            "axes": {"origin": "table_surface_center", "x": "right", "y": "up", "z": "down_screen"},
            "cube_size": .05, "table": {"width": .8, "depth": .6},
            "revision": revision, "reset_generation": generation, "cubes": list(cubes)}


def header(data, protocol=2):
    return {"stream": "CubeStack/Map", "content_type": "application/json",
            "sequence": 1, "payload_bytes": len(json.dumps(data).encode("utf-8")), "timestamp": 1.,
            "metadata": json.dumps({"protocol_version": protocol, "session_id": "test-session"})}


def wire_frame(data):
    payload = json.dumps(data).encode("utf-8")
    encoded = json.dumps(header(data)).encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded + struct.pack("<I", len(payload)) + payload


def receive_part(peer):
    def exactly(count):
        result = bytearray()
        while len(result) < count:
            chunk = peer.recv(count - len(result))
            if not chunk:
                raise AssertionError("Robot disconnected inside a status frame")
            result.extend(chunk)
        return bytes(result)
    count, = struct.unpack("<I", exactly(4))
    return exactly(count)


def receive_state(peer):
    encoded = receive_part(peer)
    payload = receive_part(peer)
    message = json.loads(encoded)
    assert message["stream"] == "CubeStack/RobotState", message
    assert message["content_type"] == "application/json", message
    assert message["payload_bytes"] == len(payload), message
    assert json.loads(message["metadata"]) == {
        "protocol_version": 2, "session_id": "test-session"}, message
    return message, json.loads(payload)


class RobotStateTests(unittest.TestCase):
    def setUp(self):
        self.queue = LiveMapQueue(report=lambda message: None)

    def accept(self, data, **kwargs):
        self.assertGreaterEqual(self.queue.accept(header(data, **kwargs), data), 0)

    def state(self):
        self.assertTrue(hasattr(self.queue, "robot_state"),
                        "The simulator must expose completed robot state")
        return self.queue.robot_state()

    def test_active_birth_never_confirms_a_newer_target(self):
        self.accept(snapshot(cube("a", x=.1)))
        self.queue.start_next()
        self.accept(snapshot(cube("a", x=.2), revision=2))
        self.assertIsNone(self.state(), "An active transfer is not an acknowledgement")
        self.queue.complete_active()
        state = self.state()
        self.assertEqual((state["session_id"], state["revision"], state["reset_generation"]),
                         ("test-session", 2, 0))
        self.assertEqual([(item["id"], item["color"]) for item in state["cubes"]], [("a", "red")])
        np.testing.assert_allclose(state["cubes"][0]["position"], [.1, .025, 0], atol=1e-12)
        np.testing.assert_allclose(state["cubes"][0]["object_to_table"],
                                   [.05, 0, 0, .1, 0, .05, 0, .025,
                                    0, 0, .05, 0, 0, 0, 0, 1], atol=1e-12)

    def test_rotation_and_actual_staging_pose_are_converted_back_to_table(self):
        rotated = cube("rotated", x=.1)
        rotated["object_to_table"] = [0, 0, .05, .1, 0, .05, 0, .025,
                                       -.05, 0, 0, 0, 0, 0, 0, 1]
        self.accept(snapshot(rotated))
        command = self.queue.start_next()
        self.queue.complete_active()
        np.testing.assert_allclose(self.state()["cubes"][0]["object_to_table"],
                                   rotated["object_to_table"], atol=1e-12)
        # Staging replaces only the workspace pose; its retained source_position
        # is intentionally stale and must not be used for acknowledgement.
        self.queue.realized["rotated"] = replace(command.placement, position=(.7, .48, .04))
        np.testing.assert_allclose(self.state()["cubes"][0]["position"],
                                   [.3, .025, .1], atol=1e-12)

    def test_delete_and_cancelled_unstarted_birth_confirm_absence(self):
        self.accept(snapshot(cube("a")))
        self.queue.start_next()
        self.queue.complete_active()
        self.accept(snapshot(revision=2))
        self.queue.start_next()
        self.assertIsNone(self.state())
        self.queue.complete_active()
        self.assertEqual(self.state()["cubes"], [])
        self.accept(snapshot(cube("never_started"), revision=3))
        self.accept(snapshot(revision=4))
        self.assertIsNone(self.queue.start_next())
        self.assertEqual(self.state()["cubes"], [])
        self.assertEqual(self.state()["revision"], 4)

    def test_reset_is_confirmed_only_after_pool_reset_finishes(self):
        self.accept(snapshot(cube("a")))
        self.queue.start_next()
        self.accept(snapshot(revision=2, generation=1))
        self.queue.cancel_active_for_reset()
        self.assertIsNone(self.state())
        self.queue.start_next()
        self.assertIsNone(self.state())
        self.queue.complete_active()
        self.assertEqual(self.state(), {"session_id": "test-session", "revision": 2,
                                       "reset_generation": 1, "cubes": []})

    def test_legacy_and_halted_receivers_cannot_claim_success(self):
        self.accept(snapshot(), protocol=1)
        self.assertIsNone(self.state())
        self.queue = LiveMapQueue(report=lambda message: None)
        self.accept(snapshot())
        self.queue.halt("trajectory failure")
        self.assertIsNone(self.state())

    def test_measured_roundoff_stays_in_status_without_queuing_skipped_moves(self):
        self.accept(snapshot(cube("a")))
        command = self.queue.start_next()
        self.queue.complete_active()
        self.queue.realized["a"] = replace(command.placement, position=(.54000001, 0., .04))
        self.assertEqual(self.queue.pending_count, 0)
        self.assertIsNone(self.queue.start_next(), "Roundoff must not cause a busy loop of skipped moves")
        np.testing.assert_allclose(self.state()["cubes"][0]["position"],
                                   [0., .025, .00000000625], atol=1e-16)
        self.queue.realized["a"] = replace(command.placement, position=(.541, 0., .04))
        self.assertEqual(self.queue.pending_count, 1)
        self.assertEqual(self.queue.start_next().kind, "move")


class RobotStateSocketTests(unittest.TestCase):
    def setUp(self):
        self.queue = LiveMapQueue(report=lambda message: None)
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.listener.settimeout(2)
        self.client = LiveStreamClient(self.queue, port=self.listener.getsockname()[1],
                                       retry_interval=0, report=lambda message: None)
        self.peer = None

    def tearDown(self):
        self.client.close()
        if self.peer is not None:
            self.peer.close()
        self.listener.close()

    def connect(self, data):
        self.client.pump()
        self.peer, _ = self.listener.accept()
        self.peer.settimeout(.05)
        self.peer.sendall(wire_frame(data))
        for _ in range(3):
            self.client.pump()
        deadline = time.monotonic() + 2
        while self.queue.revision != data["revision"] and time.monotonic() < deadline:
            self.client.pump()
        self.assertEqual(self.queue.revision, data["revision"])

    def publish(self):
        self.assertTrue(hasattr(self.client, "publish_robot_state"),
                        "Completed robot state must be sent back to the table")
        self.client.publish_robot_state()

    def test_receiving_maps_alone_never_acknowledges_and_reconnect_replays_status(self):
        data = snapshot(cube("a"))
        self.connect(data)
        for _ in range(3):
            self.client.pump()
        with self.assertRaises(socket.timeout):
            self.peer.recv(1)
        self.queue.start_next()
        self.publish()
        with self.assertRaises(socket.timeout):
            self.peer.recv(1)
        self.queue.complete_active()
        self.publish()
        first_header, first = receive_state(self.peer)
        self.assertEqual(first["cubes"][0]["id"], "a")
        self.client.close()
        self.peer.close()
        self.connect(data)
        self.publish()
        second_header, second = receive_state(self.peer)
        self.assertEqual(second, first)
        self.assertGreater(second_header["sequence"], first_header["sequence"])

    def test_heartbeat_recovers_unchanged_status_without_duplicate_motion(self):
        self.connect(snapshot())
        with patch("cube_stack_stream.time.monotonic", return_value=100):
            self.publish()
        first_header, first = receive_state(self.peer)
        with patch("cube_stack_stream.time.monotonic", return_value=100.1):
            self.publish()
        with self.assertRaises(socket.timeout):
            self.peer.recv(1)
        with patch("cube_stack_stream.time.monotonic", return_value=100.6):
            self.publish()
        second_header, second = receive_state(self.peer)
        self.assertEqual(second, first)
        self.assertGreater(second_header["sequence"], first_header["sequence"])
        self.assertIsNone(self.queue.start_next())

    def test_partial_writes_finish_whole_frame_before_newer_state(self):
        self.connect(snapshot())
        transport = self.client.socket

        class PartialSocket:
            def __init__(self):
                self.blocked = False

            def send(self, data):
                if self.blocked:
                    raise BlockingIOError()
                self.blocked = True
                return transport.send(data[:13])

            def close(self):
                transport.close()

        partial = PartialSocket()
        self.client.socket = partial
        self.publish()
        data = snapshot(cube("a"), revision=2)
        self.queue.accept(header(data), data)
        self.queue.start_next()
        self.queue.complete_active()
        for _ in range(250):
            partial.blocked = False
            self.publish()
        _, first = receive_state(self.peer)
        _, second = receive_state(self.peer)
        self.assertEqual((first["revision"], first["cubes"]), (1, []))
        self.assertEqual(second["revision"], 2)
        self.assertEqual(second["cubes"][0]["id"], "a")

    def test_live_loop_sends_completion_after_transfer_before_bounded_exit(self):
        self.connect(snapshot(cube("a", x=.1)))
        demo = ColorCubeDemo.__new__(ColorCubeDemo)
        demo.options = type("Options", (), {"pool_per_color": 30})()
        demo._start_recording = lambda: None
        demo._stop_recording = lambda settle=False: None

        def place(placement):
            self.peer.sendall(wire_frame(snapshot(cube("a", x=.2), revision=2)))
            deadline = time.monotonic() + 2
            while self.queue.revision != 2 and time.monotonic() < deadline:
                self.client.pump()
            with self.assertRaises(socket.timeout):
                self.peer.recv(1)

        demo.place_live_cube = place
        demo.run_live(self.queue, self.client, max_cubes=1)
        _, result = receive_state(self.peer)
        self.assertEqual(result["revision"], 2)
        np.testing.assert_allclose(result["cubes"][0]["position"], [.1, .025, 0], atol=1e-12)

    def test_idle_live_loop_sends_absence_without_starting_robot_work(self):
        self.connect(snapshot())
        demo = ColorCubeDemo.__new__(ColorCubeDemo)
        demo.options = type("Options", (), {"pool_per_color": 30})()
        demo._start_recording = lambda: None
        demo._stop_recording = lambda settle=False: None

        def stop_idle():
            raise KeyboardInterrupt()

        demo._step = stop_idle
        with self.assertRaises(KeyboardInterrupt):
            demo.run_live(self.queue, self.client)
        _, result = receive_state(self.peer)
        self.assertEqual(result["cubes"], [])

    def test_physical_completion_reports_entity_pose_when_target_was_missed(self):
        self.connect(snapshot(cube("a", x=.1)))
        demo = ColorCubeDemo.__new__(ColorCubeDemo)
        demo.options = SimpleNamespace(pool_per_color=30, physical=True)
        demo._start_recording = lambda: None
        demo._stop_recording = lambda settle=False: None
        demo.current_placements = {}
        actual_position = (.54, 0., .04)
        demo.blocks = {"a": SimpleNamespace(get_pos=lambda: np.array(actual_position),
                                            get_quat=lambda: np.array((1., 0., 0., 0.)))}
        # A physical grasp can return normally even though the cube was dropped.
        demo.place_live_cube = lambda placement: demo.current_placements.update(a=placement)
        demo.run_live(self.queue, self.client, max_cubes=1)
        _, result = receive_state(self.peer)
        np.testing.assert_allclose(result["cubes"][0]["position"], [0, .025, 0], atol=1e-12)
        np.testing.assert_allclose(self.queue.realized["a"].position, actual_position, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
