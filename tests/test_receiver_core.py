"""The simulator-independent receiver loop, pose snapping, and --sim launcher."""
import contextlib
import io
import math
import sys
import unittest

import numpy as np

from cube_stack_layout import CubePlacement, PiecePlacement, as_piece
from cube_stack_receiver import ResetRequested, pump_and_check, run_live, snap_to_target
from cube_stack_stream import LiveMapQueue
from fake_backend import FakeBackend
from test_live_stream import cube, header, snapshot


class _ScriptDone(Exception):
    pass


class ScriptedClient:
    """Feeds the next map only once the queue is idle, like a patient table user."""

    def __init__(self, queue, maps):
        self.queue = queue
        self.maps = list(maps)
        self.published = []

    def pump(self):
        if self.queue.active is not None or (self.queue.layout is not None and self.queue.pending_count):
            return
        if not self.maps:
            raise _ScriptDone()
        data = self.maps.pop(0)
        self.queue.accept(header(data), data)

    def publish_robot_state(self):
        state = self.queue.robot_state()
        if state is not None:
            self.published.append(state)

    def close(self):
        pass


def run_script(maps, max_cubes=0):
    messages = []
    queue = LiveMapQueue(pool_per_color=4, report=messages.append)
    client = ScriptedClient(queue, maps)
    client.pump()  # first map builds the backend, as live_main does
    backend = FakeBackend(queue.layout)
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
        try:
            run_live(backend, queue, client, max_cubes=max_cubes)
        except _ScriptDone:
            pass
    return backend, queue, client, messages, stdout.getvalue()


class RunLiveTests(unittest.TestCase):
    def test_add_move_delete_reset_through_shared_loop(self):
        maps = [
            snapshot(cube("a"), revision=1),
            snapshot(cube("a", x=0.1), revision=2),
            snapshot(revision=3),
            snapshot(cube("b", x=-0.1, color="blue"), revision=4, generation=1),
        ]
        backend, queue, client, messages, stdout = run_script(maps)
        started = [line for line in messages if line.startswith(("started:", "completed:"))]
        self.assertEqual(started, [
            "started: add a", "completed: add a",
            "started: move a", "completed: move a",
            "started: delete a", "completed: delete a",
            "started: reset reset", "completed: reset reset",
            "started: add b", "completed: add b",
        ])
        self.assertIn("ready: session=test-session pool_per_color=4", stdout)
        self.assertEqual(backend.homed, 1)
        final = client.published[-1]
        self.assertEqual(final["reset_generation"], 1)
        self.assertEqual([entry["id"] for entry in final["cubes"]], ["b"])
        np.testing.assert_allclose(final["cubes"][0]["position"], [-0.1, 0.025, 0.0], atol=1e-9)

    def test_max_cubes_ends_cleanly_with_completion_line(self):
        _, queue, _, _, stdout = run_script([snapshot(cube("a"), revision=1)], max_cubes=1)
        self.assertEqual(queue.completed_count, 1)
        self.assertIn("live complete: accepted=1 completed=1", stdout)

    def test_pump_and_check_reports_halt_and_reset(self):
        queue = LiveMapQueue(pool_per_color=4, report=lambda message: None)
        client = type("Client", (), {"pump": lambda self: None, "queue": queue})()
        pump_and_check(None)  # offline replay has no client
        pump_and_check(client)
        queue.reset_generation = 1
        with self.assertRaises(ResetRequested):
            pump_and_check(client, queue)
        queue.halt("boom")
        with self.assertRaisesRegex(RuntimeError, "HALTED: boom"):
            pump_and_check(client, queue)


def yaw(degrees):
    half = math.radians(degrees) / 2
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def tilt(degrees):
    half = math.radians(degrees) / 2
    return (math.cos(half), math.sin(half), 0.0, 0.0)


def single(position, quaternion=(1.0, 0.0, 0.0, 0.0)):
    return as_piece(CubePlacement("c", "red", (0, 0, 0), position, quaternion))


def bar(position, quaternion=(1.0, 0.0, 0.0, 0.0)):
    members = tuple(CubePlacement(f"m{i}", "red", (0, 0, 0), position, quaternion, "bar", i, 3)
                    for i in range(3))
    return PiecePlacement("bar", "red", members, (0, 0, 0), position, quaternion)


class SnapToTargetTests(unittest.TestCase):
    def test_close_pose_reports_exact_target(self):
        target = single((0.3, 0.1, 0.04), yaw(20))
        snapped = snap_to_target(single((0.305, 0.097, 0.0405), yaw(21)), target)
        self.assertEqual(snapped.position, target.position)
        self.assertEqual(snapped.quaternion, target.quaternion)

    def test_far_or_rotated_or_tipped_pose_stays_measured(self):
        target = single((0.3, 0.1, 0.04))
        for measured in (single((0.32, 0.1, 0.04)), single((0.3, 0.1, 0.04), yaw(10)),
                         single((0.3, 0.1, 0.04), tilt(10))):
            with self.subTest(measured=measured):
                self.assertIs(snap_to_target(measured, target), measured)

    def test_yaw_symmetry_follows_piece_shape(self):
        self.assertEqual(snap_to_target(single((0, 0, .04), yaw(90)), single((0, 0, .04))).quaternion,
                         (1.0, 0.0, 0.0, 0.0))
        measured = bar((0, 0, .04), yaw(90))
        self.assertIs(snap_to_target(measured, bar((0, 0, .04))), measured)
        self.assertEqual(snap_to_target(bar((0, 0, .04), yaw(180)), bar((0, 0, .04))).quaternion,
                         (1.0, 0.0, 0.0, 0.0))

    def test_no_target_keeps_measured(self):
        measured = single((0, 0, .04))
        self.assertIs(snap_to_target(measured, None), measured)


class LauncherTests(unittest.TestCase):
    def test_genesis_arguments_parse_without_importing_genesis(self):
        import robot_receiver
        args, backend = robot_receiver.parse_args(["--sim", "genesis", "--live", "--physical", "--speed", "2"])
        self.assertEqual(backend.__name__, "vr_json_cube_stacking")
        self.assertTrue(args.physical)
        self.assertEqual(args.pool_per_color, 30)
        self.assertNotIn("genesis", sys.modules)

    def test_toyblocks_arguments_parse_without_importing_isaac(self):
        import robot_receiver
        args, backend = robot_receiver.parse_args(
            ["--sim", "toyblocks", "--live", "--robot-to-near-edge", "0.2", "--speed", "10"])
        self.assertEqual(backend.__name__, "toyblocks_backend.backend")
        self.assertEqual(args.pool_per_color, 10)
        self.assertAlmostEqual(args.robot_to_near_edge, 0.2)
        for module in ("isaaclab", "toyblocks", "torch"):
            self.assertNotIn(module, sys.modules)

    def test_toyblocks_rejects_recording_before_launching_isaac(self):
        import robot_receiver
        args, backend = robot_receiver.parse_args(["--sim", "toyblocks", "--live", "--record"])
        with self.assertRaisesRegex(ValueError, "--record"):
            backend.prepare_backend(args)

    def test_mode_is_required(self):
        import robot_receiver
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            robot_receiver.parse_args(["--sim", "genesis"])


if __name__ == "__main__":
    unittest.main()
