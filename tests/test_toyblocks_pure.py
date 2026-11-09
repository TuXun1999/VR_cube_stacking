"""ToyBlocks backend pieces that need no Isaac Sim: frames and in-reach staging."""
import json
from pathlib import Path
import unittest

import numpy as np

from cube_stack_layout import parse_table_layout
from cube_stack_stream import LiveMapQueue
from fake_backend import FakeBackend
from test_live_stream import header
from cube_stack_receiver import pose_within_tolerance
from toyblocks_backend.frames import FrameMap, jaw_clear_quaternion, nearest_symmetric_quaternion
from toyblocks_backend.props import GROUND_Z, pad_boxes, pool_bodies, table_box
from toyblocks_backend.staging import PAD_CLEARANCE, ReachableStagingArea

ROOT = Path(__file__).resolve().parents[1]


def example_layout(version=3):
    data = json.loads((ROOT / "example_table_data.json").read_text(encoding="utf-8"))
    data["cubes"] = []
    data["version"] = version
    return parse_table_layout(data)


class FrameMapTests(unittest.TestCase):
    def test_translation_only_round_trip(self):
        frames = FrameMap(robot_to_near_edge=0.2, table_top=0.01)
        planner = np.array((0.3, -0.1, 0.04))
        np.testing.assert_allclose(frames.to_robot(planner), (0.44, -0.1, 0.05))
        np.testing.assert_allclose(frames.to_planner(frames.to_robot(planner)), planner)
        np.testing.assert_allclose(frames.robot_base_in_planner, (-0.14, 0.0, -0.01))

    def test_pose_vectors_keep_scalar_first_quaternions(self):
        frames = FrameMap(robot_to_near_edge=0.15)
        quaternion = (0.9238795, 0.0, 0.0, 0.3826834)
        pose = frames.to_robot_pose((0.4, 0.0, 0.04), quaternion)
        self.assertEqual(pose.shape, (7,))
        position, back = frames.to_planner_pose(pose)
        np.testing.assert_allclose(position, (0.4, 0.0, 0.04))
        self.assertEqual(back, quaternion)


def yaw(degrees):
    half = np.radians(degrees) / 2
    return (np.cos(half), 0.0, 0.0, np.sin(half))


class SymmetricQuaternionTests(unittest.TestCase):
    def test_cube_turns_at_most_45_degrees(self):
        chosen = nearest_symmetric_quaternion(yaw(80), yaw(0), length=1)
        np.testing.assert_allclose(np.abs(np.dot(chosen, yaw(90))), 1.0, atol=1e-9)
        self.assertTrue(pose_within_tolerance((0, 0, 0), chosen, (0, 0, 0), yaw(0), length=1))

    def test_bar_uses_half_turns_only(self):
        chosen = nearest_symmetric_quaternion(yaw(170), yaw(0), length=3)
        np.testing.assert_allclose(np.abs(np.dot(chosen, yaw(180))), 1.0, atol=1e-9)
        chosen = nearest_symmetric_quaternion(yaw(80), yaw(0), length=3)
        np.testing.assert_allclose(np.abs(np.dot(chosen, yaw(0))), 1.0, atol=1e-9)


class JawClearanceTests(unittest.TestCase):
    target = (0.3, 0.0, 0.04)

    def assert_same_orientation(self, first, second):
        np.testing.assert_allclose(abs(np.dot(first, second)), 1.0, atol=1e-9)

    def test_turns_the_grasp_away_from_a_touching_neighbour(self):
        neighbour = ((0.38, 0.0, 0.04), yaw(0), 1)  # touching the target's +x face
        chosen = jaw_clear_quaternion(yaw(0), yaw(0), 1, (1.0, 0.0, 0.0), self.target, [neighbour])
        self.assertAlmostEqual(abs(chosen[3]), np.sin(np.radians(45)), places=6)  # a quarter turn

    def test_without_neighbours_keeps_the_smallest_wrist_turn(self):
        chosen = jaw_clear_quaternion(yaw(10), yaw(0), 1, (1.0, 0.0, 0.0), self.target, [])
        self.assert_same_orientation(chosen, yaw(0))

    def test_neighbours_on_other_levels_or_far_away_do_not_block(self):
        below = ((0.38, 0.0, -0.04), yaw(0), 1)
        far = ((0.45, 0.0, 0.04), yaw(0), 1)
        chosen = jaw_clear_quaternion(yaw(0), yaw(0), 1, (1.0, 0.0, 0.0), self.target, [below, far])
        self.assert_same_orientation(chosen, yaw(0))

    def test_bar_keeps_its_axis_and_only_half_turns(self):
        neighbour = ((0.3, 0.08, 0.04), yaw(0), 1)  # beside the bar's long side
        chosen = jaw_clear_quaternion(yaw(0), yaw(0), 3, (1.0, 0.0, 0.0), self.target, [neighbour])
        self.assertTrue(abs(np.dot(chosen, yaw(0))) > 1 - 1e-9 or abs(np.dot(chosen, yaw(180))) > 1 - 1e-9)


class SceneGeometryTests(unittest.TestCase):
    def test_pool_matches_genesis_counts_and_ids(self):
        bodies = pool_bodies(3, 10)
        self.assertEqual(len(bodies), 3 * (10 + 5 + 3))
        self.assertTrue(all(body.key.startswith("block_") for body in bodies))
        bar = next(body for body in bodies if body.length == 3)
        self.assertEqual(bar.size, (0.08, 0.24, 0.08))
        self.assertAlmostEqual(bar.mass, 0.3)
        parks = {body.park_position[:2] for body in bodies}
        self.assertEqual(len(parks), len(bodies))  # no two bodies parked on the same spot
        self.assertTrue(all(body.park_position[0] < 0 and body.park_position[2] == GROUND_Z + .04
                            for body in bodies))
        self.assertEqual(len(pool_bodies(2, 10)), 30)

    def test_table_and_pads_are_flush_and_disjoint(self):
        layout = example_layout()
        frames = FrameMap(robot_to_near_edge=0.15, table_top=0.0)
        table = table_box(layout, frames)
        self.assertAlmostEqual(table.center[0] - table.size[0] / 2, 0.15)
        self.assertAlmostEqual(table.center[2] + table.size[2] / 2, 0.0)
        staging = ReachableStagingArea(layout, max_piece_length=3)
        for pad in pad_boxes(staging, frames):
            self.assertAlmostEqual(pad.center[2] + pad.size[2] / 2, 0.0)
            self.assertLess(pad.center[0] + pad.size[0] / 2, 0.15)  # in front of the table edge
            self.assertGreater(abs(pad.center[1]) - pad.size[1] / 2, 0.15)  # clear of the robot base


class ReachableStagingTests(unittest.TestCase):
    def setUp(self):
        self.layout = example_layout()
        self.staging = ReachableStagingArea(self.layout, max_piece_length=3)
        self.base = FrameMap().robot_base_in_planner

    def cells(self):
        return [self.staging.position(side, lane, column)
                for side in ("left", "right") for lane in self.staging.lanes
                for column in range(self.staging.columns)]

    def test_cells_stay_off_the_table_and_beside_the_robot(self):
        depth, width = self.staging.cell_footprint
        for x, y, z in self.cells():
            self.assertLessEqual(x + depth / 2, self.staging.near_x - PAD_CLEARANCE + 1e-9)
            self.assertGreaterEqual(abs(y) - width / 2, self.staging.inner_y - width / 2 - 1e-9)
            self.assertLess(np.hypot(x - self.base[0], y - self.base[1]), 0.6)
            self.assertAlmostEqual(z, 0.04)

    def test_pads_cover_their_cells(self):
        depth, width = self.staging.cell_footprint
        for side in ("left", "right"):
            x_min, x_max, y_min, y_max = self.staging.pad_extents(side)
            for lane in self.staging.lanes:
                x, y, _ = self.staging.position(side, lane, 0)
                self.assertTrue(x_min <= x - depth / 2 and x + depth / 2 <= x_max)
                self.assertTrue(y_min <= y - width / 2 and y + width / 2 <= y_max)

    def test_reservations_stack_in_the_colour_lane(self):
        levels = [self.staging.reserve(f"red_{i}", "red", "right").level for i in range(3)]
        self.assertEqual(sorted(levels), [0, 0, 1])  # one column per side: two stacks per colour

    def test_single_cube_maps_get_two_columns(self):
        staging = ReachableStagingArea(example_layout(version=2), max_piece_length=1)
        self.assertEqual(staging.columns, 2)
        first, second = (staging.position("left", "red", column)[0] for column in (0, 1))
        self.assertAlmostEqual(first - second, staging.cell_footprint[0])


class PadPlanner(FakeBackend):
    def _make_staging(self):
        return ReachableStagingArea(self.layout, max_piece_length=3 if self.layout.map_version == 3 else 1)


class PlannerOnPadsTests(unittest.TestCase):
    def test_delete_returns_piece_to_a_pad(self):
        from fake_table_publisher_path import publisher
        script = {"cube_size": 0.08, "table": {"width": 1.21, "depth": 0.680625}}
        queue = LiveMapQueue(pool_per_color=4, report=lambda message: None)
        add = publisher.build_map(script, [{"id": "r1", "color": "red", "x": 0.1, "z": -0.1}], 1, 0)
        queue.accept(header(add, protocol=3), add)
        backend = PadPlanner(queue.layout)
        backend.live_queue = queue
        backend.place_live_cube(queue.start_next().placement)
        queue.complete_active()
        delete = publisher.build_map(script, [], 2, 0)
        queue.accept(header(delete, protocol=3), delete)
        backend.delete_live_cube(queue.start_next().placement)
        destination = backend.transfers[-1][2]
        self.assertLess(destination[0], backend.layout.workspace_center[0] - backend.layout.workspace_depth / 2)
        self.assertGreaterEqual(abs(destination[1]), backend.staging.inner_y - 1e-9)


if __name__ == "__main__":
    unittest.main()
