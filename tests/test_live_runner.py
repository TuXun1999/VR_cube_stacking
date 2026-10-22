"""Live scene behavior at the Genesis entity/trajectory boundary."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from cube_stack_layout import CubePlacement, parse_table_layout
from vr_json_cube_stacking import ColorCubeDemo
from cube_stack_stream import LiveMapQueue


class Entity:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.quaternion = np.array((1, 0, 0, 0), dtype=float)

    def get_pos(self):
        return self.position.copy()

    def get_quat(self):
        return self.quaternion.copy()

    def set_pos(self, position, zero_velocity=True):
        self.position = np.array(position, dtype=float)

    def set_quat(self, quaternion, zero_velocity=True):
        self.quaternion = np.array(quaternion, dtype=float)


def cube(cube_id, position, color="blue"):
    return CubePlacement(cube_id, color, (0, 0, 0), tuple(position), (1, 0, 0, 0))


class LiveRunnerTests(unittest.TestCase):
    def setUp(self):
        data = json.loads((Path(__file__).resolve().parents[1] / "example_table_data.json").read_text())
        data["cubes"] = []
        self.demo = ColorCubeDemo.__new__(ColorCubeDemo)
        demo = self.demo
        demo.layout = parse_table_layout(data)
        demo.options = SimpleNamespace(live=True, pool_per_color=4, speed=100,
                                       physical=False)
        demo.blocks = {f"{color}_{i}": Entity((2 + index * .08, 2, .04))
                       for index, (color, i) in enumerate(
                           (color, i) for color in ("red", "green", "blue") for i in range(1, 5))}
        demo.carried_block = demo.carried_tool = demo.carried_quaternion = None
        demo.placed_placements = []
        demo.live_queue = None
        demo.live_client = None
        demo.left_arm = object()
        demo.right_arm = object()
        demo.left_tool = object()
        demo.right_tool = object()
        demo._step = lambda: None
        demo._initialize_poses = lambda: None
        demo._setup_live_pool()
        self.transfers = []

        def transfer(block, placement):
            self.transfers.append((block, tuple(block.get_pos()), placement.position))
            block.set_pos(placement.position)
            block.set_quat(placement.quaternion)

        demo._transfer_block = transfer

    def assert_outside_table(self, position):
        half_width = self.demo.layout.workspace_width / 2
        self.assertGreater(abs(position[1]) - np.sqrt(2) * .04, half_width)
        near_x = self.demo.layout.workspace_center[0] - self.demo.layout.workspace_depth / 2
        self.assertGreaterEqual(position[0], near_x)
        self.assertLessEqual(position[0], near_x + self.demo.layout.workspace_depth)

    def test_adjacent_upper_cube_does_not_obstruct_direct_move(self):
        lower = cube("moving", (.3, .2, .04))
        neighbor = cube("neighbor", (.38, .2, .12))
        for item in (lower, neighbor):
            self.demo.place_live_cube(item)
        self.transfers.clear()
        self.demo.move_live_cube(cube("moving", (.55, .2, .04)))
        self.assertEqual(len(self.transfers), 1)
        np.testing.assert_allclose(self.demo.blocks["neighbor"].get_pos(), neighbor.position)

    def test_recursively_relocated_blocker_is_not_staged_again(self):
        a = cube("a", (.30, .20, .04))
        edge = cube("edge", (.42, .20, .04), "red")
        c = cube("c", (.36, .20, .12))
        support0 = cube("support0", (.70, .20, .04), "green")
        support1 = cube("support1", (.70, .20, .12), "green")
        b = cube("b", (.70, .20, .20))
        for item in (a, edge, c, support0, support1, b):
            self.demo.place_live_cube(item)
        new_a = cube("a", b.position)
        new_b = cube("b", c.position)
        new_c = cube("c", (.30, -.20, .04))
        desired = (new_a, edge, new_c, support0, support1, new_b)
        self.demo.live_queue = SimpleNamespace(layout=replace(self.demo.layout, placements=desired),
                                               realized=dict(self.demo.current_placements))
        self.transfers.clear()
        self.demo.move_live_cube(new_a)
        self.assertEqual(len(self.transfers), 3, "each of a, b, c needs one direct transfer")
        for item in (new_a, new_b, new_c):
            np.testing.assert_allclose(self.demo.blocks[item.cube_id].get_pos(), item.position)

    def test_support_cycle_clears_upper_cube_before_picking_its_support(self):
        a = cube("a", (.50, .20, .04))
        b = cube("b", (.30, .20, .04))
        for item in (a, b):
            self.demo.place_live_cube(item)
        top_a = cube("a", (.30, .20, .12))
        self.demo.move_live_cube(top_a)
        a_block, b_block = self.demo.blocks["a"], self.demo.blocks["b"]
        desired_a = cube("a", b.position)
        desired_b = cube("b", a.position)
        self.demo.live_queue = SimpleNamespace(
            layout=replace(self.demo.layout, placements=(desired_a, desired_b)),
            realized=dict(self.demo.current_placements))
        self.transfers.clear()
        self.demo.move_live_cube(desired_a)
        self.assertIs(self.transfers[0][0], a_block, "clear the upper cube before its support")
        self.assertIs(self.transfers[1][0], b_block)
        self.assertEqual(len(self.transfers), 3)
        np.testing.assert_allclose(a_block.get_pos(), desired_a.position)
        np.testing.assert_allclose(b_block.get_pos(), desired_b.position)

    def test_new_cube_starts_outside_table_before_robot_places_it(self):
        self.demo.place_live_cube(cube("birth_1", (.3, .2, .04)))
        self.assertEqual(len(self.transfers), 1)
        self.assert_outside_table(self.transfers[0][1])
        np.testing.assert_allclose(self.demo.blocks["birth_1"].get_pos(), (.3, .2, .04))

    def test_spawns_and_deleted_returns_stay_on_matching_color_stripes(self):
        # These are the displayed stripe centers, from nearest to farthest.
        stripe_centers = {"red": .686568575, "green": .806568575, "blue": .926568575}
        for color, stripe_y in stripe_centers.items():
            for side in (-1, 1):
                with self.subTest(color=color, side=side):
                    target = cube(f"{color}_{side}", (.3, side * .2, .04), color)
                    self.demo.place_live_cube(target)
                    self.assertAlmostEqual(abs(self.transfers[-1][1][1]), stripe_y, places=6)
                    block = self.demo.blocks[target.cube_id]
                    self.demo.delete_live_cube(target)
                    self.assertAlmostEqual(abs(block.get_pos()[1]), stripe_y, places=6)
                    self.assertAlmostEqual(abs(self.transfers[-1][2][1]), stripe_y, places=6)

    def test_offline_spawn_uses_matching_color_stripe(self):
        self.demo.options.live = False
        stripe_centers = {"red": .686568575, "green": .806568575, "blue": .926568575}
        for color, stripe_y in stripe_centers.items():
            with self.subTest(color=color):
                target = cube(f"offline_{color}", (.3, .2, .04), color)
                self.demo.blocks[target.cube_id] = Entity((2, 2, .04))
                self.demo.place_live_cube(target)
                self.assertAlmostEqual(abs(self.transfers[-1][1][1]), stripe_y, places=6)

    def test_uncovering_returned_cube_keeps_blocker_on_matching_stripe(self):
        demo = self.demo
        demo.blocks = {f"blue_{index}": Entity((2, 2, .04)) for index in range(1, 9)}
        demo._setup_live_pool()
        for item in list(demo.pool_cubes.values())[:7]:
            item.staging_slot = demo.staging.reserve(item.key, item.color, "right")
            item.block.set_pos(item.staging_slot.position)
        buried = demo.pool_cubes["blue_1"]
        self.assertFalse(demo.staging.is_top(buried.key))
        demo._ensure_staging_access(buried)
        self.assertTrue(demo.staging.is_top(buried.key))
        self.assertTrue(self.transfers)
        for _, _, target in self.transfers:
            self.assertAlmostEqual(abs(target[1]), .926568575, places=6)

    def test_move_picks_existing_entity_at_its_actual_pose(self):
        self.demo.place_live_cube(cube("birth_1", (.3, .2, .04)))
        block = self.demo.blocks["birth_1"]
        block.set_pos((.31, .21, .04))
        self.transfers.clear()
        self.demo.move_live_cube(cube("birth_1", (.45, .25, .04)))
        self.assertIs(self.demo.blocks["birth_1"], block)
        self.assertEqual(len(self.transfers), 1, "move must go directly to its destination")
        self.assertEqual(self.transfers[0][1], (.31, .21, .04))
        self.assertEqual(self.transfers[0][2], (.45, .25, .04))
        np.testing.assert_allclose(block.get_pos(), (.45, .25, .04))

    def test_physical_failed_return_does_not_confirm_deletion(self):
        placement = cube("not_returned", (.3, .2, .04))
        self.demo.place_live_cube(placement)
        self.demo.options.physical = True
        self.demo._transfer_block = lambda block, target: None
        with self.assertRaisesRegex(RuntimeError, "Physical return did not remove"):
            self.demo.delete_live_cube(placement)
        self.assertIn("not_returned", self.demo.blocks)
        self.assertIsNotNone(self.demo._pool_cube("not_returned"))

    def test_delete_stays_visible_in_staging_and_is_reused(self):
        original = cube("birth_1", (.3, .2, .04))
        self.demo.place_live_cube(original)
        block = self.demo.blocks["birth_1"]
        self.demo.delete_live_cube(original)
        self.assertNotIn("birth_1", self.demo.blocks)
        staged = tuple(block.get_pos())
        self.assert_outside_table(staged)
        self.transfers.clear()
        self.demo.place_live_cube(cube("birth_2", (.4, .2, .04)))
        self.assertIs(self.demo.blocks["birth_2"], block)
        self.assertEqual(self.transfers[0][1], staged)

    def test_stack_blocker_is_evacuated_before_lower_cube(self):
        lower = cube("lower", (.3, .2, .04))
        upper = cube("upper", (.3, .2, .12))
        for item in (lower, upper):
            self.demo.place_live_cube(item)
        upper_block = self.demo.blocks["upper"]
        new_lower = cube("lower", (.5, .2, .04))
        self.demo.live_queue = SimpleNamespace(
            layout=replace(self.demo.layout, placements=(new_lower, upper)))
        self.transfers.clear()
        self.demo.move_live_cube(new_lower)
        self.assertIs(self.transfers[0][0], upper_block)
        lower_transfers = [transfer for transfer in self.transfers
                           if transfer[0] is self.demo.blocks["lower"]]
        self.assertEqual(len(lower_transfers), 1)
        self.assertEqual(lower_transfers[0][2], new_lower.position)
        np.testing.assert_allclose(self.demo.blocks["lower"].get_pos(), (.5, .2, .04))
        np.testing.assert_allclose(upper_block.get_pos(), (.3, .2, .12))

    def test_swap_temporarily_stages_other_cube_and_moves_requested_cube_directly(self):
        a = cube("a", (.3, .2, .04))
        b = cube("b", (.5, .2, .04))
        for item in (a, b):
            self.demo.place_live_cube(item)
        new_a = cube("a", b.position)
        new_b = cube("b", a.position)
        self.demo.live_queue = SimpleNamespace(
            layout=replace(self.demo.layout, placements=(new_a, new_b)))
        self.transfers.clear()
        self.demo.move_live_cube(new_a)
        self.assertEqual(len(self.transfers), 3)
        self.assertIs(self.transfers[0][0], self.demo.blocks["b"])
        self.assert_outside_table(self.transfers[0][2])
        self.assertIs(self.transfers[1][0], self.demo.blocks["a"])
        self.assertEqual(self.transfers[1][1:], (a.position, b.position))
        np.testing.assert_allclose(self.demo.blocks["a"].get_pos(), b.position)
        np.testing.assert_allclose(self.demo.blocks["b"].get_pos(), a.position)

    def test_pending_add_directly_relocates_changed_cube_still_at_its_target(self):
        old = cube("old", (.3, .2, .04))
        self.demo.place_live_cube(old)
        old_block = self.demo.blocks["old"]
        new = cube("new", old.position)
        changed_old = cube("old", (.5, .2, .04))
        queue = SimpleNamespace(layout=replace(self.demo.layout, placements=(changed_old, new)),
                                realized={"old": old})
        self.demo.live_queue = queue
        self.transfers.clear()
        self.demo.place_live_cube(new)
        self.assertIs(self.transfers[0][0], old_block)
        self.assertEqual(self.transfers[0][2], changed_old.position)
        self.assertEqual(queue.realized["old"].position, changed_old.position)
        np.testing.assert_allclose(self.demo.blocks["new"].get_pos(), new.position)
        self.demo.move_live_cube(changed_old)
        np.testing.assert_allclose(old_block.get_pos(), changed_old.position)

    def test_move_leaves_unrelated_changed_cube_for_its_later_command(self):
        first = cube("first", (.25, .2, .04))
        high = cube("high", (.45, .2, .04))
        for item in (first, high):
            self.demo.place_live_cube(item)
        new_first = cube("first", (.25, .4, .04))
        support = cube("support", (.5, -.2, .04), "red")
        new_high = cube("high", (.5, -.2, .12))
        queue = SimpleNamespace(layout=replace(self.demo.layout, placements=(new_first, support, new_high)),
                                realized={"first": first, "high": high})
        self.demo.live_queue = queue
        self.demo.move_live_cube(new_first)
        np.testing.assert_allclose(self.demo.blocks["high"].get_pos(), high.position)
        self.demo.place_live_cube(support)
        self.demo.move_live_cube(new_high)
        np.testing.assert_allclose(self.demo.blocks["high"].get_pos(), new_high.position)

    def test_group_of_unobstructed_cubes_moves_each_cube_directly(self):
        old = tuple(cube(f"cube_{i}", (.25 + .12 * (i % 2), -.3 + .15 * (i // 2), .04))
                    for i in range(4))
        for item in old:
            self.demo.place_live_cube(item)
        desired = tuple(cube(item.cube_id, (item.position[0] + .35, item.position[1], .04))
                        for item in old)
        self.demo.live_queue = SimpleNamespace(layout=replace(self.demo.layout, placements=desired),
                                               realized={item.cube_id: item for item in old})
        self.transfers.clear()
        for item in desired:
            self.demo.move_live_cube(item)
        self.assertEqual(len(self.transfers), 4, "one direct transfer per cube")
        for item in desired:
            np.testing.assert_allclose(self.demo.blocks[item.cube_id].get_pos(), item.position)

    def test_transfer_executes_validated_joint_path_without_resolving_while_carrying(self):
        demo = self.demo
        demo.left_arm = SimpleNamespace(get_dofs_position=lambda: np.zeros(13))
        demo.right_arm = SimpleNamespace(get_dofs_position=lambda: np.zeros(13))
        demo.left_tool = object()
        demo.right_tool = object()
        solutions = []

        def solve(robot, tool, position, init_qpos=None):
            if len(solutions) >= 7:
                raise RuntimeError("a new IK branch during motion is unstable")
            result = np.zeros(13)
            result[0] = len(solutions) / 10
            solutions.append(result)
            return result

        demo._solve_tool = solve
        demo._move_joints = lambda *args, **kwargs: None
        demo._transfer_block = ColorCubeDemo._transfer_block.__get__(demo)
        block = Entity((-.08, .6, .04))
        target = cube("test", (.3, .3, .04))
        demo._transfer_block(block, target)
        self.assertEqual(len(solutions), 7)
        np.testing.assert_allclose(block.get_pos(), target.position)

    def cross_table_transfer(self, occupied=()):
        """Keep real routing/pick logic; replace only expensive robot commands/IK."""
        demo = self.demo
        demo.left_arm = SimpleNamespace(get_dofs_position=lambda: np.zeros(13))
        demo.right_arm = SimpleNamespace(get_dofs_position=lambda: np.zeros(13))
        demo.left_tool, demo.right_tool = object(), object()

        def solve(robot, tool, position, init_qpos=None):
            # Each arm can reach the center and its own lateral half, but cannot
            # carry a cube all the way to the opposite edge.
            if ((robot is demo.left_arm and position[1] > .25)
                    or (robot is demo.right_arm and position[1] < -.25)):
                raise RuntimeError("outside this arm's reach")
            return np.zeros(13)

        demo._solve_tool = solve
        demo._move_joints = lambda *args, **kwargs: None
        demo._transfer_block = ColorCubeDemo._transfer_block.__get__(demo)
        for index, position in enumerate(occupied):
            demo.blocks[f"occupied_{index}"] = Entity(position)
        block = Entity((.32, .50, .04))
        target = cube("crossing", (.34, -.50, .04))
        legs = []
        actual_pick = demo._pick_and_place

        def pick(robot, tool, item, source, destination, plan=None):
            legs.append((robot, tuple(source), destination.position))
            return actual_pick(robot, tool, item, source, destination, plan)

        demo._pick_and_place = pick
        return block, target, legs

    def test_cross_arm_move_hands_off_on_table_near_center(self):
        block, target, legs = self.cross_table_transfer()
        self.demo._transfer_block(block, target)
        self.assertEqual(len(legs), 2)
        self.assertIsNot(legs[0][0], legs[1][0])
        np.testing.assert_allclose(legs[0][2], (.40031251, 0, .04), atol=1e-6)
        np.testing.assert_allclose(legs[1][1], legs[0][2])
        np.testing.assert_allclose(block.get_pos(), target.position)

    def test_handoff_avoids_occupied_center_without_moving_its_cube(self):
        occupied = (.40031251, 0, .04)
        block, target, legs = self.cross_table_transfer((occupied,))
        self.demo._transfer_block(block, target)
        self.assertEqual(len(legs), 2)
        handoff = np.asarray(legs[0][2])
        self.assertGreater(np.linalg.norm(handoff[:2] - occupied[:2]), .15)
        self.assertLess(np.linalg.norm(handoff[:2] - occupied[:2]), .31)
        self.assertGreater(handoff[0], .06)
        self.assertAlmostEqual(handoff[2], .04)
        np.testing.assert_allclose(self.demo.blocks["occupied_0"].get_pos(), occupied)
        np.testing.assert_allclose(block.get_pos(), target.position)

    def test_full_center_does_not_fall_back_to_placing_behind_robot(self):
        occupied = [(x, y, .04) for x in (.16, .28, .40, .52, .64)
                    for y in (-.36, -.24, -.12, 0, .12, .24, .36)]
        block, target, legs = self.cross_table_transfer(occupied)
        with self.assertRaisesRegex(RuntimeError, "No reachable robot transfer"):
            self.demo._transfer_block(block, target)
        self.assertEqual(legs, [], "validate a safe handoff before picking the cube")
        np.testing.assert_allclose(block.get_pos(), (.32, .50, .04))

    def test_offline_cross_arm_move_does_not_require_a_live_supply_pool(self):
        block, target, legs = self.cross_table_transfer()
        del self.demo.pool_cubes
        self.demo._transfer_block(block, target)
        self.assertEqual(len(legs), 2)
        np.testing.assert_allclose(legs[0][2], (.40031251, 0, .04), atol=1e-6)
        np.testing.assert_allclose(block.get_pos(), target.position)

    def test_unreachable_second_handoff_leg_does_not_start_first_pickup(self):
        block, target, legs = self.cross_table_transfer()
        solve = self.demo._solve_tool

        def unreachable_destination(robot, tool, position, init_qpos=None):
            if position[1] < -.4:
                raise RuntimeError("unreachable final destination")
            return solve(robot, tool, position, init_qpos)

        self.demo._solve_tool = unreachable_destination
        with self.assertRaisesRegex(RuntimeError, "No reachable robot transfer"):
            self.demo._transfer_block(block, target)
        self.assertEqual(legs, [])
        np.testing.assert_allclose(block.get_pos(), (.32, .50, .04))

    def test_high_staging_target_does_not_require_vertical_rise_to_its_height_at_pickup(self):
        target = cube("high_return", (.32, -.6866, .64))
        waypoints = self.demo._transfer_waypoints((.30, -.24, .04), target)
        np.testing.assert_allclose(waypoints[2], (.30, -.24, .27))
        self.assertGreater(waypoints[3][2], waypoints[2][2])
        self.assertGreaterEqual(waypoints[4][2], target.position[2] + .23)

    def test_reset_clears_entities_and_restores_pool_without_rebuilding_scene(self):
        item = cube("birth_1", (.3, .2, .04))
        self.demo.place_live_cube(item)
        block = self.demo.blocks["birth_1"]
        self.demo.carried_block = block
        self.demo.carried_tool = object()
        homes = []
        self.demo._initialize_poses = lambda: homes.append(True)
        self.demo.reset_live_scene()
        self.assertEqual(self.demo.blocks, {})
        self.assertEqual(self.demo.placed_placements, [])
        self.assertIsNone(self.demo.carried_block)
        self.assertIsNone(self.demo.carried_tool)
        self.assertEqual(homes, [True])
        self.assertGreater(block.get_pos()[0], 1)
        self.demo.place_live_cube(cube("after_reset", (.4, .2, .04)))
        self.assertIn("after_reset", self.demo.blocks)

    def test_reset_arriving_in_idle_step_does_not_halt(self):
        demo = self.demo
        queue = LiveMapQueue(report=lambda message: None)
        data = json.loads((Path(__file__).resolve().parents[1] / "example_table_data.json").read_text())
        data.update(cubes=[], revision=0, reset_generation=0)

        def header():
            return dict(stream="CubeStack/Map", content_type="application/json",
                        sequence=1, payload_bytes=0, timestamp=0,
                        metadata=json.dumps(dict(protocol_version=2, session_id="test")))

        queue.accept(header(), data)
        pumps = []

        def pump():
            pumps.append(True)
            if len(pumps) == 2:
                data.update(revision=1, reset_generation=1)
                queue.accept(header(), data)
            elif len(pumps) == 4:
                raise KeyboardInterrupt()

        client = SimpleNamespace(pump=pump, queue=queue, publish_robot_state=lambda: None)
        demo.scene = SimpleNamespace(step=lambda: None)
        demo.step_count = 0
        demo.recording_camera = None
        demo._start_recording = lambda: None
        demo._stop_recording = lambda settle=False: None
        demo._step = ColorCubeDemo._step.__get__(demo)
        with self.assertRaises(KeyboardInterrupt):
            demo.run_live(queue, client)
        self.assertEqual(queue.halt_reason, "KeyboardInterrupt")
        self.assertEqual(queue.reset_generation, 1)
        self.assertFalse(queue.reset_pending)


if __name__ == "__main__":
    unittest.main()
