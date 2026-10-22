"""Geometry and allocation behavior for visible table-side staging."""
from collections import Counter
import math
from types import SimpleNamespace
import unittest

from cube_stack_staging import StagingArea


def layout(width=1.21, depth=.680625, near_x=.06):
    return SimpleNamespace(workspace_width=width, workspace_depth=depth,
                           workspace_center=(near_x + depth / 2, 0., 0.))


class StagingTests(unittest.TestCase):
    def test_spawn_and_return_slots_stay_on_their_matching_color_stripe(self):
        area = StagingArea(layout())
        for color_index, color in enumerate(("red", "green", "blue")):
            for index in range(30):
                slot = area.reserve(f"{color}_{index}", color, "left")
                self.assertEqual(slot.lane, color)
                self.assertEqual(slot.color, color)
                self.assertAlmostEqual(abs(slot.position[1]),
                                       .605 + math.sqrt(2) * .04 + .025 + color_index * .12)
                self.assertAlmostEqual(slot.position[0], .08 + slot.column * .12)

    def test_staging_cubes_share_the_construction_surface_height(self):
        area = StagingArea(layout())
        self.assertEqual(area.platform_height, 0.)
        for side in ("left", "right"):
            slot = area.reserve(side, "blue", side)
            self.assertAlmostEqual(slot.position[2], .04)
            self.assertAlmostEqual(area.position(side, "red", 0, 5)[2], .44)

    def test_cells_are_beside_table_with_clearance_for_yawed_cubes(self):
        for width, depth, near_x in ((1.21, .680625, .06), (.8, .5, .12),
                                    (1.8, 1.1, .06)):
            with self.subTest(width=width, depth=depth):
                area = StagingArea(layout(width, depth, near_x))
                half_diagonal = math.sqrt(2) * .04
                positions = []
                for side in ("left", "right"):
                    for lane in area.colors:
                        for column in range(3):
                            x, y, z = area.position(side, lane, column)
                            self.assertGreaterEqual(x, near_x)
                            self.assertLessEqual(x, near_x + depth)
                            self.assertGreaterEqual(abs(y) - half_diagonal,
                                                    width / 2 + .025 - 1e-9)
                            self.assertGreater(y, 0) if side == "right" else self.assertLess(y, 0)
                            self.assertAlmostEqual(z, .04)
                            positions.append((x, y))
                for index, (x, y) in enumerate(positions):
                    for other_x, other_y in positions[index + 1:]:
                        self.assertTrue(abs(x - other_x) >= 2 * half_diagonal or
                                        abs(y - other_y) >= 2 * half_diagonal,
                                        "Yawed staging cubes must not overlap")

    def test_cell_footprint_supports_the_cube_and_stacked_tiers(self):
        area = StagingArea(layout())
        self.assertEqual(tuple(area.lanes), ("red", "green", "blue"))
        self.assertGreaterEqual(min(area.cell_footprint), math.sqrt(2) * .08)
        self.assertAlmostEqual(area.position("right", "red", 0, 4)[2], .36)

    def test_default_ninety_returns_balance_six_cells_per_color(self):
        area = StagingArea(layout())
        slots = [area.reserve(f"{color}_{index}", color, "left")
                 for color in ("red", "green", "blue") for index in range(30)]
        counts = Counter((slot.side, slot.lane, slot.column) for slot in slots)
        self.assertEqual(len(counts), 18)
        self.assertEqual(set(counts.values()), {5})
        self.assertEqual(max(slot.level for slot in slots), 4)
        self.assertTrue(all(slot.color == slot.lane for slot in slots))
        self.assertEqual(slots[0].side, "left")
        self.assertEqual(slots[0].lane, "red")
        self.assertEqual(slots[0].column, 0)

    def test_only_top_cube_can_be_released_from_its_color_lane(self):
        area = StagingArea(layout())
        first = area.reserve("first", "red", "left")
        for index in range(5):
            area.reserve(f"filler_{index}", "red", "right")
        top = area.reserve("top", "red", "left")
        self.assertEqual((top.side, top.lane, top.column),
                         (first.side, first.lane, first.column))
        self.assertFalse(area.is_top("first"))
        self.assertEqual(area.top_above("first"), "top")
        with self.assertRaisesRegex(RuntimeError, "buried"):
            area.release("first")
        area.release("top")
        self.assertTrue(area.is_top("first"))
        area.release("first")
        self.assertNotIn("first", area.slots)

    def test_relocation_excludes_only_the_original_physical_cell(self):
        area = StagingArea(layout())
        previous = area.reserve("moving", "blue", "right")
        self.assertEqual(previous.lane, "blue")
        area.release("moving")
        moved = area.reserve("moving", "blue", previous.side,
                             exclude_column=previous.column,
                             exclude_lane=previous.lane)
        self.assertNotEqual((moved.side, moved.lane, moved.column),
                            (previous.side, previous.lane, previous.column))
        self.assertEqual(moved.color, "blue")

    def test_releasing_blue_keeps_neighboring_red_cube_owned(self):
        area = StagingArea(layout())
        area.reserve("blue_0", "blue", "left")
        area.reserve("red_0", "red", "left")
        area.reserve("blue_1", "blue", "left")
        area.release("blue_1")
        self.assertTrue(area.is_top("blue_0"))
        self.assertTrue(area.is_top("red_0"))
        self.assertEqual(area.stacks["left", "red", 0], ["red_0"])
        self.assertEqual(area.stacks["left", "blue", 0], ["blue_0"])

    def test_clear_releases_all_color_cells_for_fresh_spawns(self):
        area = StagingArea(layout())
        for index in range(25):
            area.reserve(str(index), "green", "right")
        area.clear()
        self.assertEqual(area.slots, {})
        self.assertTrue(all(not stack for stack in area.stacks.values()))
        fresh = area.reserve("fresh", "red", "left")
        self.assertEqual((fresh.side, fresh.lane, fresh.column, fresh.level),
                         ("left", "red", 0, 0))


if __name__ == "__main__":
    unittest.main()
