"""Whole-bar footprints, measured members, and mixed-size storage."""
from dataclasses import replace
import math
import unittest

import numpy as np

from cube_stack_layout import parse_table_layout
from cube_stack_piece_geometry import footprints_overlap, members_at_pose
from cube_stack_staging import StagingArea
from test_piece_stream import piece_map
from test_staging import layout


class PieceGeometryTests(unittest.TestCase):
    def test_rotated_member_spacing_and_rectangle_edges(self):
        piece = parse_table_layout(piece_map()).pieces[0]
        quat = (math.cos(.4), 0, 0, math.sin(.4))
        members = members_at_pose(piece, (.4, .2, .12), quat)
        for a, b in zip(members, members[1:]):
            self.assertAlmostEqual(np.linalg.norm(np.array(a.position) - b.position), .08)
            self.assertEqual(a.quaternion, b.quaternion)
        self.assertFalse(footprints_overlap(piece, replace(piece, position=(piece.position[0] + .08, 0, .04))))
        self.assertTrue(footprints_overlap(piece, replace(piece, position=(piece.position[0] + .079, 0, .04))))

    def test_mixed_length_slots_are_disjoint_or_stacked_and_outside_table(self):
        area = StagingArea(layout(), max_piece_length=3)
        slots = []
        for index, length in enumerate((3, 2, 1) * 5):
            slot = area.reserve(str(index), "red", "left", length=length)
            piece = parse_table_layout(piece_map(length=length)).pieces[0]
            actual = replace(piece, position=slot.position, quaternion=slot.quaternion)
            self.assertGreater(abs(slot.position[1]) - .04, .605)
            rotation = np.array([[0, -1], [1, 0]])
            self.assertGreaterEqual(slot.position[0] - length * .04, .06 - 1e-8)
            self.assertLessEqual(slot.position[0] + length * .04, .740625 + 1e-8)
            for previous in slots:
                if footprints_overlap(actual, previous):
                    self.assertGreaterEqual(abs(actual.position[2] - previous.position[2]), .08 - 1e-8)
            slots.append(actual)

    def test_reach_filter_never_reserves_an_unreachable_slot(self):
        area = StagingArea(layout(), max_piece_length=3)
        with self.assertRaisesRegex(RuntimeError, "reachable"):
            area.reserve("a", "red", "left", length=3, feasible=lambda _: False)
        self.assertEqual(area.slots, {})
        self.assertTrue(all(not stack for stack in area.stacks.values()))
        slot = area.reserve("a", "red", "left", length=3,
                            feasible=lambda slot: slot.side == "right")
        self.assertEqual(slot.side, "right")
        self.assertEqual(slot.length, 3)

    def test_invalid_lengths_fail_before_mutating_storage(self):
        area = StagingArea(layout(), max_piece_length=3)
        for length in (0, 4, 1.5, True, float("nan")):
            with self.subTest(length=length), self.assertRaises(ValueError):
                area.reserve("bad", "blue", "left", length=length)
        self.assertEqual(area.slots, {})

    def test_bridge_blockers_and_supports_use_the_end_of_a_bar(self):
        from test_live_runner import Entity
        from vr_json_cube_stacking import ColorCubeDemo
        demo = ColorCubeDemo.__new__(ColorCubeDemo)
        bar = replace(parse_table_layout(piece_map()).pieces[0], position=(.4, 0, .04))
        upper = replace(parse_table_layout(piece_map(length=1, piece_id="upper")).pieces[0],
                        position=(.4, .15, .12))
        demo.blocks = {"bar": Entity(bar.position), "upper": Entity(upper.position)}
        demo.current_placements = {"bar": bar, "upper": upper}
        self.assertEqual(demo._stack_blockers({"bar"}), {"bar", "upper"})
        self.assertEqual(demo._destination_occupants(replace(bar, position=(.4, 0, .12))), {"upper"})
        demo.current_placements = {"upper": upper}
        self.assertFalse(demo._ready_to_restore(upper, {"bar": bar, "upper": upper}))


if __name__ == "__main__":
    unittest.main()
