"""One-body allocation and immutable glued-piece lifecycle at the robot boundary."""
from dataclasses import replace
import unittest

import numpy as np

from cube_stack_layout import parse_table_layout
from cube_stack_staging import StagingArea
import test_live_runner
from test_live_runner import Entity
from test_piece_stream import piece_map
from vr_json_cube_stacking import PoolCube


class PieceRunnerTests(unittest.TestCase):
    def setUp(self):
        fixture = test_live_runner.LiveRunnerTests()
        fixture.setUp()
        self.demo = fixture.demo
        self.transfers = fixture.transfers
        self.demo.staging = StagingArea(self.demo.layout, max_piece_length=3)
        self.demo._staging_feasible = lambda *args: True
        for length in (2, 3):
            key = f"red_L{length}_1"
            self.demo.pool_cubes[key] = PoolCube(key, "red", Entity((2, 2, .04)),
                                                (2, 2, .04), length=length)

    def test_whole_piece_allocates_once_returns_and_reuses_same_shape(self):
        piece = parse_table_layout(piece_map(length=3)).pieces[0]
        piece = replace(piece, position=(.4, .2, .04))
        self.demo.place_live_cube(piece)
        body = self.demo.blocks["bar"]
        self.assertEqual(len(self.transfers), 1)
        self.assertEqual(self.demo._pool_cube("bar").length, 3)
        self.assertEqual(len(self.demo.piece_members("bar")), 3)
        moved = replace(piece, position=(.4, -.2, .12))
        self.demo.move_live_cube(moved)
        self.assertEqual(len(self.transfers), 2)
        self.assertIs(self.demo.blocks["bar"], body)
        self.demo.delete_live_cube(moved)
        self.assertNotIn("bar", self.demo.blocks)
        fresh = parse_table_layout(piece_map(length=3, piece_id="fresh")).pieces[0]
        self.demo.place_live_cube(replace(fresh, position=(.4, .1, .04)))
        self.assertIs(self.demo.blocks["fresh"], body)
        self.demo.reset_live_scene()
        self.assertTrue(all(item.cube_id is None for item in self.demo.pool_cubes.values()))
        self.assertEqual(self.demo.pool_cubes["red_L3_1"].length, 3)

    def test_wrong_shape_and_tilt_fail_before_any_pickup(self):
        triple = parse_table_layout(piece_map(length=3)).pieces[0]
        self.demo.pool_cubes.pop("red_L3_1")
        with self.assertRaisesRegex(RuntimeError, "length 3"):
            self.demo.place_live_cube(triple)
        self.assertEqual(self.transfers, [])
        double = parse_table_layout(piece_map(length=2)).pieces[0]
        with self.assertRaisesRegex(ValueError, "horizontal"):
            self.demo.place_live_cube(replace(double, quaternion=(2**-.5, 2**-.5, 0, 0)))
        self.assertEqual(self.transfers, [])

    def test_piece_diagnostics_use_measured_body_pose(self):
        piece = parse_table_layout(piece_map(length=2)).pieces[0]
        self.demo.place_live_cube(piece)
        self.demo.blocks["bar"].set_pos((.4, .3, .2))
        members = self.demo.piece_members("bar")
        np.testing.assert_allclose(np.mean([m.position for m in members], axis=0), (.4, .3, .2))
        self.assertAlmostEqual(np.linalg.norm(np.array(members[1].position) - members[0].position), .08)
