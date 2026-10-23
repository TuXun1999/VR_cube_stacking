"""Glued map validation and whole-piece commands, without Genesis."""
import copy
import unittest

import numpy as np

from cube_stack_layout import parse_table_layout
from cube_stack_stream import LiveMapQueue
from test_live_stream import cube, snapshot, header


def piece_map(length=3, revision=1, generation=0, piece_id="bar", yaw=0.0):
    members = []
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    for index in range(length):
        position = rotation @ np.array([.05 * (index - (length - 1) / 2), 0, 0])
        position[1] = .025
        member = cube(f"{piece_id}_{index}", *position)
        matrix = np.eye(4)
        matrix[:3, :3] = rotation * .05
        matrix[:3, 3] = position
        member["object_to_table"] = matrix.ravel().tolist()
        member.update(piece_id=piece_id, piece_index=index, piece_size=length)
        members.append(member)
    data = snapshot(*members, revision=revision, generation=generation)
    data["version"] = 3
    return data


class PieceStreamTests(unittest.TestCase):
    def test_one_rigid_piece_preserves_flat_members(self):
        data = piece_map(yaw=.6)
        layout = parse_table_layout(data)
        self.assertEqual(len(layout.placements), 3)
        self.assertEqual(len(layout.pieces), 1)
        piece = layout.pieces[0]
        self.assertEqual(piece.length, 3)
        self.assertEqual(piece.piece_id, "bar")
        np.testing.assert_allclose(piece.position, layout.workspace_center + np.array([0, 0, .04]))

    def test_incomplete_duplicate_bent_tilted_mixed_groups_rejected(self):
        mutations = [lambda d: d["cubes"].pop(),
                     lambda d: d["cubes"][1].update(piece_index=0),
                     lambda d: d["cubes"][1].update(piece_size=2),
                     lambda d: d["cubes"][1].update(color="blue"),
                     lambda d: d["cubes"][1].pop("piece_id"),
                     lambda d: d["cubes"][1]["object_to_table"].__setitem__(3, .01)]
        for mutate in mutations:
            data = piece_map()
            mutate(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_table_layout(data)

    def test_queue_add_move_delete_and_feedback_are_whole_piece(self):
        queue = LiveMapQueue(pool_per_color=3, report=lambda _: None)
        data = piece_map()
        self.assertEqual(queue.accept(header(data, protocol=3), data), 1)
        command = queue.start_next()
        self.assertEqual(command.kind, "add")
        self.assertEqual(command.placement.length, 3)
        self.assertIsNone(queue.robot_state())
        queue.complete_active()
        self.assertEqual(queue.completed_count, 3)
        state = queue.robot_state()
        self.assertEqual(len(state["cubes"]), 3)
        self.assertEqual({c["piece_size"] for c in state["cubes"]}, {3})
        moved = piece_map(revision=2, yaw=.8)
        queue.accept(header(moved, protocol=3), moved)
        self.assertEqual(queue.start_next().kind, "move")
        queue.complete_active()
        empty = piece_map(revision=3)
        empty["cubes"] = []
        queue.accept(header(empty, protocol=3), empty)
        self.assertEqual(queue.start_next().kind, "delete")
        queue.complete_active()
        self.assertEqual(queue.robot_state()["cubes"], [])

    def test_length_mutation_halts_without_changing_accepted_map(self):
        queue = LiveMapQueue(report=lambda _: None)
        data = piece_map()
        queue.accept(header(data, protocol=3), data)
        shorter = piece_map(length=2, revision=2)
        self.assertEqual(queue.accept(header(shorter, protocol=3), shorter), -1)
        self.assertEqual(queue.accepted_count, 3)

    def test_identity_stays_immutable_after_delete_and_rebirth(self):
        queue = LiveMapQueue(report=lambda _: None)
        data = piece_map()
        queue.accept(header(data, protocol=3), data)
        queue.start_next()
        queue.complete_active()
        empty = piece_map(revision=2)
        empty["cubes"] = []
        queue.accept(header(empty, protocol=3), empty)
        queue.start_next()
        queue.complete_active()
        shorter = piece_map(length=2, revision=3)
        self.assertEqual(queue.accept(header(shorter, protocol=3), shorter), -1)
        self.assertEqual(queue.accepted_count, 0)

    def test_measured_pose_feedback_keeps_the_complete_membership(self):
        from dataclasses import replace
        queue = LiveMapQueue(report=lambda _: None)
        data = piece_map(yaw=.4)
        queue.accept(header(data, protocol=3), data)
        active = queue.start_next()
        measured = replace(active.placement, position=(.5, .1, .2))
        queue.complete_active(measured)
        state = queue.robot_state()
        self.assertEqual([c["id"] for c in state["cubes"]], [f"bar_{i}" for i in range(3)])
        positions = [np.array(c["position"]) for c in state["cubes"]]
        self.assertAlmostEqual(np.linalg.norm(positions[1] - positions[0]), .05)
        self.assertAlmostEqual(np.mean(positions, axis=0)[1], .125)

    def test_pool_is_a_constituent_cube_budget(self):
        queue = LiveMapQueue(pool_per_color=2, report=lambda _: None)
        data = piece_map()
        self.assertEqual(queue.accept(header(data, protocol=3), data), -1)
        self.assertIn("pool exhausted", queue.halt_reason)


if __name__ == "__main__":
    unittest.main()
