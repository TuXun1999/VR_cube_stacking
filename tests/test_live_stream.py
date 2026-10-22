"""Desired-state stream behavior without a Genesis installation or robot."""
import copy
import json
import socket
import struct
import time
import unittest

from cube_stack_layout import parse_table_layout
from cube_stack_stream import FrameDecoder, LiveMapQueue, LiveStreamClient, ProtocolError


def cube(cube_id, x=0.0, y=0.025, z=0.0, color="red"):
    result = {
        "color": color,
        "position": [x, y, z],
        "object_to_table": [
            0.05, 0.0, 0.0, x,
            0.0, 0.05, 0.0, y,
            0.0, 0.0, 0.05, z,
            0.0, 0.0, 0.0, 1.0,
        ],
    }
    if cube_id is not None:
        result["id"] = cube_id
    return result


def snapshot(*cubes, revision=1, generation=0):
    return {
        "format": "openup.cube_stack.map",
        "version": 2,
        "coordinate_space": "table_local",
        "units": "meters",
        "matrix_layout": "row_major",
        "axes": {"origin": "table_surface_center", "x": "right", "y": "up", "z": "down_screen"},
        "cube_size": 0.05,
        "table": {"width": 0.8, "depth": 0.6},
        "revision": revision,
        "reset_generation": generation,
        "cubes": list(cubes),
    }


def header(data, protocol=2, session="test-session", sequence=1):
    return {
        "stream": "CubeStack/Map",
        "content_type": "application/json",
        "sequence": sequence,
        "payload_bytes": len(json.dumps(data).encode("utf-8")),
        "timestamp": 1.0,
        "metadata": json.dumps({"protocol_version": protocol, "session_id": session}),
    }


def wire_frame(data, **kwargs):
    payload = json.dumps(data).encode("utf-8")
    encoded_header = json.dumps(header(data, **kwargs)).encode("utf-8")
    return struct.pack("<I", len(encoded_header)) + encoded_header + struct.pack("<I", len(payload)) + payload


class LayoutIdentityTests(unittest.TestCase):
    def test_explicit_ids_survive_middle_deletion(self):
        first = parse_table_layout(snapshot(cube("a"), cube("b", x=0.1), cube("c", x=0.2)))
        after = parse_table_layout(snapshot(cube("a"), cube("c", x=0.2)))
        self.assertEqual([placement.cube_id for placement in first.placements], ["a", "b", "c"])
        self.assertEqual([placement.cube_id for placement in after.placements], ["a", "c"])

    def test_offline_maps_without_ids_keep_legacy_color_serials(self):
        layout = parse_table_layout(snapshot(cube(None), cube(None, color="blue"), cube(None, x=0.1)))
        self.assertEqual([placement.cube_id for placement in layout.placements], ["red_1", "blue_1", "red_2"])

    def test_invalid_or_duplicate_explicit_ids_are_rejected(self):
        for cubes in ((cube("a"), cube("a", x=0.1)), (cube(""),), (cube(7),)):
            with self.subTest(cubes=cubes), self.assertRaises(ValueError):
                parse_table_layout(snapshot(*cubes))

    def test_source_centers_at_table_edges_allow_half_cube_overhang(self):
        for x, z in ((0.4, 0.0), (-0.4, 0.0), (0.0, 0.3), (0.0, -0.3)):
            with self.subTest(x=x, z=z):
                layout = parse_table_layout(snapshot(cube("edge", x=x, z=z)))
                self.assertEqual(layout.placements[0].source_position, (x, 0.025, z))

    def test_source_centers_outside_table_are_rejected(self):
        for x, z in ((0.401, 0.0), (-0.401, 0.0), (0.0, 0.301), (0.0, -0.301)):
            with self.subTest(x=x, z=z), self.assertRaises(ValueError):
                parse_table_layout(snapshot(cube("outside", x=x, z=z)))


class LiveMapQueueTests(unittest.TestCase):
    def setUp(self):
        self.queue = LiveMapQueue(report=lambda message: None)

    def accept(self, data, **kwargs):
        return self.queue.accept(header(data, **kwargs), data)

    def finish_next(self, kind, cube_id):
        command = self.queue.start_next()
        self.assertIsNotNone(command)
        self.assertEqual((command.kind, command.cube_id), (kind, cube_id))
        self.queue.complete_active()
        return command

    def test_completed_add_is_not_repeated_by_heartbeat_or_reconnect(self):
        data = snapshot(cube("a"))
        self.assertEqual(self.accept(data), 1)
        self.assertEqual(self.queue.pending_count, 1)
        self.finish_next("add", "a")
        self.assertEqual(self.accept(data, sequence=27), 0)
        self.assertEqual(self.queue.pending_count, 0)
        self.assertIsNone(self.queue.start_next())
        self.assertEqual(set(self.queue.realized), {"a"})
        self.assertEqual(self.queue.completed_count, 1)

    def test_middle_delete_returns_only_that_stable_id_to_staging(self):
        self.accept(snapshot(cube("a"), cube("b", x=0.1), cube("c", x=0.2)))
        for cube_id in ("a", "b", "c"):
            self.finish_next("add", cube_id)
        self.assertEqual(self.accept(snapshot(cube("a"), cube("c", x=0.2), revision=2)), 1)
        command = self.finish_next("delete", "b")
        self.assertEqual(command.placement.source_position, (0.1, 0.025, 0.0))
        self.assertEqual(set(self.queue.realized), {"a", "c"})
        self.assertIsNone(self.queue.start_next())

    def test_pending_birth_move_coalesces_to_latest_target(self):
        self.accept(snapshot(cube("a")))
        self.accept(snapshot(cube("a", x=0.2), revision=2))
        self.assertEqual(self.queue.pending_count, 1)
        command = self.finish_next("add", "a")
        self.assertEqual(command.placement.source_position, (0.2, 0.025, 0.0))
        self.assertIsNone(self.queue.start_next())

    def test_deleted_pending_birth_never_runs(self):
        self.accept(snapshot(cube("a")))
        self.accept(snapshot(revision=2))
        self.assertEqual(self.queue.accepted_count, 0)
        self.assertEqual(self.queue.pending_count, 0)
        self.assertIsNone(self.queue.start_next())

    def test_move_during_active_birth_waits_for_ack_then_moves(self):
        self.accept(snapshot(cube("a")))
        active = self.queue.start_next()
        self.accept(snapshot(cube("a", x=0.2), revision=2))
        self.assertIs(self.queue.active, active)
        self.assertIsNone(self.queue.start_next())
        self.assertEqual(self.queue.pending_count, 1)
        self.queue.complete_active()
        self.assertEqual(self.queue.realized["a"].source_position, (0.0, 0.025, 0.0))
        command = self.finish_next("move", "a")
        self.assertEqual(command.placement.source_position, (0.2, 0.025, 0.0))
        self.assertEqual(self.queue.completed_count, 1)

    def test_active_birth_deleted_before_ack_is_removed_after_ack(self):
        self.accept(snapshot(cube("a")))
        self.queue.start_next()
        self.accept(snapshot(revision=2))
        self.queue.complete_active()
        self.finish_next("delete", "a")
        self.assertEqual(self.queue.realized, {})

    def test_completed_move_then_new_move_uses_latest_target(self):
        self.accept(snapshot(cube("a")))
        self.finish_next("add", "a")
        self.accept(snapshot(cube("a", x=0.1), revision=2))
        active = self.queue.start_next()
        self.assertEqual(active.kind, "move")
        self.accept(snapshot(cube("a", x=0.2), revision=3))
        self.queue.complete_active()
        command = self.finish_next("move", "a")
        self.assertEqual(command.placement.source_position, (0.2, 0.025, 0.0))
        self.assertEqual(self.queue.realized["a"].source_position, (0.2, 0.025, 0.0))

    def test_reset_interrupts_active_and_discards_old_pending_births(self):
        self.accept(snapshot(cube("a"), cube("b", x=0.1)))
        self.queue.start_next()
        self.accept(snapshot(revision=2, generation=1))
        self.assertTrue(self.queue.reset_pending)
        self.queue.cancel_active_for_reset()
        command = self.finish_next("reset", "reset")
        self.assertEqual(command.reset_generation, 1)
        self.assertFalse(self.queue.reset_pending)
        self.assertEqual(self.queue.realized, {})
        self.assertEqual(self.queue.completed_count, 0)
        self.assertIsNone(self.queue.start_next())

    def test_reset_preserves_birth_received_before_reset_executes(self):
        self.accept(snapshot(cube("a")))
        self.finish_next("add", "a")
        self.accept(snapshot(revision=2, generation=1))
        self.accept(snapshot(cube("new"), revision=3, generation=1))
        self.assertEqual(self.queue.pending_count, 2)
        self.finish_next("reset", "reset")
        self.finish_next("add", "new")
        self.assertEqual(set(self.queue.realized), {"new"})
        self.assertEqual(self.queue.completed_count, 1)

    def test_new_generation_identity_is_independent_of_interrupted_old_body(self):
        self.accept(snapshot(cube("a")))
        self.queue.start_next()
        self.assertEqual(self.accept(snapshot(cube("a", color="blue"), revision=2, generation=1)), 1)
        self.assertEqual(self.accept(snapshot(cube("a", x=0.1, color="blue"), revision=3, generation=1)), 1)
        self.queue.cancel_active_for_reset()
        self.finish_next("reset", "reset")
        command = self.finish_next("add", "a")
        self.assertEqual(command.placement.color_name, "blue")

    def test_new_reset_while_reset_active_is_acknowledged_separately(self):
        self.accept(snapshot(cube("a")))
        self.accept(snapshot(revision=2, generation=1))
        first_reset = self.queue.start_next()
        self.accept(snapshot(cube("new"), revision=3, generation=2))
        self.queue.complete_active()
        self.assertEqual(first_reset.reset_generation, 1)
        self.assertTrue(self.queue.reset_pending)
        second_reset = self.finish_next("reset", "reset")
        self.assertEqual(second_reset.reset_generation, 2)
        self.finish_next("add", "new")

    def test_cancel_without_reset_is_rejected_without_losing_active(self):
        self.accept(snapshot(cube("a")))
        active = self.queue.start_next()
        with self.assertRaises(RuntimeError):
            self.queue.cancel_active_for_reset()
        self.assertIs(self.queue.active, active)

    def test_stale_revision_and_generation_do_not_undo_latest_state(self):
        old = snapshot(cube("a"), revision=5)
        self.accept(old)
        current = snapshot(cube("new"), revision=6, generation=1)
        self.accept(current)
        self.assertEqual(self.accept(old), 0)
        self.assertEqual(self.accept(snapshot(cube("old"), revision=999, generation=0)), 0)
        self.assertFalse(self.queue.halted)
        self.assertEqual([p.cube_id for p in self.queue.layout.placements], ["new"])

    def test_changed_payload_at_same_revision_halts_atomically(self):
        original = snapshot(cube("a"))
        self.accept(original)
        layout = self.queue.layout
        self.assertEqual(self.accept(snapshot(cube("a", x=0.1))), -1)
        self.assertIs(self.queue.layout, layout)
        self.assertEqual(self.queue.accepted_count, 1)

    def test_invalid_snapshot_does_not_partially_accept_valid_birth(self):
        self.accept(snapshot(cube("a")))
        layout = self.queue.layout
        self.assertEqual(self.accept(snapshot(cube("a"), cube("b", x=0.1), cube("bad", x=9.0), revision=2)), -1)
        self.assertIs(self.queue.layout, layout)
        self.assertEqual(self.queue.pending_count, 1)
        self.assertEqual(self.queue.realized, {})

    def test_live_v2_requires_unique_stable_string_ids(self):
        for cubes in ((cube(None),), (cube("a"), cube("a", x=0.1))):
            with self.subTest(cubes=cubes):
                queue = LiveMapQueue(report=lambda message: None)
                data = snapshot(*cubes)
                self.assertEqual(queue.accept(header(data), data), -1)
                self.assertIsNone(queue.layout)

    def test_session_and_table_changes_halt_even_after_explicit_reset(self):
        for change in ("session", "table"):
            with self.subTest(change=change):
                queue = LiveMapQueue(report=lambda message: None)
                data = snapshot(cube("a"))
                queue.accept(header(data), data)
                changed = snapshot(revision=2, generation=1)
                if change == "table":
                    changed["table"]["width"] = 1.0
                session = "other" if change == "session" else "test-session"
                self.assertEqual(queue.accept(header(changed, session=session), changed), -1)
                self.assertEqual(queue.accepted_count, 1)

    def test_capacity_is_simultaneous_and_deleted_supply_is_reused_first(self):
        self.queue = LiveMapQueue(pool_per_color=1, report=lambda message: None)
        self.accept(snapshot(cube("a")))
        self.finish_next("add", "a")
        self.assertEqual(self.accept(snapshot(cube("b", x=0.1), revision=2)), 2)
        self.finish_next("delete", "a")
        self.finish_next("add", "b")
        self.assertEqual(self.queue.reserved["red"], 1)
        before = self.queue.layout
        self.assertEqual(self.accept(snapshot(cube("b", x=0.1), cube("c", x=0.2), revision=3)), -1)
        self.assertIs(self.queue.layout, before)

    def test_deletions_are_top_down_and_additions_bottom_up(self):
        self.accept(snapshot(cube("high", y=0.075), cube("low")))
        self.finish_next("add", "low")
        self.finish_next("add", "high")
        self.accept(snapshot(cube("new-high", y=0.075, x=0.1), cube("new-low", x=0.1), revision=2))
        self.finish_next("delete", "high")
        self.finish_next("delete", "low")
        self.finish_next("add", "new-low")
        self.finish_next("add", "new-high")

    def test_new_support_is_added_before_existing_cube_moves_onto_it(self):
        self.accept(snapshot(cube("a")))
        self.finish_next("add", "a")
        self.accept(snapshot(cube("a", x=0.1, y=0.075), cube("support", x=0.1), revision=2))
        self.finish_next("add", "support")
        self.finish_next("move", "a")
        self.assertEqual(set(self.queue.realized), {"a", "support"})

    def test_adds_and_moves_at_equal_height_keep_producer_order(self):
        self.accept(snapshot(cube("a")))
        self.finish_next("add", "a")
        self.accept(snapshot(cube("new", x=0.1), cube("a", x=0.2), revision=2))
        self.finish_next("add", "new")
        self.finish_next("move", "a")

    def test_stable_identity_cannot_change_color(self):
        self.accept(snapshot(cube("a")))
        self.assertEqual(self.accept(snapshot(cube("a", color="blue"), revision=2)), -1)
        self.assertEqual(self.queue.layout.placements[0].color_name, "red")

    def test_revision_and_generation_require_nonnegative_integers(self):
        for field in ("revision", "reset_generation"):
            for value in (-1, True, 1.2, None):
                with self.subTest(field=field, value=value):
                    queue = LiveMapQueue(report=lambda message: None)
                    data = snapshot(cube("a"))
                    data[field] = value
                    self.assertEqual(queue.accept(header(data), data), -1)
                    self.assertIsNone(queue.layout)

    def test_legacy_protocol_keeps_append_only_synthetic_ids(self):
        first = snapshot(cube(None))
        first.pop("revision")
        first.pop("reset_generation")
        self.assertEqual(self.accept(first, protocol=1), 1)
        self.finish_next("add", "red_1")
        second = copy.deepcopy(first)
        second["cubes"].append(cube(None, x=0.1))
        self.assertEqual(self.accept(second, protocol=1), 1)
        self.finish_next("add", "red_2")
        self.assertEqual(self.accept(first, protocol=1), -1)


class FrameAndReconnectTests(unittest.TestCase):
    def test_fragmented_and_coalesced_v2_frames_decode_without_duplication(self):
        data = snapshot(cube("a"))
        wire = wire_frame(data)
        decoder = FrameDecoder()
        frames = []
        for index in range(0, len(wire), 7):
            frames.extend(decoder.feed(wire[index:index + 7]))
        frames.extend(decoder.feed(wire_frame(data, sequence=2)))
        self.assertEqual(len(frames), 2)
        queue = LiveMapQueue(report=lambda message: None)
        self.assertEqual([queue.accept(*frame) for frame in frames], [1, 0])
        self.assertEqual(queue.pending_count, 1)

    def test_payload_size_mismatch_is_rejected(self):
        data = snapshot()
        bad_header = header(data, protocol=1)
        bad_header["payload_bytes"] += 1
        encoded = json.dumps(bad_header).encode("utf-8")
        payload = json.dumps(data).encode("utf-8")
        wire = struct.pack("<I", len(encoded)) + encoded + struct.pack("<I", len(payload)) + payload
        with self.assertRaises(ProtocolError):
            FrameDecoder().feed(wire)

    def test_oversized_frame_is_rejected_before_body_arrives(self):
        with self.assertRaises(ProtocolError):
            FrameDecoder().feed(struct.pack("<I", 64 * 1024 * 1024 + 1))

    def test_real_socket_reconnect_replays_latest_map_once(self):
        queue = LiveMapQueue(report=lambda message: None)
        data = snapshot(cube("a"))
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(2.0)
        client = LiveStreamClient(queue, port=listener.getsockname()[1], retry_interval=0, report=lambda message: None)
        try:
            for sequence in (1, 2):
                client.pump()
                peer, _ = listener.accept()
                try:
                    peer.sendall(wire_frame(data, sequence=sequence))
                    deadline = time.monotonic() + 2.0
                    while time.monotonic() < deadline:
                        client.pump()
                        if queue.accepted_count == 1 or queue.halted:
                            break
                    self.assertFalse(queue.halted, queue.halt_reason)
                    self.assertEqual(queue.accepted_count, 1)
                    if sequence == 1:
                        command = queue.start_next()
                        self.assertEqual((command.kind, command.cube_id), ("add", "a"))
                        queue.complete_active()
                    else:
                        # Consume the replay even if accepted_count was already one.
                        for _ in range(3):
                            client.pump()
                        self.assertEqual(queue.pending_count, 0)
                        self.assertEqual(queue.completed_count, 1)
                finally:
                    peer.close()
                    client.close()
        finally:
            client.close()
            listener.close()


if __name__ == "__main__":
    unittest.main()
