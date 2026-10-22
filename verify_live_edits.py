"""Exercise actual Genesis pickups, staging returns, cross-arm moves and reset.

Run explicitly; this builds a headless Genesis scene and may take a few minutes.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cube_stack_layout import CubePlacement, parse_table_layout
from cube_stack_stream import LiveMapQueue
from vr_json_cube_stacking import ColorCubeDemo, DemoOptions, ResetRequested, to_numpy


def verify_completed_robot_state(demo, source):
    """Compare protocol acknowledgements with actual finished Genesis work."""
    demo.live_client = demo.live_queue = None
    demo.reset_live_scene()
    queue = LiveMapQueue(report=lambda message: None)
    demo.live_queue = queue
    cube_size = source["cube_size"]

    def record(cube_id, x, z):
        y = cube_size / 2
        return {"id": cube_id, "color": "blue", "position": [x, y, z],
                "object_to_table": [cube_size, 0, 0, x, 0, cube_size, 0, y,
                                    0, 0, cube_size, z, 0, 0, 0, 1]}

    def accept(cubes, revision, generation=0):
        data = dict(source, cubes=cubes, revision=revision, reset_generation=generation)
        header = dict(stream="CubeStack/Map", content_type="application/json", sequence=revision,
                      payload_bytes=len(json.dumps(data).encode("utf-8")), timestamp=0.,
                      metadata=json.dumps(dict(protocol_version=2, session_id="genesis-robot-state")))
        assert queue.accept(header, data) >= 0, queue.halt_reason

    def completed(command, expected, revision, generation=0):
        assert queue.robot_state() is None, "An active operation was acknowledged early"
        queue.complete_active()
        state = queue.robot_state()
        assert state["revision"] == revision and state["reset_generation"] == generation, state
        assert [cube["id"] for cube in state["cubes"]] == [cube["id"] for cube in expected], state
        for actual, wanted in zip(state["cubes"], expected):
            assert actual["color"] == wanted["color"], actual
            np.testing.assert_allclose(actual["position"], wanted["position"], atol=1e-8)
            np.testing.assert_allclose(actual["object_to_table"], wanted["object_to_table"], atol=1e-8)
            placement = queue.realized[actual["id"]]
            np.testing.assert_allclose(to_numpy(demo.blocks[actual["id"]].get_pos()),
                                       placement.position, atol=1e-4)
        print(f"verified: RobotState after actual {command.kind}: revision={revision} "
              f"generation={generation} cubes={len(expected)}", flush=True)

    try:
        first = record("ack_cube", .2, -.05)
        moved = record("ack_cube", -.2, .05)
        accept([first], 1)
        command = queue.start_next()
        # Receive a newer target while the old immutable command is active.
        accept([moved], 2)
        demo.place_live_cube(command.placement)
        completed(command, [first], 2)
        command = queue.start_next()
        demo.move_live_cube(command.placement)
        completed(command, [moved], 2)
        accept([], 3)
        command = queue.start_next()
        demo.delete_live_cube(command.placement)
        completed(command, [], 3)
        accept([], 4, 1)
        assert queue.robot_state() is None, "Reset was acknowledged before the pool reset"
        command = queue.start_next()
        demo.reset_live_scene()
        completed(command, [], 4, 1)
        fresh = record("after_reset_ack", .1, 0.)
        accept([fresh], 5, 1)
        command = queue.start_next()
        demo.place_live_cube(command.placement)
        completed(command, [fresh], 5, 1)
    finally:
        demo.live_queue = None


def main(record=None):
    data = json.loads((Path(__file__).parent / "example_table_data.json").read_text())
    data["cubes"] = []
    layout = parse_table_layout(data)
    demo = ColorCubeDemo(DemoOptions(layout=layout, live=True, pool_per_color=2,
                                    show_viewer=False, speed=12,
                                    record_path=Path(record) if record else None))
    demo._start_recording()
    transfers = []
    actual_transfer = demo._transfer_block

    def tracked_transfer(block, target):
        transfers.append((tuple(to_numpy(block.get_pos())), target.position))
        return actual_transfer(block, target)

    demo._transfer_block = tracked_transfer
    pickup_legs = []
    actual_pick = demo._pick_and_place

    def tracked_pick(robot, tool, block, source, target, plan=None):
        pickup_legs.append((robot, tuple(source), target.position))
        return actual_pick(robot, tool, block, source, target, plan)

    demo._pick_and_place = tracked_pick

    def check_pose(cube_id, position):
        np.testing.assert_allclose(to_numpy(demo.blocks[cube_id].get_pos()), position, atol=1e-4)

    try:
        # Verify actual source and return poses for every stripe on both sides.
        stripe_centers = {"red": .686568575, "green": .806568575, "blue": .926568575}
        for color, stripe_y in stripe_centers.items():
            for side in (-1, 1):
                demo.reset_live_scene()
                sample = CubePlacement(f"stripe_{color}_{side}", color, (0, 0, 0),
                                       (.30, side * .24, .04), (1, 0, 0, 0))
                transfers.clear()
                demo.place_live_cube(sample)
                check_pose(sample.cube_id, sample.position)
                assert abs(transfers[0][0][1] - side * stripe_y) < 1e-6
                returned = demo.blocks[sample.cube_id]
                demo.delete_live_cube(sample)
                assert sample.cube_id not in demo.blocks
                assert abs(to_numpy(returned.get_pos())[1] - side * stripe_y) < 1e-6
                print(f"verified: {color} spawn and deletion return on matching "
                      f"{'left' if side < 0 else 'right'} stripe", flush=True)
        demo.reset_live_scene()
        first = CubePlacement("first", "blue", (0, 0, 0), (.32, .50, .04), (1, 0, 0, 0))
        demo.place_live_cube(first)
        check_pose("first", first.position)
        print("verified: new cube picked from outside-table staging", flush=True)
        moved = replace(first, position=(.45, .26, .04))
        transfers.clear()
        demo.move_live_cube(moved)
        check_pose("first", moved.position)
        assert len(transfers) == 1 and transfers[0][1] == moved.position
        np.testing.assert_allclose(transfers[0][0], first.position, atol=1e-4)
        print("verified: existing cube moved directly old-to-new without staging", flush=True)
        demo.move_live_cube(first)  # Start at the far edge of the opposite arm.
        pickup_legs.clear()
        cross = replace(first, position=(.34, -.50, .04))
        demo.move_live_cube(cross)
        check_pose("first", cross.position)
        assert len(pickup_legs) == 2 and pickup_legs[0][0] is not pickup_legs[1][0]
        np.testing.assert_allclose(pickup_legs[0][2],
                                   (*layout.workspace_center[:2], .04), atol=1e-6)
        np.testing.assert_allclose(pickup_legs[1][1], pickup_legs[0][2], atol=1e-6)
        print(f"verified: wide cross-table handoff at table center {pickup_legs[0][2]}", flush=True)
        center = replace(first, cube_id="center_obstacle", color_name="red",
                         position=(*layout.workspace_center[:2], .04))
        demo.place_live_cube(center)
        pickup_legs.clear()
        demo.move_live_cube(first)
        check_pose("first", first.position)
        check_pose("center_obstacle", center.position)
        assert len(pickup_legs) == 2 and pickup_legs[0][0] is not pickup_legs[1][0]
        handoff = np.asarray(pickup_legs[0][2])
        assert .15 < np.linalg.norm(handoff[:2] - center.position[:2]) < .31
        assert handoff[0] > .06 and abs(handoff[2] - .04) < 1e-6
        print(f"verified: occupied center preserved; nearby clear handoff at {handoff}", flush=True)
        demo.delete_live_cube(center)
        demo.move_live_cube(cross)
        block = demo.blocks["first"]
        demo.delete_live_cube(cross)
        assert "first" not in demo.blocks
        assert abs(to_numpy(block.get_pos())[1]) - np.sqrt(2) * .04 > layout.workspace_width / 2
        assert .06 <= to_numpy(block.get_pos())[0] <= .06 + layout.workspace_depth
        staged = to_numpy(block.get_pos()).copy()
        assert abs(staged[2] - .04) < 1e-6
        print(f"verified: deletion returned cube to staging {staged}", flush=True)
        second = replace(first, cube_id="second", position=(.30, .24, .04))
        demo.place_live_cube(second)
        assert demo.blocks["second"] is block
        check_pose("second", second.position)
        print("verified: staged return reused for new birth", flush=True)
        upper = replace(first, cube_id="upper", position=(.30, .24, .12))
        demo.place_live_cube(upper)
        new_lower = replace(second, position=(.48, .24, .04))
        demo.live_queue = SimpleNamespace(layout=replace(layout, placements=(new_lower, upper)))
        demo.move_live_cube(new_lower)
        demo.live_queue = None
        check_pose("second", new_lower.position)
        check_pose("upper", upper.position)
        print("verified: stack blocker evacuated and restored", flush=True)
        demo.live_queue = SimpleNamespace(halted=False, reset_pending=True)
        demo.live_client = SimpleNamespace(pump=lambda: None)
        try:
            demo._step()
            raise AssertionError("Reset failed to interrupt at step boundary")
        except ResetRequested:
            pass
        demo.reset_live_scene()
        demo.live_queue = demo.live_client = None
        assert not demo.blocks and not demo.staging.slots
        demo.place_live_cube(replace(first, cube_id="after_reset"))
        check_pose("after_reset", first.position)
        print("verified: reset and subsequent placement in same Genesis scene", flush=True)
        # Check floor, full balanced storage, and one temporary uncovering tier
        # without adding ninety bodies or animating the entire supply batch.
        for side in ("left", "right"):
            for color in ("red", "green", "blue"):
                for column in range(3):
                    for level in (0, 4, 5):
                        high = replace(first, position=demo.staging.position(side, color, column, level))
                        arms = ((demo.left_arm, demo.left_tool), (demo.right_arm, demo.right_tool))
                        source = (.30, -.24 if side == "left" else .24, .04)
                        assert any(demo._can_transfer(arm, tool, source, high)
                                   for arm, tool in arms), (side, color, column, high.position)
        print("verified: all lateral staging cells reachable at default balanced 90-return capacity", flush=True)
        verify_completed_robot_state(demo, data)
    finally:
        demo._stop_recording(settle=False)
    print("Genesis live edits: all checks passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", help="optional MP4 output path")
    main(parser.parse_args().record)
