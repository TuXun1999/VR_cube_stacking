"""Exercise glued-piece bodies and real Genesis IK/trajectories on this machine."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from cube_stack_layout import CubePlacement, parse_table_layout
from cube_stack_piece_geometry import members_at_pose
from vr_json_cube_stacking import ColorCubeDemo, DemoOptions, to_numpy


def empty_map():
    return dict(format="openup.cube_stack.map", version=3,
                coordinate_space="table_local", units="meters", matrix_layout="row_major",
                axes=dict(origin="table_surface_center", x="right", y="up", z="down_screen"),
                cube_size=.08, table=dict(width=1.21, depth=.680625), cubes=[])


def piece(piece_id, length, color, position, yaw=0):
    data = empty_map()
    for index in range(length):
        x = (index - (length - 1) / 2) * .08
        data["cubes"].append(dict(id=f"{piece_id}_{index}", piece_id=piece_id,
            piece_index=index, piece_size=length, color=color, position=[x, .04, 0],
            object_to_table=[.08, 0, 0, x, 0, .08, 0, .04, 0, 0, .08, 0, 0, 0, 0, 1]))
    result = parse_table_layout(data).pieces[0]
    return replace(result, position=tuple(position),
                   quaternion=(float(np.cos(yaw / 2)), 0, 0, float(np.sin(yaw / 2))))


def verify(record=None):
    demo = ColorCubeDemo(DemoOptions(layout=parse_table_layout(empty_map()),
        live=True, pool_per_color=6, show_viewer=False, speed=12,
        record_path=Path(record) if record else None))
    legs = []
    samples = 0
    pick = demo._pick_and_place
    step = demo._step

    def tracked_pick(robot, tool, block, source, target, plan=None):
        legs.append((robot, target.cube_id))
        return pick(robot, tool, block, source, target, plan)

    def sampled_step():
        nonlocal samples
        step()
        for piece_id in demo.blocks:
            item = demo._pool_cube(piece_id)
            if item is None or item.piece is None:
                continue
            members = demo.piece_members(piece_id)
            assert len(members) == item.length
            for a, b in zip(members, members[1:]):
                np.testing.assert_allclose(np.linalg.norm(np.asarray(a.position) - b.position), .08, atol=1e-6)
                np.testing.assert_allclose(a.quaternion, b.quaternion, atol=1e-7)
            samples += 1

    demo._pick_and_place = tracked_pick
    demo._step = sampled_step
    demo._start_recording()

    def check(target):
        block = demo.blocks[target.cube_id]
        assert block.n_links == 1, "A glued piece must have one rigid body link"
        np.testing.assert_allclose(to_numpy(block.get_pos()), target.position, atol=1e-4)
        assert abs(np.dot(to_numpy(block.get_quat()), target.quaternion)) > 1 - 1e-6

    try:
        double = piece("double", 2, "red", (.32, -.24, .04), .3)
        triple = piece("triple", 3, "blue", (.44, .24, .04), -.5)
        for target in (double, triple):
            before = len(legs)
            demo.place_live_cube(target)
            check(target)
            assert len(legs) - before == 1, "One pickup per complete piece"
            print(f"verified: one body, one grasp for {target.length} cubes", flush=True)
        assert len(demo.blocks) == 2
        triple = replace(triple, position=(.4, .22, .12),
                         quaternion=(float(np.cos(.55)), 0, 0, float(np.sin(.55))))
        before = len(legs)
        demo.move_live_cube(triple)
        check(triple)
        assert len(legs) - before == 1
        print("verified: rotated triple stays rigid during a floating-target move", flush=True)
        demo._stage_live_cube("double")
        assert demo._pool_cube("double").staging_slot.length == 2
        demo._restore_staged_cube(double)
        check(double)
        print("verified: full double returns from oriented staging", flush=True)

        upper_position = np.asarray(demo.piece_members("triple")[-1].position) + (0, 0, .08)
        upper = CubePlacement("upper", "green", (0, 0, 0), tuple(upper_position), (1, 0, 0, 0))
        demo.place_live_cube(upper)
        triple = replace(triple, position=(.50, .20, .04))
        demo.move_live_cube(triple)
        check(triple)
        print("verified: a cube over the bar end is staged before the whole triple moves", flush=True)

        returned = demo.blocks["double"]
        demo.delete_live_cube(double)
        assert "double" not in demo.blocks
        fresh = piece("fresh", 2, "red", (.32, -.20, .04), -.4)
        demo.place_live_cube(fresh)
        assert demo.blocks["fresh"] is returned
        check(fresh)
        print("verified: atomic deletion and matching-length reuse", flush=True)

        demo.reset_live_scene()
        crossing = piece("crossing", 3, "red", (.32, .48, .04), .5)
        demo.place_live_cube(crossing)
        crossing = replace(crossing, position=(.34, -.48, .04))
        before = len(legs)
        demo.move_live_cube(crossing)
        check(crossing)
        assert len(legs) - before == 2, "Cross-table triple must exercise the center handoff"
        assert legs[-2][0] is not legs[-1][0]
        print("verified: rotated triple handed off between both actual arms", flush=True)

        # Candidate feasibility is checked before every production reservation.
        # Inspect the whole default rack, including upper tiers, with actual IK.
        reachable = 0
        rejected = 0
        item = demo._pool_cube("crossing")
        for side in ("left", "right"):
            for color in demo.staging.colors:
                for column in range(demo.staging.columns):
                    from cube_stack_staging import StagingSlot
                    for level in (0, 2, 4):
                        slot = StagingSlot(side, color, column, level,
                            demo.staging.position(side, color, column, level), color,
                            3, (2**-.5, 0, 0, 2**-.5))
                        if demo._staging_feasible(item, slot):
                            reachable += 1
                        else:
                            rejected += 1
        assert reachable > 0
        print(f"verified: actual rack IK: {reachable} reachable candidates, {rejected} rejected safely", flush=True)

        before = len(legs)
        try:
            demo.move_live_cube(replace(crossing, position=(4, 4, .04)))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Unreachable target was falsely completed")
        assert len(legs) == before
        check(crossing)
        demo.reset_live_scene()
        assert not demo.blocks and not demo.current_placements
        assert all(item.cube_id is None for item in demo.pool_cubes.values())
        assert demo.carried_tool_to_piece is None
        print(f"PASS: rigid pieces, both arms, staging, blockers, handoff, reuse/reset; {samples} pose samples", flush=True)
    finally:
        demo._stop_recording(settle=False)
        demo.scene.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path)
    verify(parser.parse_args().record)
