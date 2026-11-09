"""Phase 0 gate for the ToyBlocks backend: grasp/place accuracy and reach.

Run on the Linux GPU machine, from the VR_cube_stacking checkout, with the
ToyBlocks environment active:

    python -u toyblocks_backend/spike.py            # Isaac window
    python -u toyblocks_backend/spike.py --headless --sweep

It builds the same scene as the live receiver, then uses the live backend to
place five pieces exactly as OpenUp would request them: a flat cube, a cube
turned 30 degrees, a cube stacked on the first, a cube touching the side of
that stack, and a glued triple bar turned 30 degrees. Each must settle within
1 cm and 3 degrees. ``--sweep`` also maps
which table-height positions the arm can grasp (IK on top-down grasps) and
reports how much of the OpenUp table each robot-to-table offset would cover.

Exit status 0 means every placement passed. A failure here is the go/no-go
signal from the plan: decide between smaller cubes and a kinematic carry before
building further on physical grasping.
"""
import argparse
import csv
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from toyblocks_backend.frames import DEFAULT_ROBOT_TO_NEAR_EDGE  # noqa: E402

SCRIPT = {"table": {"width": 1.21, "depth": 0.680625}, "cube_size": 0.08}
TRIALS = [
    ("flat red cube", {"id": "r1", "color": "red", "x": 0.0, "z": -0.15}),
    ("green cube turned 30 degrees", {"id": "g1", "color": "green", "x": -0.16, "z": -0.12, "yaw_deg": 30}),
    ("blue cube stacked on the red cube", {"id": "b1", "color": "blue", "x": 0.0, "z": -0.15, "level": 1}),
    ("red cube touching the side of that stack", {"id": "r2", "color": "red", "x": 0.08, "z": -0.15}),
    ("blue triple bar turned 30 degrees",
     {"id": "bar", "color": "blue", "x": 0.22, "z": 0.0, "length": 3, "yaw_deg": 30}),
]
SWEEP_STEP = 0.04


def yaw_degrees(quaternion):
    w, x, y, z = quaternion
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def run_trials(backend, layout_for):
    from cube_stack_receiver import pose_within_tolerance

    failures = 0
    pieces = []
    for number, (name, piece) in enumerate(TRIALS, 1):
        pieces.append(piece)
        layout = layout_for(pieces, number)
        target = next(p for p in layout.pieces if p.piece_id == piece["id"])
        try:
            backend.place_live_cube(target)
        except Exception as error:  # report every trial, then fail the gate
            failures += 1
            print(f"FAIL: {name}: {error}", flush=True)
            continue
        position, quaternion = backend.blocks[target.cube_id].measured()
        error_mm = 1000 * math.dist(position, target.position)
        period = 90.0 if target.length == 1 else 180.0
        yaw_error = abs((yaw_degrees(quaternion) - yaw_degrees(target.quaternion) + period / 2) % period - period / 2)
        ok = pose_within_tolerance(position, quaternion, target.position, target.quaternion, target.length)
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}: {name}: position error {error_mm:.1f} mm, "
              f"yaw error {yaw_error:.1f} deg", flush=True)
    return failures


def sweep(backend, output):
    """IK reach over a robot-frame grid at cube heights, then table coverage per base offset."""
    cube = next(item.block for item in backend.pool_cubes.values() if item.length == 1)
    size = backend._size_of(cube)
    identity = (1.0, 0.0, 0.0, 0.0)
    backend.geo.update_worlds(backend.ctx.block_poses(), backend.ctx.table_pose(), env_ids=backend._ids)
    xs = [round(0.05 + i * SWEEP_STEP, 3) for i in range(int(0.9 / SWEEP_STEP) + 1)]
    ys = [round(-0.76 + i * SWEEP_STEP, 3) for i in range(int(1.52 / SWEEP_STEP) + 1)]
    reach = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["robot_x", "robot_y", "level", "reachable"])
        for level in range(3):
            for x in xs:
                for y in ys:
                    planner = backend.frames.to_planner((x, y, backend.frames.table_top + 0.04 + 0.08 * level))
                    ok = backend._grasp_reachable(tuple(planner), identity, size, update_world=False)
                    reach[x, y, level] = ok
                    writer.writerow([x, y, level, int(ok)])
            done = sum(reach[x, y, level] for x in xs for y in ys)
            print(f"sweep: level {level}: {done}/{len(xs) * len(ys)} grid points reachable", flush=True)
    depth, width = backend.layout.workspace_depth, backend.layout.workspace_width
    cells = [(dx, dy) for dx in xs if dx <= depth for dy in ys if abs(dy) <= width / 2]
    print("coverage of the OpenUp table on the floor level, by --robot-to-near-edge:", flush=True)
    for offset in [round(0.05 + 0.025 * i, 3) for i in range(11)]:
        covered = sum(reach.get((round(min(xs, key=lambda x: abs(x - (offset + dx))), 3), dy, 0), False)
                      for dx, dy in cells)
        print(f"  {offset:.3f} m: {100 * covered / len(cells):.0f}%", flush=True)
    print(f"sweep: wrote {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--robot-to-near-edge", type=float, default=DEFAULT_ROBOT_TO_NEAR_EDGE)
    parser.add_argument("--sweep", action="store_true", help="also map the reachable area")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "toyblocks_reach.csv")
    args = parser.parse_args(argv)

    from toyblocks_backend.app import launch
    launch(headless=args.headless, device=args.device)

    from cube_stack_layout import parse_table_layout
    from fake_table_publisher import build_map
    from toyblocks_backend.live import ToyBlocksBackend

    def layout_for(pieces, revision):
        return parse_table_layout(build_map(SCRIPT, pieces, revision, 0))

    backend = ToyBlocksBackend(layout_for([], 0), SimpleNamespace(
        live=True, pool_per_color=4, speed=1.0, physical=True,
        robot_to_near_edge=args.robot_to_near_edge, device=args.device))
    print("spike: scene ready", flush=True)
    failures = run_trials(backend, layout_for)
    if args.sweep:
        sweep(backend, args.output)
    print(f"spike: {'PASS' if not failures else f'FAIL ({failures} placement(s))'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
