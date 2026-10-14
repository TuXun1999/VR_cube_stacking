import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from cube_stack_layout import (
    CubePlacement, TableLayout, load_table_layout, parse_table_layout,
    rotation_matrix_to_quaternion,
)

gs = None  # Genesis is loaded only when a simulation is actually constructed.

# Paths and simulation settings.
SCRIPT_DIR = Path(__file__).resolve().parent
URDF_PATH = SCRIPT_DIR / "zeus_description" / "ur5e_robotiq_2f85.urdf"
DEFAULT_MAP_PATH = SCRIPT_DIR / "example_table_data.json"
DEFAULT_VIDEO_PATH = SCRIPT_DIR / "outputs" / "vr_json_cube_stacking.mp4"
SIM_DT = 0.01
TARGET_CUBE_SIZE = 0.08

CUBE_SIZE = (TARGET_CUBE_SIZE,) * 3
LEFT_SPAWN_POSITION = (0.30, -0.42, CUBE_SIZE[2] / 2)
RIGHT_SPAWN_POSITION = (0.30, 0.58, CUBE_SIZE[2] / 2)
PARKED_BLOCK_ORIGIN = (2.0, 2.0, CUBE_SIZE[2] / 2)
SPAWN_SETTLE_STEPS = 20
TABLE_LINE_WIDTH = 0.004
TABLE_LINE_HEIGHT = 0.003
COLOR_RGB = {
    "red": (0.90, 0.10, 0.10),
    "green": (0.10, 0.75, 0.20),
    "blue": (0.10, 0.35, 0.95),
}

# Rack and camera layout.
RACK_HEIGHT = 0.8
RACK_SIZE = (0.15, 0.4, RACK_HEIGHT)
RACK_MOUNT_Z = RACK_HEIGHT * 0.8
RACK_POS = (-RACK_SIZE[0] / 2, 0.0, RACK_HEIGHT / 2)
VIEWER_POS = (3.2, -2.2, 1.8)
VIEWER_LOOKAT = (0.40, 0.0, 0.35)
RECORDING_CAMERA_POS = (3.0, -2.1, 1.65)
RECORDING_CAMERA_LOOKAT = (0.40, 0.0, 0.35)

# Robot controller settings.
ARM_DOFS = np.arange(6)
GRIPPER_DOF = np.arange(6, 7)
ARM_KP = np.array([2500, 2400, 2000, 1400, 900, 700])
ARM_KV = np.array([220, 210, 180, 120, 80, 60])
ARM_FORCE_MIN = np.array([-150, -150, -150, -28, -28, -28])
ARM_FORCE_MAX = np.array([150, 150, 150, 28, 28, 28])
# Both tools remain vertical. The different wrist yaw values retain the stable
# grasp configuration established for each side of the dual-arm setup.
LEFT_TOOL_QUAT = np.array([0.0, 2**-0.5, 2**-0.5, 0.0])
RIGHT_TOOL_QUAT = np.array([0.0, 2**-0.5, -(2**-0.5), 0.0])
GRASP_OFFSET = np.array([0.0, 0.0, 0.14])
HOVER_OFFSET = GRASP_OFFSET + np.array([0.0, 0.0, 0.09])
# Keep the drop short so tall towers do not accumulate lateral drift.
RELEASE_CLEARANCE = np.array([0.0, 0.0, 0.005])
STACK_RELEASE_CLEARANCE = np.array([0.0, 0.0, 0.015])
GRIPPER_OPEN = 0.0
# An 8 cm cube nearly fills the Robotiq 2F-85 opening, so only a small closing
# command is needed. The former 0.56 rad setting was appropriate for 5 cm cubes.
GRIPPER_CLOSED = 0.10

# Stable pre-grasp homes that avoid the singular configurations used originally.
LEFT_ARM_HOME_QPOS = np.array([-0.7165, -1.0068, 1.4604, -0.4533, 2.4254, 0.0])
RIGHT_ARM_HOME_QPOS = np.array([0.2647, -1.0065, 1.4600, -0.4546, 0.2655, -3.1406])


@dataclass(frozen=True)
class DemoOptions:
    layout: TableLayout
    show_viewer: bool = True
    record_path: Path | None = None
    video_fps: int = 30
    speed: float = 1.25
    physical: bool = False
    live: bool = False
    pool_per_color: int = 30

    def validate(self):
        if self.speed <= 0:
            raise ValueError("speed must be greater than zero")
        if self.video_fps <= 0:
            raise ValueError("video_fps must be greater than zero")
        if self.pool_per_color <= 0:
            raise ValueError("pool_per_color must be greater than zero")
        if not self.live and not self.layout.placements:
            raise ValueError("The VR table map does not contain any cubes.")



def to_numpy(value):
    """Convert Genesis/Torch values to NumPy arrays."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class ColorCubeDemo:
    """Rebuild a transformed VR cube layout with the dual-arm robot."""

    def __init__(self, options: DemoOptions):
        options.validate()
        self.options = options
        self.layout = options.layout
        self.carried_block = None
        self.carried_tool = None
        self.carried_quaternion = None
        self.live_client = None
        self.live_queue = None

        global gs
        if gs is None:
            import genesis as gs_module
            gs = gs_module

        # A fixed seed makes IK behavior deterministic.
        gs.init(backend=gs.cpu, seed=0)

        self.scene = self._create_scene()
        self.blocks = self._add_blocks()
        self.placed_placements = []
        self.left_arm, self.right_arm = self._add_rack_and_arms()
        self.recording_camera = self._add_recording_camera()
        self.scene.build()

        self.left_tool = self.left_arm.get_link("tool0")
        self.right_tool = self.right_arm.get_link("tool0")
        self.render_stride = max(1, round(1.0 / (SIM_DT * options.video_fps)))
        self.step_count = 0

        self._configure_arm(self.left_arm)
        self._configure_arm(self.right_arm)
        self._initialize_poses()

    def _create_scene(self):
        return gs.Scene(
            sim_options=gs.options.SimOptions(dt=SIM_DT),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=VIEWER_POS,
                camera_lookat=VIEWER_LOOKAT,
                camera_fov=30,
            ),
            renderer=gs.renderers.Rasterizer(),
            show_viewer=self.options.show_viewer,
        )

    def _add_blocks(self):
        self.scene.add_entity(gs.morphs.Plane())
        self._add_workspace_table()

        # Every JSON cube is created before scene.build(), then parked outside the
        # workspace until its turn at the color-specific feeder.
        material = gs.materials.Rigid(rho=200.0, friction=2.0)
        blocks = {}
        placements = self.layout.placements
        if self.options.live:
            placements = tuple(
                CubePlacement(f"{color}_{serial}", color, (0, 0, 0), (0, 0, 0), (1, 0, 0, 0))
                for color in COLOR_RGB
                for serial in range(1, self.options.pool_per_color + 1)
            )
        for index, placement in enumerate(placements):
            parked_position = (
                PARKED_BLOCK_ORIGIN[0] + index * 0.08,
                PARKED_BLOCK_ORIGIN[1],
                PARKED_BLOCK_ORIGIN[2],
            )
            blocks[placement.cube_id] = self.scene.add_entity(
                gs.morphs.Box(
                    size=CUBE_SIZE,
                    pos=parked_position,
                    fixed=not self.options.physical,
                    collision=self.options.physical,
                ),
                material=material,
                surface=gs.surfaces.Rough(color=COLOR_RGB[placement.color_name]),
            )
        return blocks

    def _add_workspace_table(self):
        """Draw the uniformly scaled VR table footprint and its center axes."""
        center_x, center_y, _ = self.layout.workspace_center
        half_depth = self.layout.workspace_depth / 2
        half_width = self.layout.workspace_width / 2
        x_min, x_max = center_x - half_depth, center_x + half_depth
        y_min, y_max = center_y - half_width, center_y + half_width
        table_surface = gs.surfaces.Rough(color=(0.12, 0.18, 0.24))
        line_surface = gs.surfaces.Rough(color=(0.12, 0.65, 0.95))
        self.scene.add_entity(
            gs.morphs.Box(
                size=(self.layout.workspace_depth, self.layout.workspace_width, 0.001),
                pos=(center_x, center_y, 0.0005),
                fixed=True,
                collision=False,
            ),
            surface=table_surface,
        )

        for x in (x_min, x_max):
            self.scene.add_entity(
                gs.morphs.Box(
                    size=(TABLE_LINE_WIDTH, y_max - y_min, TABLE_LINE_HEIGHT),
                    pos=(x, center_y, TABLE_LINE_HEIGHT / 2),
                    fixed=True,
                    collision=False,
                ),
                surface=line_surface,
            )
        for y in (y_min, y_max):
            self.scene.add_entity(
                gs.morphs.Box(
                    size=(x_max - x_min, TABLE_LINE_WIDTH, TABLE_LINE_HEIGHT),
                    pos=(center_x, y, TABLE_LINE_HEIGHT / 2),
                    fixed=True,
                    collision=False,
                ),
                surface=line_surface,
            )

        # Crosshairs mark the transformed VR table-frame origin.
        axis_length = 0.08
        self.scene.add_entity(
            gs.morphs.Box(
                size=(axis_length, TABLE_LINE_WIDTH, TABLE_LINE_HEIGHT),
                pos=(center_x, center_y, TABLE_LINE_HEIGHT),
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Rough(color=(0.95, 0.25, 0.20)),
        )
        self.scene.add_entity(
            gs.morphs.Box(
                size=(TABLE_LINE_WIDTH, axis_length, TABLE_LINE_HEIGHT),
                pos=(center_x, center_y, TABLE_LINE_HEIGHT),
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Rough(color=(0.20, 0.85, 0.30)),
        )

    def _add_rack_and_arms(self):
        # The rack is visual-only so the arms cannot catch it during a transfer.
        self.scene.add_entity(
            gs.morphs.Box(
                size=RACK_SIZE,
                pos=RACK_POS,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Rough(color=(0.5, 0.5, 0.5)),
        )

        # Rotating about X points each base mounting normal away from the rack.
        left_arm = self._add_arm(y=-RACK_SIZE[1] / 2, x_rotation=90)
        right_arm = self._add_arm(y=RACK_SIZE[1] / 2, x_rotation=-90)
        return left_arm, right_arm

    def _add_arm(self, y, x_rotation):
        return self.scene.add_entity(
            gs.morphs.URDF(
                file=str(URDF_PATH),
                pos=(RACK_POS[0], y, RACK_MOUNT_Z),
                euler=(x_rotation, 0, 0),
                fixed=True,
                merge_fixed_links=False,
                links_to_keep=["tool0", "robotiq_arg2f_base_link"],
            )
        )

    def _add_recording_camera(self):
        if self.options.record_path is None:
            return None

        return self.scene.add_camera(
            res=(960, 540),
            pos=RECORDING_CAMERA_POS,
            lookat=RECORDING_CAMERA_LOOKAT,
            fov=35,
            GUI=False,
        )

    @staticmethod
    def _configure_arm(arm):
        """Apply identical arm and gripper controller gains."""
        arm.set_dofs_kp(ARM_KP, dofs_idx_local=ARM_DOFS)
        arm.set_dofs_kv(ARM_KV, dofs_idx_local=ARM_DOFS)
        arm.set_dofs_force_range(
            ARM_FORCE_MIN,
            ARM_FORCE_MAX,
            dofs_idx_local=ARM_DOFS,
        )
        arm.set_dofs_kp(np.array([12.0]), dofs_idx_local=GRIPPER_DOF)
        arm.set_dofs_kv(np.array([3.5]), dofs_idx_local=GRIPPER_DOF)

    def _initialize_poses(self):
        """Place both arms at home and actively hold both grippers open."""
        self.left_arm.set_qpos(np.concatenate([LEFT_ARM_HOME_QPOS, np.zeros(6)]))
        self.right_arm.set_qpos(np.concatenate([RIGHT_ARM_HOME_QPOS, np.zeros(6)]))
        self.left_arm.control_dofs_position(LEFT_ARM_HOME_QPOS, dofs_idx_local=ARM_DOFS)
        self.right_arm.control_dofs_position(
            RIGHT_ARM_HOME_QPOS, dofs_idx_local=ARM_DOFS
        )
        for arm in (self.left_arm, self.right_arm):
            arm.control_dofs_position([GRIPPER_OPEN], dofs_idx_local=GRIPPER_DOF)
        self.scene.step()

    def _step(self):
        if self.live_client is not None:
            self.live_client.pump()
            queue = self.live_queue if getattr(self, "live_queue", None) is not None else self.live_client.queue
            if queue.halted:
                raise RuntimeError(f"HALTED: {queue.halt_reason}")
        self.scene.step()
        self.step_count += 1
        if self.carried_block is not None:
            cube_position = to_numpy(self.carried_tool.get_pos()) - GRASP_OFFSET
            self.carried_block.set_pos(cube_position, zero_velocity=True)
            self.carried_block.set_quat(
                self.carried_quaternion,
                zero_velocity=True,
            )
        if (
            self.recording_camera is not None
            and self.step_count % self.render_stride == 0
        ):
            self.recording_camera.render()

    def keep_viewer_alive(self):
        """Advance the settled scene while the final comparison remains open."""
        self.scene.step()

    def _scaled_steps(self, steps, allow_zero=False):
        minimum = 0 if allow_zero else 1
        return max(minimum, round(steps / self.options.speed))

    def _move_joints(self, robot, target, dofs, steps, settle=30):
        """Follow a minimum-jerk joint trajectory and settle at its endpoint."""
        start = to_numpy(robot.get_dofs_position())[dofs]
        target = np.asarray(target)
        steps = self._scaled_steps(steps)
        settle = self._scaled_steps(settle, allow_zero=True)

        for time_fraction in np.linspace(0.0, 1.0, steps + 1)[1:]:
            blend = 10 * time_fraction**3 - 15 * time_fraction**4 + 6 * time_fraction**5
            waypoint = start + blend * (target - start)
            # Only drive the requested joints. The other gripper joints are mimic
            # joints and must remain free to follow the primary actuator.
            robot.control_dofs_position(waypoint, dofs_idx_local=dofs)
            self._step()

        for _ in range(settle):
            robot.control_dofs_position(target, dofs_idx_local=dofs)
            self._step()

    def _move_tool(self, robot, tool, position, steps, settle=30):
        """Solve seeded arm-only IK, validate it, and move to the result."""
        tool_quat = (
            LEFT_TOOL_QUAT if robot is self.left_arm else RIGHT_TOOL_QUAT
        )
        qpos, error = robot.inverse_kinematics(
            link=tool,
            pos=np.asarray(position),
            quat=tool_quat,
            init_qpos=to_numpy(robot.get_dofs_position()),
            dofs_idx_local=ARM_DOFS,
            max_samples=1,
            max_solver_iters=100,
            damping=0.03,
            max_step_size=0.15,
            return_error=True,
        )

        error = np.abs(to_numpy(error))
        qpos = to_numpy(qpos)
        if (not np.all(np.isfinite(error)) or not np.all(np.isfinite(qpos))
                or np.max(error[:3]) > 0.003 or np.max(error[3:]) > 0.015):
            raise RuntimeError(
                f"IK failed for target {position}; pose error was {error}"
            )

        self._move_joints(robot, qpos[:6], ARM_DOFS, steps, settle)

    def _pick_and_place(self, robot, tool, block, pick_position, placement):
        pick_position = np.asarray(pick_position)
        place_position = np.asarray(placement.position)

        # Approach and retreat vertically; transfer only after the object is clear.
        self._move_tool(robot, tool, pick_position + HOVER_OFFSET, steps=120)
        self._move_tool(robot, tool, pick_position + GRASP_OFFSET, steps=90)
        self._move_joints(
            robot,
            [GRIPPER_CLOSED],
            GRIPPER_DOF,
            steps=60,
            settle=50,
        )
        if not self.options.physical:
            self.carried_block = block
            self.carried_tool = tool
            self.carried_quaternion = placement.quaternion

        self._move_tool(robot, tool, pick_position + HOVER_OFFSET, steps=100)
        transfer_distance = np.linalg.norm((place_position - pick_position)[:2])
        if transfer_distance > 0.35:
            # A center waypoint keeps long side-to-side moves slow enough that the
            # cube cannot slip from the gripper during peak acceleration.
            transfer_midpoint = (pick_position + place_position) / 2
            transfer_midpoint[2] = max(pick_position[2], place_position[2])
            self._move_tool(
                robot,
                tool,
                transfer_midpoint + HOVER_OFFSET,
                steps=140,
            )
            self._move_tool(robot, tool, place_position + HOVER_OFFSET, steps=140)
        else:
            self._move_tool(robot, tool, place_position + HOVER_OFFSET, steps=180)
        # Release just above the requested pose so the fingertips stay clear of the
        # support while gravity only has to settle the final 5 mm in Genesis.
        clearance = (
            STACK_RELEASE_CLEARANCE
            if place_position[2] > TARGET_CUBE_SIZE
            else RELEASE_CLEARANCE
        )
        release_position = place_position + GRASP_OFFSET + clearance
        self._move_tool(robot, tool, release_position, steps=100)
        if not self.options.physical:
            block.set_pos(place_position, zero_velocity=True)
            block.set_quat(placement.quaternion, zero_velocity=True)
            self.carried_block = None
            self.carried_tool = None
            self.carried_quaternion = None
        self._move_joints(
            robot,
            [GRIPPER_OPEN],
            GRIPPER_DOF,
            steps=60,
            settle=50,
        )
        self._move_tool(robot, tool, place_position + HOVER_OFFSET, steps=100)

    def _park(self, robot, home_qpos):
        self._move_joints(robot, home_qpos, ARM_DOFS, steps=180, settle=30)

    def _start_recording(self):
        if self.recording_camera is None:
            return

        self.options.record_path.parent.mkdir(parents=True, exist_ok=True)
        self.recording_camera.start_recording()
        self.recording_camera.render()

    def _stop_recording(self, settle=True):
        if self.recording_camera is None:
            return

        # Leave the completed color layout on screen briefly before ending the video.
        if settle:
            for _ in range(round(0.75 / SIM_DT)):
                self._step()
        output_path = self.options.record_path.resolve()
        self.recording_camera.stop_recording(
            save_to_filename=str(output_path),
            fps=self.options.video_fps,
        )
        print(f"Video saved to: {output_path}")

    def _arm_for_placement(self, placement):
        """Use the target-side arm to avoid sweeping across the wider workspace."""
        if placement.position[1] >= 0.0:
            return self.right_arm, self.right_tool, RIGHT_ARM_HOME_QPOS
        return self.left_arm, self.left_tool, LEFT_ARM_HOME_QPOS

    def _spawn_block(self, block, placement, robot):
        """Move the next parked block to the selected arm's side feeder."""
        spawn_position = (
            RIGHT_SPAWN_POSITION if robot is self.right_arm else LEFT_SPAWN_POSITION
        )
        block.set_pos(spawn_position, zero_velocity=True)
        block.set_quat(placement.quaternion, zero_velocity=True)
        for _ in range(self._scaled_steps(SPAWN_SETTLE_STEPS)):
            self._step()
        return spawn_position

    def _place_colored_cubes(self):
        """Spawn and place every transformed JSON cube from bottom to top."""
        for placement in self.layout.placements:
            self.place_live_cube(placement)

    def place_live_cube(self, placement):
        """Execute one arbitrary imported pose; the caller owns queue ordering."""
        try:
            robot, tool, home = self._arm_for_placement(placement)
            block = self.blocks[placement.cube_id]
            spawn_position = self._spawn_block(block, placement, robot)

            self._pick_and_place(
                robot,
                tool,
                block,
                spawn_position,
                placement,
            )
            self._park(robot, home)
            self.placed_placements.append(placement)
        finally:
            # Never allow a failed partial move to keep moving a carried cube.
            self.carried_block = None
            self.carried_tool = None
            self.carried_quaternion = None

    def run_live(self, queue, client, max_cubes=0):
        """Only this outer loop starts commands; _step only receives/enqueues."""
        self.live_client = client
        self.live_queue = queue
        self._start_recording()
        print(f"ready: session={queue.session_id} pool_per_color={self.options.pool_per_color}", flush=True)
        try:
            while not max_cubes or queue.completed_count < max_cubes:
                client.pump()
                if queue.halted:
                    raise RuntimeError(f"HALTED: {queue.halt_reason}")
                placement = queue.start_next()
                if placement is None:
                    self._step()
                    time.sleep(SIM_DT)
                    continue
                self.place_live_cube(placement)
                queue.complete_active()
        except BaseException as error:
            queue.fail_active(str(error) or type(error).__name__)
            self.carried_block = self.carried_tool = self.carried_quaternion = None
            raise
        finally:
            # Saving a failed run must not advance the scene or issue commands.
            self._stop_recording(settle=False)
        print(f"live complete: accepted={queue.accepted_count} completed={queue.completed_count}", flush=True)

    def _print_final_positions(self):
        print("VR cubes placed. Target -> final position (meters):")
        for placement in self.layout.placements:
            block = self.blocks[placement.cube_id]
            actual = to_numpy(block.get_pos())
            error = np.linalg.norm(actual - np.asarray(placement.position))
            print(
                f"  {placement.cube_id} ({placement.color_name}): "
                f"{np.asarray(placement.position)} -> {actual}; error={error:.4f}"
            )

    def run(self):
        self._start_recording()
        self._place_colored_cubes()
        self._stop_recording()
        self._print_final_positions()


def main(
    map_path=DEFAULT_MAP_PATH,
    show_viewer=True,
    record_path=None,
    video_fps=30,
    speed=1.25,
    physical=False,
):
    layout = load_table_layout(map_path)
    print(f"Loaded VR table map: {layout.source_path}")
    print(
        f"Scale: {layout.scale:.6f}; cube: {layout.source_cube_size:.3f} m -> "
        f"{TARGET_CUBE_SIZE:.3f} m; workspace table: "
        f"{layout.workspace_width:.3f} x {layout.workspace_depth:.3f} m"
    )
    options = DemoOptions(
        layout=layout,
        show_viewer=show_viewer,
        record_path=Path(record_path).expanduser() if record_path is not None else None,
        video_fps=video_fps,
        speed=speed,
        physical=physical,
    )
    demo = ColorCubeDemo(options)
    demo.run()
    return demo


def live_main(host="127.0.0.1", port=9036, pool_per_color=30, receive_only=False,
              max_cubes=0, timeout=0, show_viewer=True, record_path=None,
              video_fps=30, speed=1.25, physical=False):
    from cube_stack_stream import LiveMapQueue, LiveStreamClient

    def report(message):
        print(message, flush=True)

    queue = LiveMapQueue(pool_per_color=pool_per_color, report=report)
    client = LiveStreamClient(queue, host=host, port=port, report=report)
    deadline = time.monotonic() + timeout if timeout else None
    try:
        print(f"waiting: CubeStack/Map at {host}:{port}", flush=True)
        while receive_only or queue.layout is None:
            client.pump()
            if queue.halted:
                raise RuntimeError(f"HALTED: {queue.halt_reason}")
            if receive_only and queue.layout is not None and max_cubes and queue.accepted_count >= max_cubes:
                print(f"receive-only complete: session={queue.session_id} accepted={queue.accepted_count} completed=0", flush=True)
                return queue
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Timed out receiving the requested live map/cube count")
            time.sleep(0.01)
        options = DemoOptions(
            layout=queue.layout, live=True, pool_per_color=pool_per_color,
            show_viewer=show_viewer,
            record_path=Path(record_path).expanduser() if record_path is not None else None,
            video_fps=video_fps, speed=speed, physical=physical,
        )
        try:
            demo = ColorCubeDemo(options)
        except Exception as error:
            queue.fail_active(error)
            raise
        demo.run_live(queue, client, max_cubes=max_cubes)
        return demo
    finally:
        client.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Rebuild a VR table-map JSON cube layout with dual UR5e arms."
    )
    parser.add_argument(
        "map_path",
        nargs="?",
        type=Path,
        default=DEFAULT_MAP_PATH,
        help="openup.cube_stack.map JSON file (default: example_table_data.json)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without opening the interactive viewer",
    )
    parser.add_argument(
        "--record",
        nargs="?",
        type=Path,
        const=DEFAULT_VIDEO_PATH,
        metavar="PATH",
        help="export an MP4, optionally to PATH",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="output video frame rate (default: 30)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.25,
        help="trajectory speed multiplier (default: 1.25)",
    )
    parser.add_argument(
        "--physical",
        action="store_true",
        help=(
            "use free rigid cubes to test stability; by default cubes follow the "
            "robot kinematically and preserve the JSON poses"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="receive append-only CubeStack/Map snapshots and place cubes continuously")
    mode.add_argument("--receive-only", action="store_true", help="validate/queue real live maps without importing Genesis or moving robots")
    parser.add_argument("--host", default="127.0.0.1", help="OpenUp table host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9036, help="OpenUp simulation stream port (default: 9036)")
    parser.add_argument("--pool-per-color", type=int, default=30, help="prebuilt 8 cm cubes per color (default: 30)")
    parser.add_argument("--max-cubes", type=int, default=0, help="stop after N completed live placements or accepted receive-only cubes (0: continuous)")
    parser.add_argument("--timeout", type=float, default=0, help="seconds to wait for startup metadata or receive-only target count (0: unlimited; never interrupts motion)")
    args = parser.parse_args(argv)
    if not 0 < args.port < 65536 or args.pool_per_color <= 0 or args.max_cubes < 0 or args.timeout < 0:
        parser.error("port must be 1..65535, pool-per-color positive, max-cubes and timeout nonnegative")
    return args


if __name__ == "__main__":
    args = parse_args()
    options = dict(
        show_viewer=not args.headless,
        record_path=args.record,
        video_fps=args.fps,
        speed=args.speed,
        physical=args.physical,
    )
    try:
        if args.live or args.receive_only:
            live_main(host=args.host, port=args.port, pool_per_color=args.pool_per_color,
                      receive_only=args.receive_only, max_cubes=args.max_cubes,
                      timeout=args.timeout, **options)
        else:
            main(map_path=args.map_path, **options)
    except KeyboardInterrupt:
        print("stopped: live session interrupted; restart explicitly to resume with a fresh simulation", flush=True)
    except Exception as error:
        print(f"failed: {error}", flush=True)
        raise SystemExit(1)
