import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import genesis as gs
import numpy as np

from stacking_panel import (
    AnchorPlacement,
    AnchorTable,
    BlockSpec,
    launch_cube_designer,
)


# Paths and simulation settings.
SCRIPT_DIR = Path(__file__).resolve().parent
URDF_PATH = SCRIPT_DIR / "zeus_description" / "ur5e_robotiq_2f85.urdf"
DEFAULT_VIDEO_PATH = SCRIPT_DIR / "outputs" / "ur5e_color_anchors.mp4"
SIM_DT = 0.01
UI_UPDATE_STRIDE = 10

# Smaller cubes leave safe finger clearance on the denser 5×8 table grid.
CUBE_SIZE = (0.05, 0.05, 0.05)
# Separate color feeders sit beyond the right/left sides of the build grid.
SPAWN_POSITIONS = {
    "red": (0.30, 0.38, CUBE_SIZE[2] / 2),
    "green": (0.30, -0.38, CUBE_SIZE[2] / 2),
    "blue": (0.30, -0.38, CUBE_SIZE[2] / 2),
}
PARKED_BLOCK_ORIGIN = (2.0, 2.0, CUBE_SIZE[2] / 2)
SPAWN_SETTLE_STEPS = 20
GRID_LINE_WIDTH = 0.003
GRID_LINE_HEIGHT = 0.002
BLOCK_SPECS = (
    BlockSpec(
        "red",
        "Red",
        CUBE_SIZE,
        (0.90, 0.10, 0.10),
    ),
    BlockSpec(
        "green",
        "Green",
        CUBE_SIZE,
        (0.10, 0.75, 0.20),
    ),
    BlockSpec(
        "blue",
        "Blue",
        CUBE_SIZE,
        (0.10, 0.35, 0.95),
    ),
)
BLOCK_SPECS_BY_NAME = {spec.name: spec for spec in BLOCK_SPECS}

# Default layout used for headless runs and available from the panel's example button.
DEFAULT_PLACEMENTS = (
    AnchorPlacement("red_1", "red", x=0.10, y=-0.28, z=0.025),
    AnchorPlacement("green_1", "green", x=0.26, y=-0.04, z=0.025),
    AnchorPlacement("blue_1", "blue", x=0.42, y=0.28, z=0.025),
    AnchorPlacement("red_2", "red", x=0.26, y=-0.04, z=0.075),
)

# Rack and camera layout.
RACK_HEIGHT = 0.8
RACK_SIZE = (0.15, 0.4, RACK_HEIGHT)
RACK_MOUNT_Z = RACK_HEIGHT * 0.8
RACK_POS = (-RACK_SIZE[0] / 2, 0.0, RACK_HEIGHT / 2)
VIEWER_POS = (3.0, -1.2, 1.5)
VIEWER_LOOKAT = (0.0, 0.0, 0.5)
RECORDING_CAMERA_POS = (2.6, -1.5, 1.35)
RECORDING_CAMERA_LOOKAT = (0.25, 0.0, 0.45)

# Robot controller settings.
ARM_DOFS = np.arange(6)
GRIPPER_DOF = np.arange(6, 7)
ARM_KP = np.array([2500, 2400, 2000, 1400, 900, 700])
ARM_KV = np.array([220, 210, 180, 120, 80, 60])
ARM_FORCE_MIN = np.array([-150, -150, -150, -28, -28, -28])
ARM_FORCE_MAX = np.array([150, 150, 150, 28, 28, 28])
# Both tools stay vertical. The left wrist is yawed 90 degrees so its widest
# geometry clears cubes in neighboring 80 mm cells; the right arm retains the
# orientation that is reachable at its side feeder.
LEFT_TOOL_QUAT = np.array([0.0, 2**-0.5, 2**-0.5, 0.0])
RIGHT_TOOL_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
GRASP_OFFSET = np.array([0.0, 0.0, 0.14])
HOVER_OFFSET = GRASP_OFFSET + np.array([0.0, 0.0, 0.09])
# Keep the drop short so tall towers do not accumulate lateral drift.
RELEASE_CLEARANCE = np.array([0.0, 0.0, 0.005])
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.56

# Stable pre-grasp homes that avoid the singular configurations used originally.
LEFT_ARM_HOME_QPOS = np.array([-0.7165, -1.0068, 1.4604, -0.4533, 2.4254, 0.0])
RIGHT_ARM_HOME_QPOS = np.array([0.2647, -1.0065, 1.4600, -0.4546, 0.2655, -3.1406])


@dataclass(frozen=True)
class DemoOptions:
    show_viewer: bool = True
    record_path: Path | None = None
    video_fps: int = 30
    speed: float = 1.25
    placements: tuple[AnchorPlacement, ...] = DEFAULT_PLACEMENTS
    ui_pump: Callable[[], None] | None = None
    live_mode: bool = False

    def validate(self):
        if self.speed <= 0:
            raise ValueError("speed must be greater than zero")
        if self.video_fps <= 0:
            raise ValueError("video_fps must be greater than zero")
        placement_errors = AnchorTable(BLOCK_SPECS).validate(
            self.placements,
            require_any=not self.live_mode,
        )
        if placement_errors:
            raise ValueError(f"Invalid cube layout: {placement_errors[0]}")


def to_numpy(value):
    """Convert Genesis/Torch values to NumPy arrays."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class ColorCubeDemo:
    """Spawn and move an ordered sequence of colored cubes."""

    def __init__(self, options: DemoOptions):
        options.validate()
        self.options = options

        # A fixed seed makes IK behavior deterministic.
        gs.init(backend=gs.cpu, seed=0)

        self.scene = self._create_scene()
        self.blocks = self._add_blocks()
        self.placed_placements = []
        self.live_blocks = {}
        self.left_arm, self.right_arm = self._add_rack_and_arms()
        self.recording_camera = self._add_recording_camera()
        self.scene.build()
        self._pump_ui()

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
        self._add_table_grid()

        # Genesis cannot add rigid bodies after a scene has been built. Live mode
        # therefore reserves a color-specific pool up front; only the requested
        # cubes are teleported into a feeder and used by the robots.
        material = gs.materials.Rigid(rho=200.0, friction=2.0)
        blocks = {}
        if self.options.live_mode:
            for color_index, spec in enumerate(BLOCK_SPECS):
                blocks[spec.name] = []
                for index in range(AnchorTable.MAX_CUBES):
                    pool_index = color_index * AnchorTable.MAX_CUBES + index
                    parked_position = (
                        PARKED_BLOCK_ORIGIN[0] + (pool_index % 15) * 0.08,
                        PARKED_BLOCK_ORIGIN[1] + (pool_index // 15) * 0.08,
                        PARKED_BLOCK_ORIGIN[2],
                    )
                    blocks[spec.name].append(
                        self.scene.add_entity(
                            gs.morphs.Box(size=spec.size, pos=parked_position),
                            material=material,
                            surface=gs.surfaces.Rough(color=spec.color),
                        )
                    )
        else:
            for index, placement in enumerate(self.options.placements):
                spec = BLOCK_SPECS_BY_NAME[placement.color_name]
                parked_position = (
                    PARKED_BLOCK_ORIGIN[0] + index * 0.08,
                    PARKED_BLOCK_ORIGIN[1],
                    PARKED_BLOCK_ORIGIN[2],
                )
                blocks[placement.cube_id] = self.scene.add_entity(
                    gs.morphs.Box(size=spec.size, pos=parked_position),
                    material=material,
                    surface=gs.surfaces.Rough(color=spec.color),
                )
        return blocks

    def _add_table_grid(self):
        """Draw the panel's 5x8 grid in Genesis without affecting physics."""
        table = AnchorTable(BLOCK_SPECS)
        x_min, x_max = table.x_edges
        y_min, y_max = table.y_edges
        line_surface = gs.surfaces.Rough(color=(0.12, 0.65, 0.95))

        x_edges = tuple(
            x_min + index * table.X_PITCH for index in range(table.ROWS + 1)
        )
        y_edges = tuple(
            y_min + index * table.Y_PITCH for index in range(table.COLUMNS + 1)
        )
        for x in x_edges:
            self.scene.add_entity(
                gs.morphs.Box(
                    size=(GRID_LINE_WIDTH, y_max - y_min, GRID_LINE_HEIGHT),
                    pos=(x, (y_min + y_max) / 2, GRID_LINE_HEIGHT / 2),
                    fixed=True,
                    collision=False,
                ),
                surface=line_surface,
            )
        for y in y_edges:
            self.scene.add_entity(
                gs.morphs.Box(
                    size=(x_max - x_min, GRID_LINE_WIDTH, GRID_LINE_HEIGHT),
                    pos=((x_min + x_max) / 2, y, GRID_LINE_HEIGHT / 2),
                    fixed=True,
                    collision=False,
                ),
                surface=line_surface,
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
        self.scene.step()
        self.step_count += 1
        if (
            self.recording_camera is not None
            and self.step_count % self.render_stride == 0
        ):
            self.recording_camera.render()
        if self.step_count % UI_UPDATE_STRIDE == 0:
            self._pump_ui()

    def _pump_ui(self):
        if self.options.ui_pump is not None:
            self.options.ui_pump()

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
        if np.max(error[:3]) > 0.003 or np.max(error[3:]) > 0.015:
            raise RuntimeError(
                f"IK failed for target {position}; pose error was {error}"
            )

        self._move_joints(robot, to_numpy(qpos)[:6], ARM_DOFS, steps, settle)

    def _pick_and_place(self, robot, tool, pick_position, place_position):
        pick_position = np.asarray(pick_position)
        place_position = np.asarray(place_position)

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
        release_position = place_position + GRASP_OFFSET + RELEASE_CLEARANCE
        self._move_tool(robot, tool, release_position, steps=100)
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

    def _stop_recording(self):
        if self.recording_camera is None:
            return

        # Leave the completed color layout on screen briefly before ending the video.
        for _ in range(round(0.75 / SIM_DT)):
            self._step()
        output_path = self.options.record_path.resolve()
        self.recording_camera.stop_recording(
            save_to_filename=str(output_path),
            fps=self.options.video_fps,
        )
        print(f"Video saved to: {output_path}")

    def _arm_for_placement(self, placement):
        """Keep each color with its feeder-side arm for the complete move."""
        if placement.color_name == "red":
            return self.right_arm, self.right_tool, RIGHT_ARM_HOME_QPOS
        return self.left_arm, self.left_tool, LEFT_ARM_HOME_QPOS

    def _spawn_block(self, block, color_name):
        """Move the next parked block to its color-specific feeder pose."""
        spawn_position = SPAWN_POSITIONS[color_name]
        block.set_pos(spawn_position, zero_velocity=True)
        block.set_quat((1.0, 0.0, 0.0, 0.0), zero_velocity=True)
        for _ in range(self._scaled_steps(SPAWN_SETTLE_STEPS)):
            self._step()
        return spawn_position

    def _place_colored_cubes(self):
        """Spawn and place every requested cube in panel order."""
        for placement in self.options.placements:
            robot, tool, home = self._arm_for_placement(placement)
            block = self.blocks[placement.cube_id]
            spawn_position = self._spawn_block(block, placement.color_name)

            place_position = (placement.x, placement.y, placement.z)
            self._pick_and_place(
                robot,
                tool,
                spawn_position,
                place_position,
            )
            self._park(robot, home)
            self.placed_placements.append(placement)

    def place_live_cube(self, placement):
        """Execute one placement received from the still-open design panel."""
        if not self.options.live_mode:
            raise RuntimeError("Live placements require live_mode=True.")
        color_blocks = self.blocks[placement.color_name]
        color_index = sum(
            item.color_name == placement.color_name for item in self.placed_placements
        )
        if color_index >= len(color_blocks):
            raise RuntimeError(f"No unused {placement.color_name} cubes remain.")

        robot, tool, home = self._arm_for_placement(placement)
        block = color_blocks[color_index]
        self.live_blocks[placement.cube_id] = block
        spawn_position = self._spawn_block(block, placement.color_name)
        self._pick_and_place(
            robot,
            tool,
            spawn_position,
            (placement.x, placement.y, placement.z),
        )
        self._park(robot, home)
        self.placed_placements.append(placement)

    def _print_final_positions(self):
        print("Colored cubes placed. Final positions:")
        placements = self.placed_placements or self.options.placements
        for placement in placements:
            block = (
                self.live_blocks[placement.cube_id]
                if self.options.live_mode
                else self.blocks[placement.cube_id]
            )
            print(
                f"  {placement.cube_id} ({placement.color_name}):",
                to_numpy(block.get_pos()),
            )

    def run(self):
        self._start_recording()
        self._place_colored_cubes()
        self._stop_recording()
        self._print_final_positions()

    def run_live(self, design_session):
        """Consume panel placements as soon as the user confirms each anchor."""
        self._start_recording()
        try:
            while not design_session.closed and not design_session.finished:
                design_session.process_events()
                placement = design_session.pop_next_placement()
                if placement is None:
                    self._step()
                    continue
                design_session.mark_placement_started(placement)
                self.place_live_cube(placement)
                design_session.mark_placement_complete(placement)
        finally:
            self._stop_recording()
        self._print_final_positions()


def main(
    show_viewer=True,
    record_path=None,
    video_fps=30,
    speed=1.25,
    placements=DEFAULT_PLACEMENTS,
    ui_pump=None,
    live_mode=False,
    autorun=True,
):
    options = DemoOptions(
        show_viewer=show_viewer,
        record_path=Path(record_path).expanduser() if record_path is not None else None,
        video_fps=video_fps,
        speed=speed,
        placements=tuple(placements),
        ui_pump=ui_pump,
        live_mode=live_mode,
    )
    demo = ColorCubeDemo(options)
    if autorun:
        demo.run()
    return demo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spawn and stack colored cubes at selected 5×8 table anchors."
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
        "--design",
        action="store_true",
        help="place cubes interactively while Genesis executes them in real time",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    design_session = None
    if args.design:
        design_session = launch_cube_designer(
            BLOCK_SPECS,
            DEFAULT_PLACEMENTS,
            live_mode=True,
        )
        design_session.mark_simulation_starting()

    try:
        demo = main(
            show_viewer=not args.headless,
            record_path=args.record,
            video_fps=args.fps,
            speed=args.speed,
            placements=() if design_session is not None else DEFAULT_PLACEMENTS,
            ui_pump=(
                design_session.process_events if design_session is not None else None
            ),
            live_mode=design_session is not None,
            autorun=design_session is None,
        )
        if design_session is not None:
            design_session.mark_simulation_ready()
            demo.run_live(design_session)
    except Exception as error:
        if design_session is not None:
            design_session.mark_build_failed(str(error))
            design_session.wait_until_closed()
        raise

    if design_session is not None and not design_session.closed:
        design_session.mark_build_complete()
        design_session.wait_until_closed(demo.keep_viewer_alive)
