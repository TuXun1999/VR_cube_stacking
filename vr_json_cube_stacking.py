import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from cube_stack_layout import (
    CubePlacement, PiecePlacement, TableLayout, as_piece, load_table_layout, parse_table_layout,
    quaternion_to_rotation, rotation_matrix_to_quaternion,
)
# Shared, simulator-independent pieces; re-exported for existing callers.
from cube_stack_planner import LiveScenePlanner, PoolCube, to_numpy
from cube_stack_receiver import ResetRequested, pump_and_check
from cube_stack_receiver import live_main as _receiver_live_main, run_live as _receiver_run_live

gs = None  # Genesis is loaded only when a simulation is actually constructed.

# Paths and simulation settings.
SCRIPT_DIR = Path(__file__).resolve().parent
URDF_PATH = SCRIPT_DIR / "zeus_description" / "ur5e_robotiq_2f85.urdf"
DEFAULT_MAP_PATH = SCRIPT_DIR / "example_table_data.json"
DEFAULT_VIDEO_PATH = SCRIPT_DIR / "outputs" / "vr_json_cube_stacking.mp4"
TABLE_MAP_PATH = SCRIPT_DIR / "assets" / "islands.png"
TABLE_MAP_MESH_PATH = SCRIPT_DIR / "assets" / "table_map.obj"
TABLE_PALETTE_WIDTH_FRACTION = 0.12
SIM_DT = 0.01
TARGET_CUBE_SIZE = 0.08

CUBE_SIZE = (TARGET_CUBE_SIZE,) * 3
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



class ColorCubeDemo(LiveScenePlanner):
    """Rebuild a transformed VR cube layout with the dual-arm robot in Genesis.

    Live planning (blockers, staging, ordering) comes from ``LiveScenePlanner``;
    this class supplies the Genesis scene, the two UR5e arms, IK, and handoffs.
    """

    idle_sleep = SIM_DT

    def __init__(self, options: DemoOptions):
        options.validate()
        self.options = options
        self.layout = options.layout
        self.carried_block = None
        self.carried_tool = None
        self.carried_quaternion = None
        self.carried_tool_to_piece = None
        self.live_client = None
        self.live_queue = None
        self.staging = self._make_staging()

        global gs
        if gs is None:
            import genesis as gs_module
            gs = gs_module

        # A fixed seed makes IK behavior deterministic.
        gs.init(backend=gs.cpu, seed=0)

        self.scene = self._create_scene()
        self.blocks = self._add_blocks()
        self.placed_placements = []
        self.current_placements = {}
        self.left_arm, self.right_arm = self._add_rack_and_arms()
        self.recording_camera = self._add_recording_camera()
        self.scene.build()
        if options.live:
            self._setup_live_pool()

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
        self._add_staging_surfaces()

        # Every JSON cube is created before scene.build(), then parked outside the
        # workspace until its turn at the color-specific feeder.
        material = gs.materials.Rigid(rho=200.0, friction=2.0)
        blocks = {}
        placements = self.layout.pieces
        self._pool_specs = {}
        if self.options.live:
            lengths = (1, 2, 3) if self.layout.map_version == 3 else (1,)
            placements = tuple(
                PiecePlacement(f"{color}_L{length}_{serial}", color,
                    tuple(CubePlacement(f"stock_{i}", color, (0, 0, 0), (0, 0, 0),
                                        (1, 0, 0, 0)) for i in range(length)),
                    (0, 0, 0), (0, 0, 0), (1, 0, 0, 0))
                for color in COLOR_RGB for length in lengths
                for serial in range(1, self.options.pool_per_color // length + 1))
        for index, placement in enumerate(placements):
            parked_position = (PARKED_BLOCK_ORIGIN[0] + index * .28,
                               PARKED_BLOCK_ORIGIN[1], PARKED_BLOCK_ORIGIN[2])
            self._pool_specs[placement.cube_id] = (placement.color_name, placement.length)
            if placement.length == 1:
                morph = gs.morphs.Box(size=CUBE_SIZE, pos=parked_position,
                                      fixed=not self.options.physical,
                                      collision=self.options.physical)
            else:
                morph = gs.morphs.URDF(
                    file=str(SCRIPT_DIR / "assets" / f"glued_piece_{placement.length}.urdf"),
                    pos=parked_position, fixed=not self.options.physical,
                    collision=self.options.physical, requires_jac_and_IK=False)
            blocks[placement.cube_id] = self.scene.add_entity(
                morph, material=material,
                surface=gs.surfaces.Rough(color=COLOR_RGB[placement.color_name]))
        return blocks

    def _add_staging_surfaces(self):
        """Draw three continuous color strips on each side of the table."""
        for side in ("left", "right"):
            positions = [self.staging.position(side, lane, column)
                         for lane in self.staging.lanes for column in range(self.staging.columns)]
            cell_depth, cell_width = self.staging.cell_footprint
            x_min = min(position[0] for position in positions) - cell_depth / 2
            x_max = max(position[0] for position in positions) + cell_depth / 2
            rows = sorted({position[1] for position in positions},
                          key=lambda y: abs(y - self.staging.center_y))
            for color, y in zip(self.staging.colors, rows):
                self.scene.add_entity(
                    gs.morphs.Box(size=(x_max - x_min, cell_width, .02),
                                  pos=((x_min + x_max) / 2, y,
                                       self.staging.platform_height - .01),
                                  fixed=True, collision=self.options.physical),
                    surface=gs.surfaces.Rough(color=tuple(.45 * v for v in COLOR_RGB[color])),
                )

    def _add_workspace_table(self):
        """Draw the VR table's island map, footprint, and center axes."""
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
                pos=(center_x, center_y, -0.0005),
                fixed=True,
                collision=False,
            ),
            surface=table_surface,
        )

        map_width = self.layout.workspace_width * (1 - TABLE_PALETTE_WIDTH_FRACTION)
        map_center_y = center_y + self.layout.workspace_width * TABLE_PALETTE_WIDTH_FRACTION / 2
        self.scene.add_entity(
            gs.morphs.Mesh(
                file=str(TABLE_MAP_MESH_PATH),
                scale=(self.layout.workspace_depth, map_width, 1.0),
                pos=(center_x, map_center_y, 0.0002),
                fixed=True,
                collision=False,
                decimate=False,
                convexify=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ImageTexture(image_path=str(TABLE_MAP_PATH)),
                double_sided=True,
            ),
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
            pump_and_check(self.live_client, getattr(self, "live_queue", None))
        self.scene.step()
        self.step_count += 1
        if self.carried_block is not None:
            tool_rotation = quaternion_to_rotation(to_numpy(self.carried_tool.get_quat()))
            offset, relative_rotation = self.carried_tool_to_piece
            self.carried_block.set_pos(to_numpy(self.carried_tool.get_pos()) + tool_rotation @ offset,
                                       zero_velocity=True)
            self.carried_block.set_quat(rotation_matrix_to_quaternion(tool_rotation @ relative_rotation),
                                        zero_velocity=True)
        if (
            self.recording_camera is not None
            and self.step_count % self.render_stride == 0
        ):
            self.recording_camera.render()

    def keep_viewer_alive(self):
        """Advance the settled scene while the final comparison remains open."""
        self.scene.step()

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

    def _solve_tool(self, robot, tool, position, init_qpos=None, tool_quaternion=None):
        """Solve arm-only IK without issuing robot commands."""
        tool_quat = (tool_quaternion if tool_quaternion is not None else
                     LEFT_TOOL_QUAT if robot is self.left_arm else RIGHT_TOOL_QUAT)
        qpos, error = robot.inverse_kinematics(
            link=tool,
            pos=np.asarray(position),
            quat=tool_quat,
            init_qpos=to_numpy(robot.get_dofs_position()) if init_qpos is None else init_qpos,
            dofs_idx_local=ARM_DOFS,
            max_samples=10,
            max_solver_iters=300,
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

        return qpos

    def _move_tool(self, robot, tool, position, steps, settle=30):
        qpos = self._solve_tool(robot, tool, position)
        self._move_joints(robot, qpos[:6], ARM_DOFS, steps, settle)

    def _transfer_height(self, pick_position, place_position):
        heights = [pick_position[2], place_position[2]]
        heights.extend(p.position[2] for p in self.placed_placements)
        heights.extend(slot.position[2] for slot in self.staging.slots.values())
        return max(heights) + GRASP_OFFSET[2] + TARGET_CUBE_SIZE / 2 + .08

    def _transfer_waypoints(self, pick_position, placement):
        pick = np.asarray(pick_position, dtype=float)
        place = np.asarray(placement.position, dtype=float)
        height = self._transfer_height(pick, place)
        rise = pick + HOVER_OFFSET
        over = np.array((place[0], place[1], height))
        midpoint = (rise + over) / 2
        midpoint[2] = height
        clearance = STACK_RELEASE_CLEARANCE if place[2] > TARGET_CUBE_SIZE else RELEASE_CLEARANCE
        if as_piece(placement).length > 1:
            height += .5 * np.linalg.norm([TARGET_CUBE_SIZE, TARGET_CUBE_SIZE * as_piece(placement).length,
                                          TARGET_CUBE_SIZE]) - TARGET_CUBE_SIZE / 2
            at_pick = np.array((pick[0], pick[1], height))
            over[2] = midpoint[2] = height
            return (pick + HOVER_OFFSET, pick + GRASP_OFFSET, rise,
                    at_pick, at_pick, midpoint, over,
                    place + GRASP_OFFSET + clearance, place + HOVER_OFFSET)
        return (pick + HOVER_OFFSET, pick + GRASP_OFFSET, rise,
                midpoint, over, place + GRASP_OFFSET + clearance, place + HOVER_OFFSET)

    def _pick_and_place(self, robot, tool, block, pick_position, placement, joint_plan=None):
        pick_position = np.asarray(pick_position)
        place_position = np.asarray(placement.position)
        if joint_plan is None:
            joint_plan = self._transfer_plan(robot, tool, pick_position, placement)
        if joint_plan is None:
            raise RuntimeError(f"Unreachable pick-and-place for {placement.cube_id}")

        def go(index, steps):
            self._move_joints(robot, joint_plan[index][:6], ARM_DOFS, steps)

        # Approach and retreat vertically; transfer only after the object is clear.
        go(0, 120)
        go(1, 90)
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
            tool_rotation = quaternion_to_rotation(to_numpy(tool.get_quat()))
            self.carried_tool_to_piece = (
                tool_rotation.T @ (to_numpy(block.get_pos()) - to_numpy(tool.get_pos())),
                tool_rotation.T @ quaternion_to_rotation(to_numpy(block.get_quat())))

        for index in range(2, len(joint_plan) - 2):
            go(index, 140)
        go(len(joint_plan) - 2, 100)
        if not self.options.physical:
            block.set_pos(place_position, zero_velocity=True)
            block.set_quat(placement.quaternion, zero_velocity=True)
            self.carried_block = None
            self.carried_tool = None
            self.carried_quaternion = None
            self.carried_tool_to_piece = None
        self._move_joints(
            robot,
            [GRIPPER_OPEN],
            GRIPPER_DOF,
            steps=60,
            settle=50,
        )
        go(len(joint_plan) - 1, 100)

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

    def _grasp_quaternion(self, robot, quaternion):
        base = LEFT_TOOL_QUAT if robot is self.left_arm else RIGHT_TOOL_QUAT
        piece_rotation = quaternion_to_rotation(quaternion)
        base_rotation = quaternion_to_rotation(base)
        angle = np.arctan2(piece_rotation[1, 0], piece_rotation[0, 0]) - np.arctan2(base_rotation[1, 0], base_rotation[0, 0])
        c, s = np.cos(angle), np.sin(angle)
        yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return rotation_matrix_to_quaternion(yaw @ base_rotation)

    def _transfer_plan(self, robot, tool, pick_position, placement, pickup_quaternion=None):
        actual = to_numpy(robot.get_dofs_position()).copy()
        home = actual.copy()
        home[:6] = LEFT_ARM_HOME_QPOS if robot is self.left_arm else RIGHT_ARM_HOME_QPOS
        for initial in (actual, home):
            seed = initial
            plan = []
            try:
                for index, position in enumerate(self._transfer_waypoints(pick_position, placement)):
                    if as_piece(placement).length > 1:
                        quaternion = (pickup_quaternion if pickup_quaternion is not None else placement.quaternion) if index < 4 else placement.quaternion
                        seed = self._solve_tool(robot, tool, position, seed,
                                                self._grasp_quaternion(robot, quaternion))
                    else:
                        seed = self._solve_tool(robot, tool, position, seed)
                    plan.append(seed.copy())
                return tuple(plan)
            except RuntimeError:
                continue
        return None

    def _can_transfer(self, robot, tool, pick_position, placement):
        return self._transfer_plan(robot, tool, pick_position, placement) is not None

    def _handoff_candidates(self, block, placement):
        """Find free table-surface spots near its center, including tool clearance."""
        center = np.asarray(self.layout.workspace_center, dtype=float)
        pitch = TARGET_CUBE_SIZE + .04
        half_diagonal = .5 * np.linalg.norm([TARGET_CUBE_SIZE, as_piece(placement).length * TARGET_CUBE_SIZE])
        # Stay in the central half of each table dimension, with the entire cube
        # on the table even for yawed placements or unusually small workspaces.
        x_limit = min(self.layout.workspace_depth / 4,
                      self.layout.workspace_depth / 2 - half_diagonal)
        y_limit = min(self.layout.workspace_width / 4,
                      self.layout.workspace_width / 2 - half_diagonal)
        offsets = sorted(((x, y) for x in range(-2, 3) for y in range(-2, 3)),
                         key=lambda offset: (offset[0] ** 2 + offset[1] ** 2, offset))
        others = {id(other): other for other in self.blocks.values() if other is not block}
        others.update({id(item.block): item.block for item in getattr(self, "pool_cubes", {}).values()
                       if item.block is not block})
        # A circumscribed footprint plus 4 cm leaves room for the open gripper.
        radii = {id(other): .5 * np.linalg.norm([TARGET_CUBE_SIZE,
                    as_piece(self.current_placements[key]).length * TARGET_CUBE_SIZE])
                 for key, other in self.blocks.items() if key in self.current_placements}
        radii.update({id(item.block): .5 * np.linalg.norm([TARGET_CUBE_SIZE, item.length * TARGET_CUBE_SIZE])
                      for item in getattr(self, "pool_cubes", {}).values()})
        for x, y in offsets:
            dx, dy = x * pitch, y * pitch
            if abs(dx) > x_limit or abs(dy) > y_limit:
                continue
            position = center + (dx, dy, TARGET_CUBE_SIZE / 2)
            if any(np.linalg.norm(to_numpy(other.get_pos())[:2] - position[:2]) <
                   half_diagonal + radii.get(id(other), TARGET_CUBE_SIZE / np.sqrt(2)) + .04
                   for other in others.values()):
                continue
            yield replace(placement, position=tuple(float(value) for value in position))

    def _transfer_block(self, block, placement):
        """Transfer directly, or set down near table center for the other arm."""
        placement = self._validate_piece(placement)
        source = to_numpy(block.get_pos())
        pickup_quaternion = tuple(to_numpy(block.get_quat()))
        arms = [(self.left_arm, self.left_tool, LEFT_ARM_HOME_QPOS),
                (self.right_arm, self.right_tool, RIGHT_ARM_HOME_QPOS)]
        if source[1] >= 0:
            arms.reverse()
        try:
            for robot, tool, home in arms:
                plan = self._transfer_plan(robot, tool, source, placement, pickup_quaternion)
                if plan is not None:
                    self._pick_and_place(robot, tool, block, source, placement, plan)
                    self._park(robot, home)
                    return
            for relay in self._handoff_candidates(block, placement):
                for first in arms:
                    first_plan = self._transfer_plan(first[0], first[1], source, relay, pickup_quaternion)
                    if first_plan is None:
                        continue
                    for second in reversed(arms):
                        if second[0] is first[0]:
                            continue
                        second_plan = self._transfer_plan(second[0], second[1], relay.position, placement, relay.quaternion)
                        if second_plan is None:
                            continue
                        self._pick_and_place(first[0], first[1], block, source, relay, first_plan)
                        self._park(first[0], first[2])
                        self._pick_and_place(second[0], second[1], block, relay.position, placement, second_plan)
                        self._park(second[0], second[2])
                        return
            raise RuntimeError(f"No reachable robot transfer for {placement.cube_id}: "
                               f"{tuple(source)} -> {placement.position}")
        finally:
            self.carried_block = self.carried_tool = self.carried_quaternion = self.carried_tool_to_piece = None

    def _staging_feasible(self, item, slot, destination=None):
        target = destination or self._staging_target(item, slot)
        source = slot.position if destination is not None else tuple(to_numpy(item.block.get_pos()))
        quaternion = slot.quaternion if destination is not None else tuple(to_numpy(item.block.get_quat()))
        return any(self._transfer_plan(robot, tool, source, target, quaternion) is not None
                   for robot, tool in ((self.left_arm, self.left_tool), (self.right_arm, self.right_tool)))

    @staticmethod
    def _footprints_overlap(first_position, first_quaternion, second_position, second_quaternion):
        def axes(quaternion):
            w, x, y, z = quaternion
            angle = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            return (np.array((np.cos(angle), np.sin(angle))),
                    np.array((-np.sin(angle), np.cos(angle))))
        first_axes, second_axes = axes(first_quaternion), axes(second_quaternion)
        delta = np.asarray(second_position[:2]) - first_position[:2]
        for axis in (*first_axes, *second_axes):
            limit = TARGET_CUBE_SIZE / 2 * sum(abs(np.dot(axis, basis))
                                               for basis in (*first_axes, *second_axes))
            if abs(np.dot(delta, axis)) >= limit - 1e-5:
                return False
        return True

    def _spawn_block(self, block, placement, robot):
        """Create a new supply cube in staging before its first robot pickup."""
        side = "right" if robot is self.right_arm else "left"
        slot = self.staging.reserve(placement.cube_id, placement.color_name, side, length=as_piece(placement).length)
        spawn_position = slot.position
        block.set_pos(spawn_position, zero_velocity=True)
        block.set_quat(slot.quaternion if self.staging.max_piece_length > 1 else placement.quaternion, zero_velocity=True)
        for _ in range(self._scaled_steps(SPAWN_SETTLE_STEPS)):
            self._step()
        return spawn_position

    def _place_colored_cubes(self):
        """Spawn and place every transformed JSON cube from bottom to top."""
        for placement in self.layout.pieces:
            self.place_live_cube(placement)

    def _place_offline(self, placement):
        """Offline JSON replay: spawn beside the target-side arm, then place."""
        robot, _, _ = self._arm_for_placement(placement)
        block = self.blocks[placement.cube_id]
        self._spawn_block(block, placement, robot)
        self._transfer_block(block, placement)
        self.staging.release(placement.cube_id)

    def run_live(self, queue, client, max_cubes=0):
        """Only this outer loop starts commands; _step only receives/enqueues."""
        _receiver_run_live(self, queue, client, max_cubes=max_cubes)

    def _print_final_positions(self):
        print("VR cubes placed. Target -> final position (meters):")
        for placement in self.layout.pieces:
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
    def build_backend(layout):
        return ColorCubeDemo(DemoOptions(
            layout=layout, live=True, pool_per_color=pool_per_color,
            show_viewer=show_viewer,
            record_path=Path(record_path).expanduser() if record_path is not None else None,
            video_fps=video_fps, speed=speed, physical=physical,
        ))

    return _receiver_live_main(build_backend, host=host, port=port, pool_per_color=pool_per_color,
                               receive_only=receive_only, max_cubes=max_cubes, timeout=timeout)


# -- robot_receiver.py backend interface ---------------------------------------------

BACKEND_DESCRIPTION = "dual UR5e arms in Genesis"
DEFAULT_POOL_PER_COLOR = 30


def add_backend_arguments(parser):
    """Genesis-only options for ``robot_receiver.py --sim genesis``."""
    group = parser.add_argument_group("genesis backend")
    group.add_argument("--record", nargs="?", type=Path, const=DEFAULT_VIDEO_PATH, metavar="PATH",
                       help="export an MP4, optionally to PATH")
    group.add_argument("--fps", type=int, default=30, help="output video frame rate (default: 30)")
    group.add_argument("--speed", type=float, default=1.25,
                       help="trajectory speed multiplier (default: 1.25)")
    group.add_argument("--physical", action="store_true",
                       help="use free rigid cubes to test stability instead of kinematic carrying")


def build_backend(layout, args):
    """Build the Genesis receiver for the first live map."""
    return ColorCubeDemo(DemoOptions(
        layout=layout, live=True, pool_per_color=args.pool_per_color,
        show_viewer=not args.headless,
        record_path=Path(args.record).expanduser() if args.record is not None else None,
        video_fps=args.fps, speed=args.speed, physical=args.physical,
    ))


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
    mode.add_argument("--live", action="store_true", help="follow CubeStack/Map additions, moves, deletions and session resets with robot pick-and-place")
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
