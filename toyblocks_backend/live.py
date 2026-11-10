"""The live ToyBlocks backend: the shared planner driving vecgen skills on one Franka.

``LiveScenePlanner`` decides which piece moves where (clearing blockers, staging,
support order). This class supplies the hooks it needs:

- ``_transfer_block``: a reach check, then ToyBlocks' cuRobo-planned pick
  (``pick_piece``) and an exact-pose place (``place_at``). One retry from the live
  state; a piece that still does not settle within tolerance halts the session.
- ``_staging_feasible``: the same reach check (IK on top-down grasps).
- ``_step``: one vecgen control step, then pump OpenUp (which may raise
  ``ResetRequested``).
- ``_initialize_poses``: teleport the arm to ToyBlocks' home pose.

Import only after ``app.launch()``.
"""
from collections import Counter

import numpy as np
import torch

import toyblocks.vecgen.geometry as vecgen_geometry
from toyblocks.vecgen.geometry import Geometry
from toyblocks.vecgen.skills.pnp import grasp_candidates

from cube_stack_layout import TARGET_CUBE_SIZE, as_piece, placements_share_pose, quaternion_to_rotation
from cube_stack_planner import LiveScenePlanner
from cube_stack_receiver import pump_and_check
from toyblocks_backend.engine import ENV, PROFILE, LiveEngine, LiveSim
from toyblocks_backend.entity import LiveWorld, ToyBlocksEntity
from toyblocks_backend.frames import FrameMap, jaw_clear_quaternion, nearest_symmetric_quaternion
from toyblocks_backend.props import pad_boxes, pool_bodies, table_box
from toyblocks_backend.scene import build_env
from toyblocks_backend.skills import (
    JAW_AXIS_LOCAL, MAX_JAW_OPENING, MIN_JAW_ALIGNMENT, PickPiece, PlaceAt, jaw_alignment,
)
from toyblocks_backend.staging import ReachableStagingArea

TRANSFER_ATTEMPTS = 2
MAX_CORRECTIONS = 2  # re-placements of a piece at an unchanged target before "unstable"
SETTLE_TICKS = 12
TABLE_TOP_TOLERANCE = 0.002
PARK_STEP = 0.03  # radians per control step when walking home after a failure
PARK_TOLERANCE = 0.02
PARK_MAX_TICKS = 300
IK_POSITION_TOLERANCE = 0.005  # Pick.begin's acceptance for a near-miss IK solution
IK_ROTATION_TOLERANCE = 0.05
PARKED_BELOW = -0.5  # planner z; pool bodies parked on the ground are not neighbours
VIEWER_EVERY = 3  # control steps (0.04 s of sim time each) between browser-viewer updates


class ToyBlocksBackend(LiveScenePlanner):
    idle_sleep = 0.0  # stepping Isaac already paces the idle loop

    def __init__(self, layout, options):
        self.layout = layout
        self.options = options
        self.viewer = None
        self._viewer_ticks = 0
        self.clear_carried()
        self.live_client = self.live_queue = None
        self.placed_placements = []
        self.current_placements = {}
        self.frames = FrameMap(robot_to_near_edge=options.robot_to_near_edge, table_top=0.0)
        bodies = pool_bodies(layout.map_version, options.pool_per_color)
        env = build_env(table_box(layout, self.frames), pad_boxes(self._make_staging(), self.frames),
                        bodies, options.device)
        device = torch.device(env.unwrapped.device)
        self.sim = LiveSim(env, device)
        self.sim.apply_profile(PROFILE)
        self._ids = torch.tensor([ENV], dtype=torch.long, device=device)
        self._check_table_top()

        dims, _centre = self.sim.measure_block_dims()
        # vecgen models the table as an axis-aligned slab of TABLE_DIMS centred on the
        # table's root; it is a module constant sized for ToyBlocks' own table.
        vecgen_geometry.TABLE_DIMS = (layout.workspace_depth, layout.workspace_width, 0.2)
        self.geo = Geometry(self.sim.num_envs, self.sim.block_names, dims, device)
        self.geo.table_top = self.sim.measure_table_top("workspace_table")
        self.engine = LiveEngine(self.sim, self.geo, {"pick_piece": PickPiece(), "place_at": PlaceAt()})
        self.ctx = self.engine.ctx

        world = LiveWorld(self.sim, self.frames)
        self._pool_specs = {body.key: (body.color, body.length) for body in bodies}
        self.blocks = {body.key: ToyBlocksEntity(world, body.key, body.length) for body in bodies}
        self._targets = {}
        self._corrections = Counter()
        self._setup_live_pool()
        self._initialize_poses()
        if getattr(options, "viewer", False):
            self._start_viewer(env.unwrapped)

    # -- browser viewer (ToyBlocks' viser viewer; needs no Vulkan) -------------------------

    def _start_viewer(self, base_env):
        from toyblocks.viewer.viewer import Viewer, ViewerCfg

        try:
            self.viewer = Viewer(ViewerCfg(minimal_gui=True))
            self.viewer.load_assets(base_env)
        except Exception as error:  # the robot keeps working without a picture
            print(f"viewer: unavailable ({error})", flush=True)
            self.viewer = None
            return
        self._viewer_env = base_env
        self._viewer_ids = torch.tensor([ENV], device=base_env.device, dtype=torch.int32)
        port = self.viewer._server.get_port()
        print(f"viewer: serving on port {port} ({self.viewer.get_server_url()})", flush=True)
        self._refresh_viewer(force=True)

    def _refresh_viewer(self, force=False):
        if self.viewer is None:
            return
        self._viewer_ticks += 1
        if not force and self._viewer_ticks % VIEWER_EVERY:
            return
        try:
            frame = self.viewer._create_trajectory_frame_from_env(self._viewer_env, self._viewer_ids)
            # view_frame skips a frame equal to the last one, but comparing two live Frames
            # compares numpy arrays and raises; forget the last frame instead.
            self.viewer._last_viewed_set = None
            self.viewer.view_frame(frame, env_id=ENV)
        except Exception as error:
            print(f"viewer: stopped after an error ({error}); the robot carries on", flush=True)
            self.viewer = None

    def _check_table_top(self):
        root = float(self.sim.object_pose("workspace_table")[ENV, 2])
        top = root + float(self.sim.measure_table_top("workspace_table")[ENV])
        if abs(top - self.frames.table_top) > TABLE_TOP_TOLERANCE:
            raise RuntimeError(f"workspace_table top is at z={top:.4f} in the robot frame; "
                               f"expected {self.frames.table_top:.4f}")

    # -- planner policy -----------------------------------------------------------------

    def _make_staging(self):
        return ReachableStagingArea(self.layout, TARGET_CUBE_SIZE,
                                    max_piece_length=3 if self.layout.map_version == 3 else 1)

    # -- simulation hooks -----------------------------------------------------------------

    def _tick(self):
        self._refresh_viewer()
        pump_and_check(self.live_client, self.live_queue)

    def _step(self):
        self.engine.tick()
        self._tick()

    def _settle(self, ticks=SETTLE_TICKS):
        for _ in range(ticks):
            self._step()

    def _initialize_poses(self):
        """Teleport the arm home with the gripper open (no pumping: also used by reset)."""
        art = self.sim.art
        joints = art.data.default_joint_pos[self._ids].clone()
        art.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=self._ids)
        art.set_joint_position_target(joints, env_ids=self._ids)
        self.sim.command_gripper(torch.zeros(1, dtype=torch.bool, device=self.sim.device), self._ids)
        self.sim.flush()
        self.sim.forward()
        self.sim.apply_profile(PROFILE, self._ids)
        self.geo.detach(self._ids)
        self.ctx.held[self._ids] = -1
        self._refresh_viewer(force=True)

    def _start_recording(self):
        pass

    def _stop_recording(self, settle=True):
        pass

    def _release_hand(self):
        self.engine.cancel(ENV)
        self.sim.command_gripper(torch.zeros(1, dtype=torch.bool, device=self.sim.device), self._ids)
        self.sim.flush()
        self.geo.detach(self._ids)
        self.ctx.held[self._ids] = -1

    def _walk_home(self):
        """After a failed command: open the hand and walk the arm home in joint space."""
        self._release_hand()
        art = self.sim.art
        home = art.data.default_joint_pos[self._ids][:, self.sim.arm_ids]
        for _ in range(PARK_MAX_TICKS):
            target = art.data.joint_pos_target[self._ids][:, self.sim.arm_ids]
            art.set_joint_position_target(target + (home - target).clamp(-PARK_STEP, PARK_STEP),
                                          joint_ids=self.sim.arm_ids, env_ids=self._ids)
            self.sim.flush()
            self.sim.step()
            self._tick()
            if float((art.data.joint_pos[self._ids][:, self.sim.arm_ids] - home).abs().amax()) < PARK_TOLERANCE:
                return

    # -- reachability -------------------------------------------------------------------

    def _size_of(self, block):
        return self.ctx.sizes[ENV, self.ctx.block_index[block.name]]

    def _grasp_reachable(self, position, quaternion, size, update_world=True):
        """Whether any allowed top-down grasp of a piece at this planner pose has an IK solution."""
        pose = torch.as_tensor(self.frames.to_robot_pose(position, quaternion), dtype=torch.float32,
                               device=self.sim.device)[None]
        size = size[None].to(pose.dtype)
        cands = grasp_candidates(pose, size, self.sim.ee_pose()[self._ids])
        wide = size[:, :2].amax(dim=1) > MAX_JAW_OPENING
        allowed = (jaw_alignment(cands, pose, size) >= MIN_JAW_ALIGNMENT) | ~wide[:, None]
        if update_world:
            self.geo.update_worlds(self.ctx.block_poses(), self.ctx.table_pose(), env_ids=self._ids)
        seeds = self.sim.joint_pos()[self._ids]
        for c in range(cands.shape[1]):
            if not bool(allowed[0, c]):
                continue
            ok, _q, pos_err, rot_err = self.geo.ik(self._ids, cands[:, c], seeds=seeds)
            if bool(ok[0]) or (float(pos_err[0]) <= IK_POSITION_TOLERANCE
                               and float(rot_err[0]) <= IK_ROTATION_TOLERANCE):
                return True
        return False

    def _staging_feasible(self, item, slot, destination=None):
        target = destination or self._staging_target(item, slot)
        source = slot.position if destination is not None else tuple(item.block.get_pos())
        quaternion = slot.quaternion if destination is not None else tuple(item.block.get_quat())
        size = self._size_of(item.block)
        return (self._grasp_reachable(source, quaternion, size)
                and self._grasp_reachable(target.position, target.quaternion, size, update_world=False))

    # -- transfers ------------------------------------------------------------------------

    def _transfer_block(self, block, placement):
        placement = self._validate_piece(placement)
        source, source_quaternion = tuple(block.get_pos()), tuple(block.get_quat())
        size = self._size_of(block)
        # Reach is checked before the pick, when the grasp (and so the best of the
        # piece's symmetric orientations) is not known yet; any of them reaches alike.
        quaternion = nearest_symmetric_quaternion(source_quaternion, placement.quaternion, placement.length)
        if not (self._grasp_reachable(source, source_quaternion, size)
                and self._grasp_reachable(placement.position, quaternion, size, update_world=False)):
            raise RuntimeError(
                f"No reachable robot transfer for {placement.cube_id}: "
                f"{tuple(round(v, 3) for v in source)} -> {tuple(round(v, 3) for v in placement.position)} "
                f"(single Franka; see toyblocks_backend/spike.py --sweep)")
        reason = ""
        for _ in range(TRANSFER_ATTEMPTS):
            pick_from = block.measured()
            ok, reason = self.engine.run_call("pick_piece", (block.name,), {}, self._tick)
            if ok:
                quaternion = self._place_orientation(block, placement)
                put = self.frames.to_robot_pose(placement.position, quaternion).tolist()
                ok, reason = self.engine.run_call("place_at", (block.name,), {"put": put}, self._tick)
                if not ok and int(self.ctx.held[ENV]) >= 0:
                    # Set the piece back down where it came from rather than dropping it.
                    back = self.frames.to_robot_pose(*pick_from).tolist()
                    self.engine.run_call("place_at", (block.name,), {"put": back}, self._tick)
            if ok:
                self._settle()
                if block.accept(placement.position, placement.quaternion):
                    return
                reason = "the piece did not settle at its target"
            self._walk_home()
        raise RuntimeError(f"unstable: {placement.cube_id} did not stay at its target after "
                           f"{TRANSFER_ATTEMPTS} attempts ({reason})")

    def _place_orientation(self, block, placement):
        """The piece's symmetric target orientation that keeps the closed fingers clear of neighbours."""
        ee_quaternion = self.sim.ee_pose()[ENV, 3:7].detach().cpu().numpy()
        jaw_world = quaternion_to_rotation(ee_quaternion) @ np.asarray(JAW_AXIS_LOCAL)
        neighbours = [(item.block.get_pos(), item.block.get_quat(), item.length)
                      for item in self.pool_cubes.values()
                      if item.block is not block and item.block.get_pos()[2] > PARKED_BELOW]
        return jaw_clear_quaternion(block.measured()[1], placement.quaternion, placement.length,
                                    jaw_world, placement.position, neighbours, TARGET_CUBE_SIZE)

    # -- live commands: guard against pieces that never stay put ---------------------------

    def _note_target(self, placement):
        piece = as_piece(placement)
        previous = self._targets.get(piece.cube_id)
        if previous is not None and placements_share_pose(previous, piece):
            self._corrections[piece.cube_id] += 1
            if self._corrections[piece.cube_id] > MAX_CORRECTIONS:
                raise RuntimeError(f"unstable: {piece.cube_id} does not stay at its target")
        else:
            self._corrections[piece.cube_id] = 0
        self._targets[piece.cube_id] = piece

    def place_live_cube(self, placement):
        if as_piece(placement).cube_id not in self.blocks:
            self._note_target(placement)
        super().place_live_cube(placement)

    def move_live_cube(self, placement):
        self._note_target(placement)
        super().move_live_cube(placement)

    def delete_live_cube(self, placement):
        self._targets.pop(placement.cube_id, None)
        self._corrections.pop(placement.cube_id, None)
        super().delete_live_cube(placement)

    def reset_live_scene(self):
        self._release_hand()
        self._targets.clear()
        self._corrections.clear()
        super().reset_live_scene()
