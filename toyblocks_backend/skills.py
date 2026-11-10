"""vecgen skills for OpenUp's 8 cm pieces: exact-pose placement and safe grasps.

``PlaceAt`` follows ``_PixelTablePlace`` in ``toyblocks/eval/ll_oracle.py``: it
sets its own target in ``begin``; transport, lowering, release and retreat are
ToyBlocks' ``Place`` phases, with the pre-place pose raised above neighbouring
pieces. ``PickPiece`` is ToyBlocks' ``Pick`` with two changes for pieces that
nearly fill the 85 mm gripper:

- on a piece wider than the gripper opens, the jaws must close across its short
  side;
- the planned motion stops above the piece, and an IK-interpolated descent that
  ignores the piece itself finishes the approach (cuRobo's open-finger spheres
  overlap an 8 cm cube at the grasp pose), then the lift clears neighbouring
  pieces before transport.

Import only after Isaac Sim is running.
"""
import torch

from toyblocks.vecgen.math import pose_compose, pose_inverse, quat_apply
from toyblocks.vecgen.skills.base import PhaseRequest
from toyblocks.vecgen.skills.pnp import (
    GRASP_NEIGHBOR_CLEARANCE, GRIP_TICKS, IK_ROWS, PLACE_Z_GAP, PRE_PLACE_STANDOFF, Pick, Place,
    grasp_candidates,
)

# The Robotiq 2F-85 pads close along its base link's x axis: in ToyBlocks' cuRobo
# model (vecgen/planner/configs/robot/fr3_model_full_robotiq_2f85_kinematics.urdf,
# generated from the simulated USD) the outer knuckles sit at x = -/+0.0306 and
# hinge about y.
JAW_AXIS_LOCAL = (1.0, 0.0, 0.0)
MAX_JAW_OPENING = 0.085
APPROACH_HEIGHT = 0.05  # planned approach ends this far above the grasp pose
LIFT_HEIGHT = 0.15  # lift the piece clear of 8 cm neighbours before transport
NEIGHBOUR_CLEARANCE = 0.10  # extra pre-place height, so transport ends above neighbours
MIN_JAW_ALIGNMENT = 0.7  # |cos| between the jaw axis and the piece's short axis
PLACE_VERIFY_TOLERANCE = 0.03  # metres; entity.POSITION_TOLERANCE decides "done"


def jaw_alignment(cands, block_pose, block_size):
    """|cos| between each candidate's jaw axis and the block's short horizontal axis, ``[B, C]``."""
    b, c = cands.shape[:2]
    dev = cands.device
    jaw = quat_apply(cands[..., 3:7].reshape(-1, 4),
                     torch.tensor(JAW_AXIS_LOCAL, device=dev).expand(b * c, 3)).reshape(b, c, 3)[..., :2]
    q = block_pose[:, 3:7]
    x = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=dev).expand(b, 3))[:, :2]
    y = quat_apply(q, torch.tensor([0.0, 1.0, 0.0], device=dev).expand(b, 3))[:, :2]
    short = torch.where((block_size[:, 0] <= block_size[:, 1])[:, None], x, y)
    jaw = jaw / jaw.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    short = short / short.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return (jaw * short[:, None, :]).sum(-1).abs()


def _only(ctx, envs, key):
    """``[B, K]`` mask selecting each env's ``ctx.params[e][key]`` block."""
    mask = torch.zeros(len(envs), len(ctx.block_names), dtype=torch.bool, device=ctx.device)
    for i, e in enumerate(envs):
        mask[i, ctx.params[e][key]] = True
    return mask


class PickPiece(Pick):
    """``Pick`` with a short-side grasp rule, a final descent, and a high lift."""

    name = "pick_piece"
    phases = ["open", "approach", "descend", "close", "lift"]

    def request(self, ctx, phase, env_ids):
        envs = env_ids.tolist()
        grasp = torch.stack([ctx.params[e]["grasp_pose"] for e in envs])
        if phase == "approach":
            goals = grasp.clone()
            goals[:, 2] += APPROACH_HEIGHT
            return PhaseRequest("plan", "pregrasp_traj", goals=goals)
        if phase == "descend":
            return PhaseRequest("ik_interp", "grasp_traj", goals=grasp, ticks=IK_ROWS,
                                exclude=_only(ctx, envs, "block"))
        if phase == "lift":
            goals = grasp.clone()
            goals[:, 2] += LIFT_HEIGHT
            return PhaseRequest("ik_interp", "grasp_delta_traj", goals=goals, ticks=2 * IK_ROWS,
                                exclude=_only(ctx, envs, "block"))
        if phase == "open":
            return PhaseRequest("gripper", "init_gripper_open_traj", closed=False, ticks=GRIP_TICKS)
        return super().request(ctx, phase, env_ids)  # "close"

    def begin(self, ctx, env_ids, args):
        b = int(env_ids.numel())
        dev = ctx.device
        j = torch.tensor([ctx.block_index[a[0]] for a in args], dtype=torch.long, device=dev)
        poses = ctx.block_poses()[env_ids]
        ar = torch.arange(b, device=dev)
        blk = poses[ar, j]
        size = ctx.sizes[env_ids, j]
        ee = ctx.sim.ee_pose()[env_ids]
        q_now = ctx.sim.joint_pos()[env_ids]
        cands = grasp_candidates(blk, size, ee)  # [B, 4, 7]
        wide = size[:, :2].amax(dim=1) > MAX_JAW_OPENING
        allowed = (jaw_alignment(cands, blk, size) >= MIN_JAW_ALIGNMENT) | ~wide[:, None]
        # The open fingers straddle the piece being picked (cuRobo's finger spheres
        # overlap an 8 cm piece), so that piece is not an obstacle for these IK checks;
        # the descend phase likewise excludes it.
        held = torch.zeros(ctx.sim.num_envs, len(ctx.block_names), dtype=torch.bool, device=dev)
        held[env_ids, j] = True
        ctx.geo.update_worlds(ctx.block_poses(), ctx.table_pose(), exclude=held, env_ids=env_ids)
        ignore = (poses[:, :, :2] - blk[:, None, :2]).norm(dim=-1) < 0.6 * size[:, None, 0]
        clear = torch.full((b, 4), float("-inf"), device=dev)
        for c in range(4):
            ok_c, q_c, perr, rerr = ctx.geo.ik(env_ids, cands[:, c], seeds=q_now)
            acc = (ok_c | ((perr <= 0.005) & (rerr <= 0.05))) & allowed[:, c]
            gap = ctx.geo.block_clearance(env_ids, q_c, ignore)
            clear[:, c] = torch.where(acc, gap, clear[:, c])
        ok = (clear > float("-inf")).any(dim=1)
        enough = clear >= GRASP_NEIGHBOR_CLEARANCE
        first_enough = torch.argmax(enough.int(), dim=1)
        pick = torch.where(enough.any(dim=1), first_enough, clear.argmax(dim=1))
        chosen = cands[ar, pick]
        for i, e in enumerate(env_ids.tolist()):
            ctx.params[e].update({"block": int(j[i]), "grasp_pose": chosen[i].clone(), "z0": float(blk[i, 2])})
        return ok


class PlaceAt(Place):
    """Place the held piece at ``ctx.params[e]["put"]``: an exact base-frame pose (x, y, z, w, qx, qy, qz)."""

    def __init__(self):
        super().__init__("table")
        self.name = "place_at"

    def begin(self, ctx, env_ids, args):
        b = int(env_ids.numel())
        dev = ctx.device
        ar = torch.arange(b, device=dev)
        j = torch.tensor([ctx.block_index[a[0]] for a in args], dtype=torch.long, device=dev)
        blk = ctx.block_poses()[env_ids][ar, j]
        ee = ctx.sim.ee_pose()[env_ids]
        in_hand = pose_compose(pose_inverse(blk), ee)  # T_obj^-1 T_ee
        target = torch.stack([torch.as_tensor(ctx.params[e]["put"], dtype=blk.dtype, device=dev)
                              for e in env_ids.tolist()])
        put = target.clone()
        put[:, 2] += PLACE_Z_GAP  # release just above the support, as Place does
        ee_place = pose_compose(put, in_hand)
        ee_pre = ee_place.clone()
        ee_pre[:, 2] += PRE_PLACE_STANDOFF + NEIGHBOUR_CLEARANCE
        for i, e in enumerate(env_ids.tolist()):
            ctx.params[e].update({"block": int(j[i]), "target": target[i].clone(),
                                  "ee_place": ee_place[i].clone(), "ee_pre": ee_pre[i].clone()})
        return ctx.held[env_ids] == j

    def verify(self, ctx, env_ids):
        envs = env_ids.tolist()
        dev = ctx.device
        j = torch.tensor([ctx.params[e]["block"] for e in envs], dtype=torch.long, device=dev)
        blk = ctx.block_poses()[env_ids][torch.arange(len(envs), device=dev), j]
        target = torch.stack([ctx.params[e]["target"] for e in envs])
        return (blk[:, :3] - target[:, :3]).norm(dim=1) <= PLACE_VERIFY_TOLERANCE
