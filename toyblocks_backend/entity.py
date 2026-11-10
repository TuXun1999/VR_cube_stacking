"""Planner entities backed by ToyBlocks rigid bodies.

``LiveScenePlanner`` talks to bodies through ``get_pos/get_quat/set_pos/set_quat``
in the planner frame and compares poses at 1e-5. Physics leaves a placed body a
few millimetres from its target, so each entity keeps a *nominal* pose: the last
pose it was teleported or placed to. While the measured body stays within the
receiver tolerance of that nominal pose the entity reports the nominal pose
exactly, which is what Genesis's kinematic mode gives the planner. Once the body
moves further (it fell, or was knocked), the entity reports the measured pose,
the piece stays pending in OpenUp, and the queue issues a corrective move.

Import only after Isaac Sim is running.
"""
import numpy as np
import torch

from cube_stack_receiver import pose_within_tolerance

# First runs on an A40 placed 8 cm cubes 1-10 mm from their targets, so ToyBlocks
# uses a wider "placed" tolerance than the receiver's 1 cm / 3 degree default.
POSITION_TOLERANCE = 0.02
ANGLE_TOLERANCE_DEG = 5.0


def _close(position, quaternion, target_position, target_quaternion, length):
    return pose_within_tolerance(position, quaternion, target_position, target_quaternion, length,
                                 POSITION_TOLERANCE, ANGLE_TOLERANCE_DEG)


class LiveWorld:
    """What every entity needs: the vecgen sim and the planner/robot frame map."""

    def __init__(self, sim, frames):
        self.sim = sim
        self.frames = frames
        self.device = sim.device if hasattr(sim, "device") else torch.device(sim.base_env.device)

    def measured_pose(self, name):
        pose = self.sim.object_pose(name)[0].detach().cpu().numpy()
        return self.frames.to_planner_pose(pose)

    def write_pose(self, name, position, quaternion):
        """Teleport a rigid body (planner frame) with zero velocity."""
        scene = self.sim.base_env.scene
        ids = torch.tensor([0], dtype=torch.long, device=self.device)
        pose = torch.as_tensor(self.frames.to_robot_pose(position, quaternion), dtype=torch.float32,
                               device=self.device)[None].clone()
        pose[:, :3] += scene.env_origins[ids]
        body = scene.rigid_objects[name]
        body.write_root_pose_to_sim(pose, ids)
        body.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.device), ids)
        self.sim.forward()


class ToyBlocksEntity:
    def __init__(self, world, name, length=1):
        self.world = world
        self.name = name
        self.length = length
        self.nominal = None

    def measured(self):
        return self.world.measured_pose(self.name)

    def _resolved(self):
        position, quaternion = self.measured()
        if self.nominal is not None and _close(position, quaternion, *self.nominal, self.length):
            return self.nominal
        return position, quaternion

    def get_pos(self):
        return np.array(self._resolved()[0])

    def get_quat(self):
        return np.array(self._resolved()[1])

    def set_pos(self, position, zero_velocity=True):
        self.teleport(position, self._resolved()[1])

    def set_quat(self, quaternion, zero_velocity=True):
        self.teleport(self._resolved()[0], quaternion)

    def teleport(self, position, quaternion):
        self.world.write_pose(self.name, position, quaternion)
        self.nominal = (tuple(float(v) for v in position), tuple(float(v) for v in quaternion))

    def accept(self, position, quaternion):
        """Adopt a target as nominal if the body physically reached it; return whether it did."""
        measured_position, measured_quaternion = self.measured()
        reached = _close(measured_position, measured_quaternion, position, quaternion, self.length)
        self.nominal = ((tuple(float(v) for v in position), tuple(float(v) for v in quaternion))
                        if reached else None)
        return reached
