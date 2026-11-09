"""Planner frame <-> ToyBlocks robot-base frame.

The planner frame is the one ``cube_stack_layout`` produces for every backend:
+X runs from the robot rack into the table, +Z is up, the table surface is at
z = 0 and its rack-facing (near) edge is at X = 0.06. ToyBlocks' vecgen works in
the Franka base frame: base at the origin, +X forward over the table, +Z up.
Both are right-handed with +X away from the robot and +Z up, so they differ only
by a translation, and both use scalar-first (w, x, y, z) quaternions.
"""
from dataclasses import dataclass
import math

import numpy as np

from cube_stack_layout import WORKSPACE_TABLE_NEAR_X, quaternion_to_rotation

# Distance from the Franka base axis to the table's near edge. The Phase 0 reach
# sweep (spike.py --sweep) is the place to tune it; override with
# --robot-to-near-edge.
DEFAULT_ROBOT_TO_NEAR_EDGE = 0.15


@dataclass(frozen=True)
class FrameMap:
    robot_to_near_edge: float = DEFAULT_ROBOT_TO_NEAR_EDGE
    table_top: float = 0.0  # table-surface height in the robot base frame

    @property
    def offset(self):
        """Planner origin expressed in the robot base frame."""
        return np.array((self.robot_to_near_edge - WORKSPACE_TABLE_NEAR_X, 0.0, self.table_top))

    @property
    def robot_base_in_planner(self):
        return -self.offset

    def to_robot(self, position):
        return np.asarray(position, dtype=float) + self.offset

    def to_planner(self, position):
        return np.asarray(position, dtype=float) - self.offset

    def to_robot_pose(self, position, quaternion):
        """A vecgen 7-vector (x, y, z, w, qx, qy, qz) for a planner pose."""
        return np.concatenate([self.to_robot(position), np.asarray(quaternion, dtype=float)])

    def to_planner_pose(self, pose):
        pose = np.asarray(pose, dtype=float)
        return tuple(float(v) for v in self.to_planner(pose[:3])), tuple(float(v) for v in pose[3:7])


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw))


def symmetric_quaternions(target, length=1):
    """Orientations that look identical to ``target``: quarter turns for a cube, half turns for a bar."""
    period = math.pi / 2 if length == 1 else math.pi
    return [tuple(float(v) for v in _quat_mul(np.asarray(target, dtype=float),
                                              (math.cos(k * period / 2), 0.0, 0.0, math.sin(k * period / 2))))
            for k in range(int(round(2 * math.pi / period)))]


def nearest_symmetric_quaternion(current, target, length=1):
    """The target orientation, turned by a symmetry of the piece, closest to ``current``.

    A cube looks the same after any quarter turn about its vertical axis and a
    glued bar after a half turn, so the robot can rotate the wrist the least.
    """
    return max(symmetric_quaternions(target, length), key=lambda q: abs(float(np.dot(q, current))))


def _inside_piece(point, position, quaternion, length, cube_size):
    local = quaternion_to_rotation(quaternion).T @ (np.asarray(point, dtype=float) - np.asarray(position, dtype=float))
    return (abs(local[0]) <= cube_size / 2 and abs(local[1]) <= length * cube_size / 2
            and abs(local[2]) <= cube_size / 2)


def jaw_clear_quaternion(current, target, length, jaw_world, target_position, neighbours,
                         cube_size=0.08, finger_reach=0.012):
    """Choose the symmetric target orientation that leaves the gripper's fingers room.

    ``current`` is the held piece's orientation and ``jaw_world`` the direction
    the jaws close in now; the grasp is rigid, so each candidate orientation
    carries the jaw axis with it. A side is blocked when a neighbouring piece
    (``(position, quaternion, length)``) occupies the space a finger needs just
    outside the piece's face. Fewest blocked sides wins, then the smallest wrist
    turn.
    """
    jaw_local = quaternion_to_rotation(current).T @ np.asarray(jaw_world, dtype=float)
    center = np.asarray(target_position, dtype=float)

    def score(candidate):
        jaw = quaternion_to_rotation(candidate) @ jaw_local
        jaw[2] = 0.0
        jaw /= max(np.linalg.norm(jaw), 1e-9)
        blocked = sum(any(_inside_piece(center + sign * jaw * (cube_size / 2 + finger_reach), *neighbour[:2],
                                        neighbour[2], cube_size) for neighbour in neighbours)
                      for sign in (1.0, -1.0))
        return blocked, -abs(float(np.dot(candidate, current)))

    return min(symmetric_quaternions(target, length), key=score)
