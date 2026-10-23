"""Rigid glued-piece poses and oriented footprints; no Genesis dependency."""
from dataclasses import replace

import numpy as np

from cube_stack_layout import TARGET_CUBE_SIZE, as_piece, quaternion_to_rotation


def members_at_pose(piece, position, quaternion):
    """Derive member centers from a measured body pose (long axis is local Y)."""
    piece = as_piece(piece)
    rotation = quaternion_to_rotation(quaternion)
    center = np.asarray(position)
    return tuple(replace(member,
                         position=tuple(center + rotation @ np.array([
                             0, (index - (piece.length - 1) / 2) * TARGET_CUBE_SIZE, 0])),
                         quaternion=tuple(quaternion))
                 for index, member in enumerate(piece.members))


def footprints_overlap(first, second):
    """SAT rectangle overlap, excluding contact along an edge or shared face."""
    first, second = as_piece(first), as_piece(second)
    axes_a = quaternion_to_rotation(first.quaternion)[:2, :2]
    axes_b = quaternion_to_rotation(second.quaternion)[:2, :2]
    half_a = np.array([1, first.length]) * TARGET_CUBE_SIZE / 2
    half_b = np.array([1, second.length]) * TARGET_CUBE_SIZE / 2
    delta = np.asarray(second.position)[:2] - np.asarray(first.position)[:2]
    for axis in (*axes_a.T, *axes_b.T):
        radius_a = np.abs(axis @ axes_a) @ half_a
        radius_b = np.abs(axis @ axes_b) @ half_b
        if abs(axis @ delta) >= radius_a + radius_b - 1e-6:
            return False
    return True
