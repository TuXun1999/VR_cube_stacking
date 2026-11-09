"""Simulator-independent live planning shared by every robot backend.

``LiveScenePlanner`` turns one live command (add, move or delete a piece) into robot
transfers. It clears cubes that block a pickup or destination, stages supply and
returned pieces, and restores staged pieces only once their supports are in place.
It never calls a physics engine. A backend mixes it in and provides these hooks:

- ``self.blocks`` and each ``PoolCube.block``: entities with ``get_pos()``,
  ``get_quat()``, ``set_pos(p, zero_velocity=True)`` and
  ``set_quat(q, zero_velocity=True)`` in the planner frame. That is the world frame
  produced by ``cube_stack_layout``: table surface at z = 0, near table edge at
  X = 0.06, scalar-first (w, x, y, z) quaternions.
- ``_transfer_block(block, placement)``: carry a body to a placement or raise.
- ``_staging_feasible(item, slot, destination=None)``: whether the robot can use a
  staging slot for ``item`` (and later carry it on to ``destination``).
- ``_step()``: advance the simulation one step and pump the live client
  (``cube_stack_receiver.pump_and_check``).
- ``_initialize_poses()``: return the robot to its home pose.
- ``_place_offline(placement)``: optional; only offline JSON replay uses it.

The backend also sets ``layout``, ``options`` (``live``, ``pool_per_color``,
``physical``, ``speed``), ``live_client``, ``live_queue`` and the ``carried_*``
attributes before calling ``_setup_live_pool()``.
"""
from dataclasses import dataclass, replace

import numpy as np

from cube_stack_layout import (
    TARGET_CUBE_SIZE, CubePlacement, PiecePlacement, as_piece, placements_share_pose,
    quaternion_to_rotation,
)
from cube_stack_piece_geometry import footprints_overlap, members_at_pose
from cube_stack_staging import StagingArea

SPAWN_SETTLE_STEPS = 20


def to_numpy(value):
    """Convert Genesis/Torch values to NumPy arrays."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass
class PoolCube:
    key: str
    color: str
    block: object
    parked_position: tuple
    cube_id: str | None = None
    staging_slot: object = None
    length: int = 1
    piece: PiecePlacement | None = None


class LiveScenePlanner:
    """Mix-in that plans live add/move/delete commands; see the module docstring."""

    spawn_settle_steps = SPAWN_SETTLE_STEPS

    # -- hooks every backend provides -------------------------------------------------

    def _transfer_block(self, block, placement):
        raise NotImplementedError

    def _staging_feasible(self, item, slot, destination=None):
        raise NotImplementedError

    def _step(self):
        raise NotImplementedError

    def _initialize_poses(self):
        raise NotImplementedError

    def _place_offline(self, placement):
        raise NotImplementedError("This backend supports live placement only")

    # -- policy a backend may override ------------------------------------------------

    def _make_staging(self):
        return StagingArea(self.layout, TARGET_CUBE_SIZE,
                           max_piece_length=3 if self.layout.map_version == 3 else 1)

    @staticmethod
    def _supply_side(position):
        """The staging side that serves a body or target at ``position``."""
        return "right" if position[1] >= 0 else "left"

    def clear_carried(self):
        """Forget any body the robot was carrying (after a reset or failure)."""
        self.carried_block = self.carried_tool = self.carried_quaternion = self.carried_tool_to_piece = None

    # -- planning ---------------------------------------------------------------------

    def _scaled_steps(self, steps, allow_zero=False):
        minimum = 0 if allow_zero else 1
        return max(minimum, round(steps / self.options.speed))

    def _setup_live_pool(self):
        self.staging = self._make_staging()
        self.pool_cubes = {
            key: PoolCube(key, getattr(self, "_pool_specs", {}).get(key, (key.split("_")[0], 1))[0],
                          block, tuple(to_numpy(block.get_pos())),
                          length=getattr(self, "_pool_specs", {}).get(key, (None, 1))[1])
            for key, block in self.blocks.items()
        }
        self.blocks = {}
        self.current_placements = {}
        self.placed_placements = []

    @staticmethod
    def _same_pose(first, second):
        return placements_share_pose(first, second)

    def piece_members(self, piece_id):
        piece = self._body_piece(piece_id)
        return members_at_pose(piece, piece.position, piece.quaternion)

    @staticmethod
    def _validate_piece(placement):
        piece = as_piece(placement)
        if piece.length > 1 and not np.allclose(quaternion_to_rotation(piece.quaternion)[:, 2],
                                               (0, 0, 1), atol=1e-5):
            raise ValueError("The robot controller supports horizontal glued pieces only")
        return piece

    def _remember(self, placement):
        placement = replace(as_piece(placement),
            position=tuple(float(v) for v in to_numpy(self.blocks[placement.cube_id].get_pos())),
            quaternion=tuple(float(v) for v in to_numpy(self.blocks[placement.cube_id].get_quat())))
        self.current_placements[placement.cube_id] = placement
        self.placed_placements = list(self.current_placements.values())
        if self.live_queue is not None and hasattr(self.live_queue, "realized"):
            self.live_queue.realized[placement.cube_id] = placement

    def _sample_physical_state(self, queue):
        """Replace command targets with measured bodies at a safe boundary."""
        self.current_placements = {
            cube_id: replace(placement,
                             position=tuple(float(v) for v in to_numpy(self.blocks[cube_id].get_pos())),
                             quaternion=tuple(float(v) for v in to_numpy(self.blocks[cube_id].get_quat())))
            for cube_id, placement in self.current_placements.items()
        }
        self.placed_placements = list(self.current_placements.values())
        queue.realized = dict(self.current_placements)

    def _pool_cube(self, cube_id):
        return next((item for item in self.pool_cubes.values() if item.cube_id == cube_id), None)

    def _staging_target(self, item, slot):
        template = item.piece or as_piece(CubePlacement(item.cube_id or item.key, item.color,
                                                       (0, 0, 0), (0, 0, 0), (1, 0, 0, 0)))
        quaternion = slot.quaternion if self.staging.max_piece_length > 1 else tuple(to_numpy(item.block.get_quat()))
        return replace(template, position=slot.position, quaternion=quaternion)

    def _reserve_staging(self, item, side, destination=None, **kwargs):
        return self.staging.reserve(item.key, item.color, side, length=item.length,
            feasible=lambda slot: self._staging_feasible(item, slot, destination), **kwargs)

    def _ensure_staging_access(self, item):
        """Unstack supply cubes with the robot before picking a buried object."""
        while not self.staging.is_top(item.key):
            blocker = self.pool_cubes[self.staging.top_above(item.key)]
            previous = blocker.staging_slot
            slot = self.staging.relocate(blocker.key,
                feasible=lambda candidate: self._staging_feasible(blocker, candidate))
            blocker.staging_slot = slot
            target = self._staging_target(blocker, slot)
            self._transfer_block(blocker.block, target)
            if blocker.cube_id is not None:
                self._remember(target)

    def _stage_live_cube(self, cube_id):
        item = self._pool_cube(cube_id)
        if item is None or item.staging_slot is not None:
            return
        side = self._supply_side(to_numpy(item.block.get_pos()))
        slot = self._reserve_staging(item, side)
        item.staging_slot = slot
        target = self._staging_target(item, slot)
        self._transfer_block(item.block, target)
        self._remember(target)

    def _restore_staged_cube(self, target):
        item = self._pool_cube(target.cube_id)
        self._ensure_staging_access(item)
        self._transfer_block(item.block, target)
        self.staging.release(item.key)
        item.staging_slot = None
        self._remember(target)

    def _detach_returned_cube(self, cube_id):
        item = self._pool_cube(cube_id)
        if item is not None and self.options.physical:
            actual = to_numpy(item.block.get_pos())
            if (item.staging_slot is None or
                    np.linalg.norm(actual - item.staging_slot.position) > TARGET_CUBE_SIZE / 8):
                raise RuntimeError(f"Physical return did not remove {cube_id} to staging")
        if item is not None:
            item.cube_id = None
        self.blocks.pop(cube_id, None)
        self.current_placements.pop(cube_id, None)
        self.placed_placements = list(self.current_placements.values())
        if self.live_queue is not None and hasattr(self.live_queue, "realized"):
            self.live_queue.realized.pop(cube_id, None)

    def _body_piece(self, cube_id):
        template = self.current_placements.get(cube_id)
        if template is None:
            item = self._pool_cube(cube_id)
            template = (item.piece if item and item.piece is not None else
                        CubePlacement(cube_id, item.color if item else "blue", (0, 0, 0),
                                      (0, 0, 0), (1, 0, 0, 0)))
        block = self.blocks[cube_id]
        return replace(as_piece(template), position=tuple(to_numpy(block.get_pos())),
                       quaternion=tuple(to_numpy(block.get_quat())))

    def _destination_occupants(self, target):
        return {cube_id for cube_id in self.blocks
                if (cube_id != target.cube_id
                    and abs(self._body_piece(cube_id).position[2] - target.position[2]) < TARGET_CUBE_SIZE - 1e-5
                    and footprints_overlap(target, self._body_piece(cube_id)))}

    def _stack_blockers(self, cube_ids):
        affected = set(cube_ids)
        while True:
            additional = set()
            for cube_id in affected:
                source = self._body_piece(cube_id)
                for other_id in self.blocks:
                    if other_id in affected:
                        continue
                    other = self._body_piece(other_id)
                    if (other.position[2] > source.position[2] + 1e-5
                            and footprints_overlap(source, other)):
                        additional.add(other_id)
            if not additional:
                return affected
            affected.update(additional)

    def _ready_to_restore(self, target, desired):
        for support in desired.values():
            if (support.position[2] < target.position[2] - 1e-5
                    and footprints_overlap(support, target)):
                actual = self.current_placements.get(support.cube_id)
                if actual is None or not self._same_pose(actual, support):
                    return False
        return True

    def _restore_ready_targets(self, cube_ids, desired):
        pending = {key for key in set(cube_ids) & desired.keys()
                   if self._pool_cube(key) is not None
                   and self._pool_cube(key).staging_slot is not None}
        while pending:
            ready = [desired[key] for key in pending
                     if self._ready_to_restore(desired[key], desired)]
            if not ready:
                return  # New supports still queued; realized staging poses retain this work.
            def order(target):
                item = self._pool_cube(target.cube_id)
                return (target.position[2], not self.staging.is_top(item.key),
                        -item.staging_slot.level, target.cube_id)
            target = min(ready, key=order)
            self._restore_staged_cube(target)
            pending.remove(target.cube_id)

    def _resolve_obstruction(self, cube_id, desired, protected, deferred):
        target = desired.get(cube_id)
        current = self.current_placements[cube_id]
        if (target is not None and not self._same_pose(current, target)
                and self._ready_to_restore(target, desired)
                and self._try_direct_move(target, desired, protected, deferred)):
            return
        # A protected upper cube can create a genuine support cycle. Clear it
        # before attempting to pick its support, even when direct relocation
        # cannot yet finish. Ordinary requested moves never take this path.
        above = self._stack_blockers({cube_id}) - {cube_id}
        for upper_id in sorted(above, key=lambda key: self.blocks[key].get_pos()[2], reverse=True):
            self._stage_live_cube(upper_id)
            deferred.add(upper_id)
        if (above and target is not None and not self._same_pose(self.current_placements[cube_id], target)
                and self._ready_to_restore(target, desired)
                and self._try_direct_move(target, desired, protected, deferred)):
            return
        self._stage_live_cube(cube_id)
        deferred.add(cube_id)

    def _try_direct_move(self, target, desired, protected, deferred):
        cube_id = target.cube_id
        current = self.current_placements.get(cube_id)
        if current is None:
            return False
        if self._same_pose(current, target):
            return True
        blockers = (self._stack_blockers({cube_id}) - {cube_id}) | self._destination_occupants(target)
        if blockers & protected:
            return False  # A swap must clear the other cube without staging this request.
        protected = protected | {cube_id}
        for blocker_id in sorted(blockers, key=lambda key: self.blocks[key].get_pos()[2], reverse=True):
            remaining = (self._stack_blockers({cube_id}) - {cube_id}) | self._destination_occupants(target)
            if blocker_id not in remaining:
                continue
            self._resolve_obstruction(blocker_id, desired, protected, deferred)
        item = self._pool_cube(cube_id)
        if item.staging_slot is not None:
            self._restore_staged_cube(target)
        else:
            self._transfer_block(item.block, target)
            self._remember(target)
        return True

    def move_live_cube(self, placement):
        """Pick the requested cube at its current pose and place it directly."""
        placement = self._validate_piece(placement)
        current = self.current_placements.get(placement.cube_id)
        if current is not None and (as_piece(current).length != placement.length or current.color_name != placement.color_name):
            raise ValueError("A glued piece cannot change length or color")
        if current is None or self._same_pose(current, placement):
            return
        desired = dict(self.current_placements)
        if self.live_queue is not None:
            desired = {p.cube_id: p for p in self.live_queue.layout.pieces}
        desired[placement.cube_id] = placement
        deferred = set()
        if not self._try_direct_move(placement, desired, set(), deferred):
            raise RuntimeError(f"Cannot clear destination for {placement.cube_id}")
        self._restore_ready_targets(deferred, desired)
        for cube_id in deferred - desired.keys():
            self._detach_returned_cube(cube_id)

    def delete_live_cube(self, placement):
        if placement.cube_id not in self.blocks:
            return
        desired = dict(self.current_placements)
        if self.live_queue is not None:
            desired = {p.cube_id: p for p in self.live_queue.layout.pieces}
        desired.pop(placement.cube_id, None)
        affected = self._stack_blockers({placement.cube_id})
        for cube_id in sorted(affected, key=lambda key: self.blocks[key].get_pos()[2], reverse=True):
            self._stage_live_cube(cube_id)
        for cube_id in affected - desired.keys():
            self._detach_returned_cube(cube_id)
        self._restore_ready_targets(affected, desired)

    def reset_live_scene(self):
        """Park every pool body and home the robot in the already built scene."""
        self.clear_carried()
        for item in self.pool_cubes.values():
            item.block.set_pos(item.parked_position, zero_velocity=True)
            item.block.set_quat((1, 0, 0, 0), zero_velocity=True)
            item.cube_id = None
            item.staging_slot = None
        self.blocks.clear()
        self.current_placements.clear()
        self.placed_placements.clear()
        self.staging.clear()
        self._initialize_poses()

    def place_live_cube(self, placement):
        """Execute one arbitrary imported pose; the caller owns queue ordering."""
        placement = self._validate_piece(placement)
        deferred = set()
        desired = {}
        try:
            if self.options.live:
                if placement.cube_id in self.blocks:
                    self.move_live_cube(placement)
                    return
                desired = dict(self.current_placements)
                if self.live_queue is not None:
                    desired = {p.cube_id: p for p in self.live_queue.layout.pieces}
                desired[placement.cube_id] = placement
                for cube_id in sorted(self._destination_occupants(placement),
                                      key=lambda key: self.blocks[key].get_pos()[2], reverse=True):
                    if cube_id not in self._destination_occupants(placement):
                        continue
                    self._resolve_obstruction(cube_id, desired, {placement.cube_id}, deferred)
                supplies = [item for item in self.pool_cubes.values()
                            if item.color == placement.color_name and item.length == placement.length and item.cube_id is None]
                if not supplies:
                    raise RuntimeError(f"No reusable {placement.color_name} supply piece of length {placement.length}")
                side = self._supply_side(placement.position)
                supplies.sort(key=lambda item: (item.staging_slot is None,
                                               item.staging_slot is not None and item.staging_slot.side != side,
                                               item.staging_slot is not None and not self.staging.is_top(item.key)))
                item = supplies[0]
                allocated = sum(candidate.length for candidate in self.pool_cubes.values()
                                if candidate.color == placement.color_name and candidate.cube_id is not None)
                if allocated + placement.length > self.options.pool_per_color:
                    raise RuntimeError(f"{placement.color_name} constituent-cube pool exhausted")
                item.piece = placement
                if item.staging_slot is None:
                    item.staging_slot = self._reserve_staging(item, side, destination=placement)
                    item.block.set_pos(item.staging_slot.position, zero_velocity=True)
                    item.block.set_quat(item.staging_slot.quaternion if self.staging.max_piece_length > 1 else placement.quaternion, zero_velocity=True)
                    for _ in range(self._scaled_steps(self.spawn_settle_steps)):
                        self._step()
                self._ensure_staging_access(item)
                item.cube_id = placement.cube_id
                self.blocks[placement.cube_id] = item.block
                self._transfer_block(item.block, placement)
                self.staging.release(item.key)
                item.staging_slot = None
            else:
                self._place_offline(placement)
            self._remember(placement)
            if deferred:
                self._restore_ready_targets(deferred, desired)
                for cube_id in deferred - desired.keys():
                    self._detach_returned_cube(cube_id)
        finally:
            # Never allow a failed partial move to keep moving a carried cube.
            self.clear_carried()

