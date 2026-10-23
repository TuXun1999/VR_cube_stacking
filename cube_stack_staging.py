"""Table-height supply/return stacks beside the construction footprint."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StagingSlot:
    side: str
    color: str
    column: int
    level: int
    position: tuple[float, float, float]
    lane: str = ""
    length: int = 1
    quaternion: tuple = (1, 0, 0, 0)


class StagingArea:
    """Six cells per cube color, aligned with the matching table-side stripe."""

    colors = ("red", "green", "blue")
    lanes = colors
    platform_height = 0.

    def __init__(self, layout, cube_size=.08, max_piece_length=1):
        if not math.isfinite(cube_size) or cube_size <= 0:
            raise ValueError("cube_size must be positive and finite")
        if type(max_piece_length) is not int or not 1 <= max_piece_length <= 3:
            raise ValueError("max_piece_length must be an integer in 1..3")
        self.cube_size = cube_size
        self.max_piece_length = max_piece_length
        self.near_x = layout.workspace_center[0] - layout.workspace_depth / 2
        self.center_y = layout.workspace_center[1]
        self.half_width = layout.workspace_width / 2
        self.cell_footprint = (cube_size * max_piece_length + .04, cube_size + .04)
        self.columns = (3 if max_piece_length == 1 else
                        max(0, int((layout.workspace_depth + .04) / self.cell_footprint[0])))
        if not self.columns:
            raise ValueError("Table depth cannot fit a complete staging piece")
        self.stacks = {(side, lane, column): []
                       for side in ("left", "right")
                       for lane in self.lanes for column in range(self.columns)}
        self.slots = {}

    def position(self, side, lane, column, level=0):
        row, along_table = self.lanes.index(lane), column
        pitch = self.cell_footprint[0]
        x = self.near_x + (.25 * self.cube_size if self.max_piece_length == 1
                           else .5 * self.max_piece_length * self.cube_size) + along_table * pitch
        half_diagonal = self.cube_size / math.sqrt(2)
        y = self.half_width + half_diagonal + .025 + row * self.cell_footprint[1]
        if side == "left":
            y = -y
        return (x, self.center_y + y,
                self.platform_height + self.cube_size * (level + .5))

    def reserve(self, key, color, side, exclude_column=None, exclude_lane=None,
                length=1, feasible=None):
        if type(length) is not int or not 1 <= length <= self.max_piece_length:
            raise ValueError("length must be an integer supported by this staging area")
        if color not in self.colors or side not in ("left", "right"):
            raise ValueError("Unknown staging color or side")
        if key in self.slots:
            raise RuntimeError(f"{key} already has a staging slot")
        excluded = (side, exclude_lane or color, exclude_column)
        candidates = sorted(
            (cell for cell in self.stacks if cell[1] == color and cell != excluded),
            key=lambda cell: (len(self.stacks[cell]), cell[0] != side,
                              cell[2]))
        quaternion = ((1, 0, 0, 0) if self.max_piece_length == 1 else
                      (2**-.5, 0, 0, 2**-.5))
        for physical in candidates:
            actual_side, lane, column = physical
            stack = self.stacks[physical]
            slot = StagingSlot(actual_side, color, column, len(stack),
                               self.position(actual_side, lane, column, len(stack)),
                               lane, length, quaternion)
            if feasible is not None and not feasible(slot):
                continue
            stack.append(key)
            self.slots[key] = slot
            return slot
        raise RuntimeError(f"No reachable staging cell for {key} ({color}, length {length})")

    def is_top(self, key):
        slot = self.slots[key]
        return self.stacks[slot.side, slot.lane, slot.column][-1] == key

    def relocate(self, key, feasible=None):
        """Move a top reservation to another cell, restoring it on rejection."""
        previous = self.slots[key]
        self.release(key)
        try:
            return self.reserve(key, previous.color, previous.side,
                                exclude_column=previous.column, exclude_lane=previous.lane,
                                length=previous.length, feasible=feasible)
        except Exception:
            self.stacks[previous.side, previous.lane, previous.column].append(key)
            self.slots[key] = previous
            raise

    def top_above(self, key):
        slot = self.slots[key]
        stack = self.stacks[slot.side, slot.lane, slot.column]
        return None if stack[-1] == key else stack[-1]

    def release(self, key):
        if not self.is_top(key):
            raise RuntimeError(f"Cannot pick buried staging cube {key}")
        slot = self.slots.pop(key)
        self.stacks[slot.side, slot.lane, slot.column].pop()

    def clear(self):
        self.slots.clear()
        for stack in self.stacks.values():
            stack.clear()
