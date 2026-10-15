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


class StagingArea:
    """Six cells per cube color, aligned with the matching table-side stripe."""

    colors = ("red", "green", "blue")
    lanes = colors
    platform_height = 0.

    def __init__(self, layout, cube_size=.08):
        self.cube_size = cube_size
        self.near_x = layout.workspace_center[0] - layout.workspace_depth / 2
        self.center_y = layout.workspace_center[1]
        self.half_width = layout.workspace_width / 2
        self.cell_footprint = (cube_size + .04,) * 2
        self.stacks = {(side, lane, column): []
                       for side in ("left", "right")
                       for lane in self.lanes for column in range(3)}
        self.slots = {}

    def position(self, side, lane, column, level=0):
        row, along_table = self.lanes.index(lane), column
        pitch = self.cell_footprint[0]
        x = self.near_x + .25 * self.cube_size + along_table * pitch
        half_diagonal = self.cube_size / math.sqrt(2)
        y = self.half_width + half_diagonal + .025 + row * pitch
        if side == "left":
            y = -y
        return (x, self.center_y + y,
                self.platform_height + self.cube_size * (level + .5))

    def reserve(self, key, color, side, exclude_column=None, exclude_lane=None):
        if key in self.slots:
            raise RuntimeError(f"{key} already has a staging slot")
        excluded = (side, exclude_lane or color, exclude_column)
        physical = min(
            (cell for cell in self.stacks if cell[1] == color and cell != excluded),
            key=lambda cell: (len(self.stacks[cell]), cell[0] != side,
                              cell[2]))
        actual_side, lane, column = physical
        stack = self.stacks[physical]
        slot = StagingSlot(actual_side, color, column, len(stack),
                           self.position(actual_side, lane, column, len(stack)), lane)
        stack.append(key)
        self.slots[key] = slot
        return slot

    def is_top(self, key):
        slot = self.slots[key]
        return self.stacks[slot.side, slot.lane, slot.column][-1] == key

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
