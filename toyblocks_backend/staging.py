"""Supply/return pads beside a single arm's base.

Genesis stages pieces on strips beyond the table's left and right ends, where its
two arms reach them. One Franka at the middle of the rack edge cannot reach
those, so this staging area puts the same colour lanes on two pads beside the
robot base, in front of the table's near edge. Everything except cell geometry
(reservation, stacking, relocation, release) is inherited unchanged.
"""
import math

from cube_stack_staging import StagingArea

PAD_CLEARANCE = 0.02  # gap between the pads and the table's near edge
DEFAULT_INNER_Y = 0.30  # lateral distance from the robot axis to the first lane


class ReachableStagingArea(StagingArea):
    def __init__(self, layout, cube_size=.08, max_piece_length=1, inner_y=DEFAULT_INNER_Y):
        super().__init__(layout, cube_size, max_piece_length)
        if not math.isfinite(inner_y) or inner_y <= 0:
            raise ValueError("inner_y must be positive and finite")
        self.inner_y = inner_y
        # Cells run back from the table's near edge, beside the robot base.
        self.columns = 2 if max_piece_length == 1 else 1
        self.first_x = self.near_x - PAD_CLEARANCE - self.cell_footprint[0] / 2
        self.stacks = {(side, lane, column): []
                       for side in ("left", "right")
                       for lane in self.lanes for column in range(self.columns)}

    def position(self, side, lane, column, level=0):
        row = self.lanes.index(lane)
        x = self.first_x - column * self.cell_footprint[0]
        y = self.inner_y + row * self.cell_footprint[1]
        if side == "left":
            y = -y
        return (x, self.center_y + y, self.platform_height + self.cube_size * (level + .5))

    def pad_extents(self, side):
        """(x_min, x_max, y_min, y_max) of one side's pad in the planner frame."""
        depth, width = self.cell_footprint
        x_max = self.first_x + depth / 2
        x_min = self.first_x - (self.columns - 1) * depth - depth / 2
        inner = self.inner_y - width / 2
        outer = self.inner_y + (len(self.lanes) - 1) * width + width / 2
        if side == "left":
            return (x_min, x_max, self.center_y - outer, self.center_y - inner)
        return (x_min, x_max, self.center_y + inner, self.center_y + outer)
