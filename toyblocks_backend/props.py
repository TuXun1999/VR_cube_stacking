"""Scene geometry for the ToyBlocks backend, computed without Isaac Sim.

Everything here is in the robot base frame (``frames.FrameMap``) unless named
``planner``. ``scene.py`` turns these boxes into Isaac Lab rigid bodies.
"""
from dataclasses import dataclass

from cube_stack_layout import COLOR_RGB, TARGET_CUBE_SIZE

TABLE_THICKNESS = 0.04
PAD_THICKNESS = 0.04
GROUND_Z = -1.05  # ToyBlocks' BaseEnv ground plane, in the robot base frame
PARK_ORIGIN = (-0.6, -0.9)  # behind the robot, on the ground
PARK_PITCH = 0.3
PARK_COLUMNS = 7
CUBE_MASS = 0.1  # kg per 8 cm cube; the Robotiq finger effort limit is low
STATIC_FRICTION = 1.0
DYNAMIC_FRICTION = 0.8


@dataclass(frozen=True)
class Box:
    name: str
    size: tuple  # full edge lengths along the body's x, y, z
    center: tuple  # robot base frame
    color: tuple


@dataclass(frozen=True)
class PoolBody:
    key: str  # ToyBlocks only treats rigid objects named "block*" as blocks
    color: str
    length: int
    size: tuple
    park_position: tuple  # robot base frame
    mass: float


def pool_bodies(map_version, pool_per_color, cube_size=TARGET_CUBE_SIZE):
    """The reusable pieces, as Genesis builds them: per colour, ``pool_per_color // n`` pieces of length n.

    Glued bars run along their local +y axis, matching the planner's piece frame.
    """
    lengths = (1, 2, 3) if map_version == 3 else (1,)
    bodies = []
    for color in COLOR_RGB:
        for length in lengths:
            for serial in range(1, pool_per_color // length + 1):
                index = len(bodies)
                row, column = divmod(index, PARK_COLUMNS)
                bodies.append(PoolBody(
                    key=f"block_{color}_L{length}_{serial}", color=color, length=length,
                    size=(cube_size, cube_size * length, cube_size),
                    park_position=(PARK_ORIGIN[0] - row * PARK_PITCH,
                                   PARK_ORIGIN[1] + column * PARK_PITCH,
                                   GROUND_Z + cube_size / 2),
                    mass=CUBE_MASS * length))
    return bodies


def table_box(layout, frames):
    """The construction table: its top at the planner's z = 0, near edge at the planner's X = 0.06."""
    center = frames.to_robot((layout.workspace_center[0], layout.workspace_center[1], -TABLE_THICKNESS / 2))
    return Box("workspace_table", (layout.workspace_depth, layout.workspace_width, TABLE_THICKNESS),
               tuple(float(v) for v in center), (0.12, 0.18, 0.24))


def pad_boxes(staging, frames):
    """The staging pads, flush with the table top."""
    boxes = []
    for side in ("left", "right"):
        x_min, x_max, y_min, y_max = staging.pad_extents(side)
        center = frames.to_robot(((x_min + x_max) / 2, (y_min + y_max) / 2, -PAD_THICKNESS / 2))
        boxes.append(Box(f"staging_pad_{side}", (x_max - x_min, y_max - y_min, PAD_THICKNESS),
                         tuple(float(v) for v in center), (0.30, 0.30, 0.30)))
    return boxes


def color_rgb(color):
    return COLOR_RGB[color]
