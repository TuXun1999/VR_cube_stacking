"""Genesis-free V2 table-map parsing shared by file and live replay."""
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

TARGET_CUBE_SIZE = 0.08
WORKSPACE_TABLE_NEAR_X = 0.06
# Right-handed table frame: right (+X) -> +Y, up (+Y) -> +Z,
# and down-screen/toward the viewer (+Z) -> +X.
VR_TO_GENESIS_AXES = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

COLOR_RGB = {
    "red": (0.90, 0.10, 0.10),
    "green": (0.10, 0.75, 0.20),
    "blue": (0.10, 0.35, 0.95),
}

@dataclass(frozen=True)
class CubePlacement:
    cube_id: str
    color_name: str
    source_position: tuple[float, float, float]
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


def placements_share_pose(first, second):
    """Use the same pose tolerance for reconciliation and robot execution."""
    return (np.allclose(first.position, second.position, atol=1e-5, rtol=0)
            and abs(np.dot(first.quaternion, second.quaternion)) > 1 - 1e-6)


@dataclass(frozen=True)
class TableLayout:
    source_path: Path
    source_cube_size: float
    scale: float
    source_width: float
    source_depth: float
    workspace_width: float
    workspace_depth: float
    workspace_center: tuple[float, float, float]
    placements: tuple[CubePlacement, ...]


def rotation_matrix_to_quaternion(rotation):
    """Convert a 3x3 rotation matrix to a normalized (w, x, y, z) quaternion."""
    matrix = np.asarray(rotation, dtype=float)
    trace = np.trace(matrix)
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return tuple(quaternion / np.linalg.norm(quaternion))


def load_table_layout(path, target_cube_size=TARGET_CUBE_SIZE):
    """Load and transform an openup.cube_stack.map JSON file for Genesis."""
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    return parse_table_layout(data, target_cube_size, source_path=path)


def parse_table_layout(data, target_cube_size=TARGET_CUBE_SIZE, source_path="<live>", *, require_ids=False):
    """Validate a decoded complete V2 map and convert its right-handed frame."""
    try:
        return _parse_table_layout(data, target_cube_size, Path(source_path), require_ids)
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError(f"Malformed table map: {error}") from error


def _parse_table_layout(data, target_cube_size, path, require_ids):
    if not isinstance(data, dict):
        raise ValueError("The table map must be a JSON object.")
    if data.get("version") != 2:
        raise ValueError("Only openup.cube_stack.map version 2 is supported.")
    if data.get("coordinate_space") != "table_local":
        raise ValueError("The table map must use table_local coordinates.")
    if not math.isfinite(target_cube_size) or target_cube_size <= 0:
        raise ValueError("target_cube_size must be positive and finite.")
    if data.get("format") != "openup.cube_stack.map":
        raise ValueError("Unsupported map format; expected openup.cube_stack.map.")
    if data.get("matrix_layout") != "row_major":
        raise ValueError("Only row-major object_to_table matrices are supported.")
    if data.get("units") != "meters":
        raise ValueError("The table map must use meters.")

    expected_axes = {
        "origin": "table_surface_center",
        "x": "right",
        "y": "up",
        "z": "down_screen",
    }
    if data.get("axes") != expected_axes:
        raise ValueError(f"Unexpected table axes: {data.get('axes')!r}.")
    cubes = data["cubes"]
    if not isinstance(cubes, list):
        raise ValueError("cubes must be a JSON list.")

    source_cube_size = float(data["cube_size"])
    if not math.isfinite(source_cube_size) or source_cube_size <= 0.0:
        raise ValueError("cube_size must be a positive finite value.")
    scale = target_cube_size / source_cube_size
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Derived cube scale must be positive and finite.")

    table = data["table"]
    source_width = float(table["width"])
    source_depth = float(table["depth"])
    if not all(math.isfinite(value) and value > 0 for value in (source_width, source_depth)):
        raise ValueError("Table width and depth must be positive and finite.")

    workspace_width = source_width * scale
    workspace_depth = source_depth * scale
    if not all(math.isfinite(value) and value > 0 for value in (workspace_width, workspace_depth)):
        raise ValueError("Derived workspace dimensions must be positive and finite.")
    workspace_center = np.array(
        [WORKSPACE_TABLE_NEAR_X + workspace_depth / 2, 0.0, 0.0]
    )
    if not np.all(np.isfinite(workspace_center)):
        raise ValueError("Derived workspace center must be finite.")

    serials = Counter()
    cube_ids = set()
    placements = []
    for cube_index, cube in enumerate(cubes):
        if not isinstance(cube, dict):
            raise ValueError(f"Cube {cube_index} must be a JSON object.")
        color_name = str(cube["color"]).lower()
        if color_name not in COLOR_RGB:
            raise ValueError(f"Cube {cube_index} has unsupported color {color_name!r}.")
        serials[color_name] += 1
        cube_id = cube.get("id", f"{color_name}_{serials[color_name]}")
        if require_ids and "id" not in cube:
            raise ValueError(f"Cube {cube_index} requires a stable id for live protocol 2.")
        if not isinstance(cube_id, str) or not cube_id:
            raise ValueError(f"Cube {cube_index} id must be a nonempty string.")
        if cube_id in cube_ids:
            raise ValueError(f"Duplicate cube id {cube_id!r}.")
        cube_ids.add(cube_id)

        source_position = np.asarray(cube["position"], dtype=float)
        if source_position.shape != (3,) or not np.all(np.isfinite(source_position)):
            raise ValueError(f"Cube {cube_index} has an invalid position.")
        half_cube = source_cube_size / 2
        # OpenUp accepts centers on the table boundary, including the natural
        # half-cube overhang at its edge.
        if abs(source_position[0]) > source_width / 2 + 1e-5:
            raise ValueError(f"Cube {cube_index} center is beyond the table width.")
        if abs(source_position[2]) > source_depth / 2 + 1e-5:
            raise ValueError(f"Cube {cube_index} center is beyond the table depth.")
        if source_position[1] < half_cube - 1e-5:
            raise ValueError(f"Cube {cube_index} extends below the table surface.")

        transform = np.asarray(cube["object_to_table"], dtype=float)
        if transform.size != 16 or not np.all(np.isfinite(transform)):
            raise ValueError(f"Cube {cube_index} has an invalid object_to_table matrix.")
        transform = transform.reshape(4, 4)
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
            raise ValueError(f"Cube {cube_index} has an invalid homogeneous matrix row.")
        if not np.allclose(transform[:3, 3], source_position, atol=1e-5):
            raise ValueError(f"Cube {cube_index} matrix and position disagree.")

        scaled_rotation = transform[:3, :3]
        axis_scales = np.linalg.norm(scaled_rotation, axis=0)
        if not np.allclose(axis_scales, source_cube_size, rtol=1e-4, atol=1e-6):
            raise ValueError(f"Cube {cube_index} matrix scale does not match cube_size.")
        source_rotation = scaled_rotation / source_cube_size
        if not np.allclose(source_rotation.T @ source_rotation, np.eye(3), atol=1e-5):
            raise ValueError(f"Cube {cube_index} contains a non-rigid rotation.")
        if not math.isclose(np.linalg.det(source_rotation), 1.0, abs_tol=1e-5):
            raise ValueError(f"Cube {cube_index} rotation contains a reflection.")

        with np.errstate(over="ignore", invalid="ignore"):
            workspace_position = workspace_center + VR_TO_GENESIS_AXES @ (
                scale * source_position
            )
        if not np.all(np.isfinite(workspace_position)):
            raise ValueError(f"Cube {cube_index} has a non-finite transformed position.")
        workspace_rotation = (
            VR_TO_GENESIS_AXES @ source_rotation @ VR_TO_GENESIS_AXES.T
        )
        placements.append(
            CubePlacement(
                cube_id=cube_id,
                color_name=color_name,
                source_position=tuple(float(value) for value in source_position),
                position=tuple(float(value) for value in workspace_position),
                quaternion=tuple(
                    float(value)
                    for value in rotation_matrix_to_quaternion(workspace_rotation)
                ),
            )
        )

    # Preserve the producer's order. In the example, this encodes the blue
    # structure's support dependencies before unrelated ground-level cubes.
    return TableLayout(
        source_path=path,
        source_cube_size=source_cube_size,
        scale=scale,
        source_width=source_width,
        source_depth=source_depth,
        workspace_width=workspace_width,
        workspace_depth=workspace_depth,
        workspace_center=tuple(float(value) for value in workspace_center),
        placements=tuple(placements),
    )
