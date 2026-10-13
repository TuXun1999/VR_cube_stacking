"""Long-edge table editor for the repeatable color-cube VR demo."""

from collections import Counter, deque
from dataclasses import dataclass
import math
import tkinter as tk
from tkinter import ttk


@dataclass(frozen=True)
class BlockSpec:
    """Geometry and appearance shared by every cube of one color."""

    name: str
    label: str
    size: tuple[float, float, float]
    color: tuple[float, float, float]


@dataclass(frozen=True)
class AnchorPlacement:
    """One uniquely identified colored cube at a table anchor and stack level."""

    cube_id: str
    color_name: str
    x: float
    y: float
    z: float


class AnchorTable:
    """A reachable 5×8 anchor grid with deterministic vertical stacking."""

    ROWS = 5
    COLUMNS = 8
    X_PITCH = 0.08
    Y_PITCH = 0.08
    X_ANCHORS = (0.10, 0.18, 0.26, 0.34, 0.42)
    Y_ANCHORS = (
        -0.28,
        -0.20,
        -0.12,
        -0.04,
        0.04,
        0.12,
        0.20,
        0.28,
    )
    MAX_STACK_SIZE = 5
    MAX_CUBES = 30

    def __init__(self, block_specs):
        self.block_specs = tuple(block_specs)
        self.specs_by_name = {spec.name: spec for spec in self.block_specs}
        if len(self.block_specs) != 3 or len(self.specs_by_name) != 3:
            raise ValueError("The color-placement demo requires three unique colors.")
        heights = {round(spec.size[2], 6) for spec in self.block_specs}
        if len(heights) != 1:
            raise ValueError("All colored cubes must have the same height.")

    @property
    def anchors(self):
        return tuple((x, y) for x in self.X_ANCHORS for y in self.Y_ANCHORS)

    @property
    def x_edges(self):
        half_pitch = self.X_PITCH / 2
        return self.X_ANCHORS[0] - half_pitch, self.X_ANCHORS[-1] + half_pitch

    @property
    def y_edges(self):
        half_pitch = self.Y_PITCH / 2
        return self.Y_ANCHORS[0] - half_pitch, self.Y_ANCHORS[-1] + half_pitch

    def nearest_anchor(self, x, y):
        return min(
            self.anchors,
            key=lambda anchor: (anchor[0] - x) ** 2 + (anchor[1] - y) ** 2,
        )

    def placement(self, cube_id, color_name, x, y, existing_placements):
        spec = self.specs_by_name[color_name]
        anchor_x, anchor_y = self.nearest_anchor(x, y)
        stack_level = sum(
            placement.x == anchor_x and placement.y == anchor_y
            for placement in existing_placements
        )
        if stack_level >= self.MAX_STACK_SIZE:
            raise ValueError(f"An anchor supports at most {self.MAX_STACK_SIZE} cubes.")
        return AnchorPlacement(
            cube_id=cube_id,
            color_name=color_name,
            x=anchor_x,
            y=anchor_y,
            z=(stack_level + 0.5) * spec.size[2],
        )

    def validate(self, placements, require_any=True):
        placements = tuple(placements)
        errors = []
        ids = [placement.cube_id for placement in placements]
        if len(set(ids)) != len(ids):
            errors.append("Every generated cube must have a unique ID.")
        if require_any and not placements:
            errors.append("Place at least one cube.")
        if len(placements) > self.MAX_CUBES:
            errors.append(f"The demo supports at most {self.MAX_CUBES} cubes.")

        stacks = {}
        for placement in placements:
            if placement.color_name not in self.specs_by_name:
                errors.append(f"Unknown cube color: {placement.color_name}.")
                continue
            if not all(
                math.isfinite(value)
                for value in (placement.x, placement.y, placement.z)
            ):
                errors.append(f"{placement.cube_id} has a non-finite coordinate.")
                continue

            nearest = self.nearest_anchor(placement.x, placement.y)
            if not (
                math.isclose(placement.x, nearest[0], abs_tol=1e-6)
                and math.isclose(placement.y, nearest[1], abs_tol=1e-6)
            ):
                errors.append(f"{placement.cube_id} is not centered on an anchor.")
            stacks.setdefault((placement.x, placement.y), []).append(placement)

        for stack in stacks.values():
            if len(stack) > self.MAX_STACK_SIZE:
                errors.append(
                    f"An anchor supports at most {self.MAX_STACK_SIZE} cubes."
                )
            ordered = sorted(stack, key=lambda placement: placement.z)
            for level, placement in enumerate(ordered):
                height = self.specs_by_name[placement.color_name].size[2]
                expected_z = (level + 0.5) * height
                if not math.isclose(placement.z, expected_z, abs_tol=1e-6):
                    errors.append(f"{placement.cube_id} has an invalid stack height.")
        return errors


class CubePlacementDesigner:
    """A 45-degree view from the long table edge with repeatable color cubes."""

    CANVAS_WIDTH = 820
    CANVAS_HEIGHT = 620
    CENTER_X = CANVAS_WIDTH / 2
    NEAR_Y = 560
    FAR_Y = 175
    NEAR_HALF_WIDTH = 365
    FAR_HALF_WIDTH = 275
    CUBE_HALF_WIDTH = 13
    CUBE_DEPTH = 7
    CUBE_HEIGHT = 25

    def __init__(self, root, block_specs, example_placements, live_mode=False):
        self.root = root
        self.block_specs = tuple(block_specs)
        self.specs_by_name = {spec.name: spec for spec in self.block_specs}
        self.table = AnchorTable(self.block_specs)
        self.example_placements = tuple(example_placements)
        self.live_mode = live_mode
        self.placements = []
        self.pending_placements = deque()
        self.completed_cube_ids = set()
        self.active_placement = None
        self.pending_color = None
        self.pending_canvas_position = None
        self.remove_mode = False
        self.serials = Counter()
        self.result = None
        self.submitted = False
        self.color_buttons = {}

        example_errors = self.table.validate(self.example_placements)
        if example_errors:
            raise ValueError(f"Invalid example layout: {example_errors[0]}")

        self._build_widgets()
        self._draw()

    def _build_widgets(self):
        self.root.title("VR multi-cube placement designer")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._cancel)

        container = ttk.Frame(self.root, padding=12)
        container.grid(sticky="nsew")
        self.canvas = tk.Canvas(
            container,
            width=self.CANVAS_WIDTH,
            height=self.CANVAS_HEIGHT,
            background="#081321",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, rowspan=13)
        self.canvas.bind("<Button-1>", self._on_table_click)
        self.canvas.bind("<Motion>", self._on_pointer_motion)

        controls = ttk.Frame(container, padding=(16, 0, 0, 0), width=260)
        controls.grid(row=0, column=1, sticky="n")
        ttk.Label(
            controls,
            text="Generate colored cubes",
            font=("TkDefaultFont", 16, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(
            controls,
            text=(
                "Click a color, then click a cell anchor. The robot starts that "
                "move immediately. Click a color again to create another cube; "
                "reusing an anchor builds a stack."
                if self.live_mode
                else "Click a color, then click a cell anchor. Click a color again "
                "to generate another cube. Reusing an anchor builds a stack. Use "
                "REMOVE CUBE, then click a stack to remove its top cube."
            ),
            wraplength=240,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(0, 14))

        for row, spec in enumerate(self.block_specs, start=2):
            button = tk.Button(
                controls,
                command=lambda name=spec.name: self._generate_cube(name),
                background=self._color_to_hex(spec.color),
                activebackground=self._color_to_hex(spec.color),
                foreground="#ffffff",
                activeforeground="#ffffff",
                font=("TkDefaultFont", 11, "bold"),
                width=20,
                relief="raised",
                borderwidth=3,
            )
            button.grid(row=row, column=0, sticky="ew", pady=4)
            self.color_buttons[spec.name] = button

        self.example_button = ttk.Button(
            controls,
            text="Load example layout",
            command=self._load_example,
        )
        self.example_button.grid(row=5, column=0, sticky="ew", pady=(18, 4))
        self.undo_button = ttk.Button(
            controls,
            text="Undo last cube",
            command=self._undo,
        )
        self.undo_button.grid(row=6, column=0, sticky="ew", pady=4)
        self.reset_button = ttk.Button(
            controls,
            text="Clear all cubes",
            command=self._reset,
        )
        self.reset_button.grid(row=7, column=0, sticky="ew", pady=4)
        self.remove_button = tk.Button(
            controls,
            text="REMOVE CUBE",
            command=self._toggle_remove_mode,
            font=("TkDefaultFont", 11, "bold"),
            width=20,
            relief="raised",
            borderwidth=3,
        )
        self.remove_button.grid(row=8, column=0, sticky="ew", pady=(18, 4))
        self.complete_button = ttk.Button(
            controls,
            text=("Finish real-time demo" if self.live_mode else "Build with robots"),
            command=self._complete,
            state="disabled",
        )
        self.complete_button.grid(row=9, column=0, sticky="ew", pady=(18, 4))
        self.cancel_button = ttk.Button(
            controls,
            text="Cancel",
            command=self._cancel,
        )
        self.cancel_button.grid(row=10, column=0, sticky="ew", pady=4)

        self.status = tk.StringVar()
        ttk.Label(
            controls,
            textvariable=self.status,
            wraplength=240,
            justify="left",
        ).grid(row=11, column=0, sticky="w", pady=(18, 0))

        if self.live_mode:
            for button in (
                self.example_button,
                self.undo_button,
                self.reset_button,
                self.remove_button,
            ):
                button.configure(state="disabled")

    def _draw(self):
        self.canvas.delete("all")
        self._draw_table()
        self._draw_anchors()
        for placement in sorted(
            self.placements,
            key=lambda item: (
                self._world_to_canvas(item.x, item.y)[1],
                item.z,
            ),
        ):
            self._draw_cube(placement)
        self._draw_pending_cube()

        errors = self.table.validate(self.placements)
        self.complete_button.configure(
            state="disabled" if errors or self.submitted else "normal"
        )
        if self.remove_mode:
            status = "Remove mode: click a cube or occupied cell to remove its top cube."
        elif self.pending_color is not None:
            label = self.specs_by_name[self.pending_color].label
            status = f"New {label.lower()} cube ready. Click a cell-center anchor."
        elif self.live_mode and self.active_placement is not None:
            queued = len(self.pending_placements)
            suffix = f" {queued} more queued." if queued else ""
            status = f"Robot is placing {self.active_placement.cube_id}.{suffix}"
        elif self.live_mode and self.pending_placements:
            status = f"{len(self.pending_placements)} cube(s) queued for the robot."
        elif errors:
            status = errors[0]
        elif self.live_mode:
            status = (
                f"{len(self.completed_cube_ids)}/{len(self.placements)} cubes built. "
                "Choose another color or finish."
            )
        else:
            status = f"{len(self.placements)} cubes ready. Add more or start Genesis."
        self.status.set(status)
        self._update_color_buttons()

    def _draw_table(self):
        corners = (
            self._normalized_to_canvas(0.0, 0.0),
            self._normalized_to_canvas(0.0, 1.0),
            self._normalized_to_canvas(1.0, 1.0),
            self._normalized_to_canvas(1.0, 0.0),
        )
        self.canvas.create_polygon(
            *(coordinate for point in corners for coordinate in point),
            fill="#20364c",
            outline="#dbeafe",
            width=3,
        )

        for row in range(self.table.ROWS + 1):
            depth = row / self.table.ROWS
            self.canvas.create_line(
                *self._normalized_to_canvas(depth, 0.0),
                *self._normalized_to_canvas(depth, 1.0),
                fill="#45627e",
            )
        for column in range(self.table.COLUMNS + 1):
            across = column / self.table.COLUMNS
            self.canvas.create_line(
                *self._normalized_to_canvas(0.0, across),
                *self._normalized_to_canvas(1.0, across),
                fill="#45627e",
            )

        self.canvas.create_text(
            20,
            20,
            anchor="nw",
            text="45° VIEW FROM THE LONG TABLE EDGE  ·  5 × 8 GRID",
            fill="#e2e8f0",
            font=("TkDefaultFont", 13, "bold"),
        )
        self.canvas.create_text(
            20,
            48,
            anchor="nw",
            text="Every glowing anchor is centered in its grid cell",
            fill="#94a3b8",
        )

    def _draw_anchors(self):
        stack_sizes = Counter((item.x, item.y) for item in self.placements)
        for x, y in self.table.anchors:
            center_x, center_y = self._world_to_canvas(x, y)
            stack_size = stack_sizes[(x, y)]
            self.canvas.create_oval(
                center_x - 10,
                center_y - 5,
                center_x + 10,
                center_y + 5,
                outline="#38bdf8",
                fill="#475569" if stack_size else "#f8fafc",
                width=2,
            )
            if stack_size:
                self.canvas.create_text(
                    center_x + 13,
                    center_y,
                    anchor="w",
                    text=f"×{stack_size}",
                    fill="#7dd3fc",
                    font=("TkDefaultFont", 8, "bold"),
                )

    def _draw_cube(self, placement):
        spec = self.specs_by_name[placement.color_name]
        ground_x, ground_y = self._world_to_canvas(placement.x, placement.y)
        level = round(placement.z / spec.size[2] - 0.5)
        base_y = ground_y - level * self.CUBE_HEIGHT
        half_width = self.CUBE_HALF_WIDTH
        depth = self.CUBE_DEPTH
        height = self.CUBE_HEIGHT

        front_left = (ground_x - half_width, base_y)
        front_right = (ground_x + half_width, base_y)
        back_left = (ground_x - half_width + 3, base_y - depth)
        back_right = (ground_x + half_width - 3, base_y - depth)
        top_front_left = (front_left[0], front_left[1] - height)
        top_front_right = (front_right[0], front_right[1] - height)
        top_back_left = (back_left[0], back_left[1] - height)
        top_back_right = (back_right[0], back_right[1] - height)

        outline = "#e2e8f0"
        self._polygon(
            (front_left, front_right, top_front_right, top_front_left),
            self._shade(spec.color, 0.75),
            outline,
        )
        self._polygon(
            (front_right, back_right, top_back_right, top_front_right),
            self._shade(spec.color, 0.58),
            outline,
        )
        self._polygon(
            (top_front_left, top_front_right, top_back_right, top_back_left),
            self._shade(spec.color, 1.18),
            outline,
        )

    def _draw_pending_cube(self):
        """Draw the newly generated cube as a translucent cursor follower."""
        if self.pending_color is None or self.pending_canvas_position is None:
            return

        spec = self.specs_by_name[self.pending_color]
        ground_x, ground_y = self.pending_canvas_position
        half_width = self.CUBE_HALF_WIDTH
        depth = self.CUBE_DEPTH
        height = self.CUBE_HEIGHT
        front_left = (ground_x - half_width, ground_y)
        front_right = (ground_x + half_width, ground_y)
        back_left = (ground_x - half_width + 3, ground_y - depth)
        back_right = (ground_x + half_width - 3, ground_y - depth)
        top_front_left = (front_left[0], front_left[1] - height)
        top_front_right = (front_right[0], front_right[1] - height)
        top_back_left = (back_left[0], back_left[1] - height)
        top_back_right = (back_right[0], back_right[1] - height)
        ghost_options = {"dash": (4, 2), "stipple": "gray50"}
        outline = "#ffffff"
        self._polygon(
            (front_left, front_right, top_front_right, top_front_left),
            self._shade(spec.color, 0.75),
            outline,
            **ghost_options,
        )
        self._polygon(
            (front_right, back_right, top_back_right, top_front_right),
            self._shade(spec.color, 0.58),
            outline,
            **ghost_options,
        )
        self._polygon(
            (top_front_left, top_front_right, top_back_right, top_back_left),
            self._shade(spec.color, 1.18),
            outline,
            **ghost_options,
        )

    def _polygon(self, points, fill, outline, **options):
        self.canvas.create_polygon(
            *(coordinate for point in points for coordinate in point),
            fill=fill,
            outline=outline,
            width=1,
            **options,
        )

    def _generate_cube(self, color_name):
        if self.submitted:
            return
        if len(self.placements) >= self.table.MAX_CUBES:
            self.status.set(f"The demo supports at most {self.table.MAX_CUBES} cubes.")
            return
        self.remove_mode = False
        self.pending_color = color_name
        pointer_x = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
        pointer_y = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
        if not (
            0 <= pointer_x <= self.CANVAS_WIDTH
            and 0 <= pointer_y <= self.CANVAS_HEIGHT
        ):
            pointer_x = self.CENTER_X
            pointer_y = (self.NEAR_Y + self.FAR_Y) / 2
        self.pending_canvas_position = (pointer_x, pointer_y)
        self._draw()

    def _on_pointer_motion(self, event):
        if self.pending_color is None or self.submitted:
            return
        self.pending_canvas_position = (event.x, event.y)
        self._draw()

    def _on_table_click(self, event):
        if self.submitted:
            return
        if self.remove_mode:
            self._remove_cube_at(event.x, event.y)
            return
        if self.pending_color is None:
            return
        world = self._canvas_to_world(event.x, event.y)
        if world is None:
            self.status.set("Click inside the tabletop.")
            return

        self.serials[self.pending_color] += 1
        cube_id = f"{self.pending_color}_{self.serials[self.pending_color]}"
        try:
            placement = self.table.placement(
                cube_id,
                self.pending_color,
                *world,
                self.placements,
            )
        except ValueError as error:
            self.serials[self.pending_color] -= 1
            self.status.set(str(error))
            return

        self.placements.append(placement)
        if self.live_mode:
            self.pending_placements.append(placement)
        self.pending_color = None
        self.pending_canvas_position = None
        self._draw()

    def _toggle_remove_mode(self):
        if self.submitted:
            return
        self.remove_mode = not self.remove_mode
        self.pending_color = None
        self.pending_canvas_position = None
        self._draw()

    def _remove_cube_at(self, canvas_x, canvas_y):
        anchor = self._stack_anchor_at_canvas(canvas_x, canvas_y)
        if anchor is None:
            return
        candidates = [
            (index, placement)
            for index, placement in enumerate(self.placements)
            if placement.x == anchor[0] and placement.y == anchor[1]
        ]
        if not candidates:
            return
        index, _ = max(candidates, key=lambda item: item[1].z)
        self.placements.pop(index)
        self._draw()

    def _stack_anchor_at_canvas(self, canvas_x, canvas_y):
        """Resolve clicks on either a drawn cube or its tabletop cell."""
        occupied = set((item.x, item.y) for item in self.placements)
        for anchor_x, anchor_y in occupied:
            stack_size = sum(
                item.x == anchor_x and item.y == anchor_y
                for item in self.placements
            )
            ground_x, ground_y = self._world_to_canvas(anchor_x, anchor_y)
            top = ground_y - stack_size * self.CUBE_HEIGHT - self.CUBE_DEPTH
            if (
                ground_x - self.CUBE_HALF_WIDTH <= canvas_x
                <= ground_x + self.CUBE_HALF_WIDTH
                and top <= canvas_y <= ground_y
            ):
                return anchor_x, anchor_y

        world = self._canvas_to_world(canvas_x, canvas_y)
        if world is None:
            return None
        anchor = self.table.nearest_anchor(*world)
        return anchor if anchor in occupied else None

    def _load_example(self):
        self.placements = list(self.example_placements)
        self.pending_color = None
        self.pending_canvas_position = None
        self.remove_mode = False
        self.serials = Counter(item.color_name for item in self.placements)
        self._draw()

    def _undo(self):
        if self.placements:
            self.placements.pop()
        self.pending_color = None
        self.pending_canvas_position = None
        self.remove_mode = False
        self._draw()

    def _reset(self):
        self.placements.clear()
        self.pending_color = None
        self.pending_canvas_position = None
        self.remove_mode = False
        self.serials.clear()
        self._draw()

    def _complete(self):
        if self.table.validate(self.placements):
            return
        self.result = tuple(self.placements)
        self.submitted = True
        self.pending_color = None
        self.pending_canvas_position = None
        self.remove_mode = False
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<Motion>")
        for button in self.color_buttons.values():
            button.configure(state="disabled", relief="raised")
        for button in (
            self.example_button,
            self.undo_button,
            self.reset_button,
            self.remove_button,
        ):
            button.configure(state="disabled")
        self.complete_button.configure(
            text=("Finishing queued moves…" if self.live_mode else "Building in Genesis…"),
            state="disabled",
        )
        self.cancel_button.configure(text="Close panel", command=self._close)
        self.status.set(
            "Layout locked. Waiting for the robot to finish queued moves."
            if self.live_mode
            else f"Layout locked. Spawning and placing {len(self.result)} cubes in order."
        )
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        if not self.live_mode:
            self.root.quit()

    def _cancel(self):
        self.result = None
        self._close()

    def _close(self):
        self.root.destroy()

    def mark_build_complete(self):
        self.root.title("VR multi-cube designer — build complete")
        self.complete_button.configure(text="Build complete", state="disabled")
        self.status.set(
            f"All {len(self.placements)} cubes are placed. Compare this layout with Genesis."
        )

    def mark_placement_started(self, placement):
        self.active_placement = placement
        self._draw()

    def mark_placement_complete(self, placement):
        self.completed_cube_ids.add(placement.cube_id)
        self.active_placement = None
        self._draw()

    def mark_build_failed(self, message):
        self.root.title("VR multi-cube designer — build failed")
        self.complete_button.configure(text="Build failed", state="disabled")
        self.status.set(f"Genesis stopped: {message}")

    def _update_color_buttons(self):
        counts = Counter(item.color_name for item in self.placements)
        for name, button in self.color_buttons.items():
            label = self.specs_by_name[name].label.upper()
            button.configure(
                text=f"{label}  +   ({counts[name]})",
                relief="sunken" if name == self.pending_color else "raised",
            )
        self.remove_button.configure(
            relief="sunken" if self.remove_mode else "raised"
        )

    def _normalized_to_canvas(self, depth, across):
        screen_y = self.NEAR_Y + depth * (self.FAR_Y - self.NEAR_Y)
        half_width = self.NEAR_HALF_WIDTH + depth * (
            self.FAR_HALF_WIDTH - self.NEAR_HALF_WIDTH
        )
        screen_x = self.CENTER_X + (2 * across - 1) * half_width
        return screen_x, screen_y

    def _world_to_canvas(self, x, y):
        depth = (x - self.table.x_edges[0]) / (
            self.table.x_edges[1] - self.table.x_edges[0]
        )
        across = (y - self.table.y_edges[0]) / (
            self.table.y_edges[1] - self.table.y_edges[0]
        )
        return self._normalized_to_canvas(depth, across)

    def _canvas_to_world(self, canvas_x, canvas_y):
        depth = (canvas_y - self.NEAR_Y) / (self.FAR_Y - self.NEAR_Y)
        if not 0.0 <= depth <= 1.0:
            return None
        half_width = self.NEAR_HALF_WIDTH + depth * (
            self.FAR_HALF_WIDTH - self.NEAR_HALF_WIDTH
        )
        across = (canvas_x - self.CENTER_X) / (2 * half_width) + 0.5
        if not 0.0 <= across <= 1.0:
            return None
        x = self.table.x_edges[0] + depth * (
            self.table.x_edges[1] - self.table.x_edges[0]
        )
        y = self.table.y_edges[0] + across * (
            self.table.y_edges[1] - self.table.y_edges[0]
        )
        return x, y

    @staticmethod
    def _shade(color, factor):
        shaded = tuple(max(0.0, min(1.0, channel * factor)) for channel in color)
        return CubePlacementDesigner._color_to_hex(shaded)

    @staticmethod
    def _color_to_hex(color):
        channels = [max(0, min(255, round(channel * 255))) for channel in color]
        return "#{:02x}{:02x}{:02x}".format(*channels)


class CubeDesignSession:
    """Bridge panel events to Genesis while keeping the panel responsive."""

    def __init__(self, root, designer):
        self.root = root
        self.designer = designer
        self.closed = False

    @property
    def placements(self):
        return tuple(self.designer.placements)

    @property
    def finished(self):
        return (
            self.designer.submitted
            and not self.designer.pending_placements
            and self.designer.active_placement is None
        )

    def pop_next_placement(self):
        if not self.designer.pending_placements:
            return None
        return self.designer.pending_placements.popleft()

    def mark_placement_started(self, placement):
        if not self.closed:
            self.designer.mark_placement_started(placement)
            self.process_events()

    def mark_placement_complete(self, placement):
        if not self.closed:
            self.designer.mark_placement_complete(placement)
            self.process_events()

    def mark_simulation_starting(self):
        if not self.closed:
            self.designer.status.set("Starting Genesis; controls will be ready shortly…")
            self.process_events()

    def mark_simulation_ready(self):
        if not self.closed:
            self.designer._draw()
            self.process_events()

    def process_events(self):
        if self.closed:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True

    def mark_build_complete(self):
        if not self.closed:
            self.designer.mark_build_complete()
            self.process_events()

    def mark_build_failed(self, message):
        if not self.closed:
            self.designer.mark_build_failed(message)
            self.process_events()

    def wait_until_closed(self, keep_genesis_alive=None):
        """Show the final comparison until the user closes the panel."""
        if self.closed:
            return

        def keep_alive():
            if self.closed:
                return
            if keep_genesis_alive is not None:
                keep_genesis_alive()
            try:
                self.root.after(20, keep_alive)
            except tk.TclError:
                self.closed = True

        keep_alive()
        try:
            self.root.mainloop()
        except tk.TclError:
            pass
        self.closed = True


def launch_cube_designer(block_specs, example_placements, live_mode=False):
    """Open the color-cube panel and return its event/session bridge."""
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError(
            "The cube designer needs a graphical display. "
            "Run without --design to use the built-in example layout."
        ) from error

    designer = CubePlacementDesigner(
        root,
        block_specs,
        example_placements,
        live_mode=live_mode,
    )
    session = CubeDesignSession(root, designer)
    if live_mode:
        session.process_events()
        return session

    root.mainloop()
    if designer.result is None:
        return None
    return session
