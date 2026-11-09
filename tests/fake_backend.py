"""A teleporting robot backend for receiver tests (no physics engine).

It is also a valid ``robot_receiver.py`` backend module, so tests can register
it as ``--sim fake``.
"""
import threading
from types import SimpleNamespace

from cube_stack_planner import LiveScenePlanner
from cube_stack_receiver import pump_and_check
from test_live_runner import Entity

BACKEND_DESCRIPTION = "teleporting test backend"
DEFAULT_POOL_PER_COLOR = 4
STOP = threading.Event()  # set by a test to end a receiver running in a thread, like Ctrl+C


class FakeBackend(LiveScenePlanner):
    """Moves bodies instantly; records every transfer for assertions."""

    idle_sleep = 0

    def __init__(self, layout, pool_per_color=DEFAULT_POOL_PER_COLOR):
        self.layout = layout
        self.options = SimpleNamespace(live=True, pool_per_color=pool_per_color, speed=1000, physical=False)
        self.clear_carried()
        self.live_client = self.live_queue = None
        self.placed_placements = []
        self.current_placements = {}
        self._pool_specs = {}
        self.blocks = {}
        lengths = (1, 2, 3) if layout.map_version == 3 else (1,)
        index = 0
        for color in ("red", "green", "blue"):
            for length in lengths:
                for serial in range(1, pool_per_color // length + 1):
                    key = f"{color}_L{length}_{serial}"
                    self._pool_specs[key] = (color, length)
                    self.blocks[key] = Entity((2 + index * .28, 2, .04))
                    index += 1
        self.transfers = []
        self.steps = 0
        self.homed = 0
        self._setup_live_pool()

    def _transfer_block(self, block, placement):
        self.transfers.append((placement.cube_id, tuple(block.get_pos()), tuple(placement.position)))
        block.set_pos(placement.position)
        block.set_quat(placement.quaternion)

    def _staging_feasible(self, item, slot, destination=None):
        return True

    def _step(self):
        if STOP.is_set():
            raise KeyboardInterrupt
        pump_and_check(self.live_client, self.live_queue)
        self.steps += 1

    def _initialize_poses(self):
        self.homed += 1

    def _start_recording(self):
        pass

    def _stop_recording(self, settle=True):
        pass


def add_backend_arguments(parser):
    parser.add_argument_group("fake backend").add_argument("--fake-flag", action="store_true")


def build_backend(layout, args):
    return FakeBackend(layout, pool_per_color=args.pool_per_color)
