"""``robot_receiver.py --sim toyblocks`` plug-in.

Importing this module never imports Isaac Sim, so argument parsing and
``--receive-only`` work anywhere. ``prepare_backend`` starts Isaac Sim before the
receiver connects (it is the slow part); ``build_backend`` builds the scene once
the first map has given the table size.
"""
from types import SimpleNamespace

from toyblocks_backend.frames import DEFAULT_ROBOT_TO_NEAR_EDGE

BACKEND_DESCRIPTION = "single Franka FR3 in AllenAI ToyBlocks / Isaac Sim (Linux + NVIDIA GPU)"
DEFAULT_POOL_PER_COLOR = 10  # each pool body is also a cuRobo obstacle


def add_backend_arguments(parser):
    group = parser.add_argument_group("toyblocks backend")
    group.add_argument("--robot-to-near-edge", type=float, default=DEFAULT_ROBOT_TO_NEAR_EDGE,
                       help="metres from the Franka base to the table's rack-facing edge "
                            f"(default: {DEFAULT_ROBOT_TO_NEAR_EDGE})")
    group.add_argument("--device", default="cuda:0", help="Isaac Sim / torch device (default: cuda:0)")
    group.add_argument("--speed", type=float, default=1.0,
                       help="accepted for compatibility with the Genesis flags; scales settle waits only")
    group.add_argument("--record", nargs="?", const="unsupported", metavar="PATH",
                       help="not supported by this backend yet")
    group.add_argument("--viewer", action="store_true",
                       help="serve ToyBlocks' browser 3D viewer (viser; port 8080 or the next free one)")


def _check(args):
    if args.record:
        raise ValueError("--record is not supported by the ToyBlocks backend yet")
    if not args.speed > 0:
        raise ValueError("speed must be greater than zero")
    if not 0 < args.robot_to_near_edge < 0.6:
        raise ValueError("robot-to-near-edge must be between 0 and 0.6 m")


def prepare_backend(args):
    _check(args)
    from toyblocks_backend.app import launch
    launch(headless=args.headless, device=args.device)


def build_backend(layout, args):
    prepare_backend(args)
    from toyblocks_backend.live import ToyBlocksBackend
    return ToyBlocksBackend(layout, SimpleNamespace(
        live=True, pool_per_color=args.pool_per_color, speed=args.speed, physical=True,
        robot_to_near_edge=args.robot_to_near_edge, device=args.device, viewer=args.viewer))
