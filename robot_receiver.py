"""Run the Cube Stack robot receiver with the simulator chosen at run time.

OpenUp Cube Stack publishes its layout on the simulation port and accepts robot
state back; it does not know which simulator answers. Start exactly one receiver
per session:

    python robot_receiver.py --sim genesis --live --host 127.0.0.1 --port 9036
    python robot_receiver.py --sim toyblocks --live --host <OpenUp PC address> --port 9036

Backends are imported only when chosen, so the Genesis environment never needs
Isaac Sim and the ToyBlocks environment never needs Genesis. A backend module
provides ``BACKEND_DESCRIPTION``, ``DEFAULT_POOL_PER_COLOR``,
``add_backend_arguments(parser)`` and ``build_backend(layout, args)``, and
optionally ``prepare_backend(args)`` for slow start-up work that does not need
the first map.
"""
import argparse
import importlib
import sys

from cube_stack_receiver import live_main

BACKENDS = {
    "genesis": "vr_json_cube_stacking",
    "toyblocks": "toyblocks_backend.backend",
}


def _load_backend(name):
    return importlib.import_module(BACKENDS[name])


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    chooser = argparse.ArgumentParser(add_help=False)
    chooser.add_argument("--sim", choices=sorted(BACKENDS), default="genesis")
    sim = chooser.parse_known_args(argv)[0].sim
    backend = _load_backend(sim)

    parser = argparse.ArgumentParser(
        description="Follow an OpenUp Cube Stack table with a robot simulator chosen at run time.")
    parser.add_argument("--sim", choices=sorted(BACKENDS), default="genesis",
                        help=f"robot simulator (this run: {sim}, {backend.BACKEND_DESCRIPTION})")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true",
                      help="follow CubeStack/Map additions, moves, deletions and resets with the robot")
    mode.add_argument("--receive-only", action="store_true",
                      help="validate and queue live maps without building a simulation")
    parser.add_argument("--host", default="127.0.0.1", help="OpenUp table host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9036, help="OpenUp simulation stream port (default: 9036)")
    parser.add_argument("--pool-per-color", type=int, default=backend.DEFAULT_POOL_PER_COLOR,
                        help=f"prebuilt 8 cm cubes per color (default for {sim}: {backend.DEFAULT_POOL_PER_COLOR})")
    parser.add_argument("--max-cubes", type=int, default=0,
                        help="stop after N completed live placements or accepted receive-only cubes (0: continuous)")
    parser.add_argument("--timeout", type=float, default=0,
                        help="seconds to wait for the first map or receive-only count (0: unlimited)")
    parser.add_argument("--headless", action="store_true", help="run without a viewer window")
    backend.add_backend_arguments(parser)
    args = parser.parse_args(argv)
    if not 0 < args.port < 65536 or args.pool_per_color <= 0 or args.max_cubes < 0 or args.timeout < 0:
        parser.error("port must be 1..65535, pool-per-color positive, max-cubes and timeout nonnegative")
    return args, backend


def main(argv=None):
    args, backend = parse_args(argv)
    try:
        prepare = getattr(backend, "prepare_backend", None)
        if args.live and prepare is not None:
            prepare(args)  # slow start-up (e.g. Isaac Sim) before connecting to the table
        live_main(lambda layout: backend.build_backend(layout, args),
                  host=args.host, port=args.port, pool_per_color=args.pool_per_color,
                  receive_only=args.receive_only, max_cubes=args.max_cubes, timeout=args.timeout)
    except KeyboardInterrupt:
        print("stopped: live session interrupted; restart explicitly to resume with a fresh simulation", flush=True)
    except Exception as error:
        print(f"failed: {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
