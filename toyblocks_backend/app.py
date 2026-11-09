"""Start Isaac Sim through ToyBlocks' launcher, before any isaaclab/omni import.

``toyblocks.cli.isaac_app.launch_app`` imports Warp first (so cuRobo uses the
venv's copy) and reads Isaac Lab's AppLauncher flags from ``sys.argv``; this
wrapper hands it exactly the flags the receiver chose.
"""
import atexit
import sys

_APP = None


def launch(headless=False, device="cuda:0"):
    global _APP
    if _APP is not None:
        return _APP
    from toyblocks.cli.isaac_app import launch_app

    argv = sys.argv
    sys.argv = [argv[0], *(["--headless"] if headless else []), "--device", device]
    try:
        _APP = launch_app()
    finally:
        sys.argv = argv
    atexit.register(_APP.close)
    return _APP
