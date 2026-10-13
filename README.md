# VR Cube Stacking with Genesis

An interactive dual-arm UR5e demonstration built with the Genesis physics
engine. A user selects red, green, or blue cubes in a 5 x 8 design panel and
places them on table anchors. In live mode, each placement is queued
immediately and executed by the corresponding robot arm.

## Features

- Real-time Tkinter design panel with cursor-following cube previews
- Multiple cubes and vertical stacking on the same anchor
- Red cubes handled directly by the right arm
- Green and blue cubes handled directly by the left arm
- Smooth minimum-jerk joint trajectories
- Sequential placement queue to avoid simultaneous-arm collisions
- Matching grid visualization in the design panel and Genesis
- Headless MP4 export

## Repository layout

```text
.
├── test_ur5e_robotiq_pose.py       # Genesis scene and robot controller
├── stacking_panel.py               # Interactive placement panel
├── vr_json_cube_stacking.py        # VR table-map importer and robot replay
├── example_table_data.json         # Example openup.cube_stack.map input
├── table_frame.png                 # VR table-frame convention
├── export_ur5e_robotiq_video.sh    # Headless video-export helper
└── zeus_description/
    ├── ur5e_robotiq_2f85.urdf
    └── meshes/                     # UR5e and Robotiq visual/collision meshes
```

The `outputs/` directory is created automatically when a video is exported.

## Requirements

- Python 3.10 (the tested version)
- A graphical desktop for `--design`
- Tkinter (`python3-tk` on Ubuntu/Debian)
- FFmpeg for MP4 export

Create an environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

On Ubuntu/Debian, the system packages can be installed with:

```bash
sudo apt-get install python3-tk ffmpeg
```

## Usage

Start the real-time interactive demonstration:

```bash
python3 test_ur5e_robotiq_pose.py --design
```

Select a color, move the virtual cube onto a grid anchor, and click to place
it. The placement enters the robot queue immediately. You can design another
cube while the current move is running. Click **Finish real-time demo** when
the pattern is complete; queued moves finish before the final comparison.

Run the built-in four-cube demonstration without the design panel:

```bash
python3 test_ur5e_robotiq_pose.py
```

Run the built-in demonstration without a viewer:

```bash
python3 test_ur5e_robotiq_pose.py --headless
```

Export a video to the default `outputs/` path:

```bash
./export_ur5e_robotiq_video.sh
```

Or choose the output path and adjust rendering options:

```bash
FPS=30 SPEED=1.5 ./export_ur5e_robotiq_video.sh demo.mp4
```

The Python entry point also accepts `--record [PATH]`, `--fps`, and `--speed`.

## Import a VR table map

Rebuild the included VR example with the dual-arm controller:

```bash
python3 vr_json_cube_stacking.py
```

Pass another compatible map as the positional argument:

```bash
python3 vr_json_cube_stacking.py path/to/table_data.json
```

The importer reads `cube_size`, `table`, `position`, and each flattened
row-major `object_to_table` matrix. It validates that the matrix rotation is
scaled by `cube_size`, then applies this frame conversion:

```text
VR +X (right)       -> Genesis +Y
VR +Y (up)          -> Genesis +Z
VR +Z (down-screen) -> Genesis -X
```

The importer preserves 8 cm cubes at their original size (the example map has
a scale factor of `1.0`). The table and all object translations use the same
scale. Its center is derived from the imported table depth so that its near
edge remains at Genesis `X = 0.06 m`, clear of the robot rack.

By default, the arms perform smooth pick-and-place trajectories while the
cubes follow the tools kinematically and are locked at the transformed JSON
poses. This faithfully replays layouts that the VR editor permits even when
they are not dynamically stable under Genesis contact physics.

Use free rigid bodies instead to test the design's physical stability:

```bash
python3 vr_json_cube_stacking.py --physical
```

The example's overhanging blue stack is expected to tip in physical mode.
Headless execution and video recording are also supported:

```bash
python3 vr_json_cube_stacking.py --headless --speed 3
python3 vr_json_cube_stacking.py --record outputs/vr_json_cube_stacking.mp4
```

## Third-party robot assets

The URDF and robot meshes in `zeus_description/` were copied from the local
working assets, whose recorded source repository is
[JuoTungChen/ROS2_pick_and_place_UR5](https://github.com/JuoTungChen/ROS2_pick_and_place_UR5).
The copied asset directory did not contain a license file. Before publishing
this repository, verify the applicable redistribution and attribution terms
from that project and the upstream Universal Robots and Robotiq description
packages. This project does not assign a new license to those third-party
assets.

