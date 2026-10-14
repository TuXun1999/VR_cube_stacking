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
├── cube_stack_layout.py           # Shared Genesis-free V2 map parser
├── cube_stack_stream.py           # Live stream decoder and acceptance queue
├── example_table_data.json         # Example openup.cube_stack.map input
├── table_frame.png                 # VR table-frame convention
├── export_ur5e_robotiq_video.sh    # Headless video-export helper
└── zeus_description/
    ├── ur5e_robotiq_2f85.urdf
    └── meshes/                     # UR5e and Robotiq visual/collision meshes
```

The `outputs/` directory is created automatically when a video is exported.

## Requirements

- Python 3.10+ (live integration tested with Python 3.11)
- A graphical desktop for `--design`
- Tkinter (`python3-tk` on Ubuntu/Debian)
- FFmpeg for MP4 export

From this repository root, create an environment and install the Python
dependencies. On Windows, use PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

These commands use the tested Python 3.11 interpreter directly and do not
require activating the environment. On Linux/macOS:

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
VR +Z (down-screen) -> Genesis +X
```

This preserves the table's right-handed frame and front/back arrangement:
down-screen cubes lie farther along Genesis +X, toward the default viewer.
A positive rotation about the table's +Y axis becomes a positive rotation
about Genesis +Z.

The importer preserves 8 cm cubes at their original size (the example map has
a scale factor of `1.0`). The table and all object translations use the same
scale. Its center is derived from the imported table depth so that the
rack-facing edge remains at Genesis `X = 0.06 m`, clear of the robot rack.

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

## Connect the OpenUp Cube Stack table live

Both OpenUp modes use the same robot receiver: the computer-only desktop
window, or a touch table with a Quest headset. Choose **one publisher** for a
simulation session. Start with an empty construction and wait for the receiver's
`ready` message before adding cubes.

The commands below use these local checkout locations:

- OpenUp: `C:\Users\wally\OpenUp\OpenUp`
- This simulation: `C:\Users\wally\OpenUp\VR_cube_stacking\VR_cube_stacking`

From this simulation directory, the OpenUp checkout is `../../OpenUp`. Adjust
the paths for your checkout layout. For the Windows build, physical table
dimensions, Quest installation, calibration, and connection details, follow
the [OpenUp live simulation guide](https://github.com/ivlab/OpenUp/blob/main/openup_apps/cube_stack/README.md#live-genesis-simulation).

### Choose the OpenUp publisher

For **computer-only mode**, run this in an OpenUp terminal after building
`cube_stack`. The preset shows the adjustable 3D scene above the overhead table;
the additional setting enables the robot stream:

```powershell
Set-Location C:\Users\wally\OpenUp\OpenUp
& .\build-cube\openup_apps\cube_stack\Release\cube_stack.exe `
    --config openup_apps/cube_stack/desktop.json `
    --set /cube_stack/simulation_port=9036
```

For **touch table and Quest mode**, run the table publisher instead. Its
configuration already enables the simulation stream on **9036** and headset
events on **9035**:

```powershell
Set-Location C:\Users\wally\OpenUp\OpenUp
& .\build-cube\openup_apps\cube_stack\Release\cube_stack.exe `
    --config openup_apps/cube_stack/table_windows.json `
    --width 1920 --height 1080
```

Adjust the window placement, resolution, and physical table dimensions using
the OpenUp guide, then connect and calibrate the Quest. For a USB-connected
Quest, only the headset event port uses ADB forwarding:
`adb reverse tcp:9035 tcp:9035`. The robot receiver connects directly to the
Windows publisher; **9036 needs no ADB forwarding** when the receiver runs on
that PC.

### Start the same robot receiver for either mode

In a separate terminal, run from this simulation checkout:

```powershell
Set-Location C:\Users\wally\OpenUp\VR_cube_stacking\VR_cube_stacking
.\.venv\Scripts\python.exe -u vr_json_cube_stacking.py --live --host 127.0.0.1 --port 9036 --pool-per-color 30
```

The viewer opens by default. Use `127.0.0.1` when both applications run on the
same PC; when Genesis runs on another computer, replace `--host 127.0.0.1`
with the OpenUp publisher PC's LAN address. The receiving computer must be able
to reach that address on TCP **9036**.

On Linux/macOS, run the same receiver from this directory with the environment
active:

```bash
python3 -u vr_json_cube_stacking.py --live --host 127.0.0.1 --port 9036 --pool-per-color 30
```

The receiver waits for the first map, including an empty table, builds Genesis
once, and prints `ready` after scene initialization. It prebuilds 30 parked
8 cm cubes for each color before `scene.build()`. Each accepted addition
reserves one pool object and enters the queue in the table's placement order.
Imported positions and rotations remain arbitrary; the VR replay chooses the
arm on the target side of the table. `queued`, `started`, and `completed`
console messages distinguish accepted work from finished robot motion.
Network ingestion continues during trajectories, and only the outer loop
starts a placement.

The updated OpenUp publisher sends its complete cumulative placement history
through `CubeStack/Map` after new placements and every 500 ms while subscribed.
Each cube's table pose and color are recorded when it is created and remain
unchanged in that history. New cubes may be added while the robot is working,
including after earlier cubes have been moved, rotated, or deleted in OpenUp.
Those edits do not block later placements or change queued or active robot work.
Moving the desktop camera and choosing the color of future cubes are supported.

Use the existing `--headless`, `--speed`, `--record`, `--fps`, and `--physical`
flags with `--live`. For a bounded headless run:

```powershell
.\.venv\Scripts\python.exe vr_json_cube_stacking.py --live --headless --max-cubes 3 --timeout 20
```

`--max-cubes N` stops after N completed placements in live mode; `0` runs
continuously. `--timeout SEC` limits the initial wait for table metadata and
does not interrupt a trajectory. Ctrl+C ends the session. A separate
receive-only diagnostic exercises the same real decoding, validation, pool
reservation, and queue acceptance without importing Genesis:

```powershell
.\.venv\Scripts\python.exe vr_json_cube_stacking.py --receive-only --host 127.0.0.1 --port 9036 --max-cubes 3 --timeout 20
```

Receive-only needs NumPy and Python but does not need Genesis. It exits with
status 0 once N cubes have been accepted, and reports `completed=0` because
no robot ran. Its timeout covers the entire diagnostic. A timeout, malformed
stream frame, pool exhaustion, or fatal map rejection exits with status 1.
Non-placement edits leave the updated publisher's placement records unchanged.
Omitting both limits keeps receiving until Ctrl+C.

The wire format is two uint32 little-endian length-prefixed blocks: a JSON
header followed by the UTF-8 V2 `openup.cube_stack.map` payload. The header
uses `stream: CubeStack/Map`, `content_type: application/json`, a matching
`payload_bytes`, and a **string** containing JSON metadata with
`protocol_version: 1` and a per-table-process `session_id`. Each block is
limited to 64 MiB. This schema and framing are unchanged; the live `cubes` array
now contains immutable placement records rather than OpenUp's current edited
map. The original table-local positions and scaled row-major `object_to_table`
matrices use the same parser as offline file replay.

Live sessions replay **new placements only**. The simulation retains each
cube's original pose and color, including cubes later deleted in OpenUp.
The desktop, physical table, headset, and F5 exports continue to show the current
edited map. A later cube keeps its actual pose and height when created; the
robot does not recompute stack heights or restack earlier cubes after edits.

Each placement permanently reserves one pool object of its color for that
session, even if the cube is later deleted in OpenUp. With `--pool-per-color 30`,
the cumulative history may contain at most 30 placements of each color. Increase
that limit before startup when needed.

Repeated snapshots and reconnecting to the same table process never duplicate
queued, active, or completed cubes. New subscribers receive the complete
placement history, and transport failures retry without clearing accepted work.
Changed table metadata, another table process, malformed stream frames,
exceeding a color pool, or an IK failure still produces a visible `HALTED`
state and stops further robot commands. A partial motion is never retried
automatically.

For legacy publishers that still send the current full map, the receiver
continues to ignore an entire snapshot if it changes or removes the previously
accepted prefix. The console reports `ignored:`, and existing robot work
continues. This compatibility fallback cannot identify new cube births within
an edited snapshot; use the updated OpenUp publisher to continue placing cubes
after edits.

To begin a fresh simulation, stop the receiver with **Ctrl+C**, restart the
chosen OpenUp publisher with an empty construction, then start a new receiver
and wait for `ready` before adding cubes. Placing rows and stacking new cubes
is supported. Kinematic replay preserves requested poses; `--physical` enables
free rigid bodies to assess stability separately.

## Third-party robot assets

The URDF and robot meshes in `zeus_description/` were copied from the local
working assets, whose recorded source repository is
[JuoTungChen/ROS2_pick_and_place_UR5](https://github.com/JuoTungChen/ROS2_pick_and_place_UR5).
The copied asset directory did not contain a license file. Before publishing
this repository, verify the applicable redistribution and attribution terms
from that project and the upstream Universal Robots and Robotiq description
packages. This project does not assign a new license to those third-party
assets.

