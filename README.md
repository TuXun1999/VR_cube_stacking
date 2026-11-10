# VR Cube Stacking with Genesis

An interactive dual-arm UR5e demonstration built with the Genesis physics
engine. A user selects red, green, or blue cubes in a 5 x 8 design panel and
places them on table anchors. In OpenUp live mode, the robot builds the current
construction and follows cube moves, deletions, and explicit session resets.

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
├── robot_receiver.py               # Live receiver; picks the simulator with --sim
├── vr_json_cube_stacking.py        # Genesis backend, VR table-map importer and robot replay
├── toyblocks_backend/              # AllenAI ToyBlocks (Isaac Sim, Franka FR3) backend
├── cube_stack_receiver.py          # Shared live loop, pump/reset checks, pose tolerance
├── cube_stack_planner.py           # Shared live planning: blockers, staging, ordering
├── cube_stack_layout.py           # Shared Genesis-free version-2/3 map parser
├── cube_stack_stream.py           # Live stream decoder and acceptance queue
├── cube_stack_staging.py          # Outside-table supply/return stack allocation
├── tools/fake_table_publisher.py   # Scripted stand-in for OpenUp, for any backend
├── verify_live_edits.py           # Headless Genesis edit/reset verification
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

## Choose a robot simulator

OpenUp Cube Stack does not know which simulator answers on its simulation port,
so the simulator is chosen when you start the receiver. `robot_receiver.py`
accepts the same live flags as `vr_json_cube_stacking.py`, plus `--sim`:

| `--sim` | Robot | Runs on | Notes |
|---|---|---|---|
| `genesis` (default) | Two UR5e + Robotiq 2F-85 | Windows/Linux, CPU | Kinematic carrying by default; `--physical`, `--record`, `--speed` |
| `toyblocks` | One Franka FR3 + Robotiq 2F-85 | Linux + NVIDIA GPU | Isaac Sim physics; see [ToyBlocks backend](#toyblocks-backend) |

```powershell
.\.venv\Scripts\python.exe -u robot_receiver.py --sim genesis --live --host 127.0.0.1 --port 9036
```

`vr_json_cube_stacking.py --live` still starts the Genesis receiver directly.
Both backends share one live loop (`cube_stack_receiver.py`) and one planner
(`cube_stack_planner.py`), so they print the same `ready:`, `started:`,
`completed:` and `live complete:` lines and report `CubeStack/RobotState` the
same way. A backend module supplies `BACKEND_DESCRIPTION`,
`DEFAULT_POOL_PER_COLOR`, `add_backend_arguments(parser)` and
`build_backend(layout, args)`, and is registered in `robot_receiver.BACKENDS`.

### Test any backend without OpenUp

`tools/fake_table_publisher.py` plays OpenUp's part: it publishes a script of
protocol-3 maps and waits for each to be confirmed by `RobotState`, using
OpenUp's own 1e-5 matrix check. The default script
(`tools/scripts/basic_edits.json`) adds, stacks, places a yawed triple bar,
moves, deletes and resets, all within one arm's reach.

```bash
python tools/fake_table_publisher.py --port 9036
python -u robot_receiver.py --sim toyblocks --live --host 127.0.0.1 --port 9036
```

It exits 0 when every step is confirmed and 1 on a step timeout.

## ToyBlocks backend

`toyblocks_backend/` drives [AllenAI ToyBlocks](https://github.com/allenai/ToyBlocks)
(Isaac Sim 5.1, Isaac Lab, cuRobo) as a live receiver. It is pinned to ToyBlocks
commit `931fd7aab1458fd58f4c23d810879ea4826b07d8`, because it relies on vecgen
engine internals.

**Requirements.** Linux with an NVIDIA GPU and driver 570 or newer, Python 3.11,
and `uv`. ToyBlocks does not document Windows support. Run the receiver on the
Linux machine and point it at the OpenUp PC: `--host <OpenUp PC LAN address>`
(the OpenUp PC must allow inbound TCP 9036).

```bash
git clone https://github.com/allenai/ToyBlocks && cd ToyBlocks
git checkout 931fd7aab1458fd58f4c23d810879ea4826b07d8 && git submodule update --init
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev,isaac,torch-cu128,vecgen]"
get-toyblocks-assets --asset-manifest asset_manifest.json
export OMNI_KIT_ACCEPT_EULA=YES
pytest tests/install --install-gpu
```

Then, from this repository with that environment active:

```bash
python -u toyblocks_backend/spike.py --headless          # Phase 0 gate: grasp, place, reach sweep
python -u robot_receiver.py --sim toyblocks --live --host <OpenUp PC> --port 9036
```

**Running it on MSI.** If you don't have a suitable Linux GPU machine, see
[toyblocks_backend/msi/README.md](toyblocks_backend/msi/README.md). It covers an
Apptainer image and scripts for MSI's Agate A40/L40S nodes, plus the SSH tunnel
back to a Windows laptop running the desktop table. Add `--viewer` to serve
ToyBlocks' browser 3D viewer (viser, port 8080), which needs no Isaac window.

**How it differs from Genesis.**

- **One arm.** A single Franka sits at the middle of the rack edge
  (`--robot-to-near-edge`, default 0.15 m). Targets it cannot reach stop the
  session with `rejected: HALTED: No reachable robot transfer ...`, the same way
  Genesis reports an unreachable transfer. `spike.py --sweep` writes the
  reachable area to `outputs/toyblocks_reach.csv`.
- **Staging beside the robot.** Supply and returned pieces use colour lanes on
  two pads beside the robot base (`toyblocks_backend/staging.py`) instead of
  strips beyond the table ends.
- **Real fingers.** An 8 cm cube leaves about 2.5 mm per side in the 85 mm
  Robotiq gripper, and the closed fingers stick out past the piece. The backend
  approaches from above, lifts pieces clear of 8 cm neighbours, and turns cubes
  (quarter turns) or bars (half turns) so the fingers avoid touching
  neighbours. A piece with neighbours touching both sides it must be gripped
  by may still fail to place.
- **Real physics.** Pieces are grasped and released physically. A piece counts
  as placed when it settles within 2 cm and 5 degrees of its target (widened
  from 1 cm / 3 degrees after the first A40 run); the receiver then reports the
  exact target so OpenUp's 1e-5 check confirms it. A
  piece that does not stay put (an overhang, for example) is retried, then the
  session halts with `unstable: ...`.
- **Not supported yet:** `--record`, `--physical` (always physical), and
  offline JSON replay.

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
once, and prints `ready` after scene initialization. It prebuilds 30 reusable
8 cm cube entities per color. The robot's workspace table displays the same
bundled island map as OpenUp Cube Stack, with matching orientation and the
left 12% palette margin. The map also appears during offline JSON replay.
Each new cube appears on a staging surface
beside the table's left or right edge before the robot picks it up. These
surfaces are level with the construction table; all cells sit beyond its lateral
footprint, near its rack-facing end. Each side shows three continuous rectangular
strips in red, green, and blue, with matching lengths and aligned ends. Pickup
cells hold complete supply and returned pieces on matching strips: red cubes
use red, green use green, and blue use blue. New cubes spawn on their own strip,
and deletion returns and temporary staging moves keep that color assignment.
Version-3 racks use long cells aligned with table depth, with two columns per
side on the default table. Whole-piece footprints and actual IK determine which
slots are available. Full or unreachable storage fails explicitly. Returned
pieces supply later placements of the same color and length.
`--pool-per-color` counts constituent cubes: a double uses two units and a triple
three. Bodies for each color and length are created before the scene is built. The robot uncovers buried objects before
picking them.

Click the active OpenUp palette color repeatedly to cycle **1 -> 2 -> 3 -> 1**.
Changing color resets the next size to one. Doubles and triples are permanently
glued: one rigid body, one center grasp across the 8 cm short side, and one
whole-piece robot command. Wrist yaw follows pickup/destination orientation;
kinematic carrying preserves the measured tool-to-piece transform. Selecting
any member selects the entire piece.

Hold a shadow in OpenUp to select, drag and release gently to request a move,
or swipe quickly and release to delete. The robot picks existing cubes from
their actual simulated poses and carries an ordinary requested move directly
to its new pose in one transfer. Deletion carries the cube to staging. Other
cubes move only when they block the pickup or destination: changed blockers
can go directly to their own ready targets, while blockers that need temporary
space use staging. A swap stages the other cube, moves the requested cube
directly, then restores the other cube. Unrelated queued moves wait their turn;
new supports are placed before cubes that depend on them. Both arms check the
complete transfer path before motion. When neither arm can reach both endpoints
in one transfer, the controller chooses a clear handoff point near the table
center and validates both arms' transfer paths before pickup.
Genuine support cycles first clear the upper cube before moving its support.

`accepted`, `started: add/move/delete/reset`, and `completed` messages identify
desired state and finished robot work. Network ingestion continues during
trajectories. Pending edits coalesce to the newest committed layout; an active
transfer finishes to a known pose before following newer edits. Held-drag
previews are not robot targets. Repeated snapshots and reconnection to the
same publisher do not duplicate work.

The live robot returns `CubeStack/RobotState` frames on the same TCP connection
after completed operations and while idle, with a 500 ms status heartbeat.
Their JSON payload contains `session_id`, `reset_generation`, the latest accepted
map `revision`, and `cubes` with actual completed `id`, `color`, `position`, and
row-major `object_to_table` values in the original table coordinates. A newer
map may already be queued, so a receiver must match each actual pose with its
current target before confirming it. Missing IDs confirm removal, including
births cancelled before the robot started them. Active trajectories and pending
resets send no completion status; receive-only diagnostics send none at all.
Reconnect replays the latest completed state without duplicating robot work.
Experimental physical mode samples body positions and orientations at these
boundaries, so missed targets remain unconfirmed. A failed physical return to
staging halts before reporting the deleted ID as absent.

The publisher sends complete `CubeStack/Map` snapshots after edits and every
500 ms while subscribed. New cubes may be added while the robot is working.
Moving the desktop camera and choosing the color of future cubes are supported.

### Reset without restarting

Tap **Reset session** below the palette, then tap it again within three
seconds to confirm. Alternatively press and release **F6 twice** within three
seconds on the OpenUp computer. Reset clears OpenUp's cubes, selection and
construction transform, restores the blue palette choice, cancels robot work
at the next simulation step, resets the reusable supply, and homes both arms.
The Genesis scene remains built and connected; headset/table calibration and
map configuration are preserved. You can start another construction immediately.
The existing **R** key still resets only the desktop camera.

Use the existing `--headless`, `--speed`, `--record`, `--fps`, and `--physical`
flags with `--live`. For a bounded headless run:

```powershell
.\.venv\Scripts\python.exe vr_json_cube_stacking.py --live --headless --max-cubes 3 --timeout 20
```

`--max-cubes N` stops after N completed additions since the latest reset; `0` runs
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
message, pool exhaustion, or rejected map exits with status 1. Omitting both
limits keeps receiving until Ctrl+C.

The wire format is two uint32 little-endian length-prefixed blocks: a JSON
header followed by the UTF-8 version-3 `openup.cube_stack.map` payload. The header
uses `stream: CubeStack/Map`, `content_type: application/json`, a matching
`payload_bytes`, and a **string** containing JSON metadata with
`protocol_version: 3` and a per-table-process `session_id`. Each cube carries a
stable string `id`, `piece_id`, `piece_index`, and declared `piece_size` (1-3).
Every snapshot must contain each complete rigid group. The map carries a monotonic `revision` and explicit
`reset_generation`. The generation survives missed intermediate reset maps,
including a reset immediately followed by a new placement. Each block is
limited to 64 MiB. The original table-local positions and scaled row-major
`object_to_table` matrices use the same parser as offline file replay.

Transport failures retry without clearing accepted work. Table dimensions and
cube colors remain fixed within a generation. Changing table metadata,
starting another publisher process, exceeding simultaneous color capacity,
malformed frames, or an unreachable robot transfer produces a visible `HALTED`
state. A failed partial motion is not retried automatically: stop the receiver,
reset OpenUp, restart the receiver, and wait for `ready`. Ordinary construction
reset uses the button/F6 and needs no restart. Version-2 maps and protocol-1/2
singleton streams remain supported; old receivers reject protocol 3 before motion.
Legacy protocol 1 publishers
remain supported for additions only; update both projects for edit/reset support.
Kinematic replay preserves the requested poses, including floating designs the
editor permits. `--physical` separately enables free rigid bodies to assess
grasp and stack stability; it has not been validated for these new edit paths.

### Verify live edits

Run the Python behavior suite (no Genesis initialization):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Exercise real headless robot trajectories, staged returns/reuse, cross-table
movement, necessary blocker handling, reset, and completed `RobotState` poses
for add/move/delete/reset in one Genesis scene:

```powershell
.\.venv\Scripts\python.exe -X utf8 -u verify_live_edits.py
```

Pass `--record outputs/live-edits.mp4` to retain a video. The OpenUp repository's
`openup_apps/cube_stack/verify_simulation.py` additionally verifies the actual
C++ publisher and this receiver over TCP, with optional `--robot` integration.

## Third-party robot assets

The URDF and robot meshes in `zeus_description/` were copied from the local
working assets, whose recorded source repository is
[JuoTungChen/ROS2_pick_and_place_UR5](https://github.com/JuoTungChen/ROS2_pick_and_place_UR5).
The copied asset directory did not contain a license file. Before publishing
this repository, verify the applicable redistribution and attribution terms
from that project and the upstream Universal Robots and Robotiq description
packages. This project does not assign a new license to those third-party
assets.



### Glued-piece verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe verify_glued_pieces.py
# Optional rendered video:
.\.venv\Scripts\python.exe verify_glued_pieces.py --record outputs/glued-pieces.mp4
```

On Windows with Python 3.11 and Genesis CPU, the glued verifier checks doubles
and triples as one-link bodies with one grasp, measured tool-relative carrying,
yawed floating targets, bar-end blockers, oriented staging, same-shape reuse,
reset, unreachable-target rejection and a real two-arm triple handoff. The full
C++ publisher/Genesis integration also checks add/move/delete/reset for both sizes.
This verifies the default kinematic controller; physical grasp/stack stability
and physical table/Quest interaction require their own hardware checks.
