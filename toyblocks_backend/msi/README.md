# Running the ToyBlocks robot on MSI (Agate)

This is how the ToyBlocks backend (`robot_receiver.py --sim toyblocks`) was set
up on the Minnesota Supercomputing Institute's Agate cluster and connected to the
OpenUp Cube Stack desktop table running on a Windows laptop. It was first done on
2026-11-10 on node `agc07` (NVIDIA A40) and reached a working live session with
ToyBlocks' browser 3D viewer.

```
Windows laptop                                      MSI Agate
┌──────────────────────────┐                        ┌──────────────────────────────┐
│ cube_stack.exe  :9036 ◄──┼── WSL ssh (one login) ─┤ login node 127.0.0.1:39036   │
│ browser  localhost:18080 ├──────────────────────► │   │ ssh -R hop               │
│ WSL Ubuntu: laptop_wsl.sh│   password + Duo once  │   ▼                          │
└──────────────────────────┘                        │ GPU node (your Slurm job)    │
                                                    │  Apptainer: Ubuntu 24.04     │
                                                    │   Isaac Sim 5.1 + ToyBlocks  │
                                                    │   robot_receiver.py → :39036 │
                                                    │   viser viewer :8080         │
                                                    └──────────────────────────────┘
```

## Why MSI

ToyBlocks runs on Isaac Sim 5.1, which needs an NVIDIA RTX-class GPU with at least
8 GB of memory (16 GB on the official requirements page), 32 GB of RAM and about
50 GB of disk. A laptop with an RTX 3060 Laptop GPU (6 GB) and 16 GB of RAM is
below that. Isaac Sim also does not support GPUs without RT cores, so on Agate use
the **A40** or **L40S** nodes (partitions `interactive-gpu` / `preempt-gpu`), not
A100, H100 or V100.

## 1. Get a GPU node

Start an interactive GPU session. We used an Open OnDemand session on
`interactive-gpu` with 1×A40, 16 cores, 60 GB of memory, 100 GB of job-local
`/tmp` and a 24-hour limit. From a login node, `squeue -u $USER` shows the node
name (ours was `agc07`). An `salloc` along these lines also works; adjust it to
your allocation:

```bash
salloc -p interactive-gpu --gres=gpu:a40:1 -c 16 --mem=60G -t 24:00:00
```

Compute nodes accept SSH only from a login node, and only while you have a job
running on them. Do the heavy work (building, installing, running Isaac) on the
compute node, never on a login node.

## 2. Log in once from the laptop (WSL)

MSI logins need your password plus a Duo approval. Windows' OpenSSH cannot reuse
one login across commands, so the laptop side uses WSL Ubuntu's `ssh` with a
shared ("ControlMaster") connection. Log in once in a terminal and leave it open:

```bash
MSI_USER=<your x500> bash toyblocks_backend/msi/laptop_wsl.sh login
```

For the next 10 hours, every `laptop_wsl.sh run|tunnel` call reuses that login
without prompting. `laptop_wsl.sh status` reports whether it is still open.

## 3. Put this repository on scratch

Work under `/scratch.global/$USER/openup-toyblocks` (override with
`OPENUP_TB_ROOT`). Home quotas are small, and scratch has plenty of room.
Scratch is not backed up, so check MSI's current scratch purge policy. Clone the
branch there:

```bash
mkdir -p /scratch.global/$USER/openup-toyblocks && cd /scratch.global/$USER/openup-toyblocks
git clone -b feat/sim-backends <this repository> VR_cube_stacking
```

We copied a `git archive` of the branch over the shared login instead, which
works the same way.

## 4. One-time setup on the GPU node

Agate runs Rocky Linux 8.10 (glibc 2.28), which is too old for Isaac Sim's pip
packages. So ToyBlocks runs inside a small **Apptainer** image (Ubuntu 24.04, the
system libraries from ToyBlocks' own `docker/vecgen.dockerfile`). Python and all
the packages live in a venv on scratch. On the GPU node:

```bash
cd /scratch.global/$USER/openup-toyblocks/VR_cube_stacking
OMNI_KIT_ACCEPT_EULA=YES bash toyblocks_backend/msi/setup_node.sh
```

Setting `OMNI_KIT_ACCEPT_EULA=YES` accepts NVIDIA's
[Isaac Sim license terms](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/licenses.html).
The script:

1. Builds `toyblocks.sif` from `toyblocks.def` with `apptainer build --fakeroot`
   (about 1 minute, 226 MB).
2. Clones ToyBlocks at the pinned commit (`TOYBLOCKS_COMMIT` in
   `toyblocks_backend/__init__.py`) with only its `curobo` submodule.
3. Creates a Python 3.11 venv with `uv`, then installs ToyBlocks with Isaac Sim,
   Isaac Lab and torch (CUDA 12.8), Warp, LeRobot and cuRobo v2. It pins every
   package with ToyBlocks' `docker/vecgen.constraints.txt`.
4. Downloads ToyBlocks' assets (about 280 MB, cached in `~/.toyblocks`).

On `agc07` the install took about 15 minutes. Compute nodes have internet access
(PyPI, GitHub, NVIDIA's registry).

`in_container.sh` runs any command inside the image with the GPU (`--nv`), the
host's NVIDIA Vulkan and EGL driver manifests, and the venv active, from the
repository root.

## 5. Check it: the spike

```bash
toyblocks_backend/msi/in_container.sh python -u toyblocks_backend/spike.py --headless
```

First run on the A40 (Isaac Sim takes a minute or two to start):

| Placement | Result |
|---|---|
| flat cube | PASS, 9.7 mm, 0.2° |
| cube turned 30° | PASS, 6.8 mm, 1.0° |
| cube stacked on the first | PASS, 5.0 mm, 0.7° |
| cube touching the side of that stack | PASS, 0.8 mm, 0.2° |
| triple bar turned 30° | FAIL, no feasible grasp |

That run led to two backend changes:

- The pick's IK check no longer treats the piece being picked as an obstacle;
  cuRobo's open-finger spheres overlap an 8 cm piece.
- ToyBlocks counts a piece as placed within **2 cm and 5°** (`entity.py`).

The bar has not been re-run since that change. The spike process can hang at
Isaac shutdown after printing its result; stop it with `kill -- -<pid>`.

## 6. Start the table, the tunnel and the robot

**On the laptop (Windows),** start the desktop table with the simulation stream:

```powershell
Set-Location C:\Users\<you>\OpenUp\OpenUp
& '.\build-cube\openup_apps\cube_stack\Release\cube_stack.exe' --config openup_apps/cube_stack/desktop.json --set /cube_stack/simulation_port=9036
```

`cube_stack.exe` listens on all interfaces. Its Windows Firewall inbound rule
(created the first time it ran) lets WSL reach it at the Windows host address,
the WSL default gateway, for example `172.31.128.1`.

**In WSL,** build the tunnel for your node:

```bash
MSI_USER=<your x500> bash toyblocks_backend/msi/laptop_wsl.sh tunnel agc07
```

That sets up three forwards, all over the one shared login:

- login node `127.0.0.1:39036` → Windows `:9036`: a reverse forward on the shared login.
- `agc07 127.0.0.1:39036` → login node `:39036`: an `ssh -R` hop started on the
  login node.
- laptop `localhost:18080` → `agc07:8080`: for the viewer.

**On the GPU node,** start the robot:

```bash
toyblocks_backend/msi/robot.sh start      # waits for "ready:"; prints the viewer port
toyblocks_backend/msi/robot.sh log        # follow started/completed/rejected lines
toyblocks_backend/msi/robot.sh stop
```

`robot.sh start` runs `robot_receiver.py --sim toyblocks --live --headless
--viewer --host 127.0.0.1 --port 39036` detached, logging to
`$OPENUP_TB_ROOT/receiver.log`. From the laptop, run it with
`laptop_wsl.sh run agc07 <script>`, where the script contains the `robot.sh`
command.

Then build on the desktop table. Cubes show as see-through until the robot has
placed them. Watch the robot at **http://localhost:18080**.

## What works and what does not

- **Works:** a live session against the desktop table, physical grasping of 8 cm
  cubes, reset, and ToyBlocks' viser web viewer (robot, table, pads, pieces).
- **Not yet reliable.** In the first interactive session the robot placed two
  single cubes, then halted on the third:
  `unstable: piece_3 did not stay at its target after 2 attempts (place_at effect check failed)`.
  Twice, that cube came to rest more than 3 cm from its target after release.
  This has not been diagnosed yet (likely suspects: a release near a neighbouring
  cube, or the cube slipping in the gripper). After a halt, reset the table, then
  `robot.sh stop` and `robot.sh start`.
- **No Isaac Sim window, cameras or RTX rendering.** Inside the container Vulkan
  fails with `ERROR_INCOMPATIBLE_DRIVER` / `vkCreateInstance failed`. Physics and
  cuRobo planning run anyway, and the viser viewer needs no Vulkan. Fixing this
  (probably by binding more of the host's NVIDIA Vulkan libraries) is the next
  step before cameras or vision policies can be used.
- **One arm.** ToyBlocks drives a single Franka FR3. Keep cubes near the middle
  of the robot-side table edge. An unreachable target halts the session
  (`rejected: HALTED: No reachable robot transfer`). Reset the table, then
  `robot.sh stop` and `robot.sh start`.

## Problems we hit, and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `ssh agc07` times out from the laptop | Compute nodes are only reachable from login nodes | Hop through the login node (`laptop_wsl.sh`) |
| `Permission denied (publickey,keyboard-interactive)` | MSI needs a password plus Duo | Log in once with `laptop_wsl.sh login`; later commands reuse it |
| Remote commands half-ran locally, `squeue` errors | PowerShell → `wsl` → `ssh` quoting mangles `$(...)`, `%` and `;` | Pipe a script file to `bash -s` (`laptop_wsl.sh run`) |
| `C:/Program Files/Git/mnt/c/...: No such file` | Git Bash rewrites `/mnt/c` paths passed to `wsl.exe` | Set `MSYS_NO_PATHCONV=1`, or call from PowerShell |
| Isaac pip packages won't install on the node | Rocky 8 glibc 2.28 | Apptainer image (`toyblocks.def`) |
| `vkCreateInstance failed` | NVIDIA Vulkan driver not fully visible in the container | Open; run headless and use the viser viewer |
| Spike prints its result, then never exits | Isaac shutdown hangs | Kill the process group |
| Viewer raises when updated live | `view_frame` compares numpy-backed `Frame`s | The backend clears `_last_viewed_set` before each update |

## Lifetimes

- **Shared login:** about 10 hours (`ControlPersist=10h`), or until you close it.
- **Robot and its tunnel hop:** end with your Slurm job.
- **Files under `/scratch.global/$USER/openup-toyblocks`:** persist between jobs
  (subject to MSI's scratch policy). A new job needs `setup_node.sh` only when the
  pin changes; otherwise just `laptop_wsl.sh tunnel <node>` and `robot.sh start`.
