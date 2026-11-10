#!/usr/bin/env bash
# One-time ToyBlocks setup on an MSI GPU node. Run it on the compute node, inside
# your job (never on a login node), from this repository's checkout on scratch:
#
#   OMNI_KIT_ACCEPT_EULA=YES bash toyblocks_backend/msi/setup_node.sh
#
# Setting OMNI_KIT_ACCEPT_EULA=YES accepts NVIDIA's Isaac Sim license terms
# (https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/licenses.html).
# Everything goes under $OPENUP_TB_ROOT (default /scratch.global/$USER/openup-toyblocks):
# the container image, the ToyBlocks clone and the Python venv. It is safe to re-run.
set -euo pipefail
: "${OMNI_KIT_ACCEPT_EULA:?Set OMNI_KIT_ACCEPT_EULA=YES to accept the NVIDIA Isaac Sim license terms}"
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${OPENUP_TB_ROOT:-/scratch.global/$USER/openup-toyblocks}
mkdir -p "$ROOT"

# Apptainer's build cache goes to the job-local disk, not the home quota.
export APPTAINER_CACHEDIR=${TMPDIR:-/tmp}/apptainer-cache APPTAINER_TMPDIR=${TMPDIR:-/tmp}/apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

if [ ! -f "$ROOT/toyblocks.sif" ]; then
  echo "== building $ROOT/toyblocks.sif $(date +%T)"
  apptainer build --fakeroot "$ROOT/toyblocks.sif" "$HERE/toyblocks.def"
fi

echo "== installing Python packages inside the container $(date +%T)"
apptainer exec --nv -B "$ROOT" -B "$HERE" "$ROOT/toyblocks.sif" \
  env OMNI_KIT_ACCEPT_EULA=YES bash "$HERE/install_python.sh" "$ROOT"
echo "== setup complete $(date +%T); start the robot with toyblocks_backend/msi/robot.sh start"
