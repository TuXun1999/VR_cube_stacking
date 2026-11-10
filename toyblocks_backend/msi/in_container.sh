#!/usr/bin/env bash
# Run a command inside the ToyBlocks container, from this repository's root, with
# the GPU (--nv), the host's NVIDIA Vulkan/EGL driver files, and the venv active:
#
#   toyblocks_backend/msi/in_container.sh python -u toyblocks_backend/spike.py --headless
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
ROOT=${OPENUP_TB_ROOT:-/scratch.global/$USER/openup-toyblocks}
binds=(-B "$ROOT" -B "$REPO")
# Isaac Sim needs Vulkan; the image has the loader but not NVIDIA's driver manifest.
[ -f /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json ] && \
  binds+=(-B /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json:/usr/share/vulkan/icd.d/nvidia_icd.json)
[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ] && \
  binds+=(-B /usr/share/glvnd/egl_vendor.d/10_nvidia.json)
exec apptainer exec --nv "${binds[@]}" "$ROOT/toyblocks.sif" bash -c \
  "source '$ROOT/venv/bin/activate' && export OMNI_KIT_ACCEPT_EULA=YES && cd '$REPO' && $*"
