#!/usr/bin/env bash
# Runs inside the ToyBlocks container (setup_node.sh calls it). Follows ToyBlocks'
# docker/vecgen.dockerfile at the pinned commit, including its constraints file.
#   install_python.sh ROOT
set -euo pipefail
ROOT=${1:?usage: install_python.sh ROOT}
HERE=$(cd "$(dirname "$0")" && pwd)
PIN=$(sed -n 's/^TOYBLOCKS_COMMIT = "\(.*\)"/\1/p' "$HERE/../__init__.py")
export UV_PYTHON_INSTALL_DIR=$ROOT/uv-python UV_CACHE_DIR=${TMPDIR:-/tmp}/uv-cache UV_LINK_MODE=copy
cd "$ROOT"

echo "== ToyBlocks $PIN $(date +%T)"
[ -d ToyBlocks/.git ] || git clone --filter=blob:none https://github.com/allenai/ToyBlocks.git
cd ToyBlocks
git fetch --quiet origin || true
git checkout --quiet "$PIN"
git submodule update --init curobo

echo "== venv (Python 3.11) $(date +%T)"
[ -x "$ROOT/venv/bin/python" ] || uv venv "$ROOT/venv" --python 3.11
source "$ROOT/venv/bin/activate"
export UV_CONSTRAINT=$PWD/docker/vecgen.constraints.txt

echo "== ToyBlocks + Isaac Sim / Isaac Lab + torch (cu128) $(date +%T)"
uv pip install -e ".[isaac,torch-cu128]" --config-settings editable_mode=strict
echo "== build tools, Warp, LeRobot $(date +%T)"
uv pip install "setuptools<81" setuptools_scm ninja wheel "warp-lang==1.12.1" av "lerobot==0.4.3" flask
echo "== cuRobo v2 $(date +%T)"
uv pip install --no-build-isolation -e "./curobo[cu12]"
echo "== assets (~280 MB, cached in ~/.toyblocks) $(date +%T)"
get-toyblocks-assets --asset-manifest asset_manifest.json
echo "== import check $(date +%T)"
python -c "import curobo.scene, warp, toyblocks, isaaclab; print('imports ok; warp', warp.__version__)"
