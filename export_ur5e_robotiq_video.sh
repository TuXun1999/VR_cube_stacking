#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_path="${1:-${script_dir}/outputs/ur5e_color_anchors.mp4}"

python3 "${script_dir}/test_ur5e_robotiq_pose.py" \
    --headless \
    --record "${output_path}" \
    --fps "${FPS:-30}" \
    --speed "${SPEED:-1.25}"
