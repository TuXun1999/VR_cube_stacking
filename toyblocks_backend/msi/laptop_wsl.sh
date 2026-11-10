#!/usr/bin/env bash
# Laptop side, run in WSL Ubuntu on the Windows PC that runs cube_stack.exe.
# MSI logins need a password plus Duo, and Windows' own OpenSSH cannot share one
# login between commands, so WSL's ssh keeps one shared ("ControlMaster") login
# open and every later command and tunnel reuses it without prompting.
#
#   laptop_wsl.sh login                 once, in a terminal: type your password, approve Duo
#   laptop_wsl.sh tunnel NODE           table + viewer forwards for a GPU node (e.g. agc07)
#   laptop_wsl.sh run NODE SCRIPT       run a local bash script on NODE (no prompts)
#   laptop_wsl.sh status                is the shared login still open?
#
# Set MSI_USER to your MSI user name if it differs from your WSL user name.
set -euo pipefail
MSI_USER=${MSI_USER:-$USER}
LOGIN=${MSI_LOGIN:-agate.msi.umn.edu}
SOCKET=${MSI_SOCKET:-$HOME/.ssh/cm-msi}
TABLE_PORT=${TABLE_PORT:-9036}            # cube_stack.exe --set /cube_stack/simulation_port
ROBOT_PORT=${ROBOT_PORT:-39036}           # loopback port on the login and GPU nodes
VIEWER_PORT=${VIEWER_PORT:-18080}         # http://localhost:VIEWER_PORT on the laptop
NODE_VIEWER_PORT=${NODE_VIEWER_PORT:-8080}
target="$MSI_USER@$LOGIN"
ctl=(-o "ControlPath=$SOCKET")

case "${1:-}" in
  login)
    mkdir -p -m 700 "$(dirname "$SOCKET")"
    exec ssh -M "${ctl[@]}" -o ControlPersist=10h -o ServerAliveInterval=60 \
      -o StrictHostKeyChecking=accept-new "$target"
    ;;
  tunnel)
    node=${2:?usage: laptop_wsl.sh tunnel NODE}
    # WSL2 (NAT networking) reaches Windows at its default gateway.
    win=$(ip route | awk '/^default/ {print $3; exit}')
    if ! timeout 5 bash -c "exec 3<>/dev/tcp/$win/$TABLE_PORT"; then
      echo "cannot reach the table at $win:$TABLE_PORT; start cube_stack.exe with" \
           "--set /cube_stack/simulation_port=$TABLE_PORT and allow it through the firewall"
      exit 1
    fi
    # login node 127.0.0.1:ROBOT_PORT -> Windows TABLE_PORT, over the shared login
    ssh "${ctl[@]}" -O forward -R "127.0.0.1:$ROBOT_PORT:$win:$TABLE_PORT" "$target" \
      || echo "(table forward already in place)"
    # laptop localhost:VIEWER_PORT -> NODE:NODE_VIEWER_PORT
    ssh "${ctl[@]}" -O forward -L "127.0.0.1:$VIEWER_PORT:$node:$NODE_VIEWER_PORT" "$target" \
      || echo "(viewer forward already in place)"
    # NODE 127.0.0.1:ROBOT_PORT -> login node ROBOT_PORT (compute nodes are reachable only
    # from login nodes, and only while you have a job on them)
    ssh "${ctl[@]}" -o BatchMode=yes "$target" bash -s -- "$node" "$ROBOT_PORT" <<'EOF'
node=$1; port=$2
if ! pgrep -u "$USER" -f "R 127.0.0.1:$port:127.0.0.1:$port $node" >/dev/null; then
  ssh -f -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 \
      -R "127.0.0.1:$port:127.0.0.1:$port" "$node" < /dev/null > /dev/null 2>&1
fi
ssh -o BatchMode=yes "$node" "timeout 5 bash -c 'exec 3<>/dev/tcp/127.0.0.1/$port'" \
  && echo "tunnel ok: $node 127.0.0.1:$port reaches the table"
EOF
    echo "viewer: http://localhost:$VIEWER_PORT (once the robot is started with --viewer)"
    ;;
  run)
    node=${2:?usage: laptop_wsl.sh run NODE SCRIPT}
    script=${3:?usage: laptop_wsl.sh run NODE SCRIPT}
    tr -d '\r' < "$script" | ssh "${ctl[@]}" -o BatchMode=yes "$target" "ssh -o BatchMode=yes $node bash -s"
    ;;
  status)
    ssh "${ctl[@]}" -O check "$target"
    ;;
  *)
    sed -n '2,13p' "$0"
    exit 2
    ;;
esac
