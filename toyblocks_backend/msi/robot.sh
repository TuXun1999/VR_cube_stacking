#!/usr/bin/env bash
# Start, stop and follow the ToyBlocks robot receiver on the MSI GPU node, inside
# your job. It runs detached, so a dropped SSH session does not stop it.
#
#   robot.sh start [PORT] [receiver args...]   connect to 127.0.0.1:PORT (default 39036), with --viewer
#   robot.sh stop
#   robot.sh status
#   robot.sh log                               follow the progress lines
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${OPENUP_TB_ROOT:-/scratch.global/$USER/openup-toyblocks}
PIDFILE=$ROOT/receiver.pid
LOG=$ROOT/receiver.log
PROGRESS="^(waiting|connection|accepted|ready|started|completed|cancelled|rejected|failed|viewer|stopped|live complete)|Traceback|Error:"

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-status}" in
  start)
    port=${2:-39036}
    extra=("${@:3}")
    if running; then echo "already running (pid $(cat "$PIDFILE"))"; exit 0; fi
    [ -f "$LOG" ] && mv "$LOG" "$ROOT/receiver.$(date +%Y%m%d-%H%M%S).log"
    nohup setsid "$HERE/in_container.sh" python -u robot_receiver.py --sim toyblocks --live --headless \
      --viewer --host 127.0.0.1 --port "$port" "${extra[@]}" > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "started (pid $!); waiting for ready (Isaac Sim takes 1-2 minutes)"
    for _ in $(seq 1 90); do
      grep -q "^ready:" "$LOG" 2>/dev/null && break
      running || break
      sleep 5
    done
    grep -E "$PROGRESS" "$LOG" | tail -6
    running || { echo "the receiver exited:"; tail -20 "$LOG"; exit 1; }
    ;;
  stop)
    if running; then
      pid=$(cat "$PIDFILE")
      kill -INT -- "-$pid" 2>/dev/null; sleep 4
      kill -TERM -- "-$pid" 2>/dev/null; sleep 3
      kill -KILL -- "-$pid" 2>/dev/null
      echo "stopped"
    else
      echo "not running"
    fi
    ;;
  status)
    if running; then echo "running (pid $(cat "$PIDFILE"))"; else echo "not running"; fi
    grep -E "$PROGRESS" "$LOG" 2>/dev/null | tail -5
    ;;
  log)
    tail -n 50 -F "$LOG" | grep --line-buffered -E "$PROGRESS"
    ;;
  *)
    sed -n '2,9p' "$0"; exit 2
    ;;
esac
