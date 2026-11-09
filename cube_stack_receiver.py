"""Simulator-independent live receiver loop shared by every robot backend.

OpenUp Cube Stack publishes ``CubeStack/Map`` snapshots; ``LiveMapQueue`` turns
them into add/move/delete/reset commands, and a backend executes each one. This
module owns the loop that connects the two, so every simulator prints the same
``ready:``, ``started:``, ``completed:`` and ``live complete:`` lines that OpenUp's
``verify_simulation.py`` checks.

A backend (see ``cube_stack_planner.LiveScenePlanner``) provides:

- ``options.pool_per_color``, and ``live_client`` / ``live_queue`` attributes;
- ``place_live_cube``, ``move_live_cube``, ``delete_live_cube`` (each takes a
  ``PiecePlacement``) and ``reset_live_scene()``;
- ``_sample_physical_state(queue)``: record actual poses in ``queue.realized``;
- ``_step()``: advance one simulation step, calling ``pump_and_check``;
- ``_start_recording()`` and ``_stop_recording(settle=False)``;
- ``clear_carried()``;
- optionally ``idle_sleep`` (seconds to sleep between idle steps).
"""
from dataclasses import replace
import math
import time

import numpy as np

from cube_stack_layout import quaternion_to_rotation


class ResetRequested(Exception):
    """Unwind a trajectory at a simulation step for an explicit session reset."""


def pump_and_check(client, queue=None):
    """Receive pending maps; stop the current motion if the session halted or reset."""
    if client is None:
        return
    client.pump()
    queue = queue if queue is not None else client.queue
    if queue.halted:
        raise RuntimeError(f"HALTED: {queue.halt_reason}")
    if queue.reset_pending:
        raise ResetRequested()


def wait_for_first_map(queue, client, receive_only=False, max_cubes=0, timeout=0):
    """Pump until the first map arrives (or, for receive-only, until ``max_cubes``).

    Returns True when a receive-only diagnostic has finished.
    """
    deadline = time.monotonic() + timeout if timeout else None
    while receive_only or queue.layout is None:
        client.pump()
        if queue.halted:
            raise RuntimeError(f"HALTED: {queue.halt_reason}")
        if receive_only and queue.layout is not None and max_cubes and queue.accepted_count >= max_cubes:
            print(f"receive-only complete: session={queue.session_id} accepted={queue.accepted_count} completed=0", flush=True)
            return True
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Timed out receiving the requested live map/cube count")
        time.sleep(0.01)
    return False


def run_live(backend, queue, client, max_cubes=0):
    """Only this outer loop starts commands; ``backend._step`` only receives/enqueues."""
    backend.live_client = client
    backend.live_queue = queue
    backend._start_recording()
    idle_sleep = getattr(backend, "idle_sleep", 0.01)
    print(f"ready: session={queue.session_id} pool_per_color={backend.options.pool_per_color}", flush=True)
    try:
        while not max_cubes or queue.completed_count < max_cubes:
            client.pump()
            if queue.halted:
                raise RuntimeError(f"HALTED: {queue.halt_reason}")
            if not queue.reset_pending:
                backend._sample_physical_state(queue)
            command = queue.start_next()
            try:
                if command is None:
                    client.publish_robot_state()
                    backend._step()
                    time.sleep(idle_sleep)
                    continue
                if command.kind == "reset":
                    backend.reset_live_scene()
                elif command.kind == "add":
                    backend.place_live_cube(command.placement)
                elif command.kind == "move":
                    backend.move_live_cube(command.placement)
                elif command.kind == "delete":
                    backend.delete_live_cube(command.placement)
                queue.complete_active()
                backend._sample_physical_state(queue)
                client.publish_robot_state()
            except ResetRequested:
                backend.clear_carried()
                queue.cancel_active_for_reset()
    except BaseException as error:
        queue.fail_active(str(error) or type(error).__name__)
        backend.clear_carried()
        raise
    finally:
        # Saving a failed run must not advance the scene or issue commands.
        backend._stop_recording(settle=False)
    print(f"live complete: accepted={queue.accepted_count} completed={queue.completed_count}", flush=True)


def live_main(build_backend, host="127.0.0.1", port=9036, pool_per_color=30, receive_only=False,
              max_cubes=0, timeout=0):
    """Connect to OpenUp, build the backend from the first map, and run it.

    ``build_backend(layout)`` returns a backend; it is called only after the first
    map arrives, because table size, staging and the pool depend on that map.
    """
    from cube_stack_stream import LiveMapQueue, LiveStreamClient

    def report(message):
        print(message, flush=True)

    queue = LiveMapQueue(pool_per_color=pool_per_color, report=report)
    client = LiveStreamClient(queue, host=host, port=port, report=report)
    try:
        print(f"waiting: CubeStack/Map at {host}:{port}", flush=True)
        if wait_for_first_map(queue, client, receive_only=receive_only,
                              max_cubes=max_cubes, timeout=timeout):
            return queue
        try:
            backend = build_backend(queue.layout)
        except Exception as error:
            queue.fail_active(error)
            raise
        run_live(backend, queue, client, max_cubes=max_cubes)
        return backend
    finally:
        client.close()


def _yaw_and_tilt(rotation):
    """Yaw about +Z and the angle between the body's +Z axis and world +Z."""
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    tilt = math.acos(max(-1.0, min(1.0, float(rotation[2, 2]))))
    return yaw, tilt


POSITION_TOLERANCE = 0.01  # metres
ANGLE_TOLERANCE_DEG = 3.0


def pose_within_tolerance(position, quaternion, target_position, target_quaternion, length=1,
                          position_tolerance=POSITION_TOLERANCE, angle_tolerance_deg=ANGLE_TOLERANCE_DEG):
    """Whether a measured physical pose counts as the target pose.

    Yaw is compared modulo 90 degrees for a single cube, whose footprint repeats
    every quarter turn, and modulo 180 degrees for a glued bar.
    """
    offset = np.asarray(position, dtype=float) - np.asarray(target_position, dtype=float)
    if np.linalg.norm(offset) > position_tolerance:
        return False
    measured_yaw, measured_tilt = _yaw_and_tilt(quaternion_to_rotation(quaternion))
    target_yaw, target_tilt = _yaw_and_tilt(quaternion_to_rotation(target_quaternion))
    period = math.pi / 2 if length == 1 else math.pi
    yaw_error = abs((measured_yaw - target_yaw + period / 2) % period - period / 2)
    limit = math.radians(angle_tolerance_deg)
    return yaw_error <= limit and abs(measured_tilt - target_tilt) <= limit


def snap_to_target(measured, target, position_tolerance=POSITION_TOLERANCE,
                   angle_tolerance_deg=ANGLE_TOLERANCE_DEG):
    """Return ``target``'s pose when a measured physical pose is close enough, else ``measured``.

    Physics engines leave a placed body millimetres from its commanded pose. The
    queue (``placements_share_pose``), the planner and OpenUp all compare poses at
    1e-5, so a backend running real physics reports the exact target for a body
    within tolerance and the measured pose otherwise. A measured pose keeps that
    piece pending in OpenUp and lets the queue issue a corrective move.
    """
    if target is None or not pose_within_tolerance(
            measured.position, measured.quaternion, target.position, target.quaternion,
            getattr(target, "length", 1), position_tolerance, angle_tolerance_deg):
        return measured
    return replace(measured, position=tuple(target.position), quaternion=tuple(target.quaternion))
