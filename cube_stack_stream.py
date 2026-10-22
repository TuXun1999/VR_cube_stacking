"""Bounded OpenUp stream decoding and desired-state robot reconciliation.

The client is pumped by its owner thread. It never executes robot commands.
"""
from collections import Counter
import copy
from dataclasses import dataclass
import errno
import json
import math
import select
import socket
import struct
import time

import numpy as np

from cube_stack_layout import (
    CubePlacement, VR_TO_GENESIS_AXES, parse_table_layout, placements_share_pose,
)

MAX_FRAME_BYTES = 64 * 1024 * 1024


class ProtocolError(ValueError):
    """A stream message cannot safely be accepted."""


def _json_object(encoded, description):
    def invalid_constant(value):
        raise ValueError(f"non-finite JSON number {value}")

    try:
        if isinstance(encoded, bytes):
            encoded = encoded.decode("utf-8")
        value = json.loads(encoded, parse_constant=invalid_constant)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ProtocolError(f"Invalid {description} JSON: {error}") from error
    if not isinstance(value, dict):
        raise ProtocolError(f"{description} must be a JSON object")
    return value


def validate_header(header):
    if not isinstance(header, dict):
        raise ProtocolError("header must be a JSON object")
    if header.get("stream") != "CubeStack/Map":
        raise ProtocolError("Expected CubeStack/Map stream")
    if header.get("content_type") != "application/json":
        raise ProtocolError("Expected application/json content_type")
    for field in ("sequence", "payload_bytes"):
        if type(header.get(field)) is not int or header[field] < 0:
            raise ProtocolError(f"Invalid {field}")
    if header["payload_bytes"] > MAX_FRAME_BYTES:
        raise ProtocolError("payload_bytes exceeds 64 MiB limit")
    timestamp = header.get("timestamp")
    if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
        raise ProtocolError("Invalid timestamp")
    if not isinstance(header.get("metadata"), str):
        raise ProtocolError("metadata must be a JSON string")
    metadata = _json_object(header["metadata"], "metadata")
    if type(metadata.get("protocol_version")) is not int or metadata["protocol_version"] not in (1, 2):
        raise ProtocolError("Unsupported protocol_version; expected 1 or 2")
    if not isinstance(metadata.get("session_id"), str) or not metadata["session_id"]:
        raise ProtocolError("metadata session_id must be a nonempty string")
    return metadata


class FrameDecoder:
    """Incrementally decode uint32LE(header), header, uint32LE(map), map."""

    def __init__(self):
        self.buffer = bytearray()
        self.header = None
        self.length = None

    def feed(self, chunk):
        self.buffer.extend(chunk)
        frames = []
        while True:
            if self.length is None:
                if len(self.buffer) < 4:
                    break
                self.length = struct.unpack_from("<I", self.buffer)[0]
                del self.buffer[:4]
                if self.length > MAX_FRAME_BYTES:
                    raise ProtocolError("Frame length exceeds 64 MiB limit")
                if self.header is not None and self.length != self.header["payload_bytes"]:
                    raise ProtocolError("payload length disagrees with payload_bytes")
            if len(self.buffer) < self.length:
                break
            block = bytes(self.buffer[:self.length])
            del self.buffer[:self.length]
            self.length = None
            if self.header is None:
                self.header = _json_object(block, "header")
                validate_header(self.header)
            else:
                frames.append((self.header, _json_object(block, "payload")))
                self.header = None
        return frames


@dataclass(frozen=True)
class LiveCommand:
    """One acknowledged robot operation; reset has no cube placement."""

    kind: str
    placement: CubePlacement | None = None
    reset_generation: int = 0

    @property
    def cube_id(self):
        return self.placement.cube_id if self.placement is not None else "reset"


class LiveMapQueue:
    """Accept whole maps atomically and reconcile only the newest desired state.

    ``realized`` records acknowledged robot results. A newer snapshot never
    mutates an active command; completing it records its actual target, after
    which the next command catches up with the newest desired map. Reset is the
    explicit exception: the owner may cancel an interrupted command and return
    the entire pool to staging before acknowledging the reset.
    """

    def __init__(self, pool_per_color=30, report=print):
        if type(pool_per_color) is not int or pool_per_color <= 0:
            raise ValueError("pool_per_color must be a positive integer")
        self.pool_per_color = pool_per_color
        self.report = report
        self.session_id = None
        self.protocol_version = None
        self.layout = None
        self._accepted = None
        self.revision = None
        self.reset_generation = 0
        self._acknowledged_generation = 0
        self.realized = {}
        self.reserved = Counter()
        self.active = None
        self.completed_count = 0
        self.completed_command_count = 0
        self.halted = False
        self.halt_reason = None

    @property
    def accepted_count(self):
        return len(self.layout.placements) if self.layout is not None else 0

    @property
    def pending_count(self):
        projected = dict(self.realized)
        active_reset_generation = None
        if self.active is not None:
            self._apply_result(projected, self.active)
            if self.active.kind == "reset":
                active_reset_generation = self.active.reset_generation
        if self.reset_pending:
            # A reset in flight already accounts for its own generation. A
            # newer reset still needs a separate acknowledgement.
            extra_reset = int(active_reset_generation != self.reset_generation)
            return extra_reset + self.accepted_count
        return len(self._commands_for(projected))

    @property
    def reset_pending(self):
        return self.reset_generation > self._acknowledged_generation

    def halt(self, reason):
        if not self.halted:
            self.halted = True
            self.halt_reason = str(reason)
            self.report(f"rejected: HALTED: {reason}; restart the live simulation to begin a new session")

    def accept(self, header, data):
        """Return changed desired IDs, 0 for a repeat/stale map, or -1 on halt."""
        if self.halted:
            return -1
        try:
            metadata = validate_header(header)
            protocol = metadata["protocol_version"]
            if not isinstance(data, dict):
                raise ProtocolError("payload must be a JSON object")
            if self.session_id is not None and metadata["session_id"] != self.session_id:
                raise ProtocolError("table session changed")
            if self.protocol_version is not None and protocol != self.protocol_version:
                raise ProtocolError("stream protocol changed")
            if self._accepted is not None:
                ignored = {"cubes", "revision", "reset_generation"} if protocol == 2 else {"cubes"}
                previous_metadata = {key: value for key, value in self._accepted.items() if key not in ignored}
                current_metadata = {key: value for key, value in data.items() if key not in ignored}
                if previous_metadata != current_metadata:
                    raise ProtocolError("table metadata changed")
            if protocol == 2:
                for field in ("revision", "reset_generation"):
                    if type(data.get(field)) is not int or data[field] < 0:
                        raise ProtocolError(f"{field} must be a nonnegative integer")
                revision = data["revision"]
                generation = data["reset_generation"]
                if self._accepted is not None:
                    if generation < self.reset_generation:
                        return 0
                    if generation == self.reset_generation:
                        if revision < self.revision:
                            return 0
                        if revision == self.revision:
                            if data != self._accepted:
                                raise ProtocolError("payload changed at the same revision")
                            return 0
            else:
                revision = 0 if self.revision is None else self.revision + 1
                generation = 0
            layout = parse_table_layout(data, require_ids=protocol == 2)
            if protocol == 1 and self._accepted is not None:
                previous = self._accepted["cubes"]
                if len(data["cubes"]) < len(previous) or data["cubes"][:len(previous)] != previous:
                    raise ProtocolError("accepted cube prefix changed or shrank")
                if data == self._accepted:
                    return 0
            proposed = Counter(placement.color_name for placement in layout.placements)
            for color, count in proposed.items():
                if count > self.pool_per_color:
                    raise ProtocolError(f"{color} pool exhausted: needs {count}, capacity {self.pool_per_color}")
            desired = {placement.cube_id: placement for placement in layout.placements}
            previous_desired = self._desired()
            known = dict(previous_desired)
            if not self.reset_pending:
                known.update(self.realized)
            if (self.active is not None and self.active.placement is not None
                    and self.active.reset_generation == generation):
                known[self.active.cube_id] = self.active.placement
            # An explicit reset starts fresh identities, but same-generation
            # edits cannot turn an allocated red body into a blue body.
            if generation == self.reset_generation:
                for cube_id in desired.keys() & known.keys():
                    if desired[cube_id].color_name != known[cube_id].color_name:
                        raise ProtocolError(f"Cube {cube_id!r} changed color")
            changed = sum(previous_desired.get(cube_id) != desired.get(cube_id)
                          for cube_id in previous_desired.keys() | desired.keys())
            # No observable queue state changes until the whole snapshot passed.
            self.session_id = metadata["session_id"]
            self.protocol_version = protocol
            self._accepted = copy.deepcopy(data)
            self.layout = layout
            self.revision = revision
            self.reset_generation = generation
            self.reserved = proposed
            self.report(f"accepted: session={self.session_id} revision={revision} reset_generation={generation} cubes={self.accepted_count} queued={self.pending_count} completed={self.completed_count}")
            return changed
        except (ValueError, KeyError, TypeError, OverflowError, RecursionError) as error:
            self.halt(error)
            return -1

    def start_next(self):
        if self.halted or self.active is not None:
            return None
        if self.reset_pending:
            self.active = LiveCommand("reset", reset_generation=self.reset_generation)
        else:
            commands = self._commands_for(self.realized)
            if not commands:
                return None
            self.active = commands[0]
        self.report(f"started: {self.active.kind} {self.active.cube_id}")
        return self.active

    def _desired(self):
        return {placement.cube_id: placement for placement in self.layout.placements} if self.layout is not None else {}

    def _commands_for(self, realized):
        desired = self._desired()
        removed = [placement for cube_id, placement in realized.items() if cube_id not in desired]
        materialized = [placement for cube_id, placement in desired.items()
                        if (cube_id not in realized or
                            placement.color_name != realized[cube_id].color_name or
                            not placements_share_pose(placement, realized[cube_id]))]
        removed.sort(key=lambda placement: -placement.position[2])
        # New support cubes and moves share one bottom-up order. Sorting keeps
        # the producer's order for equal-height targets regardless of kind.
        materialized.sort(key=lambda placement: placement.position[2])
        return ([LiveCommand("delete", placement, self.reset_generation) for placement in removed]
                + [LiveCommand("move" if placement.cube_id in realized else "add",
                               placement, self.reset_generation)
                   for placement in materialized])

    @staticmethod
    def _apply_result(realized, command):
        if command.kind == "reset":
            realized.clear()
        elif command.kind == "delete":
            realized.pop(command.cube_id, None)
        else:
            realized[command.cube_id] = command.placement

    def complete_active(self):
        if self.active is None:
            raise RuntimeError("No active command to complete")
        command = self.active
        self._apply_result(self.realized, command)
        if command.kind == "reset":
            self._acknowledged_generation = command.reset_generation
            self.completed_count = 0
            self.completed_command_count = 0
        else:
            self.completed_command_count += 1
            if command.kind == "add":
                self.completed_count += 1
        self.report(f"completed: {command.kind} {command.cube_id}")
        self.active = None

    def robot_state(self):
        """A completed actual scene, available only between robot operations.

        The latest accepted revision identifies the desired map this result was
        checked against. Actual poses may still differ from it when an older
        transfer just finished; consumers must compare each cube's target.
        """
        if (self.halted or self.protocol_version != 2 or self.layout is None
                or self.active is not None or self.reset_pending):
            return None
        cubes = []
        for cube_id, placement in sorted(self.realized.items()):
            position = (VR_TO_GENESIS_AXES.T @
                        (np.asarray(placement.position) - self.layout.workspace_center)
                        / self.layout.scale)
            w, x, y, z = placement.quaternion
            rotation = np.array([
                [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
            ])
            transform = np.eye(4)
            transform[:3, :3] = (VR_TO_GENESIS_AXES.T @ rotation @
                                 VR_TO_GENESIS_AXES) * self.layout.source_cube_size
            transform[:3, 3] = position
            cubes.append({"id": cube_id, "color": placement.color_name,
                          "position": position.tolist(),
                          "object_to_table": transform.ravel().tolist()})
        return {"session_id": self.session_id, "revision": self.revision,
                "reset_generation": self._acknowledged_generation, "cubes": cubes}

    def cancel_active_for_reset(self):
        """Forget an interrupted operation only when a whole-pool reset follows."""
        if not self.reset_pending:
            raise RuntimeError("Cannot cancel an active command without a pending reset")
        if self.active is not None:
            self.report(f"cancelled: {self.active.kind} {self.active.cube_id} for reset")
        self.active = None

    def fail_active(self, reason):
        cube_id = self.active.cube_id if self.active is not None else "initialization"
        self.report(f"failed: {cube_id}: {reason}")
        self.active = None
        self.halt(reason)


class LiveStreamClient:
    """Nonblocking map receiver and completed-state sender on the same socket."""

    def __init__(self, queue, host="127.0.0.1", port=9036, retry_interval=0.5, report=print):
        if not 0 < port < 65536:
            raise ValueError("port must be in 1..65535")
        self.queue = queue
        self.address = (socket.gethostbyname(host), port)
        self.retry_interval = retry_interval
        self.report = report
        self.socket = None
        self.connecting = False
        self.next_attempt = 0.0
        self.connect_deadline = 0.0
        self.decoder = FrameDecoder()
        self._last_connection_error = None
        self._robot_sequence = 0
        self._robot_pending = b""
        self._robot_sent = 0
        self._robot_waiting = None
        self._last_robot_payload = None
        self._last_robot_publish = 0.0
        self._received_map_on_connection = False

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        self.connecting = False
        self.decoder = FrameDecoder()  # discard only this connection's partial frame
        self._robot_pending = b""
        self._robot_sent = 0
        self._robot_waiting = None
        self._last_robot_payload = None
        self._received_map_on_connection = False

    def _disconnected(self, reason):
        self.close()
        self.next_attempt = time.monotonic() + self.retry_interval
        if reason != self._last_connection_error:
            self.report(f"connection: {reason}; retrying {self.address[0]}:{self.address[1]}")
            self._last_connection_error = reason

    def pump(self):
        """Decode/enqueue available maps; never dequeue or run Genesis."""
        if self.queue.halted:
            return
        now = time.monotonic()
        try:
            if self.socket is None:
                if now < self.next_attempt:
                    return
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.setblocking(False)
                status = self.socket.connect_ex(self.address)
                if status not in (0, errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY, 10035, 10036, 10037):
                    raise OSError(status, "connect failed")
                self.connecting = True
                self.connect_deadline = now + 3.0
            if self.connecting:
                _, writable, exceptional = select.select([], [self.socket], [self.socket], 0)
                if exceptional or now > self.connect_deadline:
                    raise OSError("connect failed or timed out")
                if not writable:
                    return
                status = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                if status:
                    raise OSError(status, "connect failed")
                self.connecting = False
                self._last_connection_error = None
                self.report(f"connection: connected {self.address[0]}:{self.address[1]}")
            for _ in range(16):  # at most 1 MiB per owner-thread pump
                try:
                    chunk = self.socket.recv(65536)
                except BlockingIOError:
                    return
                if not chunk:
                    self._disconnected("publisher disconnected")
                    return
                for header, data in self.decoder.feed(chunk):
                    self.queue.accept(header, data)
                    if self.queue.halted:
                        return
                    self._received_map_on_connection = True
        except ProtocolError as error:
            self.queue.halt(error)
            self.close()
        except OSError as error:
            self._disconnected(str(error))

    def publish_robot_state(self):
        """Send actual completed state from an outer robot-loop boundary only.

        Receiving/pumping never calls this: receive-only diagnostics and
        trajectory steps cannot acknowledge work. At most one frame in flight
        and the newest waiting state are retained, including partial writes.
        """
        state = self.queue.robot_state()
        if (state is None or self.socket is None or self.connecting
                or not self._received_map_on_connection):
            return
        now = time.monotonic()
        payload = json.dumps(state, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(payload) > MAX_FRAME_BYTES:
            self.queue.halt("RobotState exceeds 64 MiB limit")
            return
        if payload != self._last_robot_payload or now - self._last_robot_publish >= .5:
            self._robot_waiting = payload
            self._last_robot_payload = payload
            self._last_robot_publish = now
        try:
            for _ in range(16):
                if not self._robot_pending:
                    if self._robot_waiting is None:
                        return
                    body = self._robot_waiting
                    self._robot_waiting = None
                    self._robot_sequence += 1
                    message = {
                        "stream": "CubeStack/RobotState", "content_type": "application/json",
                        "sequence": self._robot_sequence, "payload_bytes": len(body),
                        "timestamp": now,
                        "metadata": json.dumps({"protocol_version": 2,
                                                "session_id": self.queue.session_id}),
                    }
                    encoded = json.dumps(message, separators=(",", ":")).encode("utf-8")
                    self._robot_pending = (struct.pack("<I", len(encoded)) + encoded
                                           + struct.pack("<I", len(body)) + body)
                    self._robot_sent = 0
                try:
                    count = self.socket.send(memoryview(self._robot_pending)[self._robot_sent:])
                except BlockingIOError:
                    return
                if count == 0:
                    raise OSError("publisher disconnected while sending RobotState")
                self._robot_sent += count
                if self._robot_sent == len(self._robot_pending):
                    self._robot_pending = b""
                    self._robot_sent = 0
        except OSError as error:
            self._disconnected(str(error))
