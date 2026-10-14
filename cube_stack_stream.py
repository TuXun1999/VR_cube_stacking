"""Bounded OpenUp stream decoding and append-only placement acceptance.

The client is pumped by its owner thread. It never executes robot commands.
"""
from collections import Counter, deque
import copy
import errno
import json
import math
import select
import socket
import struct
import time

from cube_stack_layout import parse_table_layout

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
    if type(metadata.get("protocol_version")) is not int or metadata["protocol_version"] != 1:
        raise ProtocolError("Unsupported protocol_version; expected 1")
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


class LiveMapQueue:
    """Reserve complete snapshots atomically, including queued/active cubes."""

    def __init__(self, pool_per_color=30, report=print):
        if type(pool_per_color) is not int or pool_per_color <= 0:
            raise ValueError("pool_per_color must be a positive integer")
        self.pool_per_color = pool_per_color
        self.report = report
        self.session_id = None
        self.layout = None
        self._accepted = None
        self._pending = deque()
        self.reserved = Counter()
        self.active = None
        self.completed_count = 0
        self.halted = False
        self.halt_reason = None

    @property
    def accepted_count(self):
        return len(self.layout.placements) if self.layout is not None else 0

    @property
    def pending_count(self):
        return len(self._pending)

    def halt(self, reason):
        if not self.halted:
            self.halted = True
            self.halt_reason = str(reason)
            self.report(f"rejected: HALTED: {reason}; restart the live simulation to begin a new session")

    def accept(self, header, data):
        """Return additions, 0 for a repeat, or -1 for a halted rejection."""
        if self.halted:
            return -1
        try:
            metadata = validate_header(header)
            layout = parse_table_layout(data)
            if self.session_id is not None and metadata["session_id"] != self.session_id:
                raise ProtocolError("table session changed")
            if self._accepted is not None:
                previous_metadata = {key: value for key, value in self._accepted.items() if key != "cubes"}
                current_metadata = {key: value for key, value in data.items() if key != "cubes"}
                if previous_metadata != current_metadata:
                    raise ProtocolError("table metadata changed")
                previous = self._accepted["cubes"]
                if len(data["cubes"]) < len(previous) or data["cubes"][:len(previous)] != previous:
                    raise ProtocolError("accepted cube prefix changed or shrank")
            additions = layout.placements[self.accepted_count:]
            if self._accepted is not None and not additions:
                return 0
            proposed = self.reserved + Counter(placement.color_name for placement in additions)
            for color, count in proposed.items():
                if count > self.pool_per_color:
                    raise ProtocolError(f"{color} pool exhausted: needs {count}, capacity {self.pool_per_color}")
            # No observable queue state changes until the whole snapshot passed.
            self.session_id = metadata["session_id"]
            self._accepted = copy.deepcopy(data)
            self.layout = layout
            self.reserved = proposed
            self._pending.extend(additions)
            for placement in additions:
                self.report(f"queued: {placement.cube_id} color={placement.color_name} position={placement.position} quaternion={placement.quaternion}")
            if additions or self.accepted_count == 0:
                self.report(f"accepted: session={self.session_id} cubes={self.accepted_count} queued={self.pending_count} completed={self.completed_count}")
            return len(additions)
        except (ValueError, KeyError, TypeError, OverflowError, RecursionError) as error:
            self.halt(error)
            return -1

    def start_next(self):
        if self.halted or self.active is not None or not self._pending:
            return None
        self.active = self._pending.popleft()
        self.report(f"started: {self.active.cube_id}")
        return self.active

    def complete_active(self):
        if self.active is None:
            raise RuntimeError("No active placement to complete")
        self.report(f"completed: {self.active.cube_id}")
        self.completed_count += 1
        self.active = None

    def fail_active(self, reason):
        cube_id = self.active.cube_id if self.active is not None else "initialization"
        self.report(f"failed: {cube_id}: {reason}")
        self.active = None
        self.halt(reason)


class LiveStreamClient:
    """Nonblocking TCP receiver with bounded pumps and persistent acceptance."""

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

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        self.connecting = False
        self.decoder = FrameDecoder()  # discard only this connection's partial frame

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
        except ProtocolError as error:
            self.queue.halt(error)
            self.close()
        except OSError as error:
            self._disconnected(str(error))
