"""Stand in for OpenUp Cube Stack: publish scripted maps and check robot state.

Run it where OpenUp would run, then point any receiver at it:

    python tools/fake_table_publisher.py --port 9036
    python robot_receiver.py --sim toyblocks --live --host <this machine> --port 9036

Each script step is one complete desired layout (a ``CubeStack/Map`` snapshot,
protocol 3). The publisher resends it every 0.5 s, like OpenUp, until a
``CubeStack/RobotState`` frame for the same revision and reset generation reports
exactly those cubes at their target ``object_to_table`` matrices (OpenUp's own
1e-5 check), then moves to the next step. It exits 0 when every step is
confirmed and 1 on a timeout or a disconnect it cannot recover from.

Script format (JSON), positions in OpenUp table coordinates (metres):

    {"table": {"width": 1.21, "depth": 0.680625}, "cube_size": 0.08,
     "steps": [{"name": "add", "pieces": [
                  {"id": "r1", "color": "red", "x": 0.0, "z": -0.15,
                   "level": 0, "length": 1, "yaw_deg": 0}]},
               {"name": "reset", "reset": true, "pieces": []}]}

``x`` is to the right, ``z`` is down-screen (away from the robot rack), ``level``
counts cubes below, ``length`` 2 or 3 makes a glued bar along its local x axis,
and ``yaw_deg`` turns the piece about the table's up axis.
"""
import argparse
import json
import math
from pathlib import Path
import select
import socket
import struct
import sys
import time
import uuid

import numpy as np

DEFAULT_SCRIPT = Path(__file__).resolve().parent / "scripts" / "basic_edits.json"
RESEND_SECONDS = 0.5
MATRIX_TOLERANCE = 1e-5  # OpenUp's SameTransform tolerance (model.cc)


def piece_cubes(piece, cube_size):
    """Expand one script piece into OpenUp version-3 cube entries."""
    length = int(piece.get("length", 1))
    if not 1 <= length <= 3:
        raise ValueError(f"piece {piece['id']!r} length must be 1..3")
    yaw = math.radians(float(piece.get("yaw_deg", 0.0)))
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    center = np.array([float(piece["x"]), cube_size * (int(piece.get("level", 0)) + 0.5), float(piece["z"])])
    cubes = []
    for index in range(length):
        position = center + rotation @ np.array([cube_size * (index - (length - 1) / 2), 0.0, 0.0])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation * cube_size
        matrix[:3, 3] = position
        cubes.append({
            "id": piece["id"] if length == 1 else f"{piece['id']}_{index}",
            "piece_id": piece["id"], "piece_index": index, "piece_size": length,
            "color": piece["color"],
            "position": position.tolist(),
            "object_to_table": matrix.ravel().tolist(),
        })
    return cubes


def build_map(script, pieces, revision, generation):
    cube_size = float(script.get("cube_size", 0.08))
    return {
        "format": "openup.cube_stack.map", "version": 3,
        "coordinate_space": "table_local", "units": "meters", "matrix_layout": "row_major",
        "axes": {"origin": "table_surface_center", "x": "right", "y": "up", "z": "down_screen"},
        "cube_size": cube_size,
        "table": dict(script.get("table", {"width": 1.21, "depth": 0.680625})),
        "revision": revision, "reset_generation": generation,
        "cubes": [entry for piece in pieces for entry in piece_cubes(piece, cube_size)],
    }


def encode_frame(stream, payload, sequence, metadata):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    header = json.dumps({
        "stream": stream, "content_type": "application/json", "sequence": sequence,
        "payload_bytes": len(body), "timestamp": time.monotonic(), "metadata": json.dumps(metadata),
    }, separators=(",", ":")).encode("utf-8")
    return struct.pack("<I", len(header)) + header + struct.pack("<I", len(body)) + body


class FrameReader:
    """Decode uint32LE(header), header JSON, uint32LE(payload), payload JSON."""

    def __init__(self):
        self.buffer = bytearray()
        self.header = None

    def feed(self, chunk):
        self.buffer.extend(chunk)
        frames = []
        while len(self.buffer) >= 4:
            length = struct.unpack_from("<I", self.buffer)[0]
            if len(self.buffer) < 4 + length:
                break
            block = json.loads(bytes(self.buffer[4:4 + length]).decode("utf-8"))
            del self.buffer[:4 + length]
            if self.header is None:
                self.header = block
            else:
                frames.append((self.header, block))
                self.header = None
        return frames


def state_matches(state, expected_map, session_id):
    """Whether a RobotState confirms every expected cube and nothing else."""
    if (state.get("session_id") != session_id or state.get("revision") != expected_map["revision"]
            or state.get("reset_generation") != expected_map["reset_generation"]):
        return False
    actual = {cube.get("id"): cube for cube in state.get("cubes", [])}
    expected = {cube["id"]: cube for cube in expected_map["cubes"]}
    if actual.keys() != expected.keys():
        return False
    return all(actual[key].get("color") == cube["color"]
               and np.allclose(actual[key].get("object_to_table"), cube["object_to_table"],
                               atol=MATRIX_TOLERANCE, rtol=0)
               for key, cube in expected.items())


class FakeTablePublisher:
    def __init__(self, script, listener, step_timeout=120.0, connect_timeout=600.0, report=print):
        self.script = script
        self.listener = listener
        self.step_timeout = step_timeout
        self.connect_timeout = connect_timeout
        self.report = report
        self.session_id = f"fake-table-{uuid.uuid4().hex[:12]}"
        self.connection = None
        self.reader = FrameReader()
        self.sequence = 0

    def _accept(self, deadline):
        self.listener.settimeout(max(0.01, deadline - time.monotonic()))
        try:
            self.connection, address = self.listener.accept()
        except socket.timeout:
            return False
        self.connection.setblocking(False)
        self.reader = FrameReader()
        self.report(f"connection: receiver connected from {address[0]}:{address[1]}")
        return True

    def _send(self, data):
        self.sequence += 1
        frame = encode_frame("CubeStack/Map", data, self.sequence,
                             {"protocol_version": 3, "session_id": self.session_id})
        self.connection.setblocking(True)
        try:
            self.connection.sendall(frame)
        finally:
            self.connection.setblocking(False)

    def _receive_states(self, wait):
        readable, _, _ = select.select([self.connection], [], [], wait)
        if not readable:
            return []
        chunk = self.connection.recv(65536)
        if not chunk:
            raise ConnectionError("receiver disconnected")
        return [payload for header, payload in self.reader.feed(chunk)
                if header.get("stream") == "CubeStack/RobotState"]

    def run(self, hold=False):
        if not self._accept(time.monotonic() + self.connect_timeout):
            self.report("failed: no receiver connected")
            return 1
        revision, generation = 0, 0
        steps = self.script["steps"]
        data = None
        for number, step in enumerate(steps, 1):
            if step.get("reset"):
                generation += 1
            revision += 1
            data = build_map(self.script, step.get("pieces", []), revision, generation)
            name = step.get("name", f"step {number}")
            started = time.monotonic()
            deadline = started + self.step_timeout
            next_send = 0.0
            self.report(f"step {number}/{len(steps)}: {name} (revision {revision}, generation {generation})")
            while True:
                now = time.monotonic()
                if now >= deadline:
                    self.report(f"failed: step {number} {name!r} not confirmed within {self.step_timeout:g} s")
                    return 1
                try:
                    if self.connection is None and not self._accept(deadline):
                        continue
                    if now >= next_send:
                        self._send(data)
                        next_send = now + RESEND_SECONDS
                    states = self._receive_states(0.05)
                except (ConnectionError, OSError) as error:
                    self.report(f"connection: {error}; waiting for the receiver to reconnect")
                    if self.connection is not None:
                        self.connection.close()
                    self.connection = None
                    continue
                if any(state_matches(state, data, self.session_id) for state in states):
                    self.report(f"confirmed: step {number} {name!r} in {time.monotonic() - started:.1f} s")
                    break
        self.report("complete: every step confirmed")
        while hold and data is not None:
            try:
                self._send(data)
                self._receive_states(RESEND_SECONDS)
            except (ConnectionError, OSError):
                break
        return 0

    def close(self):
        if self.connection is not None:
            self.connection.close()
        self.listener.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Publish scripted Cube Stack maps and check robot state.")
    parser.add_argument("--host", default="0.0.0.0", help="address to listen on (default: all interfaces)")
    parser.add_argument("--port", type=int, default=9036, help="simulation port (default: 9036)")
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT, help="JSON step script")
    parser.add_argument("--step-timeout", type=float, default=180.0, help="seconds allowed per step")
    parser.add_argument("--connect-timeout", type=float, default=600.0,
                        help="seconds to wait for the receiver to connect")
    parser.add_argument("--hold", action="store_true", help="keep publishing the last map after the script")
    args = parser.parse_args(argv)
    script = json.loads(args.script.read_text(encoding="utf-8"))
    listener = socket.create_server((args.host, args.port))
    print(f"listening: {args.host}:{args.port} script={args.script}", flush=True)
    publisher = FakeTablePublisher(script, listener, args.step_timeout, args.connect_timeout,
                                   report=lambda message: print(message, flush=True))
    try:
        return publisher.run(hold=args.hold)
    except KeyboardInterrupt:
        return 1
    finally:
        publisher.close()


if __name__ == "__main__":
    sys.exit(main())
