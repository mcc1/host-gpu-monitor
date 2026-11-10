import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import socket
import sys

from .daemon import MAX_REPLY, serve


DEFAULT_DIR = Path("/run/host-gpu-monitor")


def policies(items):
    result = {}
    for item in items:
        cid_text, separator, devices = item.partition("=")
        if not separator or not cid_text.isdecimal():
            raise ValueError("expected CID=GPU-UUID[,GPU-UUID]")
        cid = int(cid_text)
        uuids = set(devices.split(","))
        if cid < 3 or cid >= 2**32 - 1 or cid in result:
            raise ValueError("invalid or duplicate VM CID")
        if not all(re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", uuid) for uuid in uuids):
            raise ValueError("invalid GPU UUID")
        result[cid] = uuids
    return result


def execution_locks(items, allowed_ctids):
    result = {}
    for item in items:
        ctid, sep, value = item.partition("=")
        path = Path(value)
        if not sep or not ctid.isdecimal() or int(ctid) not in allowed_ctids or int(ctid) in result:
            raise ValueError("execution lock must specify one configured CTID")
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("execution lock path must be relative to the guest root")
        result[int(ctid)] = path
    return result


def read_snapshot(args):
    vsock = args.vsock_port is not None
    family = socket.AF_VSOCK if vsock else socket.AF_UNIX
    address = (args.cid, args.vsock_port) if vsock else str(args.socket)
    request = {"schemaVersion": 2 if vsock else 1, "op": "gpu.snapshot"}
    with socket.socket(family, socket.SOCK_STREAM) as conn:
        conn.settimeout(3)
        conn.connect(address)
        conn.sendall(json.dumps(request).encode() + b"\n")
        with conn.makefile("rb") as reader:
            line = reader.readline(MAX_REPLY + 1)
            if not line.endswith(b"\n") or len(line) > MAX_REPLY:
                raise ValueError("truncated or oversized response")
            response = json.loads(line)
    print(json.dumps(response, indent=2 if args.pretty else None))
    return 0 if response.get("status") == "ok" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="host-gpu-monitor")
    sub = parser.add_subparsers(dest="command", required=True)
    server = sub.add_parser("serve")
    server.add_argument("--socket-dir", type=Path, default=DEFAULT_DIR)
    server.add_argument("--no-unix", action="store_true")
    server.add_argument("--uid", type=int, default=os.geteuid())
    server.add_argument("--lxc-ctid", action="append", type=int, default=[])
    server.add_argument("--guest-uid", type=int, default=1000)
    server.add_argument("--lxc-execution-lock", action="append", default=[], metavar="CTID=PATH")
    server.add_argument("--vsock-port", type=int)
    server.add_argument("--vsock-allow", action="append", default=[])
    server.add_argument("--telemetry", action="store_true")
    server.add_argument("--metrics-host", default="127.0.0.1")
    server.add_argument("--metrics-port", type=int, default=9464)
    server.add_argument("--interval", type=int, default=15)
    server.add_argument("--workload", action="append", default=[])

    client = sub.add_parser("snapshot")
    client.add_argument("--socket", type=Path, default=DEFAULT_DIR / "gpu.sock")
    client.add_argument("--vsock-port", type=int)
    client.add_argument("--cid", type=int, default=2)
    client.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "snapshot":
            return read_snapshot(args)
        args.vsock_allow = policies(args.vsock_allow)
        if (args.vsock_port is None) != (not args.vsock_allow):
            parser.error("--vsock-port and at least one --vsock-allow are required together")
        if args.vsock_port is not None and not 1024 <= args.vsock_port < 2**32 - 1:
            parser.error("--vsock-port must be an unprivileged vsock port")
        if not 1 <= args.metrics_port <= 65535 or args.interval < 1:
            parser.error("invalid metrics port or sample interval")
        if not 0 <= args.uid < 2**32 - 1 or not 0 <= args.guest_uid < 2**32 - 1:
            parser.error("UID is out of range")
        if len(set(args.lxc_ctid)) != len(args.lxc_ctid) or any(ct <= 0 for ct in args.lxc_ctid):
            parser.error("LXC IDs must be distinct positive integers")
        args.lxc_execution_lock = execution_locks(args.lxc_execution_lock, args.lxc_ctid)
        if args.no_unix and not (args.lxc_ctid or args.vsock_port or args.telemetry):
            parser.error("at least one listener is required")
        asyncio.run(serve(args))
        return 0
    except (AttributeError, OSError, ValueError) as exc:
        print(f"host-gpu-monitor: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
