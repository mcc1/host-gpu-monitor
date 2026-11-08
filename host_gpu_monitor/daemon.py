"""Read-only socket daemon with bounded request and sampling work."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import stat
import struct
import sys
import time

from . import lxc, telemetry
from .sampler import Sampler, command, proc_identity, unix_snapshot, vm_snapshot

MAX_LINE = 1024
MAX_REPLY = 65536
MAX_CLIENTS = 32


def unknown(version, reason):
    return {
        "schemaVersion": version, "source": "host-processes", "status": "unknown",
        "sampledAt": time.time_ns() // 1_000_000, "reason": reason,
    }


def limit_clients(handler, limit=MAX_CLIENTS):
    active = 0

    async def handle(reader, writer):
        nonlocal active
        if active >= limit:
            writer.close()
            return
        active += 1
        try:
            await handler(reader, writer)
        finally:
            active -= 1
            writer.close()
    return handle


def request_handler(sampler, allowed_uid=None, init_pid=None, vm_policy=None):
    version = 1 if vm_policy is None else 2

    async def handle(reader, writer):
        try:
            async with asyncio.timeout(3):
                response = await respond(reader, writer)
                payload = json.dumps(response, separators=(",", ":")).encode() + b"\n"
                if len(payload) > MAX_REPLY:
                    payload = json.dumps(unknown(version, "response_too_large")).encode() + b"\n"
                writer.write(payload)
                await writer.drain()
        except (OSError, TimeoutError):
            pass
        finally:
            writer.close()

    async def respond(reader, writer):
        try:
            peer = writer.get_extra_info("socket")
            if vm_policy is None:
                pid, uid, _ = struct.unpack("3i", peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                if not authorized(pid, uid):
                    return unknown(version, "unauthorized_peer")
                identity = proc_identity(pid)
            else:
                cid, _ = peer.getpeername()
                if cid not in vm_policy:
                    return unknown(version, "unauthorized_peer")
            async with asyncio.timeout(1):
                request = json.loads(await reader.readline())
            if request != {"schemaVersion": version, "op": "gpu.snapshot"}:
                return unknown(version, "unsupported_request")
            sample = await sampler.refresh()
            if sample is None:
                return unknown(version, "probe_unavailable")
            if vm_policy is not None:
                return vm_snapshot(sample, vm_policy[cid])
            if identity != proc_identity(pid) or not authorized(pid, uid):
                return unknown(version, "peer_changed")
            return unix_snapshot(sample, pid, uid)
        except (OSError, ValueError, TimeoutError, IndexError):
            return unknown(version, "probe_unavailable")

    def authorized(pid, uid):
        return pid > 0 and uid == allowed_uid and (
            init_pid is None
            or os.readlink(f"/proc/{pid}/ns/pid") == os.readlink(f"/proc/{init_pid}/ns/pid")
        )

    return limit_clients(handle)


@contextmanager
def open_socket_directory(path):
    parts = Path(os.path.abspath(path)).parts[1:]
    if not parts:
        raise ValueError("socket directory must be dedicated to this daemon")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    uid = os.geteuid()
    directory = os.open("/", flags)
    try:
        for index, part in enumerate(parts):
            final = index == len(parts) - 1
            created = False
            if final:
                try:
                    os.mkdir(part, 0o755, dir_fd=directory)
                    created = True
                except FileExistsError:
                    pass
            child = os.open(part, flags, dir_fd=directory)
            os.close(directory)
            directory = child
            info = os.fstat(directory)
            if final:
                if info.st_uid != uid or info.st_mode & 0o022:
                    raise PermissionError("socket directory must be daemon-owned and not group/world writable")
                if created:
                    os.fchmod(directory, 0o755)
            else:
                # A root-owned sticky ancestor such as /tmp protects its entries.
                sticky_root = info.st_uid == 0 and info.st_mode & stat.S_ISVTX
                if info.st_uid not in (0, uid) or (info.st_mode & 0o022 and not sticky_root):
                    raise PermissionError("socket directory has an untrusted ancestor")
        yield directory
    finally:
        os.close(directory)


def socket_inode(directory):
    return os.stat("gpu.sock", dir_fd=directory, follow_symlinks=False)


def unlink_socket(directory, inode):
    try:
        current = socket_inode(directory)
    except FileNotFoundError:
        return
    if stat.S_ISSOCK(current.st_mode) and current.st_ino == inode:
        os.unlink("gpu.sock", dir_fd=directory)


async def start_unix_server(handler, sock):
    # The adapters own socket removal through an anchored directory FD.
    options = {"cleanup_socket": False} if sys.version_info >= (3, 13) else {}
    return await asyncio.start_unix_server(handler, sock=sock, limit=MAX_LINE, **options)


@asynccontextmanager
async def local_listener(path, uid, sampler):
    with open_socket_directory(path) as directory:
        anchored_path = f"/proc/self/fd/{directory}/gpu.sock"
        lock = os.open(".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
                       0o600, dir_fd=directory)
        inode = None
        try:
            if not stat.S_ISREG(os.fstat(lock).st_mode):
                raise PermissionError("socket lock must be a regular file")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                old = socket_inode(directory)
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISSOCK(old.st_mode) or old.st_uid != uid:
                    raise FileExistsError("refusing to replace an unrelated file")
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                    probe.settimeout(0.1)
                    try:
                        probe.connect(anchored_path)
                    except ConnectionRefusedError:
                        unlink_socket(directory, old.st_ino)
                    else:
                        raise FileExistsError("socket already has a listener")
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.bind(anchored_path)
                inode = socket_inode(directory).st_ino
                os.chmod(anchored_path, 0o600, follow_symlinks=False)
                if uid != os.geteuid():
                    os.chown("gpu.sock", uid, -1, dir_fd=directory, follow_symlinks=False)
                async with await start_unix_server(request_handler(sampler, uid), sock):
                    yield
        finally:
            try:
                if inode is not None:
                    unlink_socket(directory, inode)
            finally:
                os.close(lock)


@asynccontextmanager
async def vsock_listener(port, policy, sampler):
    with socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM) as sock:
        sock.bind((socket.VMADDR_CID_ANY, port))
        sock.setblocking(False)
        async with await asyncio.start_server(request_handler(sampler, vm_policy=policy), sock=sock, limit=MAX_LINE):
            yield


async def lxc_listener(ctid, guest_uid, sampler):
    lock = lxc.acquire_lock(ctid)
    try:
        while True:
            directory = None
            owns_socket = False
            try:
                init_pid = int((await command("/usr/bin/lxc-info", "-n", str(ctid), "-pH")).strip())
                identity = proc_identity(init_pid)
                lxc.validate_container(ctid, init_pid)
                uid = lxc.mapped_id(init_pid, guest_uid)
                root_uid = lxc.mapped_id(init_pid, 0)
                root_gid = lxc.mapped_id(init_pid, 0, "gid")
                directory = lxc.open_runtime(init_pid, root_uid, root_gid)
                lxc.remove_socket(directory)
                with lxc.bind_socket(directory, uid, root_uid, root_gid) as sock:
                    owns_socket = True
                    async with await start_unix_server(request_handler(sampler, uid, init_pid), sock):
                        while proc_identity(init_pid) == identity:
                            await asyncio.sleep(2)
            except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
                print(f"LXC {ctid}: {exc}", flush=True)
            finally:
                if directory is not None:
                    try:
                        if owns_socket:
                            lxc.remove_socket(directory)
                    finally:
                        os.close(directory)
            await asyncio.sleep(2)
    finally:
        os.close(lock)


async def serve(args, sampler=None):
    sampler = sampler if sampler is not None else Sampler()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    try:
        async with AsyncExitStack() as stack:
            if not args.no_unix:
                await stack.enter_async_context(local_listener(args.socket_dir, args.uid, sampler))
            if args.vsock_port is not None:
                await stack.enter_async_context(vsock_listener(args.vsock_port, args.vsock_allow, sampler))
            metrics_active = False
            if args.telemetry:
                try:
                    await stack.enter_async_context(await asyncio.start_server(
                        limit_clients(telemetry.handler(sampler, args.interval, set(args.workload))),
                        args.metrics_host, args.metrics_port, limit=4096,
                    ))
                except OSError as exc:
                    if args.no_unix and args.vsock_port is None and not args.lxc_ctid:
                        raise
                    print(f"Telemetry listener unavailable: {exc}", file=sys.stderr, flush=True)
                else:
                    metrics_active = True
            async with asyncio.TaskGroup() as tasks:
                background = [tasks.create_task(lxc_listener(ct, args.guest_uid, sampler)) for ct in args.lxc_ctid]
                if metrics_active:
                    background.append(tasks.create_task(telemetry.refresh_periodically(sampler, args.interval)))
                print("host-gpu-monitor ready", flush=True)
                await stop.wait()
                for task in background:
                    task.cancel()
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
