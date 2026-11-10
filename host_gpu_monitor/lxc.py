"""Optional Linux LXC runtime socket adapter."""

from contextlib import ExitStack, contextmanager
import ctypes
import fcntl
import os
from pathlib import Path
import socket
import stat

from .sampler import proc_identity

class FatalCredentialError(Exception):
    pass


def lock_stat(init_pid, relative_path):
    # The configured guest-relative path is anchored to the LXC root. Never
    # follow a guest-controlled symlink on the privileged side of the boundary.
    with ExitStack() as stack:
        def open_fd(name, flags, **kwargs):
            fd = os.open(name, flags, **kwargs)
            stack.callback(os.close, fd)
            return fd

        flags = os.O_PATH | os.O_DIRECTORY
        directory = open_fd(f"/proc/{init_pid}/root", flags)
        for part in relative_path.parts[:-1]:
            directory = open_fd(part, flags | os.O_NOFOLLOW, dir_fd=directory)
        return os.fstat(open_fd(relative_path.name, os.O_PATH | os.O_NOFOLLOW, dir_fd=directory))


def flock_owner(locks, device, inode):
    target = (os.major(device), os.minor(device), inode)
    owners = []
    for line in locks.splitlines():
        fields = line.split()
        if (len(fields) == 8 and fields[1:4] == ["FLOCK", "ADVISORY", "WRITE"]
                and fields[6:] == ["0", "EOF"]):
            major, minor, number = fields[5].split(":")
            if (int(major, 16), int(minor, 16), int(number)) == target:
                owners.append(int(fields[4]))
    return owners[0] if len(owners) == 1 and owners[0] > 0 else None


def execution_owner(init_pid, uid, relative_path):
    try:
        info = lock_stat(init_pid, relative_path)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != uid
                or info.st_nlink != 1 or info.st_mode & 0o077):
            return None
        pid = flock_owner(Path("/proc/locks").read_text(), info.st_dev, info.st_ino)
        if (pid is not None and Path(f"/proc/{pid}").stat().st_uid == uid
                and os.readlink(f"/proc/{pid}/ns/pid") == os.readlink(f"/proc/{init_pid}/ns/pid")):
            return info.st_dev, info.st_ino, pid, proc_identity(pid)[1]
    except (OSError, ValueError, IndexError):
        pass
    return None


def mapped_id(init_pid, uid, kind="uid"):
    for line in Path(f"/proc/{init_pid}/{kind}_map").read_text().splitlines():
        inside, outside, count = map(int, line.split())
        if inside <= uid < inside + count:
            return outside + uid - inside
    raise ValueError("guest UID is not mapped")


@contextmanager
def filesystem_ids(uid, gid):
    """Synchronous only: guest tmpfs requires mapped inode owners (EOVERFLOW).

    Keep ALL inode mutations at mapped guest-root privilege. Never await while
    these per-thread credentials differ. Host PID/user namespaces remain intact.
    """
    libc = ctypes.CDLL(None)
    old_gid = libc.setfsgid(gid)
    old_uid = libc.setfsuid(uid)
    try:
        if libc.setfsuid(-1) != uid or libc.setfsgid(-1) != gid:
            raise PermissionError("cannot set guest filesystem credentials")
        yield
    finally:
        libc.setfsuid(old_uid)
        libc.setfsgid(old_gid)
        if libc.setfsuid(-1) != old_uid or libc.setfsgid(-1) != old_gid:
            # Continuing with incorrect credentials would invalidate attribution.
            raise FatalCredentialError("cannot restore host filesystem credentials")


def open_runtime(init_pid, root_uid, root_gid, name="host-gpu-monitor"):
    # Anchor to the running container, reject symlinks at each guest-controlled
    # component. Never let guest filesystem content redirect privileged writes.
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root = os.open(f"/proc/{init_pid}/root", os.O_RDONLY | os.O_DIRECTORY)
    try:
        run = os.open("run", flags, dir_fd=root)
    finally:
        os.close(root)
    try:
        with filesystem_ids(root_uid, root_gid):
            try:
                os.mkdir(name, 0o755, dir_fd=run)
            except FileExistsError:
                pass
        directory = os.open(name, flags, dir_fd=run)
        info = os.fstat(directory)
        if info.st_uid != root_uid or info.st_mode & 0o022:
            os.close(directory)
            raise PermissionError("socket directory must be guest-root owned and not writable by guest user")
        try:
            with filesystem_ids(root_uid, root_gid):
                os.fchmod(directory, 0o755)
        except BaseException:
            os.close(directory)
            raise
        return directory
    finally:
        os.close(run)


def bind_socket(directory, uid, root_uid, root_gid):
    # Relative bind against an anchored directory avoids both the Unix path length
    # limit and /proc/self/fd traversal under temporary filesystem credentials.
    cwd = os.open(".", os.O_RDONLY | os.O_DIRECTORY)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        os.fchdir(directory)
        with filesystem_ids(root_uid, root_gid):
            sock.bind("gpu.sock")
            os.chmod("gpu.sock", 0o600, follow_symlinks=False)
        # CAP_CHOWN belongs to host root. lchown via the anchored directory never
        # follows a guest replacement symlink to a privileged host target.
        os.chown("gpu.sock", uid, -1, dir_fd=directory, follow_symlinks=False)
        return sock
    except BaseException:
        sock.close()
        raise
    finally:
        os.fchdir(cwd)
        os.close(cwd)


def remove_socket(directory):
    try:
        info = os.stat("gpu.sock", dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(info.st_mode):
        raise PermissionError("refusing to replace non-socket")
    os.unlink("gpu.sock", dir_fd=directory)  # only this service's runtime socket


def validate_container(ctid, init_pid):
    if os.readlink(f"/proc/{init_pid}/ns/pid") == os.readlink("/proc/self/ns/pid"):
        raise ValueError("LXC init must not be in host PID namespace")
    lines = Path(f"/proc/{init_pid}/cgroup").read_text().splitlines()
    if not any(line.startswith(f"0::/lxc/{ctid}/") for line in lines):
        raise ValueError("init PID does not belong to configured LXC")


def acquire_lock(ctid, runtime="/run"):
    # Prevent a second instance from unlinking a live monitor's socket.
    lock = os.open(f"{runtime}/host-gpu-monitor-{ctid}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock
    except BaseException:
        os.close(lock)
        raise
