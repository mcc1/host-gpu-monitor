"""One bounded NVIDIA sample shared by every listener and the metrics exporter."""

import asyncio
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Sample:
    sampled_at: int
    devices: list[dict]


def proc_identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[1]), fields[19]


def is_descendant(pid, ancestor):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        pid = proc_identity(pid)[0]
    return False


def workload_label(pid):
    try:
        parts = Path(f"/proc/{pid}/cgroup").read_text().split("/")
    except OSError:
        return None
    units = [part.strip() for part in parts if part.strip().endswith(".service")]
    return units[-1] if units else None


def mib(text):
    value, unit = (text or "").split()
    if unit != "MiB" or not value.isdecimal():
        raise ValueError("GPU memory unavailable")
    return int(value)


def optional_number(text, unit):
    try:
        value, suffix = (text or "").split()
        value = float(value)
    except ValueError:
        return None
    return value if suffix == unit and math.isfinite(value) else None


def parse_xml(xml):
    devices = []
    for gpu in ET.fromstring(xml).findall("gpu"):
        processes = []
        seen = set()
        for proc in gpu.findall("processes/process_info"):
            pid = int(proc.findtext("pid", "0"))
            if pid <= 0 or pid in seen:
                raise ValueError("GPU process attribution unavailable")
            seen.add(pid)
            processes.append({
                "pid": pid,
                "usedMb": mib(proc.findtext("used_memory")),
                "name": os.path.basename(proc.findtext("process_name", "unknown")),
                "workload": workload_label(pid),
            })
        total = mib(gpu.findtext("fb_memory_usage/total"))
        used = mib(gpu.findtext("fb_memory_usage/used"))
        free = mib(gpu.findtext("fb_memory_usage/free"))
        uuid = gpu.findtext("uuid")
        if not uuid or total <= 0 or max(used, free, sum(p["usedMb"] for p in processes)) > total:
            raise ValueError("inconsistent GPU snapshot")
        power = gpu.findtext("gpu_power_readings/power_draw") or gpu.findtext("power_readings/power_draw")
        if not power:
            power = gpu.findtext("gpu_power_readings/instant_power_draw")
        devices.append({
            "gpuUuid": uuid, "totalMb": total, "usedMb": used, "freeMb": free,
            "unattributedMb": max(0, used - sum(p["usedMb"] for p in processes)),
            "processes": processes,
            "temperatureC": optional_number(gpu.findtext("temperature/gpu_temp"), "C"),
            "utilizationPercent": optional_number(gpu.findtext("utilization/gpu_util"), "%"),
            "powerWatts": optional_number(power, "W"),
        })
    if not devices:
        raise ValueError("GPU unavailable")
    return Sample(time.time_ns() // 1_000_000, devices)


async def command(*args):
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), 2)
    except BaseException:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
        raise
    if proc.returncode:
        raise RuntimeError("probe command failed")
    return stdout


async def nvidia_xml():
    return await command("/usr/bin/nvidia-smi", "-q", "-x")


class Sampler:
    def __init__(self, read_xml=nvidia_xml):
        self.read_xml = read_xml
        self._lock = asyncio.Lock()
        self.latest = None
        self.completed_at = float("-inf")

    async def refresh(self):
        async with self._lock:
            if time.monotonic() - self.completed_at < 1:
                return self.latest
            try:
                self.latest = parse_xml(await self.read_xml())
            except (OSError, RuntimeError, TimeoutError, ValueError, ET.ParseError):
                self.latest = None
            self.completed_at = time.monotonic()
            return self.latest


def unix_snapshot(sample, pid, uid, descendant=is_descendant):
    devices = []
    for device in sample.devices:
        processes = [
            {**p, "owner": "self" if descendant(p["pid"], pid) else "external"}
            for p in device["processes"]
        ]
        self_mb = sum(p["usedMb"] for p in processes if p["owner"] == "self")
        other_mb = sum(p["usedMb"] for p in processes if p["owner"] == "external")
        devices.append({
            **{key: device[key] for key in ("gpuUuid", "totalMb", "usedMb", "freeMb", "unattributedMb")},
            "selfMb": self_mb, "othersMb": other_mb, "processes": processes,
            "callerHasContext": any(p["pid"] == pid for p in processes),
        })
    return {
        "schemaVersion": 1, "source": "host-processes", "status": "ok",
        "sampledAt": sample.sampled_at, "peerHostPid": pid, "peerHostUid": uid,
        "devices": devices,
    }


def vm_snapshot(sample, allowed_devices):
    devices = [
        {
            **{key: device[key] for key in ("gpuUuid", "totalMb", "usedMb", "freeMb", "unattributedMb")},
            "selfMb": None, "othersMb": None, "callerHasContext": None,
        }
        for device in sample.devices if device["gpuUuid"] in allowed_devices
    ]
    return {
        "schemaVersion": 2, "source": "host-processes", "scope": "host",
        "status": "ok" if devices else "unknown", "sampledAt": sample.sampled_at,
        **({"devices": devices} if devices else {"reason": "gpu_unavailable"}),
    }
