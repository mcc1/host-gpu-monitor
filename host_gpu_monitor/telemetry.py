"""Opt-in Prometheus exposition over a dedicated listener."""

import asyncio
from collections import Counter
import time


def label(value):
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def render_metrics(sample, workloads):
    lines = [
        "# TYPE host_gpu_sample_up gauge",
        f"host_gpu_sample_up {int(sample is not None)}",
    ]
    if sample is None:
        return "\n".join(lines) + "\n"

    lines.extend((
        "# TYPE host_gpu_sample_timestamp_seconds gauge",
        f"host_gpu_sample_timestamp_seconds {sample.sampled_at / 1000}",
    ))
    fields = (
        ("memory_total_bytes", "totalMb", 1048576),
        ("memory_used_bytes", "usedMb", 1048576),
        ("memory_free_bytes", "freeMb", 1048576),
        ("temperature_celsius", "temperatureC", 1),
        ("utilization_ratio", "utilizationPercent", 0.01),
        ("power_watts", "powerWatts", 1),
    )
    for metric, key, multiplier in fields:
        lines.append(f"# TYPE host_gpu_{metric} gauge")
        for device in sample.devices:
            value = device[key]
            if value is not None:
                lines.append(f"host_gpu_{metric}{{gpu_uuid={label(device['gpuUuid'])}}} {value * multiplier}")
    lines.append("# TYPE host_gpu_workload_memory_used_bytes gauge")
    for device in sample.devices:
        aggregate = Counter({name: 0 for name in (*workloads, "other")})
        for proc in device["processes"]:
            name = proc["workload"] if proc["workload"] in workloads else "other"
            aggregate[name] += proc["usedMb"] * 1048576
        for name, value in sorted(aggregate.items()):
            lines.append(
                f"host_gpu_workload_memory_used_bytes{{gpu_uuid={label(device['gpuUuid'])},workload={label(name)}}} {value}"
            )
    return "\n".join(lines) + "\n"


async def refresh_periodically(sampler, interval):
    while True:
        await sampler.refresh()
        await asyncio.sleep(interval)


def handler(sampler, interval, workloads):
    async def metrics(reader, writer):
        try:
            async with asyncio.timeout(1):
                headers = await reader.readuntil(b"\r\n\r\n")
                line = headers.split(b"\r\n", 1)[0]
                if line in (b"GET /metrics HTTP/1.0", b"GET /metrics HTTP/1.1"):
                    sample = sampler.latest if time.monotonic() - sampler.completed_at <= interval * 2 else None
                    body = render_metrics(sample, workloads).encode()
                    status = "200 OK"
                else:
                    body, status = b"Not found\n", "404 Not Found"
                writer.write(
                    f"HTTP/1.1 {status}\r\nContent-Type: text/plain; version=0.0.4; charset=utf-8\r\n"
                    f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body
                )
                await writer.drain()
        except (OSError, EOFError, asyncio.LimitOverrunError, TimeoutError):
            pass
        finally:
            writer.close()
    return metrics
