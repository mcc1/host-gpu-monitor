# host-gpu-monitor

Read-only GPU snapshots from the Linux system that owns the NVIDIA driver.
A single bounded sampler serves local clients and optional Prometheus metrics.
Python 3.11+ and NVIDIA's `nvidia-smi` are required on the sampler host.

## Run from source

```sh
python3 -m host_gpu_monitor serve --socket-dir "${XDG_RUNTIME_DIR:-/tmp}/host-gpu-monitor-$(id -u)"
python3 -m host_gpu_monitor snapshot --socket "${XDG_RUNTIME_DIR:-/tmp}/host-gpu-monitor-$(id -u)/gpu.sock" --pretty
```

Run the two commands in separate shells. The server defaults to its own effective UID.
Select a dedicated directory owned by the daemon, with no group or world write
access. The daemon can create the final directory under an existing trusted parent.
An installed copy also provides the `host-gpu-monitor` command.

## Unix snapshot v1

Send one newline-terminated UTF-8 JSON line, read one line, then close.

```json
{"schemaVersion":1,"op":"gpu.snapshot"}
```

A successful v1 response includes `source`
(`host-processes`), `status` (`ok`), `sampledAt` (Unix ms), `peerHostPid`,
`peerHostUid` and `devices`. Each device has GPU UUID, memory in MiB, process
rows and caller-relative `selfMb`, `othersMb` and `callerHasContext`. Each
process row has `peerNsPid`: its PID as numbered in the caller's PID namespace
when it lives in that same namespace (for example another process in the same
LXC), else `null`. Processes in nested namespaces, such as Docker inside an
LXC, stay `null`. A client can use it to match host rows to its own `/proc`
without guessing. The value is read when the response is built, so a PID that
exits and is reused within the one-second sample cache can be misattributed.
A caller
with no GPU context can still read the physical inventory. Unavailable data
returns `status:"unknown"` with a reason. Requests and responses are bounded.

For Linux Docker, bind mount the dedicated socket directory read-only. Match
the allowed UID as seen by the host. A user namespace can remap the container
UID. Mount the directory so the client can reconnect when the socket inode is
replaced. A client only needs the Unix socket, even with no Docker networking.

For PVE LXC, enable `--lxc-ctid <CTID> --guest-uid <UID>` on the PVE host. This
adapter requires the privileges to inspect and access the selected container
runtime. Its guest socket remains `/run/host-gpu-monitor/gpu.sock`. It checks the
peer UID and PID namespace. Run `--no-unix` if only LXC sockets are wanted.
Deploying the LXC adapter should first be tested on a disposable container.

For a workload that serializes GPU access with an exclusive Linux `flock`,
configure `--lxc-execution-lock CTID=relative/path/to/execution.lock` for that
LXC. The Unix response may then include `executionLease` with `lock` (device
number and inode) and `ownerHostPid` when the same authorized lock holder was
confirmed around the GPU sample and remains the owner at response time. This
does not change `selfMb`, `othersMb`, or the process list. A scheduler should
only use the optional proof if its execution target also uses that exact lock.

## KVM/QEMU guest VM snapshot v2

Enable a host-side `AF_VSOCK` listener with an explicit CID-to-GPU allowlist.
Configure the matching vsock device on an isolated VM before launching its client.

```sh
# Host, with a GPU UUID from the host NVIDIA driver
python3 -m host_gpu_monitor serve --no-unix --vsock-port 19465 --vsock-allow '15=GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'
# VM, whose configured CID is 15. Host CID is 2.
python3 -m host_gpu_monitor snapshot --vsock-port 19465 --cid 2 --pretty
```

The example UUID is synthetic. Use the real UUID of the visible GPU.
The CID policy applies within the trusted local hypervisor. Keep the CID
allocation current when VMs are removed or rebuilt. This protocol uses schema
version 2 and a host-scoped device summary with null caller attribution.
A VM's host CID connection cannot identify a process inside the guest.
No raw host process list is sent on vsock. A PCI-passthrough GPU is sampled
inside the guest that owns its NVIDIA driver. Treat guest and host data as
separate sources. See the verification record for the live test scope.

## Optional Prometheus telemetry

Enable with `--telemetry`. The default listener is `127.0.0.1:9464` and the
background sample interval is 15 seconds. The socket still refreshes on demand
and concurrent refreshes share a one-second cache. Scrapes read completed
snapshots and do not trigger sampling. The exporter is disabled by default.
If the metrics port cannot be opened, the daemon logs the error and keeps its
configured socket service available.

The exporter reports memory gauges in bytes, optional supported temperature,
utilization and power, a success gauge, and the source timestamp. Use
`--workload example.service` to allow stable workload labels. Other processes
are grouped under `other`. Missing or stale samples emit a failure gauge and
omit device metrics. Per-process rows remain available over the Unix socket.

See [Prometheus example](examples/prometheus.yml) and
[Grafana datasource example](examples/grafana-datasource.yml). For a remote
collector, use a trusted tunnel or explicitly configure a restricted metrics
listener. It exposes metrics without application credentials.

## Tests

```sh
python3 -m unittest discover -s tests -v
bash tests/smoke-docker.sh
bash tests/smoke-observability.sh
```

The local smoke tests use temporary containers and a synthetic NVIDIA sample.
Observability binds loopback ports 19464, 19090 and 13398. Run it on an isolated
test host with those ports free. Details and actual acceptance status are in
[VERIFICATION.md](VERIFICATION.md).
