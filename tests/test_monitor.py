import asyncio
import fcntl
import json
import os
from pathlib import Path
import socket
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

from host_gpu_monitor.__main__ import execution_locks, policies
from host_gpu_monitor.daemon import local_listener, request_handler
from host_gpu_monitor import lxc
from host_gpu_monitor.sampler import (
    DEVICE_TELEMETRY, Sampler, parse_xml, peer_namespace_pid, unix_snapshot, vm_snapshot,
)
from host_gpu_monitor.telemetry import handler, render_metrics


XML = Path(__file__).with_name("fixture.xml").read_bytes()
UUID = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


class SnapshotTests(unittest.TestCase):
    def test_unix_v1_preserves_process_attribution(self):
        xml = XML.replace(b"</gpu>", b"""<processes>
          <process_info><pid>200</pid><process_name>/example/renderer</process_name><used_memory>5010 MiB</used_memory></process_info>
          <process_info><pid>201</pid><process_name>/example/embedding</process_name><used_memory>204 MiB</used_memory></process_info>
          </processes></gpu>""")
        sample = parse_xml(xml)
        response = unix_snapshot(sample, 200, 1000, lambda p, caller: p == caller)
        device = response["devices"][0]
        self.assertEqual(response["schemaVersion"], 1)
        self.assertEqual(response["source"], "host-processes")
        self.assertEqual(response["sampledAt"], sample.sampled_at)
        self.assertEqual((device["selfMb"], device["othersMb"], device["unattributedMb"]), (5010, 204, 59))
        self.assertTrue(device["callerHasContext"])
        self.assertEqual((device["temperatureC"], device["utilizationPercent"], device["powerWatts"]),
                         (52.0, 14.0, 32.5))
        self.assertIsNone(device["smClockMhz"])  # not in the fixture: unknown, not 0
        self.assertEqual([p["owner"] for p in device["processes"]], ["self", "external"])

    def test_unix_v1_reports_peer_namespace_pid_only_when_resolved(self):
        xml = XML.replace(b"</gpu>", b"""<processes>
          <process_info><pid>200</pid><process_name>/example/renderer</process_name><used_memory>5010 MiB</used_memory></process_info>
          <process_info><pid>201</pid><process_name>/example/embedding</process_name><used_memory>204 MiB</used_memory></process_info>
          </processes></gpu>""")
        local = {200: 7}
        response = unix_snapshot(parse_xml(xml), 200, 1000, lambda p, caller: p == caller,
                                 lambda p, caller: local.get(p))
        self.assertEqual([p["peerNsPid"] for p in response["devices"][0]["processes"]], [7, None])

    def test_unix_v1_reports_device_readings_and_keeps_unavailable_null(self):
        xml = XML.replace(b"<nvidia_smi_log>", b"""<nvidia_smi_log>
          <driver_version>595.58.03</driver_version><cuda_version>13.2</cuda_version>""").replace(
            b"</gpu>", b"""<product_name>NVIDIA GeForce RTX 3060</product_name>
          <pci><pci_bus_id>00000000:0B:00.0</pci_bus_id></pci>
          <fan_speed>N/A</fan_speed><performance_state>P8</performance_state>
          <clocks><sm_clock>210 MHz</sm_clock><mem_clock>405 MHz</mem_clock></clocks>
          <max_clocks><sm_clock>2130 MHz</sm_clock></max_clocks>
          </gpu>""").replace(b"</gpu_power_readings>",
                             b"<current_power_limit>170.00 W</current_power_limit></gpu_power_readings>")
        device = unix_snapshot(parse_xml(xml), 200, 1000, lambda p, caller: False,
                               lambda p, caller: None)["devices"][0]
        self.assertEqual({key: device[key] for key in DEVICE_TELEMETRY}, {
            "name": "NVIDIA GeForce RTX 3060", "pciBusId": "00000000:0B:00.0",
            "driverVersion": "595.58.03", "cudaVersion": "13.2", "temperatureC": 52.0,
            "utilizationPercent": 14.0, "memoryUtilizationPercent": None, "powerWatts": 32.5,
            "powerLimitWatts": 170.0, "fanPercent": None, "performanceState": 8,
            "smClockMhz": 210.0, "smClockMaxMhz": 2130.0, "memClockMhz": 405.0,
        })

    def test_vm_v2_does_not_gain_device_readings(self):
        reply = vm_snapshot(parse_xml(XML), {UUID})
        self.assertNotIn("temperatureC", reply["devices"][0])

    def test_peer_namespace_pid_reads_own_namespace_and_rejects_missing(self):
        me = os.getpid()
        self.assertEqual(peer_namespace_pid(me, me), me)
        self.assertIsNone(peer_namespace_pid(2**22 + 12345, me))

    def test_execution_owner_requires_the_exact_exclusive_flock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "execution.lock"
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            relative = path.relative_to("/")
            try:
                fcntl.flock(fd, fcntl.LOCK_SH)
                self.assertIsNone(lxc.execution_owner(os.getpid(), os.getuid(), relative))
                fcntl.flock(fd, fcntl.LOCK_EX)
                dev, ino, pid, _ = lxc.execution_owner(os.getpid(), os.getuid(), relative)
                info = os.fstat(fd)
                self.assertEqual((dev, ino, pid), (info.st_dev, info.st_ino, os.getpid()))
                alias = path.with_name("alias")
                alias.symlink_to(path)
                self.assertIsNone(lxc.execution_owner(os.getpid(), os.getuid(), alias.relative_to("/")))
                self.assertIsNone(lxc.execution_owner(os.getpid(), os.getuid() + 1, relative))
                fcntl.flock(fd, fcntl.LOCK_UN)
                self.assertIsNone(lxc.execution_owner(os.getpid(), os.getuid(), relative))
            finally:
                os.close(fd)


class ExecutionLeaseSampleTests(unittest.IsolatedAsyncioTestCase):
    async def test_ownership_must_bracket_the_sample_and_still_match_on_cache_read(self):
        original = (45, 100, 40, "original")
        owner = original
        during_read = original

        def source():
            return owner

        async def read_xml():
            nonlocal owner
            owner = during_read
            return XML

        sampler = Sampler(read_xml)
        sampler.lease_sources.add(source)
        sample = await sampler.refresh()
        self.assertEqual(sampler.execution_lease(sample, source, 10), {"lock": [45, 100], "ownerHostPid": 40})
        self.assertIsNone(sampler.execution_lease(sample, source, 40))
        owner = (45, 100, 40, "reused-pid")
        self.assertIs(await sampler.refresh(), sample)
        self.assertIsNone(sampler.execution_lease(sample, source, 10))
        sampler.completed_at = float("-inf")
        during_read = (45, 100, 50, "different-owner")
        sample = await sampler.refresh()
        self.assertIsNone(sampler.execution_lease(sample, source, 10))

    async def test_unix_response_only_attests_a_live_lock_owner(self):
        async def read_xml():
            return XML

        sampler = Sampler(read_xml)
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "execution.lock"
            owner = await asyncio.create_subprocess_exec(
                sys.executable, "-c",
                "import fcntl, os, sys; fd = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o600); "
                "fcntl.flock(fd, fcntl.LOCK_EX); print('locked', flush=True); sys.stdin.read()",
                str(lock), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            )
            try:
                self.assertEqual(await asyncio.wait_for(owner.stdout.readline(), 2), b"locked\n")

                def source():
                    return lxc.execution_owner(os.getpid(), os.getuid(), lock.relative_to("/"))

                sampler.lease_sources.add(source)
                path = Path(directory) / "monitor.sock"
                handler = request_handler(sampler, os.getuid(), os.getpid(), lease_source=source)
                async with await asyncio.start_unix_server(handler, path):
                    async def query():
                        reader, writer = await asyncio.open_unix_connection(path)
                        try:
                            writer.write(b'{"schemaVersion":1,"op":"gpu.snapshot"}\n')
                            await writer.drain()
                            return json.loads(await asyncio.wait_for(reader.readline(), 2))
                        finally:
                            writer.close()
                            await writer.wait_closed()

                    info = lock.stat()
                    self.assertEqual((await query())["executionLease"],
                                     {"lock": [info.st_dev, info.st_ino], "ownerHostPid": owner.pid})
                    owner.stdin.close()
                    self.assertEqual(await asyncio.wait_for(owner.wait(), 2), 0)
                    self.assertNotIn("executionLease", await query())
            finally:
                if owner.returncode is None:
                    owner.kill()
                    await owner.wait()

    def test_execution_lock_paths_are_explicitly_scoped_to_configured_containers(self):
        self.assertEqual(execution_locks(["101=home/user/lock"], [101]), {101: Path("home/user/lock")})
        for item in ("102=home/user/lock", "101=/etc/passwd", "101=home/../lock", "101="):
            with self.assertRaises(ValueError):
                execution_locks([item], [101])

    def test_metrics_report_valid_gauges_and_preserve_unknown(self):
        metrics = render_metrics(parse_xml(XML), {"renderer.service"})
        self.assertIn(f'host_gpu_memory_total_bytes{{gpu_uuid="{UUID}"}} 12884901888', metrics)
        self.assertIn(f'host_gpu_utilization_ratio{{gpu_uuid="{UUID}"}} 0.14', metrics)
        self.assertIn(f'host_gpu_power_watts{{gpu_uuid="{UUID}"}} 32.5', metrics)
        self.assertNotIn("memory_total", render_metrics(None, set()))
        self.assertIn("host_gpu_sample_up 0", render_metrics(None, set()))

    def test_vsock_policy_requires_device_uuids_and_unambiguous_cids(self):
        self.assertEqual(policies([f"15={UUID}"]), {15: {UUID}})
        for invalid in (["2=" + UUID], ["15=*"], [f"15={UUID}", f"15={UUID}"]):
            with self.assertRaises(ValueError):
                policies(invalid)


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_unix_directory_mode_survives_restrictive_umask(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "socket"
            previous = os.umask(0o077)
            try:
                async with local_listener(path, os.geteuid(), Sampler()):
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o755)
                    self.assertEqual(stat.S_IMODE((path / "gpu.sock").stat().st_mode), 0o600)
            finally:
                os.umask(previous)

    async def test_unix_rejects_symlink_ancestor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target").mkdir()
            (root / "alias").symlink_to(root / "target", target_is_directory=True)
            with self.assertRaises(OSError):
                async with local_listener(root / "alias" / "socket", os.geteuid(), Sampler()):
                    self.fail("daemon accepted a symlink ancestor")
            self.assertFalse((root / "target" / "socket").exists())

    async def test_metrics_port_conflict_leaves_unix_available(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            fixture = Path(__file__).with_name("fixture_daemon.py").resolve()
            env = {**os.environ, "PYTHONPATH": str(fixture.parents[1])}
            daemon = await asyncio.create_subprocess_exec(
                sys.executable, str(fixture), "serve", "--socket-dir", directory,
                "--telemetry", "--metrics-port", str(occupied.getsockname()[1]),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env,
            )
            try:
                ready = await asyncio.wait_for(daemon.stdout.readline(), 3)
                self.assertIn(b"ready", ready)
                error = await asyncio.wait_for(daemon.stderr.readline(), 3)
                self.assertIn(b"Telemetry listener unavailable", error)
                reader, writer = await asyncio.open_unix_connection(Path(directory) / "gpu.sock")
                try:
                    writer.write(b'{"schemaVersion":1,"op":"gpu.snapshot"}\n')
                    await writer.drain()
                    reply = json.loads(await asyncio.wait_for(reader.readline(), 2))
                    self.assertEqual(reply["status"], "ok")
                finally:
                    writer.close()
                    await writer.wait_closed()
            finally:
                if daemon.returncode is None:
                    daemon.terminate()
                await asyncio.wait_for(daemon.communicate(), 3)

    async def test_unix_snapshot_singleflight_and_restart(self):
        calls = 0

        async def read_xml():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            return XML

        async def query(path):
            reader, writer = await asyncio.open_unix_connection(path)
            writer.write(b'{"schemaVersion":1,"op":"gpu.snapshot"}\n')
            await writer.drain()
            try:
                return json.loads(await asyncio.wait_for(reader.readline(), 2))
            finally:
                writer.close()
                await writer.wait_closed()

        sampler = Sampler(read_xml)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.sock"
            for _ in range(2):
                async with local_listener(directory, os.geteuid(), sampler):
                    replies = await asyncio.gather(*(query(path) for _ in range(10)))
                    self.assertTrue(all(r["status"] == "ok" and r["peerHostPid"] == os.getpid() for r in replies))
                    self.assertEqual(len({r["sampledAt"] for r in replies}), 1)
                self.assertFalse(path.exists())
            self.assertEqual(calls, 1)

    async def test_metrics_http_reads_cache_without_starting_sampling(self):
        async def read_xml():
            self.fail("HTTP reader must not sample the GPU")

        sampler = Sampler(read_xml)
        async with await asyncio.start_server(handler(sampler, 15, set()), "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n")
            try:
                result = await asyncio.wait_for(reader.read(), 2)
                self.assertIn(b"200 OK", result)
                self.assertIn(b"host_gpu_sample_up 0", result)
            finally:
                writer.close()
                await writer.wait_closed()

    async def test_vm_uses_peer_cid_and_hides_process_details(self):
        async def read_xml():
            return XML

        sampler = Sampler(read_xml)
        dispatch = request_handler(sampler, vm_policy={15: {UUID}})

        class Writer:
            def __init__(self, cid):
                self.cid = cid
                self.payload = b""

            def get_extra_info(self, name):
                return self

            def getpeername(self):
                return self.cid, 50000

            def write(self, payload):
                self.payload += payload

            async def drain(self):
                pass

            def close(self):
                pass

        async def query(cid, version):
            reader = asyncio.StreamReader()
            reader.feed_data(json.dumps({"schemaVersion": version, "op": "gpu.snapshot"}).encode() + b"\n")
            writer = Writer(cid)
            await dispatch(reader, writer)
            return json.loads(writer.payload)

        result = await query(15, 2)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["scope"], "host")
        self.assertIsNone(result["devices"][0]["selfMb"])
        self.assertNotIn("processes", result["devices"][0])
        self.assertEqual((await query(16, 2))["reason"], "unauthorized_peer")
        self.assertEqual((await query(15, 1))["reason"], "unsupported_request")

    async def test_failed_probe_is_coalesced_and_unknown(self):
        calls = 0

        async def read_xml():
            nonlocal calls
            calls += 1
            raise TimeoutError

        sampler = Sampler(read_xml)
        self.assertEqual(await asyncio.gather(sampler.refresh(), sampler.refresh()), [None, None])
        self.assertEqual(calls, 1)
