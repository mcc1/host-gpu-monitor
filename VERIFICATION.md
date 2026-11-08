# Verification

## Verified on an isolated Linux test host

| Check | Evidence |
| --- | --- |
| Unix v1 client compatibility | An existing v1 client parsed the standalone response and reported 5010 MiB for the caller and 204 MiB for other processes. |
| Unit and socket tests | 10 passed. Unix credentials, shared sampling, restart, restrictive umask, symlink rejection, telemetry port conflict isolation, probe failure, cached metrics and VM CID policy exercised. |
| Docker | A real Linux container with networking disabled and a read-only socket-directory bind mount queried v1 successfully. Wrong UID was rejected. The same container reconnected after the daemon restarted. The sampler used synthetic XML. |
| Default telemetry | The Docker test ran with telemetry disabled and no HTTP listener. |
| Prometheus and Grafana | In isolated local containers, Prometheus v3.5.0 scraped the exporter and Grafana 12.1.1 queried that datasource. Both returned 5529141248 bytes from the synthetic 5273 MiB sample. Test containers were removed. |

## Verified on Linux with a physical NVIDIA GPU

| Check | Evidence |
| --- | --- |
| Linux guest VM | Two disposable QEMU/KVM guests booted without a disk or network. Over AF_VSOCK, the allowed CID received a host-scoped v2 snapshot with null caller attribution and no host process rows. The disallowed CID received `unauthorized_peer`. |
| Prometheus and Grafana | An existing Prometheus instance scraped an opt-in host exporter. The target and GPU sample were healthy, and the timestamp was current. Grafana queried the existing Prometheus datasource. Both matched the host's `nvidia-smi` memory reading. |
| Cleanup | The temporary guest processes, exporter and scrape target were removed. The original Prometheus configuration was restored and the monitoring services remained healthy. |

PCI passthrough was outside the live test scope.

## Live acceptance recipe

1. On the authorized host, run the standalone daemon in a dedicated runtime
   directory. Give the test VM its own CID and authorize only the test GPU UUID.
   From the guest, query v2 and verify scope, UUID and null caller attribution.
   Repeat with a different CID and verify rejection. Keep live GPU allocation
   and the existing application's socket outside the test.
2. Opt in to telemetry for the temporary host daemon. Check the real
   `host_gpu_sample_up` gauge and device UUID. Verify the NVIDIA memory values
   against a read-only host sample. Use a trusted monitoring path or tunnel.
3. On the existing monitoring stack, register a temporary scrape target with
   `examples/prometheus.yml` as the reference. Check `up`,
   `host_gpu_sample_timestamp_seconds` and `host_gpu_memory_used_bytes` in
   Prometheus, then query those series through Grafana's Prometheus datasource.
   Remove the temporary target and daemon after the test. Keep existing
   configuration and dashboards intact.
