#!/usr/bin/env bash
set -euo pipefail
project=$(cd "$(dirname "$0")/.." && pwd)
scratch=$(mktemp -d /tmp/hgm-observability.XXXXXX)
chmod 0755 "$scratch"
suffix=${scratch##*.}
prom="hgm-prom-$suffix"
grafana="hgm-grafana-$suffix"
daemon=""
cleanup() {
    if [[ -n "$daemon" ]]; then
        kill "$daemon" 2>/dev/null || true
        wait "$daemon" 2>/dev/null || true
    fi
    docker rm -f "$grafana" "$prom" >/dev/null 2>&1 || true
}
trap cleanup EXIT
cat > "$scratch/prometheus.yml" <<'YAML'
global:
  scrape_interval: 2s
scrape_configs:
  - job_name: host-gpu-monitor-smoke
    static_configs:
      - targets: ["127.0.0.1:19464"]
YAML
cat > "$scratch/grafana.yml" <<'YAML'
apiVersion: 1
datasources:
  - name: Host GPU smoke
    uid: hgm-smoke
    type: prometheus
    access: proxy
    url: http://127.0.0.1:19090
YAML
cd "$project"
PYTHONPATH="$project" python3 tests/fixture_daemon.py serve --no-unix --telemetry --metrics-port 19464 --interval 1 > "$scratch/exporter.log" 2>&1 &
daemon=$!
for _ in $(seq 1 30); do
    if curl -fsS --max-time 1 http://127.0.0.1:19464/metrics | grep -q '^host_gpu_sample_up 1$'; then break; fi
    sleep .2
done
curl -fsS --max-time 2 http://127.0.0.1:19464/metrics | grep -q '^host_gpu_sample_up 1$'
docker run -d --rm --name "$prom" --network host --cpus 1 --memory 256m \
    --mount "type=bind,src=$scratch/prometheus.yml,dst=/etc/prometheus/prometheus.yml,readonly" \
    "${PROMETHEUS_IMAGE:-prom/prometheus:v3.5.0}" \
    --config.file=/etc/prometheus/prometheus.yml --web.listen-address=127.0.0.1:19090 --storage.tsdb.retention.time=1h >/dev/null
docker run -d --rm --name "$grafana" --network host --cpus 1 --memory 512m \
    -e GF_SERVER_HTTP_ADDR=127.0.0.1 -e GF_SERVER_HTTP_PORT=13398 \
    -e GF_AUTH_ANONYMOUS_ENABLED=true -e GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
    -e GF_ANALYTICS_REPORTING_ENABLED=false -e GF_ANALYTICS_CHECK_FOR_UPDATES=false \
    -e GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES=false -e GF_LOG_LEVEL=warn \
    --mount "type=bind,src=$scratch/grafana.yml,dst=/etc/grafana/provisioning/datasources/hgm.yml,readonly" \
    "${GRAFANA_IMAGE:-grafana/grafana:12.1.1}" >/dev/null
for _ in $(seq 1 40); do
    if curl -fsS --max-time 1 http://127.0.0.1:13398/api/health >/dev/null 2>&1; then break; fi
    sleep 1
done
python3 - <<'PY'
import json
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

expected = 5273 * 1048576
expr = "host_gpu_memory_used_bytes"
for i in range(10):
    with urlopen("http://127.0.0.1:19090/api/v1/query?" + urlencode({"query": expr}), timeout=4) as r:
        data = json.load(r)["data"]["result"]
    if data:
        break
    time.sleep(1)
assert len(data) == 1 and float(data[0]["value"][1]) == expected, data
target = data[0]["metric"]["job"]
assert target == "host-gpu-monitor-smoke", target
payload = {
    "from": "now-5m", "to": "now",
    "queries": [{
        "refId": "A", "datasource": {"type": "prometheus", "uid": "hgm-smoke"},
        "expr": expr, "instant": True, "range": False, "intervalMs": 1000,
    }],
}
req = Request("http://127.0.0.1:13398/api/ds/query", json.dumps(payload).encode(), {
    "Content-Type": "application/json",
})
with urlopen(req, timeout=10) as r:
    frames = json.load(r)["results"]["A"]["frames"]
values = frames[0]["data"]["values"][-1]
assert values and values[0] == expected, frames
print(json.dumps({"environment": "isolated-local-containers", "source": "synthetic-nvidia-xml",
                  "prometheus": target, "grafana": "datasource-query-ok", "gpuMemoryUsedBytes": expected}))
PY
