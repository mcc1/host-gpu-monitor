#!/usr/bin/env bash
set -euo pipefail
project=$(cd "$(dirname "$0")/.." && pwd)
scratch=$(mktemp -d /tmp/hgm-docker.XXXXXX)
name="hgm-unix-${scratch##*.}"
daemon=""
uid=$(id -u)
gid=$(id -g)
other_uid=$(( uid + 1 ))
cleanup() {
    if [[ -n "$daemon" ]]; then
        kill "$daemon" 2>/dev/null || true
        wait "$daemon" 2>/dev/null || true
    fi
    docker rm -f "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT

start_daemon() {
    PYTHONPATH="$project" python3 "$project/tests/fixture_daemon.py" serve \
        --socket-dir "$scratch/socket" --metrics-port 19464 > "$scratch/daemon.log" 2>&1 &
    daemon=$!
    for _ in $(seq 1 40); do
        if [[ -S "$scratch/socket/gpu.sock" ]]; then break; fi
        sleep .1
    done
    test -S "$scratch/socket/gpu.sock"
}

start_daemon
if curl -fsS --max-time 1 http://127.0.0.1:19464/metrics >/dev/null 2>&1; then
    exit 1
fi
docker run -d --rm --name "$name" --network none --cap-drop ALL \
    --security-opt no-new-privileges --user "$uid:$gid" --entrypoint /bin/sleep \
    --mount "type=bind,src=$scratch/socket,dst=/run/host-gpu-monitor,readonly" \
    --mount "type=bind,src=$project/host_gpu_monitor,dst=/app/host_gpu_monitor,readonly" \
    --workdir /app "${SMOKE_PYTHON_IMAGE:-python:3.12-slim}" 120 >/dev/null
docker exec "$name" python3 -m host_gpu_monitor snapshot > "$scratch/first.json"
if docker exec --user "$other_uid:$other_uid" "$name" python3 -m host_gpu_monitor snapshot > "$scratch/denied.log" 2>&1; then
    exit 1
fi
grep -q 'Permission denied' "$scratch/denied.log"

kill "$daemon"
wait "$daemon"
daemon=""
test ! -e "$scratch/socket/gpu.sock"
start_daemon
docker exec "$name" python3 -m host_gpu_monitor snapshot > "$scratch/second.json"
python3 - "$scratch" <<'PY'
import json
from pathlib import Path
import sys

folder = Path(sys.argv[1])
first, second = [json.loads((folder / name).read_text()) for name in ("first.json", "second.json")]
for sample in (first, second):
    assert sample["schemaVersion"] == 1 and sample["status"] == "ok", sample
    assert sample["peerHostPid"] > 0
    assert sample["devices"][0]["usedMb"] == 5273
assert second["sampledAt"] > first["sampledAt"]
print(json.dumps({"transport":"docker-unix-no-network","source":"synthetic-nvidia-xml",
                  "wrongUidDenied":True,"reconnectAfterDaemonRestart":True,"telemetryDefaultOff":True}))
PY
