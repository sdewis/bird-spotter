#!/usr/bin/env bash
# Remote development & deployment helper for British Bird Spotter (PC -> Raspberry Pi 5).
set -euo pipefail

PI5_USER="${PI5_USER:-sean}"
PI5_HOST="${PI5_HOST:-192.168.100.122}"
PI5_DIR="${PI5_DIR:-/home/sean/Projects/bird-spotter}"
PI5_SERVICE="${PI5_SERVICE:-bird-spotter.service}"
PI5_DASHBOARD_URL="http://${PI5_HOST}:8080/api/state"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SSH_TARGET="${PI5_USER}@${PI5_HOST}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5)

is_on_pi5() {
    local host_ips
    host_ips="$(hostname -I 2>/dev/null || true)"
    [[ " ${host_ips} " == *" ${PI5_HOST} "* ]]
}

run_pi5() {
    if is_on_pi5; then
        bash -c "cd '${PI5_DIR}' && $*"
    else
        ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" "cd '${PI5_DIR}' && $*"
    fi
}

cmd_sync() {
    if is_on_pi5; then
        echo "[pi5] Already running directly on Pi 5 (${PI5_HOST}); skipping rsync."
        return 0
    fi
    if [[ "${LOCAL_DIR}" == *"/kio-fuse-"* ]]; then
        echo "[pi5] Warning: Current directory is the KIO-FUSE SFTP mount (${LOCAL_DIR})."
        echo "[pi5] Open the local repository at /home/sean/Projects/bird-spotter for fast local editing."
        return 0
    fi
    echo "==> Syncing code from ${LOCAL_DIR} -> ${SSH_TARGET}:${PI5_DIR} ..."
    rsync -avz --delete \
        --exclude='.git/' \
        --exclude='.venv/' \
        --exclude='venv/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='*.egg-info/' \
        --exclude='.pytest_cache/' \
        --exclude='data/' \
        --exclude='snapshots/' \
        --exclude='logs/' \
        --exclude='recordings/' \
        --exclude='hailort.log' \
        --exclude='gstshark_*/' \
        --exclude='backups/' \
        --exclude='models/*.onnx' \
        --exclude='models/*.pt' \
        --exclude='models/*.pth' \
        --exclude='models/*.hef' \
        --exclude='config/local.toml' \
        "${LOCAL_DIR}/" "${SSH_TARGET}:${PI5_DIR}/"
    echo "==> Sync complete."
}

cmd_test_local() {
    if [[ -x "${LOCAL_DIR}/.venv/bin/python" ]]; then
        echo "==> Running fast local unit tests on PC (${LOCAL_DIR}/.venv) ..."
        PYTHONPATH="${LOCAL_DIR}/src" "${LOCAL_DIR}/.venv/bin/python" -m unittest discover -s "${LOCAL_DIR}/tests" -v
    fi
}

cmd_test() {
    cmd_test_local
    cmd_sync
    echo "==> Running unit tests on Pi 5 (${SSH_TARGET}) ..."
    run_pi5 "PYTHONPATH=src python3 -m unittest discover -s tests -v"
}

cmd_check() {
    cmd_sync
    echo "==> Running hardware & model self-check on Pi 5 (${SSH_TARGET}) ..."
    run_pi5 "PYTHONPATH=src python3 -m bird_spotter.main --check-hardware"
}

cmd_restart() {
    echo "==> Restarting ${PI5_SERVICE} on Pi 5 (${SSH_TARGET}) ..."
    run_pi5 "sudo -n systemctl restart ${PI5_SERVICE} || systemctl restart ${PI5_SERVICE}"
    sleep 2
    cmd_status
}

cmd_status() {
    echo "==> Service status on Pi 5 (${SSH_TARGET}):"
    run_pi5 "systemctl --no-pager --full status ${PI5_SERVICE} | head -n 18" || true
    echo ""
    echo "==> Dashboard health check (${PI5_DASHBOARD_URL}):"
    curl -fsS --max-time 5 "${PI5_DASHBOARD_URL}" | python3 -c "
import json, sys
d = json.load(sys.stdin)
solar = d.get('solar') or {}
print(
    f\"  HTTP OK | FPS: {d.get('fps')} | Infer: {d.get('infer_ms')}ms | \"
    f\"Backend: {d.get('classifier_backend')} | FSM: {d.get('fsm_state')} | \"
    f\"Solar daylight: {solar.get('is_daylight')}\"
)
" || echo "  Dashboard not responding on port 8080 yet."
}

cmd_logs() {
    local follow_flag=""
    if [[ "${1:-}" == "-f" || "${1:-}" == "--follow" ]]; then
        follow_flag="-f"
    fi
    run_pi5 "journalctl -u ${PI5_SERVICE} -n 60 --no-pager ${follow_flag}"
}

cmd_pull_data() {
    if is_on_pi5; then
        echo "[pi5] Already on Pi 5."
        return 0
    fi
    echo "==> Pulling runtime data/ and snapshots/ from Pi 5 -> ${LOCAL_DIR} ..."
    mkdir -p "${LOCAL_DIR}/data" "${LOCAL_DIR}/snapshots"
    rsync -avz "${SSH_TARGET}:${PI5_DIR}/data/" "${LOCAL_DIR}/data/"
    rsync -avz "${SSH_TARGET}:${PI5_DIR}/snapshots/" "${LOCAL_DIR}/snapshots/"
    echo "==> Pulled latest database and snapshots."
}

cmd_pull_models() {
    if is_on_pi5; then
        echo "[pi5] Already on Pi 5."
        return 0
    fi
    echo "==> Pulling model weights from Pi 5 -> ${LOCAL_DIR}/models/ ..."
    mkdir -p "${LOCAL_DIR}/models"
    rsync -avz "${SSH_TARGET}:${PI5_DIR}/models/" "${LOCAL_DIR}/models/"
    echo "==> Pulled model files."
}

cmd_deploy() {
    cmd_test
    cmd_restart
}

usage() {
    cat <<EOF
Usage: ./scripts/pi5.sh <command> [args...]

Commands:
  deploy        Run local & Pi 5 unit tests, rsync code to Pi 5, restart service, and check health (default)
  sync          Rsync source, config, and tests from PC to Pi 5 (preserves Pi 5 data/, snapshots/, models/)
  test          Run unit tests locally (if .venv present) and on the Pi 5
  test-local    Run unit tests only on this PC using .venv
  check         Sync and run --check-hardware on the Pi 5
  restart       Restart ${PI5_SERVICE} on the Pi 5 and verify health
  status        Show ${PI5_SERVICE} status and live /api/state summary from the Pi 5
  logs [-f]     Show recent journalctl logs for ${PI5_SERVICE} on the Pi 5 (pass -f to follow)
  pull-data     Pull SQLite database (data/) and captures (snapshots/) from Pi 5 to PC
  pull-models   Pull heavy model files (.onnx, .pt, .hef) from Pi 5 to PC
  exec <cmd>    Execute an arbitrary shell command in ${PI5_DIR} on the Pi 5
EOF
}

SUBCMD="${1:-deploy}"
shift || true

case "${SUBCMD}" in
    deploy)      cmd_deploy "$@" ;;
    sync)        cmd_sync "$@" ;;
    test)        cmd_test "$@" ;;
    test-local)  cmd_test_local "$@" ;;
    check)       cmd_check "$@" ;;
    restart)     cmd_restart "$@" ;;
    status)      cmd_status "$@" ;;
    logs)        cmd_logs "$@" ;;
    pull-data)   cmd_pull_data "$@" ;;
    pull-models) cmd_pull_models "$@" ;;
    exec|ssh)    run_pi5 "$*" ;;
    -h|--help|help) usage ;;
    *)
        echo "Unknown command: ${SUBCMD}" >&2
        usage >&2
        exit 1
        ;;
esac
