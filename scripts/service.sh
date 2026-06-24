#!/usr/bin/env bash
# service.sh — manage VisionEngine Edge as a systemd service
#
# Usage:
#   sudo bash scripts/service.sh install    — create and enable the service
#   sudo bash scripts/service.sh start      — start the service
#   sudo bash scripts/service.sh stop       — stop the service
#   sudo bash scripts/service.sh restart    — restart the service
#        bash scripts/service.sh status     — show service status
#        bash scripts/service.sh logs       — tail live logs
#   sudo bash scripts/service.sh uninstall  — stop, disable, and remove the service

set -euo pipefail

SERVICE_NAME="visionengine-edge"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# resolve project root (one level up from this script)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*" >&2; exit 1; }

# ── guards ────────────────────────────────────────────────────────────────────

require_root() {
    [ "$(id -u)" -eq 0 ] || error "This command must be run as root: sudo bash scripts/service.sh $1"
}

require_systemd() {
    command -v systemctl &>/dev/null || error "systemd not found. This script is for Linux systemd systems."
}

require_installed() {
    [ -f "$PYTHON" ] || error "Virtual environment not found. Run install.sh first."
}

# ── commands ──────────────────────────────────────────────────────────────────

cmd_install() {
    require_root install
    require_systemd
    require_installed

    SERVICE_USER="${SUDO_USER:-$(logname 2>/dev/null || echo root)}"

    info "Creating service as user '$SERVICE_USER'..."
    cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=VisionEngine Edge Agent
After=network.target
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PYTHON main.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

# keep camera device access
SupplementaryGroups=video

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable "$SERVICE_NAME"
    info "Service installed and enabled"
    echo ""
    echo "  Start now:  sudo bash scripts/service.sh start"
    echo "  View logs:       bash scripts/service.sh logs"
    echo ""
}

cmd_start() {
    require_root start
    require_systemd
    systemctl start "$SERVICE_NAME"
    info "Service started"
    systemctl status "$SERVICE_NAME" --no-pager -l | tail -6
}

cmd_stop() {
    require_root stop
    require_systemd
    systemctl stop "$SERVICE_NAME"
    info "Service stopped"
}

cmd_restart() {
    require_root restart
    require_systemd
    systemctl restart "$SERVICE_NAME"
    info "Service restarted"
    systemctl status "$SERVICE_NAME" --no-pager -l | tail -6
}

cmd_status() {
    require_systemd
    systemctl status "$SERVICE_NAME" --no-pager -l
}

cmd_logs() {
    require_systemd
    journalctl -u "$SERVICE_NAME" -f --no-pager
}

cmd_uninstall() {
    require_root uninstall
    require_systemd

    if systemctl is-active --quiet "$SERVICE_NAME"; then
        systemctl stop "$SERVICE_NAME"
        info "Service stopped"
    fi

    if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        systemctl disable "$SERVICE_NAME"
        info "Service disabled"
    fi

    if [ -f "$SERVICE_FILE" ]; then
        rm "$SERVICE_FILE"
        systemctl daemon-reload
        info "Service file removed"
    fi

    info "Uninstall complete — files in $PROJECT_DIR are untouched"
}

# ── dispatch ──────────────────────────────────────────────────────────────────

COMMAND="${1:-}"

case "$COMMAND" in
    install)   cmd_install   ;;
    start)     cmd_start     ;;
    stop)      cmd_stop      ;;
    restart)   cmd_restart   ;;
    status)    cmd_status    ;;
    logs)      cmd_logs      ;;
    uninstall) cmd_uninstall ;;
    *)
        echo ""
        echo "  Usage: bash scripts/service.sh <command>"
        echo ""
        echo "  Commands:"
        echo "    install    — register as a systemd service (run once, needs sudo)"
        echo "    start      — start the service"
        echo "    stop       — stop the service"
        echo "    restart    — restart the service"
        echo "    status     — show current status"
        echo "    logs       — tail live logs (Ctrl+C to exit)"
        echo "    uninstall  — remove the service (data and config are kept)"
        echo ""
        exit 1
        ;;
esac
