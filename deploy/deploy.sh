#!/usr/bin/env bash
# deploy/deploy.sh — Re-deploy after a git pull.
#
# Run from the project root (as ubuntu user or with sudo -u ubuntu):
#   bash deploy/deploy.sh
#
# What this does:
#   1. Pulls latest code
#   2. Refreshes Python dependencies
#   3. Reloads systemd unit files (in case they changed)
#   4. Restarts both UI services
#   5. Prints status

set -euo pipefail

REPO_DIR="/home/ubuntu/job-agent"
VENV_DIR="$REPO_DIR/.venv"

GREEN='\033[0;32m'
NC='\033[0m'
info() { echo -e "${GREEN}[deploy]${NC} $*"; }

cd "$REPO_DIR"

# ── 1. Pull latest ────────────────────────────────────────────────────────────
info "Pulling latest code..."
git pull

# ── 2. Update dependencies ────────────────────────────────────────────────────
info "Updating Python dependencies..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r requirements.txt

# ── 3. Reload systemd (picks up any changed unit files) ───────────────────────
if [[ -d /etc/systemd/system ]]; then
    info "Refreshing systemd unit files..."
    cp deploy/systemd/jobagent-ui@.service       /etc/systemd/system/
    cp deploy/systemd/jobagent-worker@.service   /etc/systemd/system/
    cp deploy/systemd/jobagent-worker@.timer     /etc/systemd/system/
    systemctl daemon-reload
fi

# ── 4. Restart UI services ────────────────────────────────────────────────────
info "Restarting UI services..."
systemctl restart jobagent-ui@manav jobagent-ui@sister

# ── 5. Status ─────────────────────────────────────────────────────────────────
info "Service status:"
systemctl status jobagent-ui@manav jobagent-ui@sister --no-pager

info "Deploy complete."
