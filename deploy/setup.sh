#!/usr/bin/env bash
# deploy/setup.sh — Idempotent setup for a fresh Ubuntu 22.04 VM.
#
# Run as root (or with sudo):
#   sudo bash deploy/setup.sh
#
# What this script does:
#   1. Installs system packages
#   2. Clones the repo (if not already present)
#   3. Creates Python venv and installs requirements
#   4. Creates profile directories
#   5. Copies systemd unit files
#   6. Enables (but does NOT start) services and timers
#   7. Opens UFW ports
#   8. Prints next-steps
#
# To add a third profile later:
#   1. Create deploy/env/<name>.env with PORT=<port>
#   2. Add: systemctl enable --now jobagent-ui@<name>
#   3. Add: systemctl enable --now jobagent-worker@<name>.timer
#   4. Open UFW port: ufw allow <port>/tcp
#   No code changes required.

set -euo pipefail

REPO_DIR="/home/ubuntu/job-agent"
REPO_URL="${REPO_URL:-}"        # set this env var if you want auto-clone
VENV_DIR="$REPO_DIR/.venv"
SYSTEMD_DIR="/etc/systemd/system"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[setup]${NC} $*"; }

# ── 1. System packages ────────────────────────────────────────────────────────
info "Installing system packages..."
apt-get update -qq
apt-get install -y python3-pip python3-venv git ufw

# ── 2. Clone repo ─────────────────────────────────────────────────────────────
if [[ ! -d "$REPO_DIR" ]]; then
    if [[ -z "$REPO_URL" ]]; then
        warn "REPO_DIR ($REPO_DIR) does not exist and REPO_URL is not set."
        warn "Either clone the repo manually to $REPO_DIR, or re-run with:"
        warn "  REPO_URL=https://github.com/youruser/job-agent sudo bash deploy/setup.sh"
        exit 1
    fi
    info "Cloning $REPO_URL → $REPO_DIR..."
    git clone "$REPO_URL" "$REPO_DIR"
    chown -R ubuntu:ubuntu "$REPO_DIR"
else
    info "Repo already present at $REPO_DIR — skipping clone."
fi

cd "$REPO_DIR"

# ── 3. Python venv ────────────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating Python venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

info "Installing Python dependencies..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r "$REPO_DIR/requirements.txt"

# ── 4. Profile directories ────────────────────────────────────────────────────
for profile in manav sister; do
    profile_dir="$REPO_DIR/profiles/$profile"
    if [[ ! -d "$profile_dir" ]]; then
        info "Creating profile directory: profiles/$profile/"
        mkdir -p "$profile_dir"
        chown ubuntu:ubuntu "$profile_dir"
    else
        info "Profile directory already exists: profiles/$profile/"
    fi
done

# ── 5. Systemd unit files ─────────────────────────────────────────────────────
info "Copying systemd unit files to $SYSTEMD_DIR..."
cp "$REPO_DIR/deploy/systemd/jobagent-ui@.service"       "$SYSTEMD_DIR/"
cp "$REPO_DIR/deploy/systemd/jobagent-worker@.service"   "$SYSTEMD_DIR/"
cp "$REPO_DIR/deploy/systemd/jobagent-worker@.timer"     "$SYSTEMD_DIR/"
systemctl daemon-reload
info "systemd units installed."

# ── 6. Enable services and timers ─────────────────────────────────────────────
info "Enabling services and timers (not starting yet)..."

# UI services — restart=always, will start when you explicitly start them
systemctl enable jobagent-ui@manav.service
systemctl enable jobagent-ui@sister.service

# Timers — Persistent=true, fire daily at 04:00 UTC
systemctl enable jobagent-worker@manav.timer
systemctl enable jobagent-worker@sister.timer

info "Services and timers enabled."

# ── 7. UFW firewall ───────────────────────────────────────────────────────────
info "Opening UFW ports 8501 and 8502..."
ufw allow 8501/tcp comment "Job Agent — manav dashboard"
ufw allow 8502/tcp comment "Job Agent — sister dashboard"
# Enable UFW if not already active (non-interactively)
if ! ufw status | grep -q "Status: active"; then
    ufw --force enable
fi
info "UFW configured."

# ── 8. Next steps ─────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Setup complete. Next steps:"
echo "================================================================"
echo ""
echo "  1. Fill in API keys:"
echo "     cp $REPO_DIR/.env.example $REPO_DIR/.env"
echo "     nano $REPO_DIR/.env"
echo ""
echo "  2. Fill in each profile's config:"
echo "     # Use the Streamlit onboarding UI, OR copy an example:"
echo "     # profiles/manav/config.yaml"
echo "     # profiles/sister/config.yaml"
echo ""
echo "  3. Start the dashboards:"
echo "     systemctl start jobagent-ui@manav jobagent-ui@sister"
echo ""
echo "  4. Start the daily timers:"
echo "     systemctl start jobagent-worker@manav.timer"
echo "     systemctl start jobagent-worker@sister.timer"
echo ""
echo "  5. Verify:"
echo "     systemctl status jobagent-ui@manav"
echo "     systemctl status jobagent-worker@manav.timer"
echo "     curl http://localhost:8501"
echo ""
echo "  To add a third profile (e.g. 'alice' on port 8503):"
echo "    echo 'PORT=8503' > $REPO_DIR/deploy/env/alice.env"
echo "    mkdir -p $REPO_DIR/profiles/alice"
echo "    systemctl enable --now jobagent-ui@alice"
echo "    systemctl enable --now jobagent-worker@alice.timer"
echo "    ufw allow 8503/tcp"
echo "================================================================"
