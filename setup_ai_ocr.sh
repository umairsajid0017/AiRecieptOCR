#!/usr/bin/env bash
set -euo pipefail

# Cross-distro setup script for AiRecieptOCR
# - Installs system packages (python3, venv, pip) using detected package manager
# - Creates a Python venv at ./myenv and installs pip requirements
# - Generates a systemd service unit (ai_ocr.service) using current user & project path
# - Enables and starts the service (if systemd is available)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/myenv"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"
SERVICE_NAME="ai_ocr.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"

error() { echo "ERROR: $*" >&2; exit 1; }

detect_pkg_mgr() {
  if command -v apt-get >/dev/null 2>&1; then
    echo "apt"
  elif command -v dnf >/dev/null 2>&1; then
    echo "dnf"
  elif command -v yum >/dev/null 2>&1; then
    echo "yum"
  elif command -v pacman >/dev/null 2>&1; then
    echo "pacman"
  elif command -v zypper >/dev/null 2>&1; then
    echo "zypper"
  else
    echo "unknown"
  fi
}

install_packages() {
  PKG_MGR="$1"
  echo "Using package manager: $PKG_MGR"
  case "$PKG_MGR" in
    apt)
      sudo apt-get update
      sudo apt-get install -y python3 python3-venv python3-pip git
      ;;
    dnf)
      sudo dnf install -y python3 python3-venv python3-pip git
      ;;
    yum)
      sudo yum install -y python3 python3-venv python3-pip git
      ;;
    pacman)
      sudo pacman -Sy --noconfirm python python-virtualenv python-pip git
      ;;
    zypper)
      sudo zypper install -y python3 python3-venv python3-pip git
      ;;
    *)
      echo "Unknown package manager; please ensure python3 and pip are installed.";
      ;;
  esac
}

create_venv_and_install() {
  if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  else
    echo "Virtualenv already exists at $VENV_DIR"
  fi

  echo "Installing pip requirements..."
  "$VENV_DIR/bin/pip" install --upgrade pip
  if [ -f "$REQUIREMENTS_FILE" ]; then
    "$VENV_DIR/bin/pip" install -r "$REQUIREMENTS_FILE"
  else
    echo "No requirements.txt found; installing gunicorn as a fallback"
    "$VENV_DIR/bin/pip" install gunicorn
  fi
}

write_systemd_unit() {
  SERVICE_USER="${SUDO_USER:-$(whoami)}"
  cat <<EOF | sudo tee "$SERVICE_PATH" >/dev/null
[Unit]
Description=Acculedge OCR Server (AiRecieptOCR)
After=network.target

[Service]
User=$SERVICE_USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$VENV_DIR/bin/python -m gunicorn -w 1 -b 0.0.0.0:5050 api:app
Restart=always
Environment="PATH=$VENV_DIR/bin"

[Install]
WantedBy=multi-user.target
EOF

  echo "Wrote systemd unit to $SERVICE_PATH"
}

enable_and_start_service() {
  if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl daemon-reload
    sudo systemctl enable --now "$SERVICE_NAME"
    echo "Service enabled and started: $SERVICE_NAME"
    sudo systemctl status --no-pager "$SERVICE_NAME" || true
  else
    echo "systemctl not available on this system. Please start the service manually or use an alternative init system."
  fi
}

main() {
  echo "Project directory: $PROJECT_DIR"
  PKG_MGR=$(detect_pkg_mgr)
  if [ "$PKG_MGR" != "unknown" ]; then
    install_packages "$PKG_MGR"
  else
    echo "Could not detect package manager; skipping package installation.";
  fi

  create_venv_and_install

  if [ ! -w "$(dirname "$SERVICE_PATH")" ]; then
    echo "Note: This script requires sudo to write system service files and enable services. You may be prompted for your password."
  fi

  if [ -f "$SERVICE_PATH" ]; then
    echo "Backing up existing $SERVICE_PATH to ${SERVICE_PATH}.bak"
    sudo cp "$SERVICE_PATH" "${SERVICE_PATH}.bak"
  fi

  write_systemd_unit
  enable_and_start_service

  echo "Done. If something failed, inspect the systemd journal: sudo journalctl -u $SERVICE_NAME -e"
}

main "$@"
