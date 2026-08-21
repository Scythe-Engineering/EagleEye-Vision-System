#!/usr/bin/env bash
# EagleEye Vision System installer.
#
# One-line install (tested on Raspberry Pi OS Lite 64-bit, Debian 12, arm64):
#
#   curl -fsSL https://raw.githubusercontent.com/Scythe-Engineering/EagleEye-Vision-System/main/install.sh | bash
#
# Run as the normal (non-root) sudo-capable user that should own the install.
# Every function below is defined before "main" runs at the very bottom, so a
# truncated download cannot execute a partial install.

set -euo pipefail

REPO_URL="${EAGLEEYE_REPO_URL:-https://github.com/Scythe-Engineering/EagleEye-Vision-System.git}"
REPO_DIR_NAME="EagleEye-Vision-System"
SERVICE_NAME="eagleeye"
WEB_SERVER_PORT="5001"
WEB_SERVER_READY_TIMEOUT="600"
NODE_MAJOR="20"
SUPPORTED_ARCH="aarch64"
SUPPORTED_OS_ID="debian"
SUPPORTED_OS_VERSION="12"
CAMERA_GROUPS="video plugdev dialout"

APT_PACKAGES="git curl ca-certificates gnupg build-essential pkg-config cmake libssl-dev python3 python3-dev python3-venv v4l-utils libgl1 libglib2.0-0"

STEP_NUMBER=0
STEP_TOTAL=11

log_step() {
    STEP_NUMBER=$((STEP_NUMBER + 1))
    printf '\n\033[36m[%d/%d] %s\033[0m\n' "$STEP_NUMBER" "$STEP_TOTAL" "$1"
}

log_info() {
    printf '      %s\n' "$1"
}

log_warn() {
    printf '\033[33mWARNING: %s\033[0m\n' "$1" >&2
}

log_error() {
    printf '\033[31mERROR: %s\033[0m\n' "$1" >&2
}

# Read an /etc/os-release field without sourcing the whole file.
os_release_value() {
    local field_name="$1"
    local os_release_path="${2:-/etc/os-release}"
    [ -r "$os_release_path" ] || return 0
    sed -n "s/^${field_name}=//p" "$os_release_path" | head -n 1 | tr -d '"'
}

# Warn (never fail) when the platform is not the tested one.
check_platform() {
    local detected_arch="$1"
    local detected_os_id="$2"
    local detected_os_version="$3"

    if [ "$detected_arch" != "$SUPPORTED_ARCH" ]; then
        log_warn "Untested architecture '$detected_arch' (tested: $SUPPORTED_ARCH). Continuing anyway."
    fi
    if [ "$detected_os_id" != "$SUPPORTED_OS_ID" ] ||
        [ "$detected_os_version" != "$SUPPORTED_OS_VERSION" ]; then
        log_warn "Untested OS '$detected_os_id $detected_os_version' (tested: Raspberry Pi OS Lite 64-bit / Debian 12). Continuing anyway."
    fi
}

# Refuse to touch an existing install; updates belong to the Web UI updater.
check_not_already_installed() {
    local install_dir="$1"
    if [ -e "$install_dir" ]; then
        log_error "An install already exists at $install_dir."
        log_error "This installer only performs fresh installs."
        log_error "To update, open the EagleEye Web UI (http://<pi-address>:${WEB_SERVER_PORT}) and use Settings -> System Update."
        log_error "To reinstall from scratch, remove $install_dir first."
        return 1
    fi
    return 0
}

check_no_system_artifacts() {
    local service_path="${1:-/etc/systemd/system/${SERVICE_NAME}.service}"
    local sudoers_path="${2:-/etc/sudoers.d/${SERVICE_NAME}}"
    if [ -e "$service_path" ] || [ -e "$sudoers_path" ]; then
        log_error "A previous EagleEye service or sudoers policy still exists."
        log_error "Remove $service_path and $sudoers_path before a fresh install."
        return 1
    fi
    return 0
}

check_user() {
    if [ "$(id -u)" = "0" ]; then
        log_error "Do not run this installer as root or with sudo."
        log_error "Run it as the normal user that should own the install; it calls sudo itself when needed."
        return 1
    fi
    if ! command -v sudo >/dev/null 2>&1; then
        log_error "sudo is required but was not found on PATH."
        return 1
    fi
    return 0
}

# Remove only paths created by this invocation so failed fresh installs can be
# retried without changing the deliberate completed-install refusal.
cleanup_failed_install() {
    local exit_status="$1"
    if [ "$exit_status" -ne 0 ]; then
        if [ "${service_installed:-0}" = "1" ]; then
            sudo systemctl disable --now "${SERVICE_NAME}.service" >/dev/null 2>&1 || true
            sudo rm -f "/etc/systemd/system/${SERVICE_NAME}.service"
            sudo systemctl daemon-reload
        fi
        if [ "${sudoers_installed:-0}" = "1" ]; then
            sudo rm -f "/etc/sudoers.d/${SERVICE_NAME}"
        fi
        if [ -n "${staging_dir:-}" ] && [ -e "$staging_dir" ]; then
            log_warn "Install failed; removing incomplete staging directory $staging_dir."
            rm -rf -- "$staging_dir" || log_warn "Could not remove staging directory $staging_dir."
        fi
        if [ "${install_dir_created:-0}" = "1" ] && [ -e "${install_dir:-}" ]; then
            log_warn "Install failed; removing incomplete install directory $install_dir."
            rm -rf -- "$install_dir" || log_warn "Could not remove install directory $install_dir."
        fi
    fi
    exit "$exit_status"
}

# Render the systemd unit for the installing user and install path.
render_service_unit() {
    local service_user="$1"
    local service_home="$2"
    local install_dir="$3"

    cat <<UNIT
[Unit]
Description=EagleEye Vision System Backend
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${service_user}
WorkingDirectory=${install_dir}
Environment=PYTHONUNBUFFERED=1
Environment=HOME=${service_home}
Environment=PATH=${service_home}/.local/bin:${service_home}/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=${install_dir}/.venv/bin/python -m src.main_backend
Restart=on-failure
RestartSec=5s
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
UNIT
}

# Use a numeric sudoers user spec so even an unusual account name cannot be
# parsed as a sudoers alias. This grants only the backend's fixed commands.
render_sudoers_policy() {
    local service_user="$1"
    local service_uid="$2"
    case "$service_uid" in
        '' | *[!0-9]*)
            log_error "Invalid service UID for sudoers policy: $service_uid"
            return 1
            ;;
    esac
    cat <<POLICY
# EagleEye backend privileges for ${service_user} (UID ${service_uid}).
# Do not add shells, editors, wildcards, or general package commands here.
#${service_uid} ALL=(root) NOPASSWD: /usr/bin/apt update, /usr/bin/env DEBIAN_FRONTEND=noninteractive apt upgrade -y, /usr/bin/systemctl restart ${SERVICE_NAME}, /usr/sbin/reboot
POLICY
}

install_sudoers_policy() {
    local service_user="$1"
    local service_uid="$2"
    local sudoers_path="/etc/sudoers.d/${SERVICE_NAME}"
    local sudoers_temp
    sudoers_temp="$(mktemp)"
    render_sudoers_policy "$service_user" "$service_uid" >"$sudoers_temp"
    if ! sudo visudo -cf "$sudoers_temp"; then
        rm -f -- "$sudoers_temp"
        log_error "Refusing to install an invalid sudoers policy."
        return 1
    fi
    sudo install -o root -g root -m 0440 "$sudoers_temp" "$sudoers_path"
    rm -f -- "$sudoers_temp"
}

install_apt_packages() {
    log_step "Installing system packages with apt"
    sudo apt-get update
    # shellcheck disable=SC2086 # intentional word splitting of the package list
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y $APT_PACKAGES
}

install_uv() {
    log_step "Installing uv (Python package manager)"
    if command -v uv >/dev/null 2>&1; then
        log_info "uv already installed: $(command -v uv)"
        return 0
    fi
    curl -fsSL https://astral.sh/uv/install.sh | sh
}

install_node() {
    log_step "Installing Node.js ${NODE_MAJOR} and npm"
    if command -v node >/dev/null 2>&1 &&
        [ "$(node --version | sed 's/^v//' | cut -d. -f1)" -ge "$NODE_MAJOR" ] 2>/dev/null; then
        log_info "Node.js already installed: $(node --version)"
        return 0
    fi
    # Debian 12 ships Node 18; the WebUI build needs 20+, so use NodeSource.
    curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | sudo -E bash -
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs
}

install_rust() {
    log_step "Installing Rust toolchain"
    if command -v cargo >/dev/null 2>&1; then
        log_info "Rust already installed: $(cargo --version)"
        return 0
    fi
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal
}

clone_repository() {
    local install_dir="$1"
    log_step "Cloning EagleEye into $install_dir"
    git clone "$REPO_URL" "$install_dir"
}

install_python_dependencies() {
    local install_dir="$1"
    log_step "Installing Python dependencies (includes MemryX; this takes a while)"
    # pyproject.toml declares memryx for linux with its own index, so a plain
    # sync installs it on every install here.
    (cd "$install_dir" && uv sync)
}

install_frontend() {
    local install_dir="$1"
    log_step "Installing frontend dependencies and building the WebUI"
    (cd "$install_dir" && npm install && npm run build)
}

configure_camera_permissions() {
    local service_user="$1"
    log_step "Adding $service_user to camera device groups"
    for group_name in $CAMERA_GROUPS; do
        if getent group "$group_name" >/dev/null 2>&1; then
            sudo usermod -aG "$group_name" "$service_user"
            log_info "added to group: $group_name"
        else
            log_warn "Group '$group_name' does not exist on this system; skipping."
        fi
    done
}

install_service() {
    local service_user="$1"
    local service_home="$2"
    local install_dir="$3"
    log_step "Installing the ${SERVICE_NAME} systemd service"
    render_service_unit "$service_user" "$service_home" "$install_dir" |
        sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" >/dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable "${SERVICE_NAME}.service"
    sudo systemctl restart "${SERVICE_NAME}.service"
}

print_service_journal() {
    log_error "Recent ${SERVICE_NAME} service journal:"
    sudo journalctl -u "${SERVICE_NAME}.service" -n 50 --no-pager >&2 || true
}

wait_for_web_server() {
    local readiness_url="http://127.0.0.1:${WEB_SERVER_PORT}/"
    local readiness_deadline
    readiness_deadline=$(( $(date +%s) + WEB_SERVER_READY_TIMEOUT ))
    while ! curl --fail --silent --show-error --connect-timeout 2 --max-time 5 \
        "$readiness_url" >/dev/null 2>&1; do
        if [ "$(date +%s)" -ge "$readiness_deadline" ]; then
            log_error "Web UI did not become ready at $readiness_url within ${WEB_SERVER_READY_TIMEOUT}s."
            print_service_journal
            return 1
        fi
        sleep 2
    done
    log_info "OK   Web UI ready at $readiness_url"
}

verify_install() {
    local install_dir="$1"
    log_step "Verifying the install"

    verification_failed=0
    if [ -x "$install_dir/.venv/bin/python" ]; then
        log_info "OK   python venv: $install_dir/.venv"
    else
        log_error "missing python venv at $install_dir/.venv"
        verification_failed=1
    fi

    if [ -f "$install_dir/src/webui/static/bundle.js" ]; then
        log_info "OK   WebUI build: src/webui/static/bundle.js"
    else
        log_error "missing WebUI build output (src/webui/static/bundle.js)"
        verification_failed=1
    fi

    if systemctl is-enabled --quiet "${SERVICE_NAME}.service"; then
        log_info "OK   service enabled at boot"
    else
        log_error "service is not enabled"
        verification_failed=1
    fi

    if systemctl is-active --quiet "${SERVICE_NAME}.service"; then
        log_info "OK   service running"
        if ! wait_for_web_server; then
            verification_failed=1
        fi
    else
        log_error "service is not running"
        print_service_journal
        verification_failed=1
    fi

    return "$verification_failed"
}

print_summary() {
    local install_dir="$1"
    local host_name
    host_name="$(hostname 2>/dev/null || echo "<pi-address>")"

    printf '\n\033[32mEagleEye install complete.\033[0m\n\n'
    printf 'Install path : %s\n' "$install_dir"
    printf 'Web UI       : http://%s:%s\n' "$host_name" "$WEB_SERVER_PORT"
    printf 'Service      : sudo systemctl status %s\n' "$SERVICE_NAME"
    printf 'Logs         : journalctl -u %s -f\n\n' "$SERVICE_NAME"
    printf 'Next steps in the Web UI:\n'
    printf '  1. Calibrate your camera and note its bus ID.\n'
    printf '  2. Open the "2026_apriltag_starter" pipeline and fill in the camera bus ID,\n'
    printf '     intrinsics, extrinsics, and the 2026 AprilTag map path. It is intentionally\n'
    printf '     incomplete and will stay inactive until you finish it.\n'
    printf '  3. Use Settings -> System Update for future updates; do not rerun this installer.\n\n'
    printf 'Note: your current shell does not yet have the new camera groups.\n'
    printf 'Log out and back in (or reboot) before running EagleEye tools by hand.\n'
}

main() {
    check_user

    service_user="$(id -un)"
    service_uid="$(id -u)"
    service_home="$HOME"
    install_dir="${service_home}/${REPO_DIR_NAME}"
    staging_dir=""
    install_dir_created=0
    sudoers_installed=0
    service_installed=0

    printf '\033[36mEagleEye Vision System installer\033[0m\n'
    log_info "user:    $service_user"
    log_info "install: $install_dir"

    detected_arch="$(uname -m)"
    detected_os_id="$(os_release_value ID)"
    detected_os_version="$(os_release_value VERSION_ID)"
    check_platform "$detected_arch" "$detected_os_id" "$detected_os_version"
    check_not_already_installed "$install_dir"
    check_no_system_artifacts
    trap 'cleanup_failed_install $?' EXIT
    staging_dir="$(mktemp -d "${service_home}/.${REPO_DIR_NAME}.installing.XXXXXX")"

    install_apt_packages
    install_uv
    install_node
    install_rust

    # uv and rustup land in per-user bin directories that are not on PATH yet.
    PATH="${service_home}/.local/bin:${service_home}/.cargo/bin:${PATH}"
    export PATH

    clone_repository "$staging_dir"
    install_python_dependencies "$staging_dir"
    install_frontend "$staging_dir"
    configure_camera_permissions "$service_user"
    log_step "Installing narrowly scoped backend sudo permissions"
    install_sudoers_policy "$service_user" "$service_uid"
    sudoers_installed=1
    mv "$staging_dir" "$install_dir"
    staging_dir=""
    install_dir_created=1
    service_installed=1
    install_service "$service_user" "$service_home" "$install_dir"

    if verify_install "$install_dir"; then
        print_summary "$install_dir"
        install_dir_created=0
        trap - EXIT
    else
        log_error "Install finished with failed verification checks (see above)."
        return 1
    fi
}

# Allow tests to source this file for its functions without installing anything.
if [ -z "${EAGLEEYE_INSTALL_LIB_ONLY:-}" ]; then
    main "$@"
fi
