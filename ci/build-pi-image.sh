#!/usr/bin/env bash
# Build a flashable Raspberry Pi OS Lite image with EagleEye preinstalled.
#
# Runs as root on a native arm64 host (GitHub ubuntu-24.04-arm runner), so the
# chroot needs no QEMU emulation. Produces eagleeye-<date>.img.xz and, for
# tagged builds, a Raspberry Pi Imager manifest in the current directory.

set -euxo pipefail

BUILD_STARTED=$SECONDS
PHASE_STARTED=$SECONDS
PHASE_NAME="startup"

# Print the elapsed time for each build phase and start the next one.
phase() {
    local now=$SECONDS
    printf '<== %s took %dm %ds\n' "$PHASE_NAME" "$(((now - PHASE_STARTED) / 60))" "$(((now - PHASE_STARTED) % 60))"
    PHASE_NAME="$1"
    PHASE_STARTED=$now
    printf '==> %s\n' "$PHASE_NAME"
}

BASE_IMAGE_URL="${BASE_IMAGE_URL:-https://downloads.raspberrypi.com/raspios_lite_arm64_latest}"
IMAGE_GROW_BYTES="${IMAGE_GROW_BYTES:-8G}"
XZ_PRESET="${XZ_PRESET:-9}"
IMAGE_USER="eagleeye"
IMAGE_PASSWORD="eagleeye"
IMAGE_HOSTNAME="eagleeye"
REPO_SRC="${REPO_SRC:-$(pwd)}"
OUT_NAME="eagleeye-$(date +%Y-%m-%d).img"
RELEASE_URL="${RELEASE_URL:-}"
if [ -z "$RELEASE_URL" ] && [ "${GITHUB_REF_TYPE:-}" = tag ]; then
    RELEASE_URL="https://github.com/${GITHUB_REPOSITORY}/releases/download/${GITHUB_REF_NAME}/${OUT_NAME}.xz"
fi

# Build outside the repo checkout (the repo gets copied into the image, and
# the image must not contain itself). Prefer /mnt, the runner's larger disk.
if [ -d /mnt ] && [ -w /mnt ]; then
    BUILD_DIR=/mnt/eagleeye-image-build
else
    BUILD_DIR="$(mktemp -d)"
fi
BUILD_CACHE_DIR="${BUILD_CACHE_DIR:-$BUILD_DIR/cache}"
mkdir -p "$BUILD_DIR" "$BUILD_CACHE_DIR"
OUT_IMG="$BUILD_DIR/$OUT_NAME"
BASE_IMAGE_CACHE="$BUILD_CACHE_DIR/raspios-lite-arm64.img.xz"

MNT=""
LOOP=""

cleanup() {
    set +e
    if [ -n "$MNT" ]; then
        umount "$MNT/home/$IMAGE_USER/.npm" 2>/dev/null
        umount "$MNT/home/$IMAGE_USER/.cache" 2>/dev/null
        umount "$MNT/var/cache/apt/archives" 2>/dev/null
        for fs in run dev/pts dev sys proc; do umount "$MNT/$fs" 2>/dev/null; done
        umount "$MNT/boot/firmware" 2>/dev/null
        umount "$MNT" 2>/dev/null
        rmdir "$MNT" 2>/dev/null
    fi
    [ -n "$LOOP" ] && losetup -d "$LOOP" 2>/dev/null
}
trap cleanup EXIT

phase "Downloading base image"
curl -fL --etag-save "$BASE_IMAGE_CACHE.etag" --etag-compare "$BASE_IMAGE_CACHE.etag" \
    -o "$BASE_IMAGE_CACHE" "$BASE_IMAGE_URL"
xz -dc "$BASE_IMAGE_CACHE" > "$OUT_IMG"

phase "Growing image and root filesystem"
truncate -s +"$IMAGE_GROW_BYTES" "$OUT_IMG"
LOOP="$(losetup -fP --show "$OUT_IMG")"
parted -s "$LOOP" resizepart 2 100%
e2fsck -pf "${LOOP}p2" || {
    status=$?
    [[ $status == 1 || $status == 2 ]] || exit "$status"
}
resize2fs "${LOOP}p2"

phase "Mounting image"
MNT="$(mktemp -d)"
mount "${LOOP}p2" "$MNT"
mount "${LOOP}p1" "$MNT/boot/firmware"
for fs in proc sys dev dev/pts run; do mount --bind "/$fs" "$MNT/$fs"; done
cp /etc/resolv.conf "$MNT/etc/resolv.conf"

# Keep services from starting inside the chroot, and disable the Pi's
# ld.so.preload shim which can break processes on non-Pi hardware.
printf '#!/bin/sh\nexit 101\n' > "$MNT/usr/sbin/policy-rc.d"
chmod +x "$MNT/usr/sbin/policy-rc.d"
sed -i 's/^\([^#]\)/#\1/' "$MNT/etc/ld.so.preload" || true

phase "Creating the $IMAGE_USER user"
chroot "$MNT" /bin/bash -euxc "
    useradd -m -s /bin/bash '$IMAGE_USER'
    echo '$IMAGE_USER:$IMAGE_PASSWORD' | chpasswd
    for g in sudo adm video plugdev dialout i2c gpio spi input render netdev users; do
        getent group \"\$g\" >/dev/null && usermod -aG \"\$g\" '$IMAGE_USER' || true
    done
    # Match stock Raspberry Pi OS first-user behavior.
    echo '$IMAGE_USER ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/010_${IMAGE_USER}-nopasswd
    chmod 440 /etc/sudoers.d/010_${IMAGE_USER}-nopasswd
"

# Keep package-manager caches on the host disk, not in the image.
mkdir -p "$BUILD_CACHE_DIR/user-cache" "$BUILD_CACHE_DIR/npm" "$BUILD_CACHE_DIR/apt-archives/partial"
mkdir -p "$MNT/home/$IMAGE_USER/.cache" "$MNT/home/$IMAGE_USER/.npm"
mount --bind "$BUILD_CACHE_DIR/user-cache" "$MNT/home/$IMAGE_USER/.cache"
mount --bind "$BUILD_CACHE_DIR/npm" "$MNT/home/$IMAGE_USER/.npm"
mount --bind "$BUILD_CACHE_DIR/apt-archives" "$MNT/var/cache/apt/archives"
chroot "$MNT" chown -R "$IMAGE_USER:$IMAGE_USER" "/home/$IMAGE_USER/.cache" "/home/$IMAGE_USER/.npm"

phase "Copying repository into the image"
rm -rf "$MNT/tmp/eagleeye-src"
rsync -a --exclude='*.img' --exclude='*.img.xz' --exclude=node_modules \
    --exclude=.venv "$REPO_SRC/" "$MNT/tmp/eagleeye-src/"
chroot "$MNT" chown -R "$IMAGE_USER:$IMAGE_USER" /tmp/eagleeye-src

phase "Running the EagleEye installer inside the chroot"
chroot "$MNT" su - "$IMAGE_USER" -c "
    git config --global --add safe.directory '*' 2>/dev/null || true
    export EAGLEEYE_IMAGE_BUILD=1
    export EAGLEEYE_REPO_URL=/tmp/eagleeye-src
    bash /tmp/eagleeye-src/install.sh
    git -C '/home/$IMAGE_USER/EagleEye-Vision-System' remote set-url origin \
        https://github.com/Scythe-Engineering/EagleEye-Vision-System.git
"

phase "Configuring first boot"
echo "$IMAGE_HOSTNAME" > "$MNT/etc/hostname"
sed -i "s/127.0.1.1.*/127.0.1.1\t$IMAGE_HOSTNAME/" "$MNT/etc/hosts" ||
    echo -e "127.0.1.1\t$IMAGE_HOSTNAME" >> "$MNT/etc/hosts"
touch "$MNT/boot/firmware/ssh"
echo "$IMAGE_USER:$(openssl passwd -6 "$IMAGE_PASSWORD")" > "$MNT/boot/firmware/userconf.txt"
chroot "$MNT" /bin/bash -euxc "
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y avahi-daemon openssh-server rpi-usb-gadget
    systemctl enable ssh
    cloud-init clean --logs
    apt-get clean
"
cat > "$MNT/etc/cloud/cloud.cfg.d/90-eagleeye-usb-gadget.cfg" <<'EOF'
rpi:
  enable_usb_gadget: true
enable_ssh: true
EOF
mkdir -p "$MNT/etc/polkit-1/rules.d" "$MNT/var/lib/NetworkManager"
cat > "$MNT/var/lib/NetworkManager/NetworkManager.state" <<'EOF'
[main]
NetworkingEnabled=true
WirelessEnabled=true
WWANEnabled=true
EOF
cat > "$MNT/etc/polkit-1/rules.d/49-eagleeye-network-manager.rules" <<EOF
polkit.addRule(function(action, subject) {
    if (subject.user == "$IMAGE_USER" &&
        (action.id == "org.freedesktop.NetworkManager.enable-disable-wifi" ||
         action.id == "org.freedesktop.NetworkManager.network-control" ||
         action.id == "org.freedesktop.NetworkManager.settings.modify.system" ||
         action.id == "org.freedesktop.NetworkManager.wifi.scan")) {
        return polkit.Result.YES;
    }
});
EOF

phase "Cleaning up inside image"
rm -rf "$MNT/tmp/eagleeye-src"
rm -f "$MNT/usr/sbin/policy-rc.d" "$MNT/etc/resolv.conf"
sed -i 's/^#//' "$MNT/etc/ld.so.preload" || true

phase "Verifying image contents"
# The venv python is a symlink that only resolves inside the image root.
chroot "$MNT" test -x "/home/$IMAGE_USER/EagleEye-Vision-System/.venv/bin/python"
test -f "$MNT/home/$IMAGE_USER/EagleEye-Vision-System/src/webui/static/bundle.js"
test -f "$MNT/etc/systemd/system/eagleeye.service"
test "$(chroot "$MNT" git -C "/home/$IMAGE_USER/EagleEye-Vision-System" remote get-url origin)" = \
    "https://github.com/Scythe-Engineering/EagleEye-Vision-System.git"
test -L "$MNT/etc/systemd/system/multi-user.target.wants/eagleeye.service"
test -L "$MNT/etc/systemd/system/multi-user.target.wants/ssh.service"
chroot "$MNT" test -x /usr/bin/cloud-init
chroot "$MNT" test -x /usr/sbin/NetworkManager
chroot "$MNT" test -x /usr/sbin/netplan
chroot "$MNT" test -x /usr/bin/rpi-usb-gadget
grep -Fxq '  enable_usb_gadget: true' "$MNT/etc/cloud/cloud.cfg.d/90-eagleeye-usb-gadget.cfg"
grep -Fxq 'WirelessEnabled=true' "$MNT/var/lib/NetworkManager/NetworkManager.state"
grep -Fq 'org.freedesktop.NetworkManager.enable-disable-wifi' "$MNT/etc/polkit-1/rules.d/49-eagleeye-network-manager.rules"
grep -Fq 'org.freedesktop.NetworkManager.network-control' "$MNT/etc/polkit-1/rules.d/49-eagleeye-network-manager.rules"
grep -Fq 'org.freedesktop.NetworkManager.settings.modify.system' "$MNT/etc/polkit-1/rules.d/49-eagleeye-network-manager.rules"
grep -Fq 'org.freedesktop.NetworkManager.wifi.scan' "$MNT/etc/polkit-1/rules.d/49-eagleeye-network-manager.rules"
find "$MNT/usr/lib/python3/dist-packages/cloudinit/config" -name cc_raspberry_pi.py -print -quit | grep -q .

phase "Unmounting and hashing image"
sync
umount "$MNT/home/$IMAGE_USER/.npm"
umount "$MNT/home/$IMAGE_USER/.cache"
umount "$MNT/var/cache/apt/archives"
for fs in run dev/pts dev sys proc; do umount "$MNT/$fs"; done
umount "$MNT/boot/firmware"
umount "$MNT"
e2fsck -fn "${LOOP}p2"
losetup -d "$LOOP"
rmdir "$MNT"
MNT=""
LOOP=""
trap - EXIT
EXTRACT_SIZE="$(stat -c %s "$OUT_IMG")"
EXTRACT_SHA256="$(sha256sum "$OUT_IMG" | cut -d' ' -f1)"
phase "Compressing image"
xz -T0 "-$XZ_PRESET" "$OUT_IMG"
DOWNLOAD_SIZE="$(stat -c %s "$OUT_IMG.xz")"
DOWNLOAD_SHA256="$(sha256sum "$OUT_IMG.xz" | cut -d' ' -f1)"
OUTPUT_DIR="${GITHUB_WORKSPACE:-$REPO_SRC}"
mv "$OUT_IMG.xz" "$OUTPUT_DIR/"

if [ -n "$RELEASE_URL" ]; then
    MANIFEST="$OUTPUT_DIR/${OUT_NAME%.img}.rpi-imager-manifest"
    cat > "$MANIFEST" <<EOF
{
  "os_list": [
    {
      "name": "EagleEye Vision System",
      "description": "Raspberry Pi OS Lite 64-bit with EagleEye preinstalled",
      "icon": "https://downloads.raspberrypi.com/raspios_armhf/Raspberry_Pi_OS_(32-bit).png",
      "url": "$RELEASE_URL",
      "extract_size": $EXTRACT_SIZE,
      "extract_sha256": "$EXTRACT_SHA256",
      "image_download_size": $DOWNLOAD_SIZE,
      "image_download_sha256": "$DOWNLOAD_SHA256",
      "release_date": "$(date +%Y-%m-%d)",
      "init_format": "cloudinit-rpi",
      "architecture": "arm64",
      "devices": ["pi5-64bit", "pi4-64bit", "pi3-64bit"],
      "capabilities": []
    }
  ]
}
EOF
    python3 -m json.tool "$MANIFEST" >/dev/null
fi

ls -lh "$OUTPUT_DIR/$OUT_NAME.xz" "$OUTPUT_DIR"/*.rpi-imager-manifest 2>/dev/null || true
phase "Build complete"
printf '<== Total build time: %dm %ds\n' "$(((SECONDS - BUILD_STARTED) / 60))" "$(((SECONDS - BUILD_STARTED) % 60))"
