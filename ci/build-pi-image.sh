#!/usr/bin/env bash
# Build a flashable Raspberry Pi OS Lite image with EagleEye preinstalled.
#
# Runs as root on a native arm64 host (GitHub ubuntu-24.04-arm runner), so the
# chroot needs no QEMU emulation. Produces eagleeye-<date>.img.xz in the
# current directory.

set -euxo pipefail

BASE_IMAGE_URL="${BASE_IMAGE_URL:-https://downloads.raspberrypi.com/raspios_lite_arm64_latest}"
IMAGE_GROW_BYTES="${IMAGE_GROW_BYTES:-5G}"
IMAGE_USER="eagleeye"
IMAGE_PASSWORD="eagleeye"
IMAGE_HOSTNAME="eagleeye"
REPO_SRC="${REPO_SRC:-$(pwd)}"
OUT_NAME="eagleeye-$(date +%Y-%m-%d).img"

# Build outside the repo checkout (the repo gets copied into the image, and
# the image must not contain itself). Prefer /mnt, the runner's larger disk.
if [ -d /mnt ] && [ -w /mnt ]; then
    BUILD_DIR=/mnt/eagleeye-image-build
else
    BUILD_DIR="$(mktemp -d)"
fi
mkdir -p "$BUILD_DIR"
OUT_IMG="$BUILD_DIR/$OUT_NAME"

MNT=""
LOOP=""

cleanup() {
    set +e
    if [ -n "$MNT" ]; then
        for fs in run dev/pts dev sys proc; do umount "$MNT/$fs" 2>/dev/null; done
        umount "$MNT/boot/firmware" 2>/dev/null
        umount "$MNT" 2>/dev/null
        rmdir "$MNT" 2>/dev/null
    fi
    [ -n "$LOOP" ] && losetup -d "$LOOP" 2>/dev/null
}
trap cleanup EXIT

echo "==> Downloading base image"
curl -fL "$BASE_IMAGE_URL" | xz -dc > "$OUT_IMG"

echo "==> Growing image and root filesystem"
truncate -s +"$IMAGE_GROW_BYTES" "$OUT_IMG"
LOOP="$(losetup -fP --show "$OUT_IMG")"
parted -s "$LOOP" resizepart 2 100%
e2fsck -pf "${LOOP}p2"
resize2fs "${LOOP}p2"

echo "==> Mounting"
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

echo "==> Creating the $IMAGE_USER user"
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

echo "==> Copying repository into the image"
rm -rf "$MNT/tmp/eagleeye-src"
rsync -a --exclude='*.img' --exclude='*.img.xz' --exclude=node_modules \
    --exclude=.venv "$REPO_SRC/" "$MNT/tmp/eagleeye-src/"
chroot "$MNT" chown -R "$IMAGE_USER:$IMAGE_USER" /tmp/eagleeye-src

echo "==> Running the EagleEye installer inside the chroot"
chroot "$MNT" su - "$IMAGE_USER" -c "
    git config --global --add safe.directory '*' 2>/dev/null || true
    export EAGLEEYE_IMAGE_BUILD=1
    export EAGLEEYE_REPO_URL=/tmp/eagleeye-src
    bash /tmp/eagleeye-src/install.sh
"

echo "==> First-boot configuration"
echo "$IMAGE_HOSTNAME" > "$MNT/etc/hostname"
sed -i "s/127.0.1.1.*/127.0.1.1\t$IMAGE_HOSTNAME/" "$MNT/etc/hosts" ||
    echo -e "127.0.1.1\t$IMAGE_HOSTNAME" >> "$MNT/etc/hosts"
touch "$MNT/boot/firmware/ssh"
echo "$IMAGE_USER:$(openssl passwd -6 "$IMAGE_PASSWORD")" > "$MNT/boot/firmware/userconf.txt"
chroot "$MNT" /bin/bash -euxc "
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y avahi-daemon
    apt-get clean
"

echo "==> Cleanup inside image"
rm -rf "$MNT/tmp/eagleeye-src"
rm -f "$MNT/usr/sbin/policy-rc.d" "$MNT/etc/resolv.conf"
sed -i 's/^#//' "$MNT/etc/ld.so.preload" || true

echo "==> Verifying image contents"
test -x "$MNT/home/$IMAGE_USER/EagleEye-Vision-System/.venv/bin/python"
test -f "$MNT/home/$IMAGE_USER/EagleEye-Vision-System/src/webui/static/bundle.js"
test -f "$MNT/etc/systemd/system/eagleeye.service"
test -L "$MNT/etc/systemd/system/multi-user.target.wants/eagleeye.service"

echo "==> Unmounting and compressing"
cleanup
trap - EXIT
MNT=""
LOOP=""
xz -T0 -3 "$OUT_IMG"
ls -lh "$OUT_IMG.xz"
mv "$OUT_IMG.xz" "${GITHUB_WORKSPACE:-$REPO_SRC}/"
