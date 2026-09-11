#!/usr/bin/env bash
# Fix FAT32 disk, free backup folders, copy packed train data to USB.
# Usage:
#   sudo bash tools/copy_train_data_to_usb.sh
# Optional env:
#   USB_DEV=/dev/sda4
#   USB_MNT=/media/xue/XUE_DISK1
#   PACK_DIR=/home/xue/work/_isaac_factory_pack

set -euo pipefail

USB_DEV="${USB_DEV:-/dev/sda4}"
USB_MNT="${USB_MNT:-/media/xue/XUE_DISK1}"
PACK_DIR="${PACK_DIR:-/home/xue/work/_isaac_factory_pack}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_NAME="isaac_factory_train_data"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "请用 sudo 运行：sudo bash $0" >&2
  exit 1
fi

REAL_USER="${SUDO_USER:-xue}"
REAL_HOME="$(getent passwd "$REAL_USER" | cut -d: -f6)"
PACK_DIR="${PACK_DIR/#\~/$REAL_HOME}"

echo "==> device=$USB_DEV mount=$USB_MNT pack=$PACK_DIR"

if [[ ! -d "$PACK_DIR" ]] || ! ls "$PACK_DIR"/isaac_factory_train_data.tar.*.tarpart >/dev/null 2>&1; then
  echo "找不到分包：$PACK_DIR/isaac_factory_train_data.tar.*.tarpart" >&2
  echo "请先在仓库机器上打包（见 README）。" >&2
  exit 1
fi

echo "==> unmount"
udisksctl unmount -b "$USB_DEV" 2>/dev/null || umount "$USB_MNT" 2>/dev/null || true
sleep 1

echo "==> fsck.vfat -a (auto repair)"
fsck.vfat -a "$USB_DEV" || true

echo "==> mount rw"
udisksctl mount -b "$USB_DEV" || mount -o rw,uid="$(id -u "$REAL_USER"),gid="$(id -g "$REAL_USER")" "$USB_DEV" "$USB_MNT"
# ensure path
if [[ ! -d "$USB_MNT" ]]; then
  USB_MNT="$(lsblk -no MOUNTPOINT "$USB_DEV" | head -1)"
fi
echo "mounted at $USB_MNT"
mount | grep "$USB_DEV"

echo "==> delete backup + linux_backup"
rm -rf "$USB_MNT/backup" "$USB_MNT/linux_backup" "$USB_MNT/linux-backup" || true
sync
df -h "$USB_MNT"

DEST="$USB_MNT/$DEST_NAME"
mkdir -p "$DEST"
echo "==> copy pack parts → $DEST"
rsync -ah --info=progress2 "$PACK_DIR"/isaac_factory_train_data.tar.*.tarpart "$DEST/"
rsync -ah "$PACK_DIR"/README_RESTORE.md "$DEST/" 2>/dev/null || true
rsync -ah "$PACK_DIR"/MANIFEST.txt "$DEST/" 2>/dev/null || true
# also copy restore helper
cp -f "$REPO_ROOT/tools/restore_train_data_from_usb.sh" "$DEST/" 2>/dev/null || true
sync
chown -R "$REAL_USER:$REAL_USER" "$DEST" 2>/dev/null || true

echo "==> done"
df -h "$USB_MNT"
ls -lh "$DEST"
