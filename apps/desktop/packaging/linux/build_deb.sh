#!/usr/bin/env bash
# Package the AppDir that build_appimage.sh assembled (frozen app + bundled tesseract) as a .deb
# for Ubuntu / Debian: AppImages need libfuse2, which Ubuntu 22.04+ doesn't install by default.
# Usage, from apps/desktop after build_appimage.sh:  packaging/linux/build_deb.sh <version>
set -euo pipefail
VERSION="$1"
APPDIR="build/AppDir"
ROOT="build/deb"
OUT="dist/khmerclip_${VERSION}_amd64.deb"
[ -x "$APPDIR/AppRun" ] || { echo "run packaging/linux/build_appimage.sh first"; exit 1; }

rm -rf "$ROOT"
mkdir -p "$ROOT/opt" "$ROOT/usr/bin" "$ROOT/DEBIAN"
cp -a "$APPDIR" "$ROOT/opt/khmerclip"
rm -f "$ROOT/opt/khmerclip/khmerclip.desktop" "$ROOT/opt/khmerclip/khmerclip.png" "$ROOT/opt/khmerclip/.DirIcon"
ln -s /opt/khmerclip/AppRun "$ROOT/usr/bin/khmerclip"  # AppRun resolves its own path via readlink -f
install -Dm644 "$APPDIR/usr/share/icons/hicolor/256x256/apps/khmerclip.png" \
  "$ROOT/usr/share/icons/hicolor/256x256/apps/khmerclip.png"
mkdir -p "$ROOT/usr/share/applications"
sed 's|^Exec=.*|Exec=khmerclip|' packaging/linux/khmerclip.desktop > "$ROOT/usr/share/applications/khmerclip.desktop"
rm -rf "$ROOT/opt/khmerclip/usr/share/applications" "$ROOT/opt/khmerclip/usr/share/icons"

# Qt's xcb platform plugin loads these from the system (same list the CI build installs).
cat > "$ROOT/DEBIAN/control" <<CONTROL
Package: khmerclip
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: amd64
Depends: libegl1, libgl1, libxkbcommon0, libxkbcommon-x11-0, libxcb-cursor0, libxcb-icccm4, libxcb-keysyms1, libxcb-shape0, libdbus-1-3, libfontconfig1
Installed-Size: $(du -sk "$ROOT" | cut -f1)
Maintainer: Kim Cheng SHEANG <cheng900@gmail.com>
Homepage: https://github.com/KimChengSHEANG/khmerclip
Description: Khmer OCR screen capture
 Select any part of the screen and KhmerClip reads the Khmer (and English)
 text in it and copies it, ready to paste. Bundles Tesseract; the TrOCR
 model downloads on first use.
CONTROL
dpkg-deb --root-owner-group --build "$ROOT" "$OUT"
echo "built $OUT"
