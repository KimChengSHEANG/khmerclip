#!/usr/bin/env bash
# Package dist/KhmerClip (PyInstaller onedir) + the system Tesseract into an AppImage.
# Usage, from apps/desktop after pyinstaller:  packaging/linux/build_appimage.sh <version>
set -euo pipefail
VERSION="$1"
APPDIR="build/AppDir"
TOOLS="build/tools"
OUT="dist/KhmerClip-${VERSION}-x86_64.AppImage"
export APPIMAGE_EXTRACT_AND_RUN=1  # CI runners have no FUSE

mkdir -p "$TOOLS"
fetch() { [ -x "$TOOLS/$1" ] || { curl -fsSL -o "$TOOLS/$1" "$2"; chmod +x "$TOOLS/$1"; }; }
fetch linuxdeploy https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
fetch appimagetool https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage

rm -rf "$APPDIR"
cp resources/icon.png build/khmerclip.png
# 1) Tesseract and its shared libs, rpath-patched by linuxdeploy. Only tesseract goes through
#    linuxdeploy: patching PyInstaller's binaries would corrupt its appended archive.
#    The desktop file and icon are placed by hand: linuxdeploy insists that Exec=KhmerClip
#    exists in usr/bin, but the app lives in usr/lib/khmerclip and starts through AppRun.
"$TOOLS/linuxdeploy" --appdir "$APPDIR" --executable "$(command -v tesseract)"
install -Dm644 packaging/linux/khmerclip.desktop "$APPDIR/usr/share/applications/khmerclip.desktop"
install -Dm644 build/khmerclip.png "$APPDIR/usr/share/icons/hicolor/256x256/apps/khmerclip.png"
cp packaging/linux/khmerclip.desktop "$APPDIR/khmerclip.desktop"
cp build/khmerclip.png "$APPDIR/khmerclip.png"
# 2) The frozen app, copied untouched.
mkdir -p "$APPDIR/usr/lib"
cp -a dist/KhmerClip "$APPDIR/usr/lib/khmerclip"
rm -f "$APPDIR/AppRun"
install -m 755 packaging/linux/AppRun "$APPDIR/AppRun"
ARCH=x86_64 "$TOOLS/appimagetool" "$APPDIR" "$OUT"
echo "built $OUT"
