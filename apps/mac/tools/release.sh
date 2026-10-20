#!/usr/bin/env bash
# Build a distributable KhmerClipper-<version>.dmg (Tesseract + TrOCR bundled).
#
#   SIGN_ID="Developer ID Application: Name (TEAMID)" NOTARY_PROFILE=khmerclipper tools/release.sh
#
# SIGN_ID defaults to "-" (ad-hoc: runs only after the user clears Gatekeeper). With a real
# Developer ID it signs every nested Mach-O with the hardened runtime and a secure timestamp,
# then, if NOTARY_PROFILE is set, notarizes and staples the dmg. Create the profile once with
#   xcrun notarytool store-credentials khmerclipper --apple-id <id> --team-id <TEAMID>
# (it prompts for an app-specific password and keeps it in the keychain).
set -euo pipefail
cd "$(dirname "$0")/.."

SIGN_ID=${SIGN_ID:--}
APP=build/Build/Products/Release/KhmerClipper.app

make gen
xcodebuild -project KhmerClipper.xcodeproj -scheme KhmerClipper -configuration Release \
  -derivedDataPath build CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO build | tail -1

sign=(codesign --force --sign "$SIGN_ID" --options runtime)
[ "$SIGN_ID" = "-" ] || sign+=(--timestamp)

# Inside-out: loose Mach-O files under Resources (the vendored tesseract + its dylibs) are
# not bundles, so --deep would miss them, and notarization rejects unsigned code.
find "$APP/Contents/Resources" -type f -print0 | while IFS= read -r -d '' f; do
  if file -b "$f" | grep -q Mach-O; then "${sign[@]}" "$f"; fi
done
if [ -d "$APP/Contents/Frameworks" ]; then
  find "$APP/Contents/Frameworks" -maxdepth 1 \( -name '*.framework' -o -name '*.dylib' \) -print0 |
    while IFS= read -r -d '' f; do "${sign[@]}" "$f"; done
fi
"${sign[@]}" --entitlements Resources/KhmerClipper.entitlements "$APP"
codesign --verify --deep --strict "$APP"

VERSION=$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "$APP/Contents/Info.plist")
DMG=dist/KhmerClipper-$VERSION.dmg
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
ditto "$APP" "$STAGE/KhmerClipper.app"
ln -s /Applications "$STAGE/Applications"
mkdir -p dist
rm -f "$DMG"
hdiutil create -volname KhmerClipper -srcfolder "$STAGE" -fs HFS+ -format UDZO -ov "$DMG" >/dev/null
[ "$SIGN_ID" = "-" ] || codesign --force --sign "$SIGN_ID" --timestamp "$DMG"

if [ "$SIGN_ID" != "-" ] && [ -n "${NOTARY_PROFILE:-}" ]; then
  xcrun notarytool submit "$DMG" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$DMG"
  spctl --assess --type open --context context:primary-signature -v "$DMG"
fi
ls -lh "$DMG"
