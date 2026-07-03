# Stable signing identity keeps the Accessibility (TCC) grant across rebuilds.
SIGN_ID = 
APP      = build/Build/Products/Debug/Netra.app

.PHONY: gen build test run clean

gen:
	xcodegen generate

test: gen
	xcodebuild -project Netra.xcodeproj -scheme NetraTests -configuration Debug \
		-derivedDataPath build \
		CODE_SIGN_STYLE=Manual \
		CODE_SIGN_IDENTITY="$(SIGN_ID)" \
		PROVISIONING_PROFILE_SPECIFIER="" DEVELOPMENT_TEAM="" \
		test

build: gen
	xcodebuild -project Netra.xcodeproj -scheme Netra -configuration Debug \
		-derivedDataPath build \
		CODE_SIGN_STYLE=Manual \
		CODE_SIGN_IDENTITY="$(SIGN_ID)" \
		PROVISIONING_PROFILE_SPECIFIER="" DEVELOPMENT_TEAM="" \
		build

run: build
	@pkill -x Netra 2>/dev/null || true
	@while pgrep -x Netra >/dev/null; do sleep 0.1; done
	@/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$(APP)"
	@for i in 1 2 3 4 5; do \
		open "$(APP)" && break || { echo "open failed (LaunchServices not ready), retrying..."; sleep 0.5; }; \
	done

clean:
	rm -rf build Netra.xcodeproj
