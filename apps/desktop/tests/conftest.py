import os

# Must be set before any Qt import so tests run headless on every OS and in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
