# Copy the Chocolatey (UB Mannheim) Tesseract build -- exe + DLLs -- next to the fetched
# tessdata so PyInstaller bundles it under resources/tesseract. Run from apps/desktop.
$ErrorActionPreference = "Stop"
$src = "C:\Program Files\Tesseract-OCR"
$dst = "resources\tesseract"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item "$src\tesseract.exe" $dst
Copy-Item "$src\*.dll" $dst
& "$dst\tesseract.exe" --version
