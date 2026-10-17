; Per-user installer (no admin). Build from apps/desktop after PyInstaller:
;   $env:KHMERCLIP_VERSION = "0.1.0"; & "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" packaging\windows\khmerclip.iss
#define AppVersion GetEnv("KHMERCLIP_VERSION")
#if AppVersion == ""
  #error KHMERCLIP_VERSION is not set
#endif

[Setup]
AppId={{6E0B3C52-6C4E-4C1B-9C7B-6B2E7A4F1D10}
AppName=KhmerClip
AppVersion={#AppVersion}
AppPublisher=KhmerClip
DefaultDirName={localappdata}\Programs\KhmerClip
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\..\dist
OutputBaseFilename=KhmerClip-{#AppVersion}-windows-x64-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
UninstallDisplayIcon={app}\KhmerClip.exe

[Files]
Source: "..\..\dist\KhmerClip\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{userprograms}\KhmerClip"; Filename: "{app}\KhmerClip.exe"

[Registry]
; The app writes this value when "Launch at login" is on; remove it on uninstall.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueName: "KhmerClip"; ValueType: none; Flags: uninsdeletevalue dontcreatekey

[Run]
Filename: "{app}\KhmerClip.exe"; Description: "Launch KhmerClip"; Flags: nowait postinstall skipifsilent
