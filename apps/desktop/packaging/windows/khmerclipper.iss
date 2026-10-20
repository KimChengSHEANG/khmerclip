; Per-user installer (no admin). Build from apps/desktop after PyInstaller:
;   $env:KHMERCLIPPER_VERSION = "0.1.0"; & "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" packaging\windows\khmerclipper.iss
#define AppVersion GetEnv("KHMERCLIPPER_VERSION")
#if AppVersion == ""
  #error KHMERCLIPPER_VERSION is not set
#endif

[Setup]
AppId={{10873922-A408-43F4-BB58-D7E365530FBF}
AppName=KhmerClipper
AppVersion={#AppVersion}
AppPublisher=KhmerClipper
DefaultDirName={localappdata}\Programs\KhmerClipper
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\..\dist
OutputBaseFilename=KhmerClipper-{#AppVersion}-windows-x64-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
UninstallDisplayIcon={app}\KhmerClipper.exe

[Files]
Source: "..\..\dist\KhmerClipper\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{userprograms}\KhmerClipper"; Filename: "{app}\KhmerClipper.exe"

[Registry]
; The app writes this value when "Launch at login" is on; remove it on uninstall.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueName: "KhmerClipper"; ValueType: none; Flags: uninsdeletevalue dontcreatekey

[Run]
Filename: "{app}\KhmerClipper.exe"; Description: "Launch KhmerClipper"; Flags: nowait postinstall skipifsilent
