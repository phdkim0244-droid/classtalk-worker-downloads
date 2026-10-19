#define AppVersion "0.2.4"
[Setup]
AppId={{F83654D2-FA72-447A-B51C-6C46E25A8492}
AppName=ClassTalk Worker
AppVersion={#AppVersion}
AppPublisher=ClassTalk
AppPublisherURL=https://classtalk-web-three.vercel.app
DefaultDirName={localappdata}\Programs\ClassTalk Worker
DefaultGroupName=ClassTalk Worker
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installers
OutputBaseFilename=ClassTalk-Worker-{#AppVersion}-windows-x64-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
UninstallDisplayIcon={app}\ClassTalkWorker.exe
LicenseFile=THIRD_PARTY_NOTICES.md

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked

[Files]
Source: "..\dist\ClassTalkWorker\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\ClassTalk Worker"; Filename: "{app}\ClassTalkWorker.exe"
Name: "{autodesktop}\ClassTalk Worker"; Filename: "{app}\ClassTalkWorker.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\ClassTalkWorker.exe"; Description: "Start ClassTalk Worker"; Flags: nowait postinstall skipifsilent
