; Qwen3-TTS Rust Windows Installer
; Built with Inno Setup 6 (https://jrsoftware.org/isinfo.php)

#define MyAppName "Qwen3-TTS Rust"
#define MyAppVersion "0.2.0"
#define MyAppPublisher "qwen3tts-rs"
#define MyAppURL "https://github.com/stevenke1981/qwen3tts-rs"

[Setup]
AppId={{A7E3F2B1-9C4D-4E5F-8A6B-1D2C3E4F5A6B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir=..\dist
OutputBaseFilename=qwen3tts-rs-{#MyAppVersion}-setup
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
PrivilegesRequired=lowest
SetupIconFile=
UninstallDisplayIcon={app}\qwen3tts-gui.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "tradchinese"; MessagesFile: "compiler:Languages\TraditionalChinese.isl"

[Files]
; Release binaries
Source: "..\target\release\qwen3tts-gui.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\target\release\convert-gguf.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\target\release\examples\synthesize.exe"; DestDir: "{app}"; DestName: "qwen3tts-synthesize.exe"; Flags: ignoreversion

; Installer scripts
Source: "download_weights.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "qwen3tts-env.cmd"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Qwen3-TTS GUI"; Filename: "{app}\qwen3tts-gui.exe"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\qwen3tts-gui.exe"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[Code]
// Download model weights after installation
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  ScriptPath: String;
begin
  if CurStep = ssPostInstall then
  begin
    ScriptPath := ExpandConstant('{app}\download_weights.ps1');
    if MsgBox('是否要下載 Qwen3-TTS 模型權重？（約 1.8 GB，需要網路連線）', mbConfirmation, MB_YESNO) = IDYES then
    begin
      Exec('powershell.exe', '-NoProfile -ExecutionPolicy Bypass -File "' + ScriptPath + '" -InstallDir "' + ExpandConstant('{app}') + '"', '', SW_SHOW, ewWaitUntilTerminated, ResultCode);
      if ResultCode <> 0 then
        MsgBox('權重下載失敗。您可以稍後手動執行 download_weights.ps1。', mbError, MB_OK);
    end;
  end;
end;
