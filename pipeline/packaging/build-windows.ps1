$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
# Resemblyzer's obsolete Python 2 backport conflicts with PyInstaller.
python -m pip uninstall --yes typing
if ($LASTEXITCODE -ne 0) { throw "Legacy typing cleanup failed" }
python -m piplicenses --with-license-file --with-notice-file --format=json --output-file=packaging/third-party-licenses.json
if ($LASTEXITCODE -ne 0) { throw "License inventory failed" }
python -m PyInstaller --clean --noconfirm packaging/worker.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
$report = Join-Path (Get-Location) "dist/self-test.json"
python packaging/run-check.py --executable "dist/ClassTalkWorker/ClassTalkWorker.exe" --report $report
if ($LASTEXITCODE -ne 0) { throw "Packaged smoke test failed" }
$iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if ($iscc) { $compiler = $iscc.Source } else { $compiler = "${env:ProgramFiles(x86)}/Inno Setup 6/ISCC.exe" }
& $compiler packaging/windows-installer.iss
if ($LASTEXITCODE -ne 0) { throw "Installer compilation failed" }
# Install the actual distribution into a disposable per-user directory, then test it.
if ($env:RUNNER_TEMP) { $testDir = Join-Path $env:RUNNER_TEMP "classtalk-installer-smoke" }
else { $testDir = Join-Path $env:TEMP "classtalk-installer-smoke" }
$installer = "dist/installers/ClassTalk-Worker-0.2.3-windows-x64-Setup.exe"
$install = Start-Process -FilePath $installer -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/DIR=`"$testDir`"" -PassThru
if (!$install.WaitForExit(600000)) { Stop-Process -Id $install.Id -Force; throw "Installer timed out after 10 minutes" }
if ($install.ExitCode -ne 0) { throw "Installer execution failed" }
python packaging/run-check.py --executable (Join-Path $testDir "ClassTalkWorker.exe") --report $report
if ($LASTEXITCODE -ne 0) { throw "Installed application smoke test failed" }
$modelReport = Join-Path (Get-Location) "dist/model-smoke.json"
$syntheticAudio = Join-Path (Get-Location) "desktop/tests/fixtures/synthetic-ko.wav"
python packaging/run-check.py --executable (Join-Path $testDir "ClassTalkWorker.exe") --audio $syntheticAudio --regressions desktop/tests/fixtures/model-regressions.json --report $modelReport
if ($LASTEXITCODE -ne 0) { throw "Installed application Korean transcription test failed" }
$uninstall = Start-Process -FilePath (Join-Path $testDir "unins000.exe") -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" -PassThru
if (!$uninstall.WaitForExit(120000)) { Stop-Process -Id $uninstall.Id -Force; throw "Uninstaller timed out" }
if ($uninstall.ExitCode -ne 0) { throw "Uninstaller failed" }
