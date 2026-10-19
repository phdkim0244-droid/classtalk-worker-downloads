#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Resemblyzer's legacy dependency is redundant on Python 3 and blocks PyInstaller.
python -m pip uninstall --yes typing
python -m piplicenses --with-license-file --with-notice-file --format=json > packaging/third-party-licenses.json
python -m PyInstaller --clean --noconfirm packaging/worker.spec
mkdir -p dist/installers
python packaging/run-check.py --executable "dist/ClassTalk Worker.app/Contents/MacOS/ClassTalkWorker" --report dist/self-test.json
python packaging/run-check.py --executable "dist/ClassTalk Worker.app/Contents/MacOS/ClassTalkWorker" --audio desktop/tests/fixtures/synthetic-ko.wav --regressions desktop/tests/fixtures/model-regressions.json --report dist/model-smoke.json
task_arch="$(uname -m)"
if [ "$task_arch" = "x86_64" ]; then task_arch=x64; fi
task_image="dist/installers/ClassTalk-Worker-0.2.4-macos-${task_arch}.dmg"
task_stage="$(mktemp -d)"
trap 'rm -rf "$task_stage"' EXIT
ditto "dist/ClassTalk Worker.app" "$task_stage/ClassTalk Worker.app"
ln -s /Applications "$task_stage/Applications"
cp packaging/THIRD_PARTY_NOTICES.md "$task_stage/"
python packaging/create-dmg.py --source "$task_stage" --output "$task_image"
if [ -n "${MACOS_SIGN_IDENTITY:-}" ]; then
  codesign --force --sign "$MACOS_SIGN_IDENTITY" "$task_image"
fi
if [ -n "${MACOS_NOTARY_PROFILE:-}" ]; then
  xcrun notarytool submit "$task_image" --keychain-profile "$MACOS_NOTARY_PROFILE" --wait
  xcrun stapler staple "$task_image"
fi
