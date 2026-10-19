# Run from pipeline/: python -m PyInstaller --clean --noconfirm packaging/worker.spec
import os
import sys
import platform
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
sys.path.insert(0, str(root))
from app.transcribe.short_voice_verifier import prepare_model
short_voice_model = prepare_model(root / "build" / "speaker-models")
datas = [(str(short_voice_model), "speaker-models")]
binaries = []
hiddenimports = ["tkinter", "keyring.backends.macOS", "keyring.backends.Windows"]
for package in ("faster_whisper", "resemblyzer", "librosa", "certifi"):
    datas += collect_data_files(package)
for package in ("ctranslate2", "av", "onnxruntime", "sherpa_onnx"):
    binaries += collect_dynamic_libs(package)
hiddenimports += collect_submodules("sklearn")
hiddenimports += collect_submodules("librosa")
hiddenimports += collect_submodules("resemblyzer")
hiddenimports += collect_submodules("sherpa_onnx")
for package in ("keyring", "platformdirs", "requests", "faster-whisper", "resemblyzer", "torch", "sherpa-onnx", "sherpa-onnx-core"):
    datas += copy_metadata(package)
datas += [(str(root / "packaging" / "THIRD_PARTY_NOTICES.md"), "."),
          (str(root / "packaging" / "CAMPPLUS-LICENSE.txt"), ".")]
if sys.platform == "darwin" and platform.machine() == "arm64":
    # MLX kernels and Whisper tokenizer/mel assets must work without Python,
    # ffmpeg, or a model hub request at inference time.
    for package in ("mlx", "mlx_whisper", "tiktoken"):
        datas += collect_data_files(package)
        binaries += collect_dynamic_libs(package)
        hiddenimports += collect_submodules(package)
    hiddenimports += collect_submodules("tiktoken_ext")
    for package in ("mlx", "mlx-metal", "mlx-whisper", "tiktoken"):
        datas += copy_metadata(package)
if (root / "packaging" / "third-party-licenses.json").exists():
    datas += [(str(root / "packaging" / "third-party-licenses.json"), ".")]

a = Analysis(
    [str(root / "desktop" / "main.py")], pathex=[str(root)], binaries=binaries,
    datas=datas, hiddenimports=hiddenimports,
    excludes=["app.config", "app.supabase_io", "app.pipeline_core", "app.worker", "app.run",
              "dotenv", "pytest", "IPython", "matplotlib", "tensorboard", "torchvision", "torchaudio"],
    noarchive=False,
)
if sys.platform == "win32":
    # CTranslate2 and PyTorch wheels both ship Intel OpenMP. Loading copies from
    # separate vendor directories can hang when both inference engines run.
    # Resolve every consumer against one runtime in PyInstaller's DLL directory.
    openmp = [item for item in a.binaries if Path(item[0]).name.lower() == "libiomp5md.dll"]
    if openmp:
        selected = next((item for item in openmp if "ctranslate2" in item[0]), openmp[0])
        a.binaries = [item for item in a.binaries if Path(item[0]).name.lower() != "libiomp5md.dll"]
        a.binaries.append(("libiomp5md.dll", selected[1], "BINARY"))
        print(f"Bundling one Intel OpenMP runtime instead of {len(openmp)} vendor copies")
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="ClassTalkWorker",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False,
    codesign_identity=os.environ.get("MACOS_SIGN_IDENTITY") or None,
    entitlements_file=str(root / "packaging" / "entitlements.plist") if sys.platform == "darwin" else None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="ClassTalkWorker")
if sys.platform == "darwin":
    app = BUNDLE(
        coll, name="ClassTalk Worker.app", bundle_identifier="kr.ydaasam.classtalk.worker",
        info_plist={"CFBundleShortVersionString": "0.2.4", "CFBundleVersion": "6",
                    "LSMinimumSystemVersion": "13.0", "NSHighResolutionCapable": True},
    )
