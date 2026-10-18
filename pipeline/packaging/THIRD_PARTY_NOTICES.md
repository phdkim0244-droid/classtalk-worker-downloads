# ClassTalk Worker — third-party notices

The installer bundles the Python runtime and open-source libraries. The full
license texts for the exact build are included in `third-party-licenses.json`.
That file is generated from the installed distributions during every build.

Major components:

| Component | Upstream | License |
| --- | --- | --- |
| Python | https://www.python.org/ | PSF License |
| Tcl/Tk | https://www.tcl.tk/ | Tcl/Tk BSD-style license |
| faster-whisper | https://github.com/SYSTRAN/faster-whisper | MIT |
| MLX / MLX Metal | https://github.com/ml-explore/mlx | MIT |
| mlx-whisper | https://github.com/ml-explore/mlx-examples | MIT |
| tiktoken | https://github.com/openai/tiktoken | MIT |
| CTranslate2 | https://github.com/OpenNMT/CTranslate2 | MIT |
| OpenAI Whisper / converted large-v3 model | https://github.com/openai/whisper | MIT |
| Silero VAD | https://github.com/snakers4/silero-vad | MIT |
| Resemblyzer / pretrained voice encoder | https://github.com/resemble-ai/Resemblyzer | Apache-2.0 |
| PyTorch | https://github.com/pytorch/pytorch | BSD-3-Clause |
| PyAV | https://github.com/PyAV-Org/PyAV | BSD-3-Clause |
| FFmpeg libraries in PyAV wheels | https://ffmpeg.org/legal.html | LGPL-2.1-or-later; see bundled library notices |
| NumPy | https://numpy.org/ | BSD-3-Clause |
| scikit-learn | https://scikit-learn.org/ | BSD-3-Clause |
| librosa | https://librosa.org/ | ISC |
| ONNX Runtime | https://onnxruntime.ai/ | MIT |
| keyring | https://github.com/jaraco/keyring | MIT |
| requests | https://requests.readthedocs.io/ | Apache-2.0 |
| PyInstaller bootloader | https://pyinstaller.org/ | GPL-2.0-or-later with bootloader distribution exception |

The first model download comes from the public Systran/faster-whisper-large-v3
repository on Hugging Face (about 3.1 GB), pinned to revision
`edaa852ec7e145841d8ffdb056a99866b5f0a478`. The 128-band preprocessor config is
downloaded with the weights. It does not receive classroom recordings. The model
is cached locally for later runs. No Hugging Face account or token is required.

ClassTalk's app connection token is stored only in the macOS Keychain or Windows
Credential Manager. It is never embedded in this distribution. Disconnect in the
app or revoke the device from the ClassTalk website before disposing of a shared
computer. Removing the app does not remove cached models or OS credential items.

Apple Silicon GPU mode downloads `mlx-community/whisper-large-v3-mlx`, pinned to
`49e6aa286ad60c14352c404340ded53710378a11` (`config.json`, `weights.npz`, about 3.1 GB).
This download is public and does not upload audio. GPU and CPU model caches are separate.
