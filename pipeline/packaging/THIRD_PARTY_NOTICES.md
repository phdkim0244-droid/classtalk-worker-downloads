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
| sherpa-onnx / sherpa-onnx-core | https://github.com/k2-fsa/sherpa-onnx | Apache-2.0 |
| 3D-Speaker CAM++ Chinese/English advanced model | https://github.com/modelscope/3D-Speaker | Apache-2.0 |
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

Short-reply verification also bundles the public CAM++ ONNX model from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
(`3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx`, 28,281,164 bytes).
SHA-256: `aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2`.
The model runs locally with sherpa-onnx on CPU; classroom audio is not uploaded.
Upstream model family: https://huggingface.co/funasr/campplus (Apache-2.0).
The complete Apache-2.0 license accompanies this model in `CAMPPLUS-LICENSE.txt`.
