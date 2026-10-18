"""GUI entry point. PyInstaller includes Python, Tk and native STT dependencies."""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import webbrowser
from pathlib import Path

from desktop import PRODUCTION_ORIGIN, VERSION


def self_test(report_path: str | None) -> int:
    """Exercise bundled native libraries/data without network, credentials or a display."""
    report = {"version": VERSION, "platform": sys.platform, "ok": False}
    try:
        import importlib
        import tempfile
        import wave
        import tkinter
        import ctranslate2
        import numpy as np
        from faster_whisper.vad import get_speech_timestamps
        from resemblyzer import VoiceEncoder
        from app.transcribe.faster_whisper_diar_backend import _load_audio_16k_mono
        from sklearn.cluster import AgglomerativeClustering
        importlib.import_module("keyring.backends.macOS" if sys.platform == "darwin"
                                else "keyring.backends.Windows")
        assert tkinter.Tcl().eval("info patchlevel")
        assert ctranslate2.get_supported_compute_types("cpu")
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "smoke.wav"
            with wave.open(str(audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\0\0" * 16000)
            decoded = _load_audio_16k_mono(str(audio))
            assert len(decoded) == 16000
            assert get_speech_timestamps(decoded) == []
        encoder = VoiceEncoder(device="cpu", verbose=False)
        embedding = encoder.embed_utterance(np.zeros(16000, dtype=np.float32))
        assert embedding.shape == (256,)
        assert len(AgglomerativeClustering(n_clusters=2).fit_predict(
            np.array([[0., 0.], [0.1, 0.], [10., 10.]]))) == 3
        from app.transcribe.short_voice_verifier import ShortVoiceEncoder, bundled_model, valid_model
        model = bundled_model()
        if model is not None:
            assert valid_model(model), "Bundled short-speaker model is missing or corrupt"
            verifier = ShortVoiceEncoder(model)
            # Exercise native inference (not voice accuracy) with deterministic nonzero PCM.
            test_pcm = np.random.default_rng(17).normal(0, 0.01, 32000).astype(np.float32)
            assert verifier.embed(test_pcm, 0, 2000).shape == (192,)
            report["shortSpeakerModel"] = "bundled CPU inference passed"
        report.update(ok=True, checks=["tk", "keyring", "ctranslate2", "audio_decode",
                                      "vad_model", "speaker_model", "clustering"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if report_path:
        Path(report_path).write_text(output, encoding="utf-8")
    if sys.stdout:
        print(output)
    return 0 if report["ok"] else 1


def model_smoke(audio_path: str, report_path: str | None, regressions_path: str | None = None,
                cache_path: str | None = None, acceleration: str = "auto") -> int:
    """Explicit CI opt-in: download the public model and transcribe synthetic speech.

    This never constructs an API client, accesses a credential store or submits audio.
    """
    import tempfile
    import time
    import faulthandler
    from contextlib import nullcontext
    report = {"version": VERSION, "platform": sys.platform, "ok": False}
    started = time.monotonic()
    stack_log = None
    def progress(stage):
        report.update(stage=stage, elapsedSeconds=round(time.monotonic() - started, 1))
        if report_path:
            Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        progress("starting")
        if report_path:
            stack_log = Path(report_path).with_suffix(".stacks.txt").open("w", encoding="utf-8")
            faulthandler.dump_traceback_later(60, repeat=True, file=stack_log)
        from desktop.engine import Engine, MODEL_REVISIONS
        with (nullcontext(cache_path) if cache_path else tempfile.TemporaryDirectory(prefix="classtalk-model-smoke-")) as cache:
            engine = Engine(Path(cache), acceleration=acceleration)
            engine.prepare(progress)
            progress("transcribing_synthetic_audio")
            segments = engine.transcribe(Path(audio_path), {"groupNo": 1, "students": 1},
                                         {"sessionType": "class", "studentsPerGroup": 1})
            progress("checking_transcript")
            text = " ".join(segment["text"] for segment in segments)
            assert segments and any(word in text for word in ("음성", "기능", "수업", "실제")), "Synthetic Korean speech was not recognized"
            assert all(segment["speaker"] == "학생1" and segment["endMs"] >= segment["startMs"]
                       for segment in segments), "Speaker or timestamp mapping failed"
            checks = ["public_model_download", "whisper_transcription", "korean_text", "speaker_labels"]
            if regressions_path:
                from desktop.model_checks import check_model_regressions
                progress("checking_synthetic_regressions")
                report["regressions"] = check_model_regressions(engine, Path(regressions_path))
                checks.append("synthetic_numeric_regression")
            report.update(ok=True, model=engine.model_size, segmentCount=len(segments),
                          backend=engine.backend, checks=checks)
            if engine.backend == "mlx":
                from desktop.acceleration import MLX_REVISION
                report["modelRevision"] = MLX_REVISION
            else:
                report["modelRevision"] = MODEL_REVISIONS[engine.model_size]
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        faulthandler.cancel_dump_traceback_later()
        if stack_log:
            stack_log.close()
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if report_path:
        Path(report_path).write_text(output, encoding="utf-8")
    if sys.stdout:
        print(output)
    return 0 if report["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="ClassTalk 전사 워커")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preview-ui", action="store_true", help="계정·모델 없이 화면만 미리 보기")
    parser.add_argument("--self-test-report")
    parser.add_argument("--model-smoke-audio", help="CI 전용: 공개 모델을 내려받고 한국어 합성 음성 파일을 전사")
    parser.add_argument("--model-smoke-regressions", help="CI 전용: 추가 합성 음성·기대값 manifest (전사 후 검사)")
    parser.add_argument("--model-smoke-cache", help="검증 전용: 공개 모델 캐시 재사용 경로")
    parser.add_argument("--acceleration", choices=("auto", "cpu"), default="auto", help="검증 전용: 로컬 처리 방식")
    parser.add_argument("--dev-server", help="개발 전용: http(s)://localhost:<port>")
    args = parser.parse_args()
    if args.model_smoke_regressions and not args.model_smoke_audio:
        parser.error("--model-smoke-regressions requires --model-smoke-audio")
    if args.self_test:
        return self_test(args.self_test_report)
    if args.model_smoke_audio:
        return model_smoke(args.model_smoke_audio, args.self_test_report, args.model_smoke_regressions,
                           args.model_smoke_cache, args.acceleration)

    import tkinter as tk
    from tkinter import messagebox
    from desktop.ui import WorkerView, preview
    if args.preview_ui:
        preview(tk.Tk())
        return 0
    from platformdirs import user_cache_path
    from desktop.client import Api, WorkerError
    from desktop.controller import Controller
    from desktop.engine import Engine
    from desktop.instance import InstanceLock
    from desktop.secure_store import TokenStore

    root = tk.Tk()
    view = WorkerView(root)
    status, pair_code = view.status, view.pair_code
    start_button, stop_button = view.start_button, view.stop_button
    disconnect_button, browser_button = view.disconnect_button, view.browser_button
    stopping = False
    events: queue.Queue = queue.Queue()
    controller = None
    verification = {"url": None}
    instance = None

    try:
        api = Api(args.dev_server or PRODUCTION_ORIGIN, allow_local=bool(args.dev_server))
        instance = InstanceLock(user_cache_path("ClassTalkWorker"), api.origin)
        if not instance.acquire():
            messagebox.showinfo("이미 실행 중입니다", "ClassTalk 전사 워커가 이미 실행 중입니다. 열린 앱을 확인해 주세요.", parent=root)
            root.destroy()
            return 0
        store = TokenStore(api.origin)
        controller = Controller(api, store, Engine(user_cache_path("ClassTalkWorker") / "models"),
                                lambda event, value: events.put((event, value)))
    except (ValueError, WorkerError, OSError) as exc:
        messagebox.showerror("워커를 시작할 수 없습니다", str(exc), parent=root)
        if instance:
            instance.release()
        root.destroy()
        return 1

    def start():
        nonlocal stopping
        stopping = False
        view.clear_progress()
        controller.engine.acceleration = "cpu" if view.acceleration.get().startswith("CPU") else "auto"
        view.acceleration_select.configure(state="disabled")
        start_button.configure(state="disabled")
        disconnect_button.configure(state="disabled")
        stop_button.configure(state="normal")
        status.set("보안 저장소와 계정 연결을 확인하는 중…")
        controller.start()

    def stop():
        nonlocal stopping
        stopping = True
        view.clear_progress()
        controller.stop()
        stop_button.configure(state="disabled")
        status.set("중지 요청됨 · 새 작업을 받지 않습니다. 현재 음성 구간 처리가 끝날 때까지 잠시 기다려 주세요.")

    def disconnect():
        start_button.configure(state="disabled")
        disconnect_button.configure(state="disabled")
        status.set("계정 연결을 해제하는 중…")
        def background():
            try:
                controller.disconnect()
            except WorkerError as exc:
                events.put(("error", str(exc)))
            finally:
                events.put(("stopped", None))
        threading.Thread(target=background, daemon=True).start()

    start_button.configure(command=start)
    stop_button.configure(command=stop)
    disconnect_button.configure(command=disconnect)
    browser_button.configure(command=lambda: webbrowser.open(
        verification["url"] or f"{api.origin}/worker"))

    def drain():
        try:
            while True:
                event, value = events.get_nowait()
                if event == "progress":
                    if not stopping:
                        view.set_progress(value)
                elif event in ("status", "error"):
                    view.clear_progress()
                    status.set(value)
                elif event == "pair":
                    verification["url"] = value["url"]
                    pair_code.set(f"연결 코드: {value['code']} · 10분 안에 브라우저에서 승인해 주세요.")
                    status.set("브라우저에서 로그인 후 이 기기 연결을 승인하세요.")
                    browser_button.configure(text="계정 연결 화면 열기")
                elif event == "connected":
                    pair_code.set("본인 계정 연결됨 · 이 기기에 허용된 수업만 처리합니다.")
                    verification["url"] = None
                    browser_button.configure(text="ClassTalk 열기")
                elif event == "disconnected":
                    pair_code.set("")
                    status.set("계정 연결을 해제했습니다.")
                elif event == "stopped":
                    view.clear_progress()
                    view.acceleration_select.configure(state="readonly")
                    start_button.configure(state="normal")
                    stop_button.configure(state="disabled")
                    disconnect_button.configure(state="normal")
        except queue.Empty:
            pass
        root.after(150, drain)

    def close():
        controller.stop()
        # In-flight jobs expire server-side; they cannot be submitted after cancel.
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    root.after(150, drain)
    try:
        root.mainloop()
    finally:
        instance.release()
    return 0


if __name__ == "__main__":
    # PyInstaller overrides this to handle frozen subprocess entry points.
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
