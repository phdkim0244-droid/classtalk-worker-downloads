"""Explicit colours on a portable ttk theme, including macOS dark appearance."""
from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk, font

from desktop import VERSION
from desktop.progress import Progress, clock_text

BACKGROUND = "#f5f2eb"
SURFACE = "#ffffff"
TEXT = "#203b35"
SECONDARY = "#52645e"
ACCENT = "#216451"


class WorkerView:
    def __init__(self, root):
        root.title("ClassTalk 전사 워커")
        root.geometry("740x790")
        root.minsize(640, 760)
        root.configure(background=BACKGROUND)
        # Aqua/Windows native themes can ignore configured backgrounds while
        # retaining our foreground, producing dark-on-dark labels. Clam paints
        # both colours consistently even when the OS appearance changes.
        style = ttk.Style(root)
        style.theme_use("clam")
        self.body_font = font.nametofont("TkDefaultFont").copy()
        self.body_font.configure(size=13)
        self.title_font = self.body_font.copy()
        self.title_font.configure(size=28, weight="bold")
        self.status_font = self.body_font.copy()
        self.status_font.configure(size=16, weight="bold")
        style.configure("TFrame", background=BACKGROUND)
        style.configure("TLabel", background=BACKGROUND, foreground=TEXT, font=self.body_font)
        style.configure("Card.TFrame", background=SURFACE)
        style.configure("Card.TLabel", background=SURFACE, foreground=TEXT)
        style.configure("Secondary.TLabel", foreground=SECONDARY)
        style.configure("Detail.TLabel", background=SURFACE, foreground=SECONDARY)
        style.configure("TButton", background=SURFACE, foreground=TEXT,
                        padding=(14, 11), font=self.body_font, borderwidth=1)
        style.map("TButton", background=[("disabled", "#e4e7e3"), ("pressed", "#cbded5"),
                                         ("active", "#e3eee7")],
                  foreground=[("disabled", "#55615b"), ("!disabled", TEXT)])
        style.configure("Primary.TButton", background=ACCENT, foreground="#ffffff")
        style.map("Primary.TButton", background=[("disabled", "#e4e7e3"),
                                                  ("pressed", "#164536"), ("active", "#19513f")],
                  foreground=[("disabled", "#55615b"), ("!disabled", "#ffffff")])
        style.configure("Horizontal.TProgressbar", background=ACCENT,
                        troughcolor="#e1e8e2", bordercolor="#e1e8e2",
                        lightcolor=ACCENT, darkcolor=ACCENT)
        style.configure("TCombobox", fieldbackground=SURFACE, background=SURFACE,
                        foreground=TEXT, font=self.body_font)
        style.map("TCombobox", fieldbackground=[("readonly", SURFACE), ("disabled", "#e4e7e3")],
                  foreground=[("disabled", SECONDARY), ("readonly", TEXT)],
                  selectbackground=[("readonly", SURFACE)], selectforeground=[("readonly", TEXT)])
        root.option_add("*TCombobox*Listbox.background", SURFACE)
        root.option_add("*TCombobox*Listbox.foreground", TEXT)

        content = ttk.Frame(root, padding=28)
        content.pack(fill="both", expand=True)
        self.wrapped = []

        def label(parent, **kwargs):
            widget = ttk.Label(parent, wraplength=570, justify="left", **kwargs)
            self.wrapped.append(widget)
            return widget

        label(content, text="ClassTalk", font=self.title_font).pack(anchor="w")
        label(content, text="내 수업을 위한 전사 워커").pack(anchor="w", pady=(2, 16))
        label(content, style="Secondary.TLabel", text=(
            "본인 계정에 연결한 수업만 이 컴퓨터에서 전사합니다.\n"
            "웹에서 ‘전사 시작’을 누른 뒤 앱과 컴퓨터를 켜 두세요."
        )).pack(anchor="w", pady=(0, 16))

        card = ttk.Frame(content, style="Card.TFrame", padding=20)
        card.pack(fill="x")
        self.status = tk.StringVar(value="계정을 연결하고 전사를 시작하세요.")
        label(card, textvariable=self.status, style="Card.TLabel",
              font=self.status_font).pack(anchor="w")
        self.context = tk.StringVar()
        label(card, textvariable=self.context, style="Detail.TLabel").pack(anchor="w", pady=(8, 0))
        self.bar = ttk.Progressbar(card, maximum=100)
        self.bar.pack(fill="x", pady=12)
        self.detail = tk.StringVar(value="처음에는 약 3.1GB 모델을 받습니다. 다음부터는 저장된 모델을 사용합니다.")
        label(card, textvariable=self.detail, style="Detail.TLabel").pack(anchor="w")
        self.elapsed = tk.StringVar()
        label(card, textvariable=self.elapsed, style="Detail.TLabel").pack(anchor="w", pady=(8, 0))
        self.remaining = tk.StringVar()
        label(card, textvariable=self.remaining, style="Card.TLabel").pack(anchor="w", pady=(8, 0))
        label(card, text="남은 시간은 현재 단계 기준입니다. 이후 단계·다음 파일·AI 분석은 별도입니다.",
              style="Detail.TLabel").pack(anchor="w", pady=(4, 0))
        settings = ttk.Frame(content)
        settings.pack(fill="x", pady=(12, 0))
        ttk.Label(settings, text="전사 처리").pack(side="left", padx=(0, 8))
        self.acceleration = tk.StringVar(value="자동 · 지원 Mac에서 GPU 가속")
        self.acceleration_select = ttk.Combobox(settings, textvariable=self.acceleration, state="readonly",
                                               width=33, values=("자동 · 지원 Mac에서 GPU 가속", "CPU 호환 · 기존 처리 방식"))
        self.acceleration_select.pack(side="left")
        self.pair_code = tk.StringVar()
        label(content, textvariable=self.pair_code).pack(anchor="w", pady=(16, 10))
        buttons = ttk.Frame(content)
        buttons.pack(anchor="w", pady=(0, 12))
        self.start_button = ttk.Button(buttons, text="계정 연결 / 시작", style="Primary.TButton")
        self.start_button.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(buttons, text="중지", state="disabled")
        self.stop_button.pack(side="left", padx=(0, 8))
        self.disconnect_button = ttk.Button(buttons, text="연결 해제")
        self.disconnect_button.pack(side="left")
        self.browser_button = ttk.Button(content, text="ClassTalk 열기")
        self.browser_button.pack(anchor="w")
        label(content, style="Secondary.TLabel", text=(
            f"v{VERSION} · Python 설치 불필요\n"
            "음성은 이 컴퓨터에서 처리하며 전사 결과만 저장합니다.\n"
            "파일의 전사·화자 구분 후 웹에 결과가 표시됩니다."
        )).pack(anchor="w", side="bottom", pady=(16, 0))
        self.current: Progress | None = None

        def resize(event):
            if event.widget is content:
                for widget in self.wrapped:
                    widget.configure(wraplength=max(200, event.width - 96))
        content.bind("<Configure>", resize)
        self.root = root
        self.tick()

    def set_progress(self, value: Progress):
        self.current = value
        self.status.set(value.heading)
        self.context.set(f"{value.title} · 파일 {value.file_index}/{value.file_count} · {value.engine}"
                         if value.file_count else "모델 준비")
        self.detail.set(value.detail)
        mode = "indeterminate" if value.percent is None else "determinate"
        if str(self.bar.cget("mode")) != mode:
            self.bar.stop()
            self.bar.configure(mode=mode, value=0)
            if mode == "indeterminate":
                self.bar.start(25)
        if value.percent is not None:
            self.bar.configure(value=value.percent)
        if value.stage == "complete":
            self.elapsed.set(f"수업 처리 시간 {clock_text(time.monotonic() - value.started_at)}")
        self.update_elapsed()

    def clear_progress(self):
        self.current = None
        self.bar.stop()
        self.bar.configure(mode="determinate", value=0)
        self.context.set("")
        self.detail.set("")
        self.elapsed.set("")
        self.remaining.set("")

    def update_elapsed(self):
        if self.current and self.current.stage != "complete":
            scope = "수업 처리" if self.current.file_count else "모델 준비"
            self.elapsed.set(f"{scope} 경과 {clock_text(time.monotonic() - self.current.started_at)}"
                             " · 처리 속도는 구간마다 다를 수 있습니다.")
            self.remaining.set(self.current.remaining_text(time.monotonic()))
        elif self.current:
            self.remaining.set("")

    def tick(self):
        self.update_elapsed()
        self.root.after(1000, self.tick)


def preview(root):
    """Offline UI inspection; never construct a controller, token store or model."""
    view = WorkerView(root)
    root.title("ClassTalk 화면 미리보기 — 실제 전사 아님")
    view.set_progress(Progress("transcribe", 138000, 360000, "과학 · 화면 예시", 1, 2,
                               time.monotonic() - 52, time.monotonic(), 84, 2.6,
                               "Mac GPU 가속 · large-v3"))
    view.pair_code.set("화면 미리보기 · 실제 계정이나 음성을 사용하지 않습니다.")
    for button in (view.start_button, view.stop_button, view.disconnect_button, view.browser_button):
        button.configure(state="disabled")
    root.mainloop()
