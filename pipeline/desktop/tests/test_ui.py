"""Opt-in native Tk checks: CLASSTALK_UI_TEST=1 python -m unittest desktop.tests.test_ui."""
import os
import time
import unittest


@unittest.skipUnless(os.environ.get("CLASSTALK_UI_TEST") == "1", "requires a desktop display")
class WorkerUITests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from desktop.ui import WorkerView
        self.root = tk.Tk()
        self.view = WorkerView(self.root)
        self.root.update()

    def tearDown(self):
        self.root.destroy()

    def test_unknown_stage_animates_then_real_progress_stops_animation(self):
        from desktop.progress import Progress
        self.view.set_progress(Progress("download", started_at=time.monotonic() - 62))
        self.assertEqual(str(self.view.bar.cget("mode")), "indeterminate")
        self.assertIn("1:02", self.view.elapsed.get())
        self.view.set_progress(Progress("transcribe", 1000, 4000, started_at=time.monotonic()))
        self.assertEqual(str(self.view.bar.cget("mode")), "determinate")
        self.assertEqual(float(self.view.bar.cget("value")), 25)
        self.assertIn("25%", self.view.status.get())
        self.view.clear_progress()
        self.assertEqual(float(self.view.bar.cget("value")), 0)
        self.assertIsNone(self.view.current)
        self.assertEqual(self.view.elapsed.get(), "")

    def test_small_window_keeps_wrapped_pairing_error_and_buttons_visible(self):
        self.root.geometry("640x760")
        self.view.status.set("중지 요청됨 · 새 작업을 받지 않습니다. 현재 음성 구간 처리가 끝날 때까지 잠시 기다려 주세요.")
        self.view.pair_code.set("연결 코드: ABCD-EFGH · 10분 안에 브라우저에서 승인해 주세요.")
        self.root.update()
        for widget in [*self.view.wrapped, self.view.browser_button, self.view.disconnect_button]:
            with self.subTest(widget=str(widget)):
                self.assertGreater(widget.winfo_height(), 1)
                self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(),
                                     self.root.winfo_rooty() + self.root.winfo_height())

    def test_remaining_time_is_distinct_from_elapsed_and_cleared_on_stop(self):
        from desktop.progress import Progress
        now = time.monotonic()
        self.view.set_progress(Progress("transcribe", 3000, 10000, "수업", 1, 2,
                                      now - 30, now, 70, .1, "Mac GPU 가속"))
        self.assertIn("전사 단계 남음 약", self.view.remaining.get())
        self.assertIn("경과", self.view.elapsed.get())
        self.assertIn("GPU", self.view.context.get())
        self.view.clear_progress()
        self.assertEqual(self.view.remaining.get(), "")

    def test_theme_paints_both_colours_with_readable_text_contrast(self):
        from tkinter import ttk
        style = ttk.Style(self.root)
        self.assertEqual(style.theme_use(), "clam")
        def luminance(colour):
            channels = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
            linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
            return sum(c * weight for c, weight in zip(linear, (0.2126, 0.7152, 0.0722)))
        for name in ("TLabel", "Card.TLabel", "Secondary.TLabel", "Detail.TLabel", "TButton", "Primary.TButton"):
            with self.subTest(style=name):
                values = sorted(luminance(style.lookup(name, key)) for key in ("background", "foreground"))
                self.assertGreaterEqual((values[1] + 0.05) / (values[0] + 0.05), 4.5)


if __name__ == "__main__":
    unittest.main()
