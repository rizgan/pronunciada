"""Record the command-line tool scoring a recording as an animation (docs/assets/terminal.webp).

    pip install playwright          # drives the installed Google Chrome
    python scripts/graphics/make_cli_gif.py

A real shell session is recorded by record_terminal.py and replayed in a page with
asciinema-player. The page is captured frame by frame at twice the display size and the frames
are joined with ffmpeg. Nothing in the session is edited.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

from common import ROOT, font_css

OUT = ROOT / "docs/assets/terminal.webp"
PLAYER = "https://cdn.jsdelivr.net/npm/asciinema-player@3.8.0/dist/bundle/asciinema-player"
SIZE = {"width": 920, "height": 380}
FPS = 12
PAGE = """<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="%(player)s.css"><style>%(fonts)s
html,body{margin:0;height:100%%}
body{background:linear-gradient(rgba(4,6,10,.34),rgba(4,6,10,.34)),#06080c url('%(gradient)s') center/cover}
.win{position:absolute;inset:38px 44px;border-radius:14px;background:rgba(9,11,16,.9);border:1px solid rgba(255,255,255,.14);
  box-shadow:0 26px 60px rgba(2,4,12,.5),inset 0 1px 0 rgba(255,255,255,.06);overflow:hidden}
.bar{height:38px;display:flex;align-items:center;gap:8px;padding:0 16px;border-bottom:1px solid rgba(255,255,255,.07)}
.bar i{width:11px;height:11px;border-radius:50%%;background:rgba(255,255,255,.2)}
#term{position:absolute;inset:58px 26px 20px}
.ap-wrapper,.ap-player{background:transparent!important}
.ap-overlay{display:none!important}
.asciinema-player-theme-pron{--term-color-foreground:#ececee;--term-color-background:transparent;--term-color-0:#0b0b0c;--term-color-1:#7db8ff;--term-color-2:#5eead4;--term-color-3:#ececee;--term-color-4:#7aa2f7;--term-color-5:#c4a7e7;--term-color-6:#5fd3c6;--term-color-7:#ececee;--term-color-8:#8a8a93;--term-color-9:#7db8ff;--term-color-10:#5eead4;--term-color-11:#ffffff;--term-color-12:#7aa2f7;--term-color-13:#c4a7e7;--term-color-14:#5fd3c6;--term-color-15:#ffffff}
</style></head><body><div class="win"><div class="bar"><i></i><i></i><i></i></div><div id="term"></div></div>
<script src="%(player)s.min.js"></script><script>
window.player = AsciinemaPlayer.create({data: %(cast)s}, document.getElementById('term'),
  {autoPlay: false, preload: true, controls: false, fit: 'width', theme: 'pron', terminalFontFamily: "'Geist Mono', Menlo, monospace", terminalLineHeight: 1.33});
window.at = t => window.player.seek(t).then(() => window.player.pause());
</script></body></html>"""


def main(hold: float = 4.5) -> None:
    with tempfile.TemporaryDirectory() as d, sync_playwright() as play:
        tmp = Path(d)
        cast = tmp / "cli.cast"
        subprocess.run([sys.executable, str(Path(__file__).with_name("record_terminal.py")), str(cast)], check=True)
        text = cast.read_text(encoding="utf-8")
        events = [json.loads(line) for line in text.strip().split("\n")[1:]]
        end = events[-1][0]                                          # the moment the prompt comes back
        page_file = tmp / "cli.html"
        page_file.write_text(PAGE % {"player": PLAYER, "fonts": font_css(), "cast": json.dumps(text),
                                     "gradient": (ROOT / "site/assets/gradient.jpg").as_uri()}, encoding="utf-8")
        browser = play.chromium.launch(channel="chrome", headless=True)
        page = browser.new_context(viewport=SIZE, device_scale_factor=2).new_page()
        page.goto(page_file.as_uri())
        page.wait_for_function("window.player && document.fonts.ready.then(() => true)")
        frames = tmp / "frames"
        frames.mkdir()
        n = 0
        t = 0.0
        while t <= end:
            page.evaluate("t => window.at(t)", t)
            page.wait_for_timeout(30)
            page.screenshot(path=str(frames / f"{n:04d}.png"))
            n += 1
            t += 1 / FPS
        page.evaluate("t => window.at(t)", end)                       # the finished screen
        page.wait_for_timeout(60)
        page.screenshot(path=str(frames / f"{n:04d}.png"))
        n += 1
        for _ in range(int(hold * FPS)):                              # hold the finished screen before the loop restarts
            (frames / f"{n:04d}.png").write_bytes((frames / f"{n - 1:04d}.png").read_bytes())
            n += 1
        browser.close()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "%04d.png"),
                        "-c:v", "libwebp_anim", "-lossless", "0", "-q:v", "86", "-compression_level", "6", "-loop", "0", str(OUT)], check=True)
    print(f"wrote {OUT.relative_to(ROOT)} ({n / FPS:.1f} s, {n} frames, {OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
