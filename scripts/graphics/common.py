"""Shared helpers for the graphics scripts: fonts, colors and headless-Chrome rendering.

Fonts (Geist, Geist Mono, Inter; SIL Open Font License) are downloaded on first use into
scripts/graphics/.fonts/ and are not part of the repository.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FONT_DIR = Path(__file__).resolve().parent / ".fonts"
FONTS = {
    "Geist.ttf": "https://github.com/google/fonts/raw/main/ofl/geist/Geist%5Bwght%5D.ttf",
    "GeistMono.ttf": "https://github.com/google/fonts/raw/main/ofl/geistmono/GeistMono%5Bwght%5D.ttf",
    "Inter.ttf": "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf",
}
CHROME = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser",
]

INK = "#0b0b0c"        # near-black used for dark cards and charts
PAPER = "#fdfcfc"      # page background
FLAG = "#7db8ff"       # flagged words on dark backgrounds


def fonts() -> Path:
    FONT_DIR.mkdir(exist_ok=True)
    for name, url in FONTS.items():
        dest = FONT_DIR / name
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)
    return FONT_DIR


def font_css() -> str:
    d = fonts()
    return "".join(
        f"@font-face{{font-family:'{fam}';src:url('file://{d / file}');font-weight:100 900;font-style:normal}}"
        for fam, file in (("Geist", "Geist.ttf"), ("Geist Mono", "GeistMono.ttf"), ("Inter", "Inter.ttf"))
    )


def chrome() -> str:
    for c in CHROME:
        if Path(c).exists() or shutil.which(c):
            return c
    raise SystemExit("Google Chrome or Chromium is needed to render the graphics")


def render(html: str, out: Path, width: int, height: int, scale: int = 2, transparent: bool = False) -> None:
    """Render an HTML string to a PNG of width x height CSS pixels at the given device scale."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "page.html"
        page.write_text(html, encoding="utf-8")
        cmd = [chrome(), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
               f"--force-device-scale-factor={scale}", f"--window-size={width},{height}",
               "--virtual-time-budget=4000", f"--screenshot={out}"]
        if transparent:
            cmd.append("--default-background-color=00000000")
        png = out if out.suffix == ".png" else Path(tmp) / "shot.png"
        cmd[cmd.index(f"--screenshot={out}")] = f"--screenshot={png}"
        subprocess.run(cmd + [page.as_uri()], check=True, capture_output=True)
        if out.suffix in (".jpg", ".jpeg"):       # photographic backgrounds are far smaller as JPEG
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(png), "-q:v", "2", str(out)], check=True)


def height_of(html: str, width: int) -> int:
    """Height in CSS pixels that an HTML string takes at the given width, measured in headless Chrome."""
    probe = ("<script>addEventListener('load',()=>document.fonts.ready.then(()=>{"
             "document.title=String(Math.ceil(document.documentElement.getBoundingClientRect().height))}))</script>")
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "page.html"
        page.write_text(html.replace("</body>", probe + "</body>"), encoding="utf-8")
        out = subprocess.run([chrome(), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
                              f"--window-size={width},2000", "--virtual-time-budget=4000", "--dump-dom", page.as_uri()],
                             check=True, capture_output=True, text=True).stdout
    return int(re.search(r"<title>(\d+)</title>", out).group(1))


def render_card(body: str, css: str, out: Path, width: int = 808) -> None:
    """Render one figure in the site's light style as a PNG with a transparent surround, as tall as its content."""
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>0</title><style>{font_css()}{css}"
            "html,body{background:transparent;margin:0}.chart{margin:3px}</style></head>"
            f"<body>{body}</body></html>")
    render(html, out, width + 6, height_of(html, width + 6), transparent=True)
