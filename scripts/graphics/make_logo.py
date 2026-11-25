"""Write the Pronunciada logo as SVG outlines (no font needed to display it).

    python scripts/graphics/make_logo.py

The mark is a short waveform in which one bar has shrunk to a dot: a sound that was dropped,
which is what the scorer exists to find. The lettering is Geist at weight 560 with tightened
spacing. Outputs, in docs/assets/: logo-light.svg and logo-dark.svg (mark and name, for light
and dark backgrounds), logo-mark.svg (the mark alone) and avatar.png (the mark on a dark tile,
for a profile picture). In site/assets/: logo-light.svg, logo-dark.svg, favicon.svg and
apple-touch-icon.png.
"""
from __future__ import annotations

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen

from common import ROOT, fonts

WEIGHT = 560
TRACKING = -0.022      # em


def outline(text: str, weight: float = WEIGHT, tracking: float = TRACKING):
    """Return (svg path data, advance width, ascent, descent) in font units, y down."""
    blob = hb.Blob.from_file_path(str(fonts() / "Geist.ttf"))
    face = hb.Face(blob)
    font = hb.Font(face)
    font.set_variations({"wght": weight})
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf, {"kern": True, "liga": False})
    upem = face.upem
    x = 0.0
    parts = []
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        pen = SVGPathPen(None)
        font.draw_glyph_with_pen(info.codepoint, pen)
        d = pen.getCommands()
        if d:
            parts.append(f'<path transform="translate({x + pos.x_offset:.1f} 0) scale(1 -1)" d="{d}"/>')
        x += pos.x_advance + tracking * upem
    x -= tracking * upem
    ext = font.get_font_extents("ltr")
    return "".join(parts), x, ext.ascender, -ext.descender, upem


SHAPE = (0.5, None, 1.0, 0.72, 0.4)        # bar heights as a share of the mark's height; None is the dot
ON_LIGHT, ON_DARK = ("#1d4ed8", "#0ea5a0"), ("#7db8ff", "#5eead4")      # the dot's gradient


def mark(height: float, bar: float, gap: float, color: str, grad=ON_LIGHT, uid: str = "pron-dot") -> tuple[str, float]:
    """The mark centered on y = 0 with its left edge at x = 0. Returns (svg, width)."""
    out = [f'<linearGradient id="{uid}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{grad[0]}"/>'
           f'<stop offset="1" stop-color="{grad[1]}"/></linearGradient>']
    x = 0.0
    for h in SHAPE:
        if h is None:
            out.append(f'<circle cx="{x + bar / 2:.1f}" cy="0" r="{bar * 0.66:.1f}" fill="url(#{uid})"/>')
        else:
            out.append(f'<rect x="{x:.1f}" y="{-h * height / 2:.1f}" width="{bar:.1f}" height="{h * height:.1f}" rx="{bar / 2:.1f}" fill="{color}"/>')
        x += bar + gap
    return "".join(out), x - gap


def wordmark(color: str, with_mark: bool = True) -> str:
    """The logo: mark and name. `color` is the lettering; the dot's gradient follows it."""
    paths, width, asc, desc, upem = outline("Pronunciada")
    cap = 0.72 * upem                      # trim to cap height plus descender of the "p"
    top, bottom = cap + 0.06 * upem, 0.24 * upem
    sign, shift = "", 0.0
    if with_mark:
        light = color.lower() in ("#fff", "#ffffff")
        shapes, w = mark(0.8 * upem, 0.105 * upem, 0.075 * upem, color, ON_DARK if light else ON_LIGHT)
        sign = f'<g transform="translate(0 {-cap / 2:.0f})">{shapes}</g>'
        shift = w + 0.3 * upem
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {-top:.0f} {width + shift:.0f} {top + bottom:.0f}" '
            f'role="img" aria-label="Pronunciada">{sign}<g fill="{color}" transform="translate({shift:.0f} 0)">{paths}</g></svg>\n')


def tile(background: str | None = "#0b0b0c", size: int = 100, big: bool = False) -> str:
    """The mark alone: on a rounded dark tile (favicon, profile picture) or, with background=None, in black.
    big: the larger mark used for site icons, which search results show at about 18 px inside a circle; it
    still fits inside the circle inscribed in the tile."""
    color, grad = ("#fff", ON_DARK) if background else ("#000", ON_LIGHT)
    h, bar, gap = (0.62, 0.1, 0.072) if big else (0.54, 0.088, 0.064)
    shapes, w = mark(h * size, bar * size, gap * size, color, grad)
    back = f'<rect width="{size}" height="{size}" rx="{0.24 * size:.0f}" fill="{background}"/>' if background else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" role="img" aria-label="Pronunciada">{back}'
            f'<g transform="translate({(size - w) / 2:.1f} {size / 2})">{shapes}</g></svg>\n')


if __name__ == "__main__":
    from common import render
    docs, site = ROOT / "docs/assets", ROOT / "site/assets"
    for folder in (docs, site):
        (folder / "logo-light.svg").write_text(wordmark("#000000"), encoding="utf-8")
        (folder / "logo-dark.svg").write_text(wordmark("#ffffff"), encoding="utf-8")
    (docs / "logo-mark.svg").write_text(tile(None), encoding="utf-8")
    (site / "favicon.svg").write_text(tile(big=True), encoding="utf-8")
    square = "<!doctype html><html><body style='margin:0;background:#0b0b0c'>" + tile().replace("<svg ", "<svg width='256' height='256' ") + "</body></html>"
    render(square, docs / "avatar.png", 256, 256, scale=2)

    def icon(px: int, out):
        """The site icon as an opaque square PNG of px x px: iOS and search engines round or crop it themselves."""
        page = ("<!doctype html><html><body style='margin:0;background:#0b0b0c'>"
                + tile(big=True).replace("<svg ", f"<svg width='{px}' height='{px}' ") + "</body></html>")
        render(page, out, px, px, scale=1)

    icon(180, site / "apple-touch-icon.png")
    icon(192, site / "icon-192.png")
    icon(48, site / "favicon-48.png")
    # /favicon.ico at the site root, which crawlers request without reading the page
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(site / "favicon-48.png"), str(site.parent / "favicon.ico")], check=True)
    (site / "favicon-48.png").unlink()
    print("wrote logo-light.svg, logo-dark.svg, logo-mark.svg, favicon.svg, icon-192.png, "
          "apple-touch-icon.png, ../favicon.ico and avatar.png")
