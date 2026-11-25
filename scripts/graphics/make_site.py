"""Fill the generated parts of site/index.html: the key figures and the charts.

    python scripts/graphics/make_site.py

The page is static HTML. Everything between `<!-- gen:NAME -->` and `<!-- /gen:NAME -->` is
rewritten here from the tables in results/SCOREBOARD.md, so the figures on the page cannot
drift from the published results.
"""
from __future__ import annotations

import html
import json
import re

from common import ROOT
from make_charts import detection, english, table, tally, zipa

PAGE = ROOT / "site/index.html"
esc = html.escape
NUMBER = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


def stats() -> str:
    """The figures under the headline, read from the scoreboard and the timing run."""
    rows = tally()
    total, better, same = (sum(r[k] for r in rows) for k in (1, 2, 3))
    quiet = ROOT / "results/tables/learner_benchmark_cost/quiet/pronunciada-fp32__t4.json"     # the benchmark page's own timing
    ms = round(json.loads(quiet.read_text(encoding="utf-8"))["p50_s"] * 1000)
    cells = [
        (f"{better + same}<small> of {total}</small>", "comparisons level with or ahead of Azure"),
        (f"{ms}<small> ms</small>", "per sentence on a laptop CPU"),
        (f"{len(rows)}", "languages"),
    ]
    return '<div class="stats">' + "".join(f"<div><b>{n}</b><span>{label}</span></div>" for n, label in cells) + "</div>"


# ---- charts -------------------------------------------------------------------------------

def pairs(title: str, rows: list[dict], other: str, foot: str) -> str:
    """Two bars per metric: Pronunciada and one other system, on a 0 to 1 scale."""
    groups = []
    for r in rows:
        a, b = r["ours"][0], r["azure"][0]
        ns = "" if r["sig"] else '<sup>n.s.</sup>'
        groups.append(
            f'<div class="ch-g"><div class="ch-name">{r["label"]}<small>{r["what"]}</small></div><div class="ch-bars">'
            f'<div class="ch-row own"><span class="ch-track"><i style="width:{a * 100:.1f}%"></i></span><b>{a:.2f}{ns}</b></div>'
            f'<div class="ch-row"><span class="ch-track"><i style="width:{b * 100:.1f}%"></i></span><b>{b:.2f}</b></div>'
            f'</div></div>')
    if all(r["sig"] for r in rows):
        sig = f"All {NUMBER[len(rows)]} differences are statistically significant."
    else:
        sig = "Differences are statistically significant except where marked n.s."
    key = f'<span><i class="own"></i>Pronunciada</span><span><i></i>{other}</span>'
    return (f'<figure class="chart"><div class="ch-head"><figcaption>{title}</figcaption><div class="ch-key">{key}</div></div>'
            f'{"".join(groups)}<p class="ch-foot">{foot} {sig}</p></figure>')


def pick(rows: list[dict], label: str, name: str, what: str) -> dict:
    r = next(r for r in rows if r["label"] == label)
    return {**r, "label": name, "what": what}


def chart_azure() -> str:
    e, d = english(), detection()
    rows = [
        pick(e, "Phoneme", "English sounds", "agreement with expert scores"),
        pick(e, "Sentence accuracy", "English sentences", "agreement with expert scores"),
        pick(d, "English phonemes, L2-ARCTIC", "English mispronunciations", "finding the sounds experts marked"),
        pick(d, "Mandarin words, tone errors", "Mandarin tones", "finding the tones teachers marked"),
        pick(e, "Sentence prosody", "English intonation", "agreement with expert scores"),
    ]
    return pairs("Learner speech scored by experts", rows, "Azure",
                 "Higher is better. Agreement is the Pearson correlation with expert scores, and finding errors is the area under the ROC curve. "
                 "Test sets: speechocean762, L2-ARCTIC and OMPAL.")


def chart_zipa() -> str:
    z = zipa()
    rows = [
        pick(z, "English phonemes, expert scores (r)", "English sounds", "agreement with expert scores"),
        pick(z, "English sentences, expert scores (r)", "English sentences", "agreement with expert scores"),
        pick(z, "English errors, L2-ARCTIC (AUC)", "English mispronunciations", "finding the sounds experts marked"),
        pick(z, "Mandarin errors, tone (AUC)", "Mandarin tones", "finding the tones teachers marked"),
    ]
    return pairs("The same tests, against ZIPA alone", rows, "ZIPA alone",
                 "Higher is better. ZIPA alone is the unmodified model with the standard goodness-of-pronunciation score.")


def chart_tally() -> str:
    rows = tally()
    top = max(r[1] for r in rows)
    out = []
    for label, n, ahead, same, behind in rows:
        seg = "".join(f'<i class="{cls}" style="width:{count / top * 100:.2f}%">{count}</i>'
                      for count, cls in ((ahead, "own"), (same, "same"), (behind, "behind")) if count)
        out.append(f'<div class="tl-row"><span class="tl-name">{label}</span><span class="tl-bar">{seg}</span><span class="tl-n">of {n}</span></div>')
    total = [sum(r[k] for r in rows) for k in (1, 2, 3, 4)]
    key = ('<span><i class="own"></i>Pronunciada significantly better</span><span><i class="same"></i>No significant difference</span>'
           '<span><i class="behind"></i>Azure significantly better</span>')
    return (f'<figure class="chart"><div class="ch-head stack"><figcaption>Comparisons with Azure, by language</figcaption><div class="ch-key">{key}</div></div>'
            f'<div class="tl">{"".join(out)}</div><p class="ch-foot">{total[0]} comparisons on held-out public test sets. '
            'Significance is from paired tests at p &lt; 0.05.</p></figure>')


def speed() -> tuple[float, float, float]:
    """English row of the long-clip table: (Pronunciada seconds, Azure seconds, clip length in seconds)."""
    row = next(r for r in table("Longer clips next to Azure") if r[0] == "en")
    ours = re.match(r"(\d+) ms \((\d+\.\d+) s\)", row[1])
    azure = re.match(r"(\d+\.\d+) s", row[2])
    return int(ours.group(1)) / 1000, float(azure.group(1)), float(ours.group(2))


def chart_speed() -> str:
    ours, azure, clip = speed()
    top = max(ours, azure)
    row = lambda cls, who, sub, v: (f'<div class="sp-row {cls}"><span class="sp-who">{who}<small>{sub}</small></span>'
                                    f'<span class="ch-track"><i style="width:{v / top * 100:.1f}%"></i></span><b>{v:.2f} s</b></div>')
    return (f'<figure class="chart"><div class="ch-head"><figcaption>Time to score a {clip:.0f}-second recording</figcaption></div>'
            f'{row("own", "Pronunciada", "on the laptop, one CPU thread", ours)}{row("", "Azure", "request to the cloud and back", azure)}'
            f'<p class="ch-foot">Lower is better. Medians over English recordings of about {clip:.0f} seconds from the FLEURS test set. The two systems '
            'were timed on different recordings of similar length. The Azure time is the whole request from the same laptop, upload included.</p></figure>')


def figure_word() -> str:
    """One word of the learner recording, sound by sound: expected, recognized instead, probability."""
    word = next(w for w in json.loads((ROOT / "site/samples/en-learner-1.json").read_text(encoding="utf-8"))["result"]["words"] if w["prob"] < 0.5)
    phones = word["phones"]
    def heard(p):
        if p["heard"] is None:
            return '<span class="h">–</span>'
        return f'<span class="h sub">{esc(p["heard"]) or "nothing"}</span>'
    rows = ['<span class="l">Expected sound</span>' + "".join(f'<span class="s">{esc(p["ipa"])}</span>' for p in phones),
            '<span class="l">Recognized instead</span>' + "".join(heard(p) for p in phones),
            '<span class="l">Probability a rater accepts it</span>' + "".join(
                f'<span><span class="n{" low" if p["prob"] < 0.5 else ""}">{p["prob"]:.2f}</span></span>' for p in phones)]
    return (f'<figure class="chart wf"><div class="ch-head"><figcaption>How one word is scored</figcaption>'
            f'<div class="ch-key">“{esc(word["text"])}”, said without its “l”</div></div>'
            f'<div class="wf-grid">{"".join(rows)}</div>'
            f'<p class="ch-foot">The speech model heard no “l” and a different vowel, so those two sounds score low and the word as a whole '
            f'gets {word["prob"]:.2f}.</p></figure>')


CARD_CSS = """
*{box-sizing:border-box;margin:0}
body{margin:0;font-family:'Geist',sans-serif;color:#fff;-webkit-font-smoothing:antialiased}
.card{position:relative;width:920px;background:linear-gradient(rgba(4,6,10,.40),rgba(4,6,10,.40)),#06080c url(GRADIENT) center/cover}
.arch .head{position:absolute;left:40px;top:34px;display:flex;align-items:center;gap:12px;font-size:21px;letter-spacing:-.02em;color:rgba(255,255,255,.66)}
.arch .head svg{height:17.5px;display:block}
.cd{position:absolute;width:124px;height:132px;border-radius:16px;background:rgba(6,10,24,.5);border:1px solid rgba(255,255,255,.17);box-shadow:inset 0 1px 0 rgba(255,255,255,.07),0 14px 34px rgba(2,4,12,.34)}
.cd svg{position:absolute;left:18px;top:26px}
.cd b{position:absolute;left:0;right:0;bottom:17px;text-align:center;font-weight:500;font-size:14.5px;letter-spacing:-.01em}
.cd.io{background:rgba(6,10,24,.32)}
.cd.io b{font-weight:400;color:rgba(255,255,255,.92)}
.cd.zipa{background:rgba(255,255,255,.11);border-color:rgba(255,255,255,.62)}
.cd.zipa b{font-size:16.5px;letter-spacing:.02em}
.chip{position:absolute;width:124px;height:44px;border-radius:12px;display:flex;align-items:center;justify-content:center;gap:9px;background:rgba(6,10,24,.32);border:1px solid rgba(255,255,255,.17);font-size:14px;color:rgba(255,255,255,.92)}
.chip.own{background:rgba(6,10,24,.5);font-weight:500;color:#fff}
.region{position:absolute;border-radius:22px;background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.13)}
.region b,.tag{font:500 10.5px/1 'Geist Mono',monospace;letter-spacing:.13em;text-transform:uppercase;color:rgba(255,255,255,.62)}
.region b{position:absolute;left:18px;top:17px}
.tag{position:absolute;display:flex;align-items:center;justify-content:center;gap:7px;color:rgba(255,255,255,.78)}
svg.wires{position:absolute;left:0;top:0}
svg.wires path{fill:none;stroke:rgba(255,255,255,.52);stroke-width:1.25;stroke-linecap:round;stroke-linejoin:round}
svg.wires polygon{fill:rgba(255,255,255,.78)}
.stats{display:grid;grid-template-columns:1.25fr .8fr 1fr 1.15fr;gap:0 36px;padding:40px 44px 38px}
.stats b{display:block;font-weight:300;font-size:58px;line-height:1;letter-spacing:-.04em;white-space:nowrap;font-variant-numeric:tabular-nums}
.stats b small{font-size:25px;letter-spacing:-.02em;color:rgba(255,255,255,.72);margin-left:5px}
.stats span{display:block;margin-top:14px;font-size:14.5px;line-height:20px;color:rgba(255,255,255,.78)}
"""
def figure_architecture() -> str:
    """The architecture as one signal path: illustrated cards for the main stages, slim chips for the side inputs."""
    import soundfile as sf
    from make_logo import wordmark

    W, H, CW, CH = 920, 462, 124, 132
    X = [40 + i * 143 for i in range(6)]
    top, mid = 190, 190 + CH // 2
    y_pro, y_low, chip = 118, 366, 44
    G = "url(#ag)"
    # glyphs, all in an 88 x 52 box
    audio, sr = sf.read(str(ROOT / "site/samples/en-learner-1.mp3"), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    n = 18
    lv = [float(abs(mono[i * (len(mono) // n):(i + 1) * (len(mono) // n)]).max()) for i in range(n)]
    wave = "".join(f'<rect x="{i * 5:.1f}" y="{26 - max(1.5, (v / max(lv)) ** .7 * 22):.1f}" width="2.6" height="{2 * max(1.5, (v / max(lv)) ** .7 * 22):.1f}" rx="1.3" fill="#fff"/>'
                   for i, v in enumerate(lv))
    layers = "".join(f'<rect x="10" y="{3 + i * 10}" width="68" height="6" rx="3" fill="#fff" opacity="{o}"/>' for i, o in enumerate((1, .78, .58, .42, .28)))
    grid = "".join(f'<circle cx="{24 + c * 10}" cy="{6 + r * 10}" r="{3.1 if r == c else 2}" fill="{G if r == c else "#fff"}" opacity="{1 if r == c else .34}"/>'
                   for r in range(5) for c in range(5))
    cols, rows_ = (8, 32, 56, 80), (8, 26, 44)
    path = (1, 0, 1, 2)
    edges = "".join(f'<line x1="{cols[i]}" y1="{a}" x2="{cols[i + 1]}" y2="{b}" stroke="#fff" stroke-opacity=".16"/>' for i in range(3) for a in rows_ for b in rows_)
    best = " ".join(f"{cols[i]},{rows_[k]}" for i, k in enumerate(path))
    dots = "".join(f'<circle cx="{x}" cy="{y}" r="{3.6 if path[i] == k else 2.3}" fill="{G if path[i] == k else "#fff"}" opacity="{1 if path[i] == k else .5}"/>'
                   for i, x in enumerate(cols) for k, y in enumerate(rows_))
    lattice = f'{edges}<polyline points="{best}" fill="none" stroke="{G}" stroke-width="2"/>{dots}'
    curve = ('<path d="M8 3V49H84" fill="none" stroke="#fff" stroke-opacity=".34" stroke-width="1.2"/>'
             f'<path d="M11 45C40 45 46 8 82 8" fill="none" stroke="{G}" stroke-width="2.3" stroke-linecap="round"/>')
    bars = "".join(f'<rect x="6" y="{6 + i * 16}" width="76" height="7" rx="3.5" fill="#fff" opacity=".16"/>'
                   f'<rect x="6" y="{6 + i * 16}" width="{w}" height="7" rx="3.5" fill="#fff" opacity=".92"/>' for i, w in enumerate((62, 20, 48)))
    flake = ('<svg width="15" height="15" viewBox="-8 -8 16 16" fill="none" stroke="#fff" stroke-width="1.15" stroke-linecap="round" stroke-linejoin="round">'
             + "".join(f'<g transform="rotate({a})"><path d="M0 -6.6V6.6M-1.9 -4.9L0 -3.2L1.9 -4.9M-1.9 4.9L0 3.2L1.9 4.9"/></g>' for a in (0, 60, 120)) + "</svg>")
    card = lambda i, name, glyph, cls="": (f'<div class="cd {cls}" style="left:{X[i]}px;top:{top}px"><svg width="88" height="52" viewBox="0 0 88 52">{glyph}</svg>'
                                           f'<b>{name}</b></div>')
    cards = (card(0, "Audio", wave, "io") + card(1, "ZIPA", layers, "zipa") + card(2, "Adapter", grid) + card(3, "Lattice scoring", lattice)
             + card(4, "Calibration", curve) + card(5, "Scores", bars, "io"))
    chips = (f'<div class="chip own" style="left:{X[3]}px;top:{y_pro}px">Prosody heads</div>'
             f'<div class="chip" style="left:{X[0]}px;top:{y_low}px">Text</div>'
             f'<div class="chip own" style="left:{X[2]}px;top:{y_low}px">Lexicon + G2P</div>')
    tip = {"r": "-6,-3.6 0,0 -6,3.6", "u": "-3.6,6 0,0 3.6,6", "d": "-3.6,-6 0,0 3.6,-6"}
    def wire(d: str, way: str) -> str:
        x, y = d.split()[-2:]
        return f'<path d="{d}"/><polygon points="{tip[way]}" transform="translate({x} {y})"/>'
    xl, xc, xz = X[3] + CW // 2, X[4] + CW // 2, X[1] + CW // 2
    wires = "".join([
        *(wire(f"M{X[c] + CW} {mid} L {X[c + 1] - 2} {mid}", "r") for c in range(5)),
        wire(f"M{X[0] + CW} {y_low + chip // 2} L {X[2] - 2} {y_low + chip // 2}", "r"),
        wire(f"M{X[2] + CW} {y_low + chip // 2} H{xl - 12} Q{xl} {y_low + chip // 2} {xl} {y_low + chip // 2 - 12} L {xl} {top + CH + 2}", "u"),
        wire(f"M{xz} {top} V{y_pro + chip // 2 + 12} Q{xz} {y_pro + chip // 2} {xz + 12} {y_pro + chip // 2} L {X[3] - 2} {y_pro + chip // 2}", "r"),
        wire(f"M{X[3] + CW} {y_pro + chip // 2} H{xc - 12} Q{xc} {y_pro + chip // 2} {xc} {y_pro + chip // 2 + 12} L {xc} {top - 2}", "d"),
    ])
    region = (f'<div class="region" style="left:{X[2] - 14}px;top:{y_pro - 40}px;width:{X[4] + CW - X[2] + 28}px;'
              f'height:{y_low + chip - y_pro + 56}px"><b>Pronunciada</b></div>')
    frozen = f'<div class="tag" style="left:{X[1]}px;top:{top + CH + 12}px;width:{CW}px">{flake}Frozen</div>'
    defs = ('<defs><linearGradient id="ag" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6aa6ff"/><stop offset="1" stop-color="#2fe0c4"/>'
            '</linearGradient></defs>')
    return (f'<div class="card arch" style="height:{H}px"><svg width="0" height="0" style="position:absolute">{defs}</svg>'
            f'<div class="head">{wordmark("#ffffff")}<span>Architecture</span></div>'
            f'{region}<svg class="wires" width="{W}" height="{H}">{wires}</svg>{cards}{chips}{frozen}</div>')


def figure_stats() -> str:
    """The key figures as one strip."""
    rows = tally()
    ahead, behind = sum(r[2] for r in rows), sum(r[4] for r in rows)
    quiet = json.loads((ROOT / "results/tables/learner_benchmark_cost/quiet/pronunciada-int8__t4.json").read_text(encoding="utf-8"))
    size = round(next(w["mb"] for w in quiet["weight_files"] if w["file"].startswith("model")))
    cells = [(f"{ahead}–{behind}", "significant wins and losses against Azure"), (f"{len(rows)}", "languages"),
             (f"{size}<small>MB</small>", "small model"), (f"{round(quiet['p50_s'] * 1000)}<small>ms</small>", "per sentence on a laptop CPU")]
    return '<div class="card stats">' + "".join(f"<div><b>{n}</b><span>{label}</span></div>" for n, label in cells) + "</div>"


def readme_images() -> None:
    """Dark renders of the worked example and the languages card for the README, using the page's own styles."""
    from common import INK, font_css, render
    page = PAGE.read_text(encoding="utf-8")
    css = re.search(r"<style>(.*?)</style>", page, re.S).group(1)
    dark = (":root{--text:#fff;--soft:rgba(255,255,255,.62);--faint:rgba(255,255,255,.5);--rule:rgba(255,255,255,.14);"
            f"--page:{INK};--card:{INK};--pale:#34343a;--accent:linear-gradient(90deg,#7db8ff,#5eead4)}}"
            f"body{{background:{INK}}}.chart{{box-shadow:none;margin:0;border-radius:0;padding:36px 40px 30px}}.langs{{margin:0;border-radius:0}}")
    head = f"<!doctype html><html><head><meta charset='utf-8'><style>{font_css()}{css}{dark}</style></head><body>"
    gradient = (ROOT / "site/assets/gradient.jpg").as_uri()
    card = f"<!doctype html><html><head><meta charset='utf-8'><style>{font_css()}{CARD_CSS.replace('GRADIENT', gradient)}</style></head><body>"
    render(card + figure_architecture() + "</body></html>", ROOT / "docs/assets/pipeline.jpg", 920, 462)
    render(card + figure_stats() + "</body></html>", ROOT / "docs/assets/stats.jpg", 920, 196)
    langs = re.search(r'<div class="langs">.*?</div>', page, re.S).group(0)
    from common import height_of
    card_html = head.replace("url(assets/gradient.jpg)", f"url('{(ROOT / 'site/assets/gradient.jpg').as_uri()}')") + langs + "</body></html>"
    render(card_html, ROOT / "docs/assets/languages.jpg", 920, height_of(card_html, 920))


BLOCKS = {"stats": stats, "figure-word": figure_word, "chart-tally": chart_tally, "chart-azure": chart_azure, "chart-speed": chart_speed, "chart-zipa": chart_zipa}


if __name__ == "__main__":
    page = PAGE.read_text(encoding="utf-8")
    for name, make in BLOCKS.items():
        pattern = re.compile(rf"(<!-- gen:{name} -->).*?(<!-- /gen:{name} -->)", re.S)
        assert pattern.search(page), f"marker gen:{name} is missing from site/index.html"
        block = make()
        page = pattern.sub(lambda m: f"{m.group(1)}\n{block}\n{m.group(2)}", page)
    PAGE.write_text(page, encoding="utf-8")
    space = ROOT / "huggingface/space_static/index.html"          # the Hugging Face Space page repeats the key figures
    text = space.read_text(encoding="utf-8")
    space.write_text(re.sub(r"(<!-- gen:stats -->).*?(<!-- /gen:stats -->)", lambda m: f"{m.group(1)}\n{stats()}\n{m.group(2)}", text, flags=re.S),
                     encoding="utf-8")
    readme_images()
    print(f"filled {len(BLOCKS)} blocks in site/index.html and the figures in huggingface/space_static/index.html; wrote docs/assets/pipeline.jpg, stats.jpg and languages.jpg")
