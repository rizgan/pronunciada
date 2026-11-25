"""Write the real-learner benchmark page, site/benchmarks/index.html, and its two charts for GitHub.

    python scripts/graphics/make_benchmarks.py

Every number on the page is read from results/tables/learner_benchmark_*.json (written by
`python -m bench.learnerbench.report`) and results/tables/learner_benchmark_cost/, so the page
cannot drift from the published tables. The page borrows the home page's stylesheet, header and
footer. Outputs: site/benchmarks/index.html, docs/assets/benchmark-size.png, docs/assets/benchmark-cost.png.
"""
from __future__ import annotations

import html
import json
import math
import re

from common import INK, ROOT, font_css, height_of, render

HOME = ROOT / "site/index.html"
OUT = ROOT / "site/benchmarks/index.html"
TABLES = ROOT / "results/tables"
RES = json.loads((TABLES / "learner_benchmark_results.json").read_text(encoding="utf-8"))
SAMPLE = json.loads((TABLES / "learner_benchmark_sample.json").read_text(encoding="utf-8"))
TPS = json.loads((TABLES / "learner_benchmark_sample_toneperfect.json").read_text(encoding="utf-8"))
COST_DIRS = ("quiet", "rerun")            # newest quiet pass first
GITHUB = "https://github.com/moglearn/pronunciada"
FULL = f"{GITHUB}/blob/main/results/learner_benchmark.md"
RUN_DATE = "November 25, 2026"

FP, I8 = "pronunciada-fp32", "pronunciada-int8"
OP, OPD = "openpronounce-1.1.0", "openpronounce-1.1.0-student"
OP04 = "openpronounce-0.4.0"             # the release current at Pronunciada 1.0.0; in the full tables
ZIPA, AZ, LX, CX, TP = "zipa-alone", "azure", "lingolix", "chivox", "toneperfect"
NAME = {FP: "Pronunciada", I8: "Pronunciada int8", OP: "OpenPronounce", OPD: "OpenPronounce distilled", OP04: "OpenPronounce 0.4.0",
        AZ: "Azure", LX: "Lingolix", ZIPA: "ZIPA alone", CX: "Chivox", TP: "TonePerfect"}
LANG = {"en": "English", "zh": "Mandarin", "es": "Spanish", "de": "German"}
esc = html.escape


# ---- data ---------------------------------------------------------------------------------

def find(test: str, metric: str, unit: str, rows=None) -> dict:
    hits = [r for r in (rows or RES["rows"]) if r["test"].startswith(test) and r["metric"] == metric and r["unit"] == unit]
    assert len(hits) == 1, (test, metric, unit, len(hits))
    return hits[0]


def val(row: dict, system: str):
    return row["systems"].get(system, {}).get("value")


def diff(row: dict, system: str):
    """(Pronunciada minus the system, p) from the paired speaker bootstrap, or (None, None)."""
    d = row["systems"].get(system, {}).get("base_minus")
    return (d["value"], d["p"]) if d else (None, None)


COST_FILE = {OP: "openpronounce-default", OPD: "openpronounce-student", OP04: "openpronounce-0.4.0"}


def cost(system: str, threads: int):
    for folder in COST_DIRS:
        f = TABLES / "learner_benchmark_cost" / folder / f"{COST_FILE.get(system, system)}__t{threads}.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
    return None


def size_mb(system: str) -> float:
    if system == ZIPA:                    # the backbone file Pronunciada loads
        return next(w["mb"] for w in cost(FP, 1)["weight_files"] if w["file"].startswith("model"))
    return cost(system, 1)["weights_mb"]


def num(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}"


def mb(v: float) -> str:
    return f"{round(v):,} MB"


def pval(p: float) -> str:
    return "p &lt; 0.001" if p < 0.001 else f"p {p:.3f}" if p < 0.01 else f"p {p:.2f}"


# ---- chart 1: score against size ------------------------------------------------------------

PANELS = [
    ("English sounds", "agreement with expert scores", ("speechocean762 test", "Pearson", "sound")),
    ("English words", "agreement with expert scores", ("speechocean762 test", "Pearson", "word")),
    ("English sentences", "agreement with expert accuracy scores", ("speechocean762 test", "Pearson", "sentence accuracy")),
    ("English mispronunciations", "finding the sounds annotators marked", ("L2-ARCTIC test", "AUC", "sound")),
]
LOCAL = (I8, FP, OPD, OP, ZIPA)


def panel(row: dict, w: int = 360, h: int = 236) -> str:
    left, lane, top, bottom = 36, 104, 16, 30
    x0, x1, y0, y1 = left, w - lane, top, h - bottom
    lo, hi = math.log10(80), math.log10(3200)
    sx = lambda m: x0 + (math.log10(m) - lo) / (hi - lo) * (x1 - x0)
    local = [(s, sx(size_mb(s)), val(row, s)) for s in LOCAL if val(row, s) is not None]
    cloud = [(s, val(row, s)) for s in (AZ, LX) if val(row, s) is not None]
    values = [v for _, _, v in local] + [v for _, v in cloud]
    step = 0.05 if max(values) - min(values) < 0.22 else 0.1
    ylo = math.floor((min(values) - 0.02) / step) * step
    yhi = math.ceil((max(values) + 0.02) / step) * step
    sy = lambda v: y1 - (v - ylo) / (yhi - ylo) * (y1 - y0)
    o = [f'<svg class="sc" viewBox="0 0 {w} {h}" role="img" xmlns="http://www.w3.org/2000/svg">']
    t = ylo
    while t <= yhi + 1e-9:
        o.append(f'<line class="sc-grid" x1="{x0}" y1="{sy(t):.1f}" x2="{w - lane + 14}" y2="{sy(t):.1f}"/>')
        o.append(f'<text class="sc-tick" x="{x0 - 8}" y="{sy(t) + 4:.1f}" text-anchor="end">{t:.2f}</text>')
        t += step
    for m, lab in ((100, "100 MB"), (300, "300 MB"), (1000, "1 GB"), (3000, "3 GB")):
        o.append(f'<line class="sc-grid" x1="{sx(m):.1f}" y1="{y1}" x2="{sx(m):.1f}" y2="{y1 + 4}"/>')
        o.append(f'<text class="sc-tick" x="{sx(m):.1f}" y="{y1 + 18}" text-anchor="middle">{lab}</text>')
    xc = x1 + 26
    o.append(f'<text class="sc-tick" x="{xc - 5}" y="{y1 + 18}">cloud</text>')
    # cloud services: a marker in their own lane and a faint line across the plot
    placed = []
    for s, v in sorted(cloud, key=lambda c: -c[1]):
        y = sy(v)
        ty = y + 4
        if placed and ty - placed[-1] < 14:
            ty = placed[-1] + 14
        placed.append(ty)
        o.append(f'<line class="sc-cloud" x1="{x0}" y1="{y:.1f}" x2="{xc - 7:.1f}" y2="{y:.1f}"/>')
        o.append(f'<rect class="sc-dia" x="{xc - 4.2:.1f}" y="{y - 4.2:.1f}" width="8.4" height="8.4" transform="rotate(45 {xc} {y:.1f})"/>')
        o.append(f'<text class="sc-val" x="{xc + 11}" y="{ty:.1f}">{NAME[s]} {num(v)}</text>')
    # the two builds of each local system are joined by a thin line
    at = {s: (x, sy(v)) for s, x, v in local}
    for a, b, cls in ((I8, FP, "own"), (OPD, OP, "oth")):
        if a in at and b in at:
            o.append(f'<line class="sc-join {cls}" x1="{at[a][0]:.1f}" y1="{at[a][1]:.1f}" x2="{at[b][0]:.1f}" y2="{at[b][1]:.1f}"/>')
    # value labels go above the point, or below it when a close neighbor sits higher
    side = {s: "above" for s in at}
    names = list(at)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if abs(at[a][0] - at[b][0]) < 36 and abs(at[a][1] - at[b][1]) < 26:
                upper, lower = (a, b) if at[a][1] <= at[b][1] else (b, a)
                side[upper], side[lower] = "above", "below"
    for s, x, v in local:
        y = sy(v)
        cls = "own" if s in (FP, I8) else "ring" if s == ZIPA else "oth"
        o.append(f'<circle class="sc-pt {cls}" cx="{x:.1f}" cy="{y:.1f}" r="{4.6 if cls != "ring" else 4}"/>')
        ty = y - 10 if side[s] == "above" else y + 19
        o.append(f'<text class="sc-val {"own" if cls == "own" else ""}" x="{x:.1f}" y="{ty:.1f}" text-anchor="middle">{num(v)}</text>')
    o.append("</svg>")
    return "".join(o)


def chart_size() -> str:
    cells = []
    for title, what, key in PANELS:
        cells.append(f'<div class="sc-cell"><div class="ch-name">{title}<small>{what}</small></div>{panel(find(*key))}</div>')
    key = ('<span><i class="own"></i>Pronunciada: int8 model, then full model</span>'
           '<span><i class="oth"></i>OpenPronounce: distilled model, then default</span>'
           '<span><i class="ring"></i>ZIPA alone</span><span><i class="dia"></i>Cloud service</span>')
    so, l2 = find("speechocean762 test", "Pearson", "sentence accuracy"), find("L2-ARCTIC test", "AUC", "sound")
    foot = ("Higher is better, and further left is smaller. The horizontal axis is the size on disk of every model file a system "
            "loads to score English, on a logarithmic scale. Cloud services have no download and are drawn at the right. "
            f"Agreement is the Pearson correlation with expert scores on the speechocean762 test set ({so['n_items']:,} recordings, "
            f"{so['n_speakers']} learners). Mispronunciations are the area under the ROC curve on the L2-ARCTIC test set "
            f"({l2['n_items']:,} sounds, {l2['n_speakers']} learners). ZIPA alone is the speech model Pronunciada is built on, scored without Pronunciada.")
    return ('<figure class="chart size"><div class="ch-head stack"><figcaption>Score against size on disk</figcaption>'
            f'<div class="ch-key">{key}</div></div><div class="sc-grid4">{"".join(cells)}</div><p class="ch-foot">{foot}</p></figure>')


def size_text() -> str:
    """The paragraph under the chart, written from the data: each comparison says what the paired test found."""
    snd, wrd, acc, l2 = (find(*k) for _, _, k in PANELS)
    ep = find("EpaDB test", "AUC", "sound")

    def sig(r, s):
        d, p = diff(r, s)
        return None if d is None else "ahead" if d > 0 and p < 0.05 else "behind" if d < 0 and p < 0.05 else "level"

    rivals = [s for s in (OP, OPD, AZ, LX) if any(val(r, s) is not None for r in (snd, wrd, acc))]
    lead = [lab for lab, r in (("sounds", snd), ("words", wrd), ("sentence accuracy", acc))
            if all(sig(r, s) in (None, "ahead") for s in rivals + [ZIPA])]
    out = [f"<p>Pronunciada loads {mb(size_mb(I8))} of files with its int8 model and {mb(size_mb(FP))} with the full model. "
           f"OpenPronounce loads {mb(size_mb(OPD))} with its distilled English model and {mb(size_mb(OP))} in default mode.</p>"]
    para = []
    if lead:
        what = ", ".join(lead[:-1]) + (" and " if len(lead) > 1 else "") + lead[-1]
        names = [NAME[s] for s in rivals] + [NAME[ZIPA]]
        para.append(f"On the speechocean762 test set, Pronunciada agrees with the experts more closely than "
                    f"{', '.join(names[:-1])} and {names[-1]} on {what}, and each of those differences is significant.")
    both = [(sig(l2, x), sig(ep, x)) for x in (OP, OPD)]
    if all(b == ("behind", "behind") for b in both):
        para.append(f"OpenPronounce is significantly better at finding the sounds annotators marked as mispronounced: "
                    f"{num(val(l2, OP))} and {num(val(l2, OPD))} against Pronunciada's {num(val(l2, FP))} on L2-ARCTIC, "
                    f"and {num(val(ep, OP))} and {num(val(ep, OPD))} against {num(val(ep, FP))} on EpaDB.")
    else:
        para.append(f"On finding the sounds annotators marked as mispronounced, Pronunciada scores {num(val(l2, FP))} on L2-ARCTIC "
                    f"and {num(val(ep, FP))} on EpaDB, against {num(val(l2, OP))} and {num(val(ep, OP))} for OpenPronounce.")
    return "".join(out) + "<p>" + " ".join(para) + "</p>"


# ---- chart 2: memory and time ---------------------------------------------------------------

COSTED = (I8, FP, OPD, OP)
COST_NAME = {I8: ("Pronunciada", "int8 model"), FP: ("Pronunciada", "full model"),
             OPD: ("OpenPronounce", "distilled English model"), OP: ("OpenPronounce", "default mode")}


def chart_cost() -> str:
    def group(title: str, getter, fmt, sub=None) -> str:
        vals = {s: getter(s) for s in COSTED}
        top = max(v for v in vals.values() if v is not None)
        rows = []
        for s in COSTED:
            v = vals[s]
            name, kind = COST_NAME[s]
            extra = sub(s) if sub else ""
            bar = f'<i style="width:{v / top * 100:.1f}%"></i>' if v is not None else ""
            rows.append(f'<div class="sp-row {"own" if s in (FP, I8) else ""}"><span class="sp-who">{name}<small>{kind}{extra}</small></span>'
                        f'<span class="ch-track">{bar}</span><b>{fmt(v) if v is not None else "–"}</b></div>')
        return f'<div class="cg"><div class="cg-h">{title}</div>{"".join(rows)}</div>'

    def seconds(threads):
        return lambda s: (cost(s, threads) or {}).get("p50_s")

    def four(s):
        v = seconds(4)(s)
        return f" · 4 threads: {v:.3f} s" if v is not None else ""

    one = cost(FP, 1)
    foot = (f"Lower is better. Measured on one {one['machine']} laptop, CPU only, a fresh process for each run, on {one['n_clips']} speechocean762 "
            f"recordings with a mean length of {one['mean_clip_seconds']:.1f} seconds. Time is the median for one recording, audio decoding included. "
            "The cloud services were not timed here.")
    return ('<figure class="chart cost"><div class="ch-head"><figcaption>Disk, memory and time</figcaption></div>'
            + group("Model files on disk", size_mb, mb)
            + group("Peak memory while scoring", lambda s: (cost(s, 1) or {}).get("peak_rss_mb"), mb)
            + group("Time per recording, one CPU thread", seconds(1), lambda v: f"{v:.3f} s", four)
            + f'<p class="ch-foot">{foot}</p></figure>')


# ---- tables ---------------------------------------------------------------------------------

def tp_row(row: dict):
    """TonePerfect's row for the same test and measure, from its own run (speechocean762: a sample)."""
    test = row["test"]
    key = ("speechocean762 test, TonePerfect sample" if test.startswith("speechocean762 test")
           else "OMPAL test, TonePerfect sample" if test.startswith("OMPAL test") else None)
    if key is None:
        return None
    hits = [r for r in TPS["rows"] if r["test"].startswith(key) and r["metric"] == row["metric"] and r["unit"] == row["unit"]]
    return hits[0] if hits else None


TP_EN = TPS["toneperfect"]["speechocean762"]["answered"]
TP_ZH = TPS["toneperfect"]["ompal"]["answered"]


def cell(row: dict, system: str) -> str:
    dagger = ""
    if system == TP:
        alt = tp_row(row)
        if alt is None:
            return '<td class="na">not run<i></i></td>'
        if alt["n_items"] != row["n_items"]:
            dagger = "<sup>†</sup>"
        row = alt
    entry = row["systems"].get(system, {})
    if "value" not in entry:
        status = entry.get("status", "")
        word = ("not supported" if "not supported" in status else "not run" if "not run" in status
                else "not given" if "not given" in status or "not run" == status else "–")
        return f'<td class="na">{word}<i></i></td>'
    mark = "<i></i>"
    d, p = diff(row, system)
    if system != FP and p is not None and p < 0.05:
        mark = '<i class="dn" title="significantly lower than Pronunciada">▼</i>' if d > 0 else '<i class="up" title="significantly higher than Pronunciada">▲</i>'
    return f'<td{" class=own" if system == FP else ""}>{num(entry["value"])}{dagger}{mark}</td>'


def result_table(rows: list[tuple[str, str, dict]], systems: tuple[str, ...]) -> str:
    """rows = (label, small print, data row). A system with no number in any row is left out unless it is a full system."""
    keep = [s for s in systems if s in (OP, AZ, LX, TP) or any(val(r, s) is not None for _, _, r in rows)]
    head = "<tr><th>Test</th>" + "".join(f"<th>{NAME[s]}</th>" for s in keep) + "</tr>"
    body = "".join(f"<tr><td>{label}<small>{small}</small></td>{''.join(cell(r, s) for s in keep)}</tr>" for label, small, r in rows)
    return f'<div class="tw"><table class="bt"><thead>{head}</thead><tbody>{body}</tbody></table></div>'


ALL = (FP, I8, OP, OPD, AZ, LX, TP, ZIPA)


def n_of(r: dict, what: str) -> str:
    return f"{r['n_items']:,} {what}"


def sig_list(row: dict, systems) -> list[str]:
    return [NAME[s] for s in systems if (diff(row, s)[1] or 1) < 0.05]


def tables() -> str:
    so = lambda unit: find("speechocean762 test", "Pearson", unit)
    t1 = result_table([
        ("Sounds", n_of(so("sound"), "sounds"), so("sound")),
        ("Words", n_of(so("word"), "words"), so("word")),
        ("Sentence, accuracy", n_of(so("sentence accuracy"), "recordings"), so("sentence accuracy")),
        ("Sentence, total score", "", so("sentence total")),
        ("Sentence, fluency", "", so("sentence fluency")),
        ("Sentence, prosody", "", so("sentence prosodic")),
    ], ALL)
    l2, ep = find("L2-ARCTIC test", "AUC", "sound"), find("EpaDB test", "AUC", "sound")
    t2 = result_table([
        ("L2-ARCTIC", f"{l2['n_speakers']} learners, six first languages", l2),
        ("EpaDB", f"{ep['n_speakers']} Spanish-speaking learners", ep),
    ], ALL)
    om_w = find("OMPAL test (Mandarin", "AUC", "word (character): any error")
    om_s = find("OMPAL test (Mandarin", "Pearson", "sentence accuracy")
    t3 = result_table([
        ("Characters with any error", f"area under the ROC curve, {n_of(om_w, 'characters')}", om_w),
        ("Sentence accuracy", f"Pearson with teacher scores, {n_of(om_s, 'recordings')}", om_s),
    ], ALL)
    he, he_s = find("Heroico USMA", "AUC", "es/heroico"), find("Heroico USMA", "speaker_AUC", "es/heroico")
    t4 = result_table([
        ("Recordings", f"{n_of(he, 'recordings')}, compared within each prompt", he),
        ("Speakers", f"{he_s['n_speakers']} speakers, mean score of each", he_s),
    ], ALL)
    le, le_s = find("LeaP de read stories", "AUC", "de/leap"), find("LeaP de read stories", "speaker_AUC", "de/leap")
    t5 = result_table([
        ("Recordings", f"{n_of(le, 'sentences')}, compared within each sentence", le),
        ("Speakers", f"{le_s['n_speakers']} speakers, mean score of each", le_s),
    ], ALL)
    saa, lpe = find("Speech Accent Archive (<= 30 s)", "AUC", "en/saa"), find("LeaP en read stories", "AUC", "en/leap")
    t6 = result_table([
        ("Speech Accent Archive", f"{saa['n_items']} speakers reading one paragraph", saa),
        ("LeaP English", f"{n_of(lpe, 'sentences')}, {lpe['n_speakers']} speakers", lpe),
    ], ALL)
    legend = ('<p class="note">▲ significantly higher than Pronunciada, ▼ significantly lower (paired test over speakers, p &lt; 0.05). '
              "No mark means no significant difference. Pronunciada is the full model unless the column says int8. "
              f"† TonePerfect was run on {TP_EN:,} of the {so('sentence accuracy')['n_items']:,} recordings, with every speaker included; "
              "its figure and mark are computed on those recordings.</p>")
    he_sig = sig_list(he, (OP, AZ, LX))
    he_s_sig = sig_list(he_s, (OP, AZ, LX))
    spanish = ("With 18 speakers the intervals are wide. " +
               (f"On recordings, the difference to {' and '.join(he_sig)} is significant; " if he_sig else "On recordings, no difference is significant; ") +
               (f"by speaker, the differences to {' and '.join(he_s_sig)} are." if len(he_s_sig) > 1 else
                f"by speaker, the difference to {he_s_sig[0]} is." if he_s_sig else "by speaker, none is."))
    return f"""
<h3>English, scored by experts</h3>
<p>The speechocean762 test set has {so('sentence total')['n_items']:,} recordings by {so('sentence total')['n_speakers']} Mandarin-speaking learners of English, many of them children. Five experts scored every sound, word and sentence. The figure is the Pearson correlation between a system's score and the experts' score.</p>
{t1}{legend}
<h3>English, sounds marked as mispronounced</h3>
<p>In L2-ARCTIC and EpaDB, annotators marked each sound a learner mispronounced. The figure is the area under the ROC curve for separating the marked sounds from the others. Neither corpus was used to train or tune Pronunciada or OpenPronounce. Lingolix gives no score for individual sounds, and neither it nor TonePerfect was run on these sets.</p>
{t2}
<h3>Mandarin</h3>
<p>The OMPAL test set has {om_w['n_speakers']} speakers, {om_s['n_speakers']} of them French-speaking learners of Mandarin. Teachers marked every character and scored every sentence. OpenPronounce and Lingolix do not support Mandarin. TonePerfect scored all {TP_ZH} recordings; it states that it used other OMPAL recordings during its development.</p>
{t3}
<h3>Spanish</h3>
<p>Heroico has no expert scores. It has the same prompts read by 10 learners and 8 native speakers of Spanish, so the test is whether a system scores the native recording of a prompt above the learner's (area under the ROC curve). {spanish}</p>
{t4}
<h3>German</h3>
<p>LeaP has the same stories read by {le['n_speakers'] - 7} learners and 7 native speakers of German, again without expert scores. The corpus states no license, so its recordings were scored on the laptop only and were not sent to any cloud service.</p>
{t5}
<h3>English, learners against native speakers</h3>
<p>Two further English sets with the same kind of label: learners and native speakers reading the same text.</p>
{t6}"""


def sample_table() -> str:
    rows = SAMPLE["rows"]
    f = lambda test, metric, unit: find(test, metric, unit, rows)
    so, om = "speechocean762 test, Chivox sample", "OMPAL test, Chivox sample"
    picks = [
        ("English words", "Pearson with expert scores", f(so, "Pearson", "word")),
        ("English sentence, total", "Pearson", f(so, "Pearson", "sentence total")),
        ("English sentence, accuracy", "Pearson", f(so, "Pearson", "sentence accuracy")),
        ("English sentence, fluency", "Pearson", f(so, "Pearson", "sentence fluency")),
        ("English sentence, prosody", "Pearson", f(so, "Pearson", "sentence prosodic")),
        ("Mandarin characters, any error", "area under the ROC curve", f(om, "AUC", "word (character): any error")),
        ("Mandarin characters, consonant or vowel", "area under the ROC curve", f(om, "AUC", "word (character): consonant or vowel error")),
        ("Mandarin characters, tone only", "area under the ROC curve", f(om, "AUC", "word (character): tone-only error")),
        ("Mandarin sentence, accuracy", "Pearson with teacher scores", f(om, "Pearson", "sentence accuracy")),
        ("Mandarin sentence, fluency", "Pearson with teacher scores", f(om, "Pearson", "sentence fluency")),
    ]
    n_en, n_zh = SAMPLE["chivox"]["speechocean762"]["answered"], SAMPLE["chivox"]["ompal"]["answered"]
    table = result_table(picks, (FP, I8, CX, AZ))
    return f"""
<p>Chivox was measured on a free trial, which covered {n_en} speechocean762 recordings and {n_zh} OMPAL recordings chosen at random with every speaker represented. The table compares the systems on exactly those recordings, so its figures differ from the full-set tables above. Chivox accepts only MP3 audio and received the recordings as 64 kbit/s MP3, while the other systems scored the original files. Its English sentence engine returns no scores for individual sounds.</p>
{table}"""


# ---- the two summary tables ------------------------------------------------------------------

def langs_of(e: dict) -> str:
    return ", ".join(LANG[lg] for lg in sorted(e["languages"], key=["en", "zh", "es", "de"].index))


def combined_rows():
    """(system, where it runs, combined entry) for every system compared with Pronunciada."""
    rows = []
    for s, where, src in ((AZ, "Cloud", RES), (LX, "Cloud", RES), (TP, "Cloud", TPS), (CX, "Cloud", SAMPLE),
                          (OP, "Device", RES), (OPD, "Device", RES), (ZIPA, "Device", RES)):
        e = src["combined"]["systems"].get(s, {})
        if "value" in e:
            rows.append((s, where, e, src))
    return rows


def combined_table() -> str:
    body = []
    for s, where, e, src in combined_rows():
        d = e["base_minus"]
        note = ""
        if src is TPS:
            note = f"English on {TP_EN:,} recordings"
        elif src is SAMPLE:
            note = f"{SAMPLE['chivox']['speechocean762']['answered']} + {SAMPLE['chivox']['ompal']['answered']} recordings"
        kind = "on the device" if where == "Device" else "cloud"
        body.append(f"<tr><td>{NAME[s]}<small>{kind} · {langs_of(e)}{f'; {note}' if note else ''}</small></td>"
                    f"<td class=own>{num(e['base_value'])}</td><td>{num(e['value'])}</td>"
                    f"<td>{d['value']:+.3f}<small>{pval(d['p'])}</small></td></tr>")
    head = "<tr><th>Compared with</th><th>Pronunciada</th><th>Other</th><th>Difference</th></tr>"
    return f'<div class="tw"><table class="bt fit"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>'


def combined_text() -> str:
    rows = combined_rows()
    ahead = [s for s, _, e, _ in rows if e["base_minus"]["value"] > 0]
    sig = [s for s, _, e, _ in rows if e["base_minus"]["value"] > 0 and e["base_minus"]["p"] < 0.05]
    behind = [s for s, _, e, _ in rows if e["base_minus"]["value"] <= 0]
    full = RES["combined"]["systems"][FP]
    parts = []
    if not behind:
        parts.append(f"Pronunciada has the higher combined score against every system, and the difference is significant against "
                     + ", ".join(NAME[s] for s in sig[:-1]) + (" and " if len(sig) > 1 else "") + NAME[sig[-1]] + ".")
    else:
        parts.append("Pronunciada has the higher combined score against " + ", ".join(NAME[s] for s in ahead) + ".")
    ns = [s for s in ahead if s not in sig]
    if ns:
        parts.append(" Against " + " and ".join(NAME[s] for s in ns) + " the difference is not significant.")
    return (f"<p>{''.join(parts)} Over all four languages Pronunciada's combined score is {num(full['value'])}.</p>"
            "<p class=\"note\">Checks of the construction: weighting the languages by their "
            "number of speakers, or scoring every level the judges marked (sounds, words, sentences, marked sounds, Mandarin characters), "
            "keeps Pronunciada ahead of every commercial system; against OpenPronounce the difference is not significant in any version "
            f'(<a href="{GITHUB}/blob/main/results/tables/learner_benchmark_combined_variants.json">all versions</a>). '
            "Pronunciada used the training parts of speechocean762 and OMPAL for its own fitting; the Spanish and German tests are small.</p>")


ON_DEVICE = (I8, FP, OP, OPD, ZIPA)


def ondevice_table() -> str:
    C = RES["combined_en_es_de"]["systems"]
    body = []
    for s in ON_DEVICE:
        e = C[s] if "value" in C[s] else RES["combined"]["systems"][s]       # the distilled model: English only
        c1 = cost(s if s != ZIPA else FP, 1) or {}
        size = size_mb(s)
        mem = c1.get("peak_rss_mb") if s != ZIPA else None
        t = c1.get("p50_s") if s != ZIPA else None
        sub = {I8: "int8 model", FP: "full model", OP: "default mode", OPD: "distilled model, English only",
               ZIPA: "the speech model Pronunciada is built on"}[s]
        name = "Pronunciada" if s in (I8, FP) else NAME[s]
        score = num(e["value"]) + (f"<small>English only; Pronunciada {num(e['base_value'])}</small>" if s == OPD else "")
        body.append(f"<tr{' class=mine' if s in (I8, FP) else ''}><td>{name}<small>{sub}</small></td>"
                    f"<td{' class=own' if s in (I8, FP) else ''}>{score}</td><td>{mb(size)}</td>"
                    f"<td>{f'{t:.2f} s' if t else '–'}</td></tr>")
    head = ("<tr><th>System</th><th>Combined score<br>English, Spanish, German</th><th>Model files</th>"
            "<th>Time per recording</th></tr>")
    return f'<div class="tw"><table class="bt fit"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table></div>'


def ondevice_text() -> str:
    C = RES["combined_en_es_de"]["systems"]
    best = max((s for s in ON_DEVICE if "value" in C[s]), key=lambda s: C[s]["value"])
    assert best in (I8, FP), "the on-device sentence below would be false"
    smallest = min((I8, FP, OP, OPD), key=size_mb)
    fastest = min((I8, FP, OP, OPD), key=lambda s: cost(s, 1)["p50_s"])
    assert smallest == I8 and fastest in (I8, FP)
    dst = RES["combined"]["systems"][OPD]
    assert dst["base_minus"]["value"] > 0
    one = cost(FP, 1)
    return (f"<p>Among the systems that run on the learner's device, Pronunciada has the highest combined score and the smallest model. "
            f"Its int8 model is {size_mb(OP) / size_mb(I8):.0f} times smaller than OpenPronounce in default mode and scores a recording "
            f"{cost(OP, 1)['p50_s'] / cost(I8, 1)['p50_s']:.0f} times faster; against the distilled model it is {size_mb(OPD) / size_mb(I8):.1f} times smaller "
            f"and {cost(OPD, 1)['p50_s'] / cost(I8, 1)['p50_s']:.1f} times faster. OpenPronounce has no Mandarin, so the score here covers the three languages "
            "both systems support, and the distilled model covers English only.</p>"
            f"<p class=\"note\">Model files are every file a system loads to score English; peak memory is in the chart below. Time: one CPU thread, the median over "
            f"{one['n_clips']} speechocean762 recordings of {one['mean_clip_seconds']:.1f} seconds on average, {one['machine']}, audio decoding included. "
            "ZIPA alone uses Pronunciada's backbone file and was not timed separately.</p>")


# ---- where another system is ahead ----------------------------------------------------------

UNIT = {"sound": "sounds", "word": "words", "sentence total": "sentence total", "sentence accuracy": "sentence accuracy",
        "sentence fluency": "sentence fluency", "sentence prosodic": "sentence prosody"}
SET = {"es/heroico": "Spanish, Heroico", "es/heroico+heroico_natives": "Spanish, Heroico with 30 more native speakers",
       "en/saa": "English, Speech Accent Archive", "en/saa_all": "English, Speech Accent Archive, all recordings",
       "zh/ompal_nl": "Mandarin, OMPAL learners against natives", "de/leap": "German, LeaP",
       "de/leap_exact": "German, LeaP, sentences read exactly as printed", "de/leap_storyB": "German, LeaP, story B",
       "de/leap_storyC": "German, LeaP, story C", "de/leap_superlearner": "German, LeaP, very advanced against other learners",
       "en/leap": "English, LeaP", "en/leap_exact": "English, LeaP, sentences read exactly as printed",
       "en/leap_superlearner": "English, LeaP, very advanced against other learners",
       "en/speechocean762": "English, speechocean762, sentence total", "zh/ompal": "Mandarin, OMPAL, sentence accuracy"}
HOW = {"AUC": "by recording", "speaker_AUC": "by speaker", "macro_AUC": "mean over prompts", "C": "concordance"}


def row_name(r: dict) -> str:
    if r["unit"] in SET:
        return f"{SET[r['unit']]} ({HOW[r['metric']]})"
    test = r["test"]
    if test.startswith("speechocean762"):
        return f"English, speechocean762, {UNIT[r['unit']]} ({r['metric']})"
    if test.startswith(("L2-ARCTIC", "EpaDB")):
        return f"English, {test.split()[0]}, marked sounds"
    if test.startswith("OMPAL"):
        what = {"word (character): any error": "characters with any error", "word (character): tone-only error": "characters with only a tone error",
                "word (character): consonant or vowel error": "characters with a consonant or vowel error"}.get(r["unit"], r["unit"])
        return f"Mandarin, OMPAL, {what} ({r['metric']})"
    raise KeyError(test)


def ahead() -> str:
    sig, more = [], 0
    def scan(rows, systems, tag=""):
        nonlocal more
        for r in rows:
            if "(-duration)" in r["unit"] or r["metric"] in ("F1", "speaker_Spearman", "speaker_mean_diff"):
                continue
            for s in systems:
                d, p = diff(r, s)
                if d is None or d >= 0:
                    continue
                if p < 0.05:
                    sig.append((row_name(r) + tag, s, val(r, s), val(r, FP), p))
                else:
                    more += 1
    scan(RES["rows"], (OP, OPD, AZ, LX))
    scan([r for r in TPS["rows"] if r["test"].startswith("speechocean762")], (TP,), ", TonePerfect sample")
    scan([r for r in TPS["rows"] if r["test"].startswith("OMPAL")], (TP,))
    scan(SAMPLE["rows"], (CX,), ", Chivox sample")
    body = "".join(f"<tr><td>{name}</td><td>{NAME[s]} {num(v)}</td><td class=own>{num(ours)}</td><td>{pval(p).replace('p ', '')}</td></tr>"
                   for name, s, v, ours, p in sig)
    head = "<tr><th>Test</th><th>System ahead</th><th>Pronunciada</th><th>p</th></tr>"
    return (f'<p>Out of every test and every system in the full tables, these are the rows where another system scores significantly higher than Pronunciada.</p>'
            f'<div class="tw"><table class="bt"><thead>{head}</thead><tbody>{body}</tbody></table></div>'
            f"<p>In {more} further rows another system's figure is higher without a significant difference. "
            f'They are listed in the <a href="{FULL}">full tables</a>.</p>')


# ---- page -----------------------------------------------------------------------------------

EXTRA_CSS = """
h3{font-family:var(--display);font-weight:400;font-size:22px;line-height:28px;letter-spacing:-.015em;margin:56px 0 14px}
.sc-grid4{display:grid;grid-template-columns:1fr 1fr;gap:26px 24px;border-top:1px solid var(--rule);padding-top:20px}
.sc-cell .ch-name{margin-bottom:6px}
svg.sc{display:block;width:100%;height:auto;overflow:visible}
.sc-grid{stroke:var(--rule);stroke-width:1}
.sc-tick{font:400 10.5px var(--sans);fill:var(--soft)}
.sc-val{font:400 11px var(--sans);fill:var(--soft);font-variant-numeric:tabular-nums;paint-order:stroke;stroke:var(--card);stroke-width:3.5px;stroke-linejoin:round}
.sc-val.own{fill:var(--text);font-weight:600}
.sc-cloud{stroke:var(--faint);stroke-width:1;stroke-dasharray:2 4;opacity:.7}
.sc-dia{fill:var(--card);stroke:var(--soft);stroke-width:1.4}
.sc-pt.own{fill:#1763c6}
.sc-pt.oth{fill:#8c8782}
.sc-pt.ring{fill:var(--card);stroke:#8c8782;stroke-width:1.6}
.sc-join{stroke-width:1.4;opacity:.45}
.sc-join.own{stroke:#1763c6}
.sc-join.oth{stroke:#8c8782}
.size .ch-key i.own{background:#1763c6}
.size .ch-key i.oth{background:#8c8782}
.size .ch-key i.ring{background:var(--card);box-shadow:inset 0 0 0 1.6px #8c8782}
.size .ch-key i.dia{background:var(--card);box-shadow:inset 0 0 0 1.4px var(--soft);border-radius:1px;transform:rotate(45deg) scale(.82)}
.cost .sp-row{grid-template-columns:272px 1fr 104px}
.cost .cg + .cg{margin-top:18px}
.cg-h{font-size:13px;line-height:19px;color:var(--soft);padding:0 0 10px}
.cost .sp-row{padding:13px 0}
.tw{overflow-x:auto;margin:26px 0 30px;-webkit-overflow-scrolling:touch}
table.bt{border-collapse:collapse;width:100%;min-width:640px;font-size:14.5px;line-height:21px;font-variant-numeric:tabular-nums}
.bt th{font-weight:500;font-size:12.5px;line-height:17px;color:var(--soft);text-align:right;vertical-align:bottom;padding:0 0 10px 16px}
.bt td{padding:11px 0 11px 16px;border-top:1px solid var(--rule);text-align:right;white-space:nowrap;vertical-align:top}
.bt th:first-child,.bt td:first-child{text-align:left;padding-left:0;white-space:normal}
.bt td small{display:block;font-size:12.5px;line-height:18px;color:var(--soft)}
.bt td.own{font-weight:600}
.bt td.na{color:var(--faint);font-size:12.5px}
.bt i{display:inline-block;width:13px;text-align:right;font-style:normal;font-size:8.5px;vertical-align:1.5px}
.bt th:not(:first-child){padding-right:13px}
.bt td:first-child{min-width:188px}
.bt i.dn{color:var(--faint)}
.back{font-size:15px}
table.bt.fit{min-width:0}
@media (max-width:660px){
  h3{font-size:20px;line-height:26px;margin:44px 0 12px}
  .sc-grid4{grid-template-columns:1fr}
  .cost .sp-row{grid-template-columns:1fr 104px}
  table.bt.fit{font-size:13.5px}
  .bt.fit td,.bt.fit th{padding-left:7px}
  .bt.fit th:not(:first-child){padding-right:4px}
  .bt.fit td:first-child{min-width:0;white-space:normal}
  .bt.fit td{white-space:normal}
}
"""

LIMITS = """
<ul>
<li>The kind of label differs by language. Expert or teacher scores exist for English and Mandarin, and marked sounds for English only. Spanish and German have only the distinction between learners and native speakers, which rewards anything that differs between the two groups, speaking rate included.</li>
<li>Several sets have few speakers: 6 in L2-ARCTIC, 20 in EpaDB, 18 learners in OMPAL, 10 learners and 8 native speakers in Heroico, 7 native speakers in LeaP German. One speaker can move such a figure by several points.</li>
<li>There is no learner data here for Japanese, Korean, Russian, Portuguese, French or Italian. These tests say nothing about those languages.</li>
<li>The commercial systems are incomplete. Azure was not run on German, Lingolix on speechocean762 and Heroico only, TonePerfect on OMPAL and part of speechocean762, and Chivox on a sample of two sets. No other commercial service was run.</li>
<li>Pronunciada and OpenPronounce both used speechocean762 to fit their scores. Neither used L2-ARCTIC, EpaDB, Heroico, the Speech Accent Archive or LeaP. TonePerfect states that it used other OMPAL recordings during its development.</li>
<li>Disk, memory and time were measured on one laptop. Nothing was measured on a phone.</li>
</ul>"""


def page() -> str:
    home = HOME.read_text(encoding="utf-8")
    css = re.search(r"<style>(.*?)</style>", home, re.S).group(1).replace("url(assets/", "url(../assets/")
    fonts = "\n".join(re.findall(r'<link[^>]+fonts\.g[^>]+>', home))
    one = cost(FP, 1)
    desc = ("Pronunciada, OpenPronounce, Azure Pronunciation Assessment, Lingolix, TonePerfect and Chivox scored on the same "
            "recordings of language learners in English, Mandarin, Spanish and German, with size on disk, memory and time.")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmarks on real learner speech · Pronunciada</title>
<meta name="description" content="{desc}">
<meta property="og:title" content="Benchmarks on real learner speech · Pronunciada">
<meta property="og:description" content="{desc}">
<meta property="og:type" content="article">
<meta property="og:url" content="https://pronunciada.dev/benchmarks/">
<meta property="og:image" content="https://pronunciada.dev/assets/social.jpg">
<link rel="canonical" href="https://pronunciada.dev/benchmarks/">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="../assets/favicon.svg" type="image/svg+xml">
<link rel="icon" href="../assets/icon-192.png" type="image/png" sizes="192x192">
<link rel="icon" href="/favicon.ico" sizes="48x48">
<link rel="apple-touch-icon" href="../assets/apple-touch-icon.png">
{fonts}
<style>{css}{EXTRA_CSS}</style>
</head>
<body>
<header class="top"><div class="in">
  <a class="name" href="../" aria-label="Pronunciada"><img src="../assets/logo-light.svg" alt="Pronunciada"></a>
  <nav class="nav">
    <a class="link" href="../#benchmarks">Overview</a>
    <a class="link" href="../#how">How it works</a>
    <a class="link" href="../#start">Get started</a>
    <a class="pill small dark" href="{GITHUB}">GitHub</a>
  </nav>
</div></header>

<main>
<p class="crumb">Benchmarks · {RUN_DATE}</p>
<h1>Benchmarks on real learner speech</h1>
<p class="lead">Pronunciada, OpenPronounce, Azure Pronunciation Assessment, Lingolix, TonePerfect and Chivox scored the same recordings of language learners in English, Mandarin, Spanish and German. No synthetic speech is used on this page.</p>
<p>The comparison covers complete pronunciation-assessment systems: a library, app or API that takes a recording and the text that was read, and returns scores for sounds, words and the sentence. OpenPronounce is the other open-source system of that kind. Azure, Lingolix, TonePerfect and Chivox are commercial cloud services. Every system received the same audio and text.</p>

<h2>Combined score</h2>
<p>One number per system: how closely its scores follow human judgement, averaged over the four languages with learner recordings. In English and Mandarin the judges are experts who graded the pronunciation accuracy of each recording. In Spanish and German they are native speakers, whose recordings a system should score above the learners' recordings of the same text. The number is the share of pairs of recordings that a system puts in the same order as the judges: 0.5 is chance and 1 is perfect agreement. Each system is compared with Pronunciada on the languages both of them cover.</p>
{combined_table()}
{combined_text()}

<h2>On the device</h2>
<p>The same score for the systems that run on the learner's own device, next to what they cost to run there.</p>
{ondevice_table()}
{ondevice_text()}

<h2>Accuracy and size</h2>
<p>The chart places each system by the size of its model files and by its score on four English tests. Cloud services have no download and are drawn at the right.</p>
{chart_size()}
{size_text()}

<h2>Memory and time</h2>
<p>The same four builds, on the same laptop.</p>
{chart_cost()}

<h2>Results by language</h2>
{tables()}

<h2>Chivox, on a sample</h2>
{sample_table()}

<h2>Where another system is ahead</h2>
{ahead()}

<h2>What these tests do not show</h2>
{LIMITS}

<h2>Method</h2>
<p>Versions: Pronunciada 1.1.0, with the full model and the int8 model. OpenPronounce 1.1.0, the latest release on the day of the runs, in default mode and with its distilled English phone model; the <a href="{FULL}">full tables</a> also have OpenPronounce 0.4.0, the release that was current when Pronunciada 1.0 came out. Azure Pronunciation Assessment through its REST API, region westus, November 2026. Lingolix API, model 2026-11. TonePerfect API version 2 with fallback off, 24 and 25 November 2026. Chivox speech evaluation API, sentence engines for English and Mandarin. ZIPA alone is zipa-small-crctc-500k with the standard goodness-of-pronunciation score.</p>
<p>Every interval and every difference comes from {RES['n_boot']:,} resamples of speakers, with the systems compared on the same resamples. A difference is called significant at p &lt; 0.05, without correction for the number of comparisons. Disk, memory and time were measured on an {one['machine']} with no GPU.</p>
<p>The <a href="{FULL}">full tables</a> on GitHub have every row with its interval, the rank-correlation versions of the tests, the construction of the German set and the commands to reproduce the page.</p>
<p class="back"><a href="../">Back to the overview</a></p>

<footer>
  <div class="f-top">
    <img src="../assets/logo-light.svg" alt="Pronunciada" height="17">
    <nav>
      <a href="{GITHUB}">GitHub</a>
      <a href="https://pypi.org/project/pronunciada/">PyPI</a>
      <a href="https://huggingface.co/spaces/kimitonatsu/pronunciada">Hugging Face</a>
      <a href="{GITHUB}/blob/main/results/SCOREBOARD.md">Scoreboard</a>
      <a href="{GITHUB}/blob/main/docs/DESIGN.md">Design</a>
      <a href="{GITHUB}/blob/main/LICENSE">License</a>
    </nav>
  </div>
  <p>Code under Apache-2.0. Azure is a trademark of Microsoft. OpenPronounce, Lingolix, TonePerfect and Chivox are the names of their owners' products. This project is not affiliated with or endorsed by any of them.</p>
</footer>
</main>
</body>
</html>
"""


def github_charts() -> None:
    """Dark renders of the two charts for the write-up on GitHub, using the page's own styles."""
    home = HOME.read_text(encoding="utf-8")
    css = re.search(r"<style>(.*?)</style>", home, re.S).group(1)
    dark = (":root{--text:#fff;--soft:rgba(255,255,255,.62);--faint:rgba(255,255,255,.5);--rule:rgba(255,255,255,.14);"
            f"--page:{INK};--card:{INK};--pale:#34343a;--accent:linear-gradient(90deg,#7db8ff,#5eead4)}}"
            f"body{{background:{INK}}}.chart{{box-shadow:none;margin:0;border-radius:0;padding:36px 40px 30px}}"
            ".sc-pt.own{fill:#fff}.sc-join.own{stroke:#fff}.size .ch-key i.own{background:#fff}"
            ".sc-pt.oth,.size .ch-key i.oth{fill:#77777f;background:#77777f}.sc-join.oth{stroke:#77777f}"
            ".sc-pt.ring{stroke:#77777f}.size .ch-key i.ring{box-shadow:inset 0 0 0 1.6px #77777f}"
            ".sp-row.own .ch-track i{background:#fff}.ch-row b,.sp-row b{color:rgba(255,255,255,.5)}.ch-row.own b,.sp-row.own b{color:#fff}")
    head = f"<!doctype html><html><head><meta charset='utf-8'><style>{font_css()}{css}{EXTRA_CSS}{dark}</style></head><body>"
    for body, name in ((chart_size(), "benchmark-size.png"), (chart_cost(), "benchmark-cost.png")):
        html_ = head + body + "</body></html>"
        render(html_, ROOT / "docs/assets" / name, 920, height_of(html_, 920))


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page(), encoding="utf-8")
    github_charts()
    print(f"wrote {OUT.relative_to(ROOT)}, docs/assets/benchmark-size.png and benchmark-cost.png")
