"""LeaP corpus (Learning Prosody in a Foreign Language, Universität Bielefeld, 2001-2003): learners and
native speakers of German and of English reading the same short stories, with manual Praat annotation.

    python -m bench.learners.leap [--lang de|en|both]   # fetch (resumable, sha256 checked) + build + select

Source: https://sourceforge.net/projects/leapcorpus/files/ (files "LeaP corpus - German.zip",
"LeaP corpus - English.zip", "LeapCorpus_Manual.pdf"; sizes and sha256 below, sha256 of the English
archive as published by SourceForge, the others measured on download).

LICENCE: the corpus states no licence, neither on SourceForge nor in its manual. Default copyright
applies. It is used here for held-out evaluation only: nothing is fitted on it, and no audio, text,
annotation or per-recording output from it is committed or redistributed. The story texts are read
from the manual at build time and kept with the manifests outside the repository.

What is used: the read-speech recordings only (file names ``<spk>_<l1>_<target>_<sex>_read_<group>``),
whose text is printed in the manual's appendix: story A ("The tiger and the mouse", English), stories
B and C (German; C is a loose translation of A). Retellings, interviews and nonsense word lists are not
used. Group codes (manual, "File names"): ``na`` native speaker, ``sl`` (written ``s1`` in the file
names) "superlearner" = very advanced learner, ``e1`` / ``e2`` before / after a stay abroad, ``c1`` /
``c2`` / ``c3`` before / after one or two prosody courses, ``ot`` other learner. The metadata of the
corpus has no proficiency rating and no self-rating; it gives age, sex, native language(s), age of
onset (``AgeOfOnset``) and type of learning. ``labels.group``, ``labels.age_of_onset`` and
``labels.superlearner`` are kept for the pre-declared secondary analysis only.

CUTTING RULE (no manual edits; fixed before any system was run):
  1. The story text is split into sentences after ``.``, ``!`` or ``?`` (closing quotes included)
     when the next word starts with a capital letter or an opening quote. A first line without final
     punctuation is the title and is not used. A piece of fewer than 4 words is joined to the next
     piece (the last piece to the previous one).
  2. The word tier of the recording's TextGrid (orthographic, lower case, umlauts written ae / oe /
     ue, ß as ss) is aligned to the story's words by unit-cost edit distance on normalised tokens
     (lower case, umlauts and ß written as in the tier, apostrophes and other non-letters dropped).
  3. An item is the stretch from the start of the tier word aligned to the sentence's first word to
     the end of the tier word aligned to its last word, widened by up to 0.15 s on each side but never
     past the midpoint of the gap to the neighbouring tier word. The item text is the sentence as
     printed in the manual (the text the speaker was given).
  4. Excluded (counted in build.json): the sentence's first or last word not found in the tier
     exactly (the cut would be unsafe); more read deviations (tier words inserted, story words left
     out or replaced, inside the stretch) than half the sentence's words; items shorter than 0.5 s.
     ``labels.read_deviations`` keeps the count for every kept item.

TEST SELECTION (seed 23): one reading per speaker, the first in the order na, sl, ot, e1, c1 (a
speaker's later readings e2, c2, c3 go to ``prepost.jsonl``, secondary analysis only). Every speaker
contributes up to 14 items, assigned by rotating through a seeded order of the story's sentences from
an offset that advances by 14 per speaker, separately for natives and for learners, so that each
sentence is read by several natives and several learners (a sentence that was excluded for a speaker
is skipped). There is no dev part: nothing is fitted, and all speakers are needed for the test.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bench.data.common import cache_root, write_flac
from bench.learners import fetch, hours, make_record, root, sha256, write_jsonl

BASE = "https://downloads.sourceforge.net/project/leapcorpus/"
FILES = {
    "manual": ("LeapCorpus_Manual.pdf", 386116, None),
    "de": ("LeaP corpus - German.zip", 1518787812, None),
    "en": ("LeaP corpus - English.zip", 1262987879, "6ea8a4b4c21c071e4b57b1f75f7b9dd118859cf6f145ec7c39e99f207662e96d"),
}
LICENCE = ("No licence is stated by the corpus (SourceForge page and manual). Used for held-out "
           "evaluation only; nothing from it is redistributed or committed.")
SEED, PER_SPEAKER, PAD, MIN_WORDS, MIN_DUR = 23, 14, 0.15, 4, 0.5
GROUP_ORDER = ("na", "s1", "sl", "ot", "e1", "c1", "e2", "c2", "c3")
STORIES = {"en": ("A",), "de": ("B", "C")}
CORPUS = {"de": "leap-de", "en": "leap-en"}
TARGET = {"de": "ger", "en": "eng"}


def downloads() -> Path:
    return cache_root() / "learnerbench" / "downloads" / "leap"


def tool_python() -> Path:
    """A private virtualenv holding only pypdf, used to read the manual's text (not a scoring tool)."""
    return cache_root() / "learnerbench" / "venv-tools" / "bin" / "python"


# ----------------------------------------------------------------------------- story texts

def manual_text(pdf: Path) -> str:
    out = downloads() / "manual.txt"
    if not out.exists():
        code = ("import sys, pypdf; r = pypdf.PdfReader(sys.argv[1]); "
                "sys.stdout.write('\\n'.join(p.extract_text() for p in r.pages))")
        txt = subprocess.run([str(tool_python()), "-I", "-c", code, str(pdf)], capture_output=True, check=True,
                             text=True).stdout
        out.write_text(txt, encoding="utf-8")
    return out.read_text(encoding="utf-8")


def story_lines(manual: str, name: str) -> list[str]:
    lines = manual.split("\n")
    heads = {k: i for i, ln in enumerate(lines) for k in ("A", "B", "C") if ln.strip() == f"Story {k}"}
    start = heads[name] + 1
    end = next(i for i in range(start, len(lines)) if lines[i].strip().startswith(("Story ", "Word list")))
    body = [ln.strip() for ln in lines[start:end]]
    return [ln for ln in body if ln and not re.fullmatch(r"\d+", ln)]


SPLIT = re.compile(r"(?<=[.!?])([\"“”»«„']*)\s+(?=[\"„“»«]|[A-ZÄÖÜ])")


def sentences(lines: list[str]) -> tuple[str | None, list[str]]:
    title = None
    if lines and not re.search(r"[.!?\"“”]\s*$", lines[0]):
        title, lines = lines[0], lines[1:]
    text = " ".join(lines)
    pieces, last = [], 0
    for m in SPLIT.finditer(text):
        pieces.append(text[last:m.end(1)].strip())
        last = m.end()
    pieces.append(text[last:].strip())
    pieces = [p for p in pieces if p]
    merged: list[str] = []
    carry = ""
    for p in pieces:
        p = (carry + " " + p).strip() if carry else p
        if len(tokens(p)) < MIN_WORDS:
            carry = p
            continue
        merged.append(p)
        carry = ""
    if carry:
        merged[-1] = merged[-1] + " " + carry
    return title, merged


def norm(w: str) -> str:
    w = w.lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    w = unicodedata.normalize("NFKD", w)
    return "".join(c for c in w if c.isalpha() and c.isascii())


def tokens(s: str) -> list[str]:
    return [t for t in (norm(x) for x in re.split(r"[\s\-–—]+", s)) if t]


# ----------------------------------------------------------------------------- TextGrid

def read_textgrid(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    t = path.read_bytes().decode("latin-1").replace("\r", "")
    tiers = {}
    for blk in re.split(r"\n\s*item \[\d+\]:", t)[1:]:
        m = re.search(r'name = "(.*?)"', blk)
        if not m:
            continue
        iv = re.findall(r'xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "(.*?)"', blk)
        tiers[m.group(1).strip()] = [(float(a), float(b), c) for a, b, c in iv]
    return tiers


def word_tier(tiers: dict) -> list[tuple[float, float, str]] | None:
    for k in ("words", "word"):
        if k in tiers:
            return [(a, b, norm(w)) for a, b, w in tiers[k] if norm(w)]
    return None


def align(ref: list[str], hyp: list[str]) -> list[tuple[int, int]]:
    """Edit-distance alignment; list of (ref index or -1, hyp index or -1) in order."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + (ri != hyp[j - 1]), D[i - 1, j] + 1, D[i, j - 1] + 1)
    out, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            out.append((i - 1, -1))
            i -= 1
        else:
            out.append((-1, j - 1))
            j -= 1
    return out[::-1]


def cut(sents: list[str], words: list[tuple[float, float, str]]):
    """Yield (sentence index, start, end, deviations) or (sentence index, None, reason, None)."""
    ref, owner = [], []
    for k, s in enumerate(sents):
        for t in tokens(s):
            ref.append(t)
            owner.append(k)
    hyp = [w[2] for w in words]
    pairs = align(ref, hyp)
    r2h = {r: h for r, h in pairs if r >= 0 and h >= 0 and ref[r] == hyp[h]}
    pos = {p: n for n, p in enumerate(pairs)}
    for k, s in enumerate(sents):
        idx = [i for i, o in enumerate(owner) if o == k]
        i0, i1 = idx[0], idx[-1]
        if i0 not in r2h or i1 not in r2h:
            yield k, None, "first or last word of the sentence not found in the word tier", None
            continue
        j0, j1 = r2h[i0], r2h[i1]
        a = pos[(i0, j0)]
        b = pos[(i1, j1)]
        dev = sum(1 for r, h in pairs[a:b + 1] if r < 0 or h < 0 or ref[r] != hyp[h])
        if dev > len(idx) / 2:
            yield k, None, "more read deviations than half the sentence's words", None
            continue
        start, end = words[j0][0], words[j1][1]
        lo = (words[j0 - 1][1] + start) / 2 if j0 > 0 else 0.0
        hi = (end + words[j1 + 1][0]) / 2 if j1 + 1 < len(words) else end + PAD
        start, end = max(start - PAD, lo, 0.0), min(end + PAD, hi)
        if end - start < MIN_DUR:
            yield k, None, "shorter than 0.5 s", None
            continue
        yield k, start, end, dev


# ----------------------------------------------------------------------------- metadata

def meta(xml: Path) -> dict:
    if not xml.exists():
        return {}
    t = xml.read_bytes().decode("utf-8", "replace")
    head = t[: t.find("<layer")] if "<layer" in t else t[:20000]
    d = {}
    for n, v in re.findall(r"<desc><name>(.*?)</name><val>(.*?)</val></desc>", head):
        if v.strip() and n in ("Age", "Sex", "AgeOfOnset", "TypeOfLearning") and n not in d:
            d[n] = v.strip()
    return d


def parse_name(stem: str) -> dict | None:
    m = re.fullmatch(r"([a-zA-Z]{2})_([a-z]{3})_([a-z]{3})_([fm])_read_([a-z0-9]{2})", stem)
    if not m:
        return None
    spk, l1, tgt, sex, grp = m.groups()
    return {"spk": spk.lower(), "l1": l1, "target": tgt, "sex": sex, "group": grp}


# ----------------------------------------------------------------------------- build

def extract(zpath: Path, dest: Path) -> Path:
    flag = dest / f".extracted-{zpath.stem}"
    if not flag.exists():
        with zipfile.ZipFile(zpath) as z:
            for n in z.namelist():
                p = Path(n)
                if n.endswith("/") or p.is_absolute() or ".." in p.parts:
                    continue
                if "_read_" not in p.name and not p.name.endswith(".pdf"):
                    continue
                out = dest / p
                if out.exists() and out.stat().st_size == z.getinfo(n).file_size:
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(n) as src, open(out, "wb") as dst:
                    dst.write(src.read())
        flag.touch()
    return dest


def build(lang: str) -> dict:
    import soundfile as sf

    dl = downloads()
    shas = {}
    for key in ("manual", lang):
        fn, size, sha = FILES[key]
        p = fetch(BASE + fn.replace(" ", "%20"), dl / fn, size=size, sha=sha)
        shas[fn] = sha256(p)
    manual = manual_text(dl / FILES["manual"][0])
    raw = extract(dl / FILES[lang][0], dl / "raw")
    sub = {"de": "German", "en": "English"}[lang]
    story_sents = {}
    titles = {}
    for st in STORIES[lang]:
        titles[st], story_sents[st] = sentences(story_lines(manual, st))
    out = root(CORPUS[lang])
    excl = Counter()
    recs = []
    readings = []
    for wav in sorted((raw / f"{sub}-wav").glob("*_read_*.wav")):
        info = parse_name(wav.stem)
        if info is None or info["target"] != TARGET[lang]:
            excl["file name not in the documented pattern"] += 1
            continue
        tg = raw / f"{sub}-Praat" / f"{wav.stem}.TextGrid"
        words = word_tier(read_textgrid(tg)) if tg.exists() else None
        if not words:
            excl["recording without a word tier (whole recording)"] += 1
            continue
        # which story: the one whose words the tier matches best
        best = None
        for st, ss in story_sents.items():
            ref = [t for s in ss for t in tokens(s)]
            hyp = [w[2] for w in words]
            ok = sum(1 for r, h in align(ref, hyp) if r >= 0 and h >= 0 and ref[r] == hyp[h]) / len(ref)
            if best is None or ok > best[1]:
                best = (st, ok)
        st, cover = best
        if cover < 0.5:
            excl["recording matches less than half of any story (whole recording)"] += 1
            continue
        native = info["group"] == "na"
        md = meta(raw / f"{sub}-xml" / f"{wav.stem}.xml")
        x, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        spk = f"leap{lang}-{info['spk']}"
        readings.append({"file": wav.name, "speaker": spk, "group": info["group"], "story": st,
                         "story_words_matched": round(cover, 3)})
        for k, a, b, dev in cut(story_sents[st], words):
            if a is None:
                excl[b] += 1
                continue
            rid = f"{CORPUS[lang]}-{info['spk']}-{info['group']}-{st}{k:02d}"
            dst = out / "audio" / spk / f"{rid}.flac"
            seg = x[int(a * sr): int(b * sr)]
            if float(np.abs(seg).max(initial=0.0)) < 1e-4:
                excl["silent segment"] += 1
                continue
            dur = sf.info(str(dst)).duration if dst.exists() else write_flac(dst, seg, sr)
            lab = {"native": native, "l1": "german" if (native and lang == "de") else
                   "english" if (native and lang == "en") else info["l1"],
                   "l1_code": info["l1"], "group": info["group"], "superlearner": info["group"] in ("s1", "sl"),
                   "story": st, "prompt_id": f"{st}{k:02d}", "read_deviations": int(dev), "sex": info["sex"],
                   "age_of_onset": _num(md.get("AgeOfOnset")), "type_of_learning": md.get("TypeOfLearning"),
                   "source_file": wav.name, "start": round(a, 3), "end": round(b, 3)}
            recs.append(make_record(rid, dst, lang, story_sents[st][k], "pool", spk, dur, lab))
    write_jsonl(out / "manifest.jsonl", recs)
    test, prepost = select(recs)
    for r in test:
        r["split"] = "test"
    for r in prepost:
        r["split"] = "prepost"
    write_jsonl(out / "test.jsonl", test)
    write_jsonl(out / "prepost.jsonl", prepost)
    (out / "stories.json").write_text(json.dumps({"titles": titles, "sentences": story_sents}, indent=1,
                                                 ensure_ascii=False))
    summary = {"corpus": CORPUS[lang], "source": BASE, "files_sha256": shas, "licence": LICENCE, "seed": SEED,
               "stories": {st: len(v) for st, v in story_sents.items()},
               "readings": readings,
               "pool": {"items": len(recs), "speakers": len({r["speaker"] for r in recs}), "hours": hours(recs)},
               "excluded_items": dict(excl), "test": describe(test), "prepost": describe(prepost)}
    (out / "build.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    s2 = {k: v for k, v in summary.items() if k != "readings"}
    print(json.dumps(s2, indent=1, ensure_ascii=False))
    return summary


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def describe(recs: list[dict]) -> dict:
    g = defaultdict(list)
    for r in recs:
        g[(r["labels"]["story"], "native" if r["labels"]["native"] else "learner")].append(r)
    d = {f"story {a}/{b}": {"items": len(v), "speakers": len({r['speaker'] for r in v}), "hours": hours(v)}
         for (a, b), v in sorted(g.items())}
    return d | {"total_items": len(recs), "total_speakers": len({r["speaker"] for r in recs}),
                "total_hours": hours(recs), "prompts": len({r["labels"]["prompt_id"] for r in recs})}


def select(recs: list[dict]) -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    by = defaultdict(lambda: defaultdict(dict))   # speaker -> group -> prompt -> record
    for r in recs:
        by[r["speaker"]][r["labels"]["group"]][r["labels"]["prompt_id"]] = r
    first = {s: min(g, key=GROUP_ORDER.index) for s, g in by.items()}
    test, prepost = [], []
    for st in sorted({r["labels"]["story"] for r in recs}):
        prompts = sorted({r["labels"]["prompt_id"] for r in recs if r["labels"]["story"] == st})
        rng.shuffle(prompts)
        n = len(prompts)
        for native in (True, False):
            spk = sorted(s for s in by if next(iter(by[s][first[s]].values()))["labels"]["story"] == st
                         and (first[s] == "na") == native)
            off = rng.randrange(n)
            for s in spk:
                have = by[s][first[s]]
                got = [have[prompts[(off + i) % n]] for i in range(n) if prompts[(off + i) % n] in have][:PER_SPEAKER]
                off = (off + PER_SPEAKER) % n
                test.extend(got)
                chosen = {r["labels"]["prompt_id"] for r in got}
                for g, items in by[s].items():
                    if g != first[s]:
                        prepost.extend(items[p] for p in sorted(chosen) if p in items)
    return test, prepost


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="both", choices=("de", "en", "both"))
    a = ap.parse_args()
    for lang in (("de", "en") if a.lang == "both" else (a.lang,)):
        build(lang)


if __name__ == "__main__":
    main()
