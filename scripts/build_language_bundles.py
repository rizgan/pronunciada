"""Per-language data bundles for score_phones: the smallest set of package data files it reads.

    python scripts/build_language_bundles.py [--langs all] [--n 6] [--out DIR] [--check] [--table out.json]

For each language, score_phones (pronunciada.core) runs in a fresh process
(pronunciada.tools.core_child) on N practice clips (bench/opt/clips.json, FLEURS train) plus a
synthetic clip with a number in the prompt; every file under pronunciada/data/ that Python opened is
recorded. ONNX Runtime opens its model files natively (not visible to the audit hook), so for every
opened `<name>.json` / `<name>.npz` a sibling `<name>.onnx` is added when it exists (the stress, tone
and English sentence-head networks). The bundle is written to `<out>/<lang>/pronunciada/data/...` with
`<out>/<lang>/manifest.json`: file, bytes, sha256, licence, source (pronunciada/data/LICENSES_SHIPPED.json),
whether the file is shared by every language, and why scoring reads it.

--check proves sufficiency: a copy of the package's Python code with ONLY that bundle as its data
folder, and a model cache holding only the backbone, scores the same clips in a fresh process; the
outputs must equal the full installation's field for field (prints identical / differing per language).

Nothing large is committed; the size table (backbone int8 / fp32 + one language) goes to --table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LANGS = ["en", "de", "es", "fr", "it", "pt", "ru", "ja", "ko", "zh"]
DEFAULT_OUT = Path("~/.cache/pronunciada/opt/mobile/bundles").expanduser()
DIGIT_PROMPT = {"en": "I have 31 books.", "de": "Ich habe 31 Bücher.", "es": "Tengo 31 libros.",
                "fr": "J'ai 31 livres.", "it": "Ho 31 libri.", "pt": "Tenho 31 livros.", "ru": "У меня 31 книга.",
                "ja": "本が31冊あります。", "ko": "책이 31권 있어요.", "zh": "我有31本书。"}
WHY = [  # (path prefix, why scoring reads it)
    ("zipa_tokens.txt", "token list (column order of the posteriors; IPA validation)"),
    ("adapters/", "per-language output adapter on the backbone log-posteriors"),
    ("native/", "native statistics per phone and position (feature z-scores)"),
    ("calib/sentence", "sentence-level calibrator / head"),
    ("calib/seqhead", "English sequence sentence head (network + normalisation)"),
    ("calib/identity/", "sound-identity word rule"),
    ("calib/prosody/", "tone / stress channel word calibrator"),
    ("calib/", "sound / word / sentence calibration models and configuration"),
    ("coach/identity/", "sound-identity head (word factor in the score)"),
    ("coach/", "head read by scoring (identity, contrast, tense or attribute evidence in the score)"),
    ("length/", "length channel parameters"),
    ("prosody/", "stress / tone model and its calibration"),
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _why(rel: str) -> str:
    return next((w for k, w in WHY if rel.startswith(k)), "read by scoring")


def _clean_env(extra=None) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION",
                                                            "LINGOLIX_API_KEY", "PYTHONPATH")}
    env.update(extra or {})
    return env


def items_for(lang: str, n: int, tmp: Path) -> list[dict]:
    """[{id, samples (.npy path), target}] for the trace / check runs."""
    import numpy as np

    from bench.opt.common import practice_clips
    from pronunciada.audio import load_audio
    from pronunciada.core import phones_for

    out = []
    for c in practice_clips([lang], n=n):
        p = tmp / f"{c['id']}.npy"
        np.save(p, load_audio(c["audio"]))
        out.append({"id": c["id"], "samples": str(p), "target": phones_for(c["text"], lang)})
    rng = np.random.default_rng(31)
    t = np.arange(48000) / 16000.0
    x = (0.1 * np.sin(2 * np.pi * 160 * t) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
    p = tmp / f"synthetic_{lang}.npy"
    np.save(p, x)
    out.append({"id": f"synthetic-{lang}", "samples": str(p), "target": phones_for(DIGIT_PROMPT[lang], lang)})
    return out


def run_child(pkg_root: Path, lang: str, items: list[dict], tmp: Path, cache: str | None = None,
              kw: dict | None = None) -> dict:
    job = tmp / f"job_{lang}_{abs(hash(str(pkg_root))) % 10**8}.json"
    job.write_text(json.dumps({"lang": lang, "items": items, "kw": kw or {"threads": 2}, "rescore": [0.0, 1.0],
                               "details": True}, ensure_ascii=False), encoding="utf-8")
    env = _clean_env({"PYTHONPATH": str(pkg_root)} | ({"PRONUNCIADA_CACHE": cache} if cache else {}))
    r = subprocess.run([sys.executable, "-m", "pronunciada.tools.core_child", str(job)], capture_output=True,
                       text=True, env=env, cwd=tmp, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-4000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def data_files(rep: dict, data_dir: Path) -> list[str]:
    dd = str(data_dir.resolve()) + os.sep
    rel = sorted({str(Path(p).resolve())[len(dd):] for p in rep["opened"]
                  if str(Path(p).resolve()).startswith(dd) and Path(p).is_file()})
    extra = []
    for r in rel:
        stem = r.rsplit(".", 1)[0]
        if r.endswith((".json", ".npz")) and (data_dir / f"{stem}.onnx").is_file():
            extra.append(f"{stem}.onnx")
    return sorted(set(rel) | set(extra))


def licences() -> dict:
    p = ROOT / "pronunciada/data/LICENSES_SHIPPED.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for key, v in d.items():
        if isinstance(v, list):
            for e in v:
                if isinstance(e, dict) and "path" in e:
                    out.setdefault(e["path"], e)
    for e in (d.get("files") or {}).values() if isinstance(d.get("files"), dict) else d.get("files") or []:
        if isinstance(e, dict) and "path" in e:
            out.setdefault(e["path"], e)
    return out


def package_copy(dest: Path, bundle_data: Path) -> Path:
    """The package's Python code (no data) + the bundle as its data folder."""
    pkg = dest / "pronunciada"
    shutil.copytree(ROOT / "pronunciada", pkg, ignore=shutil.ignore_patterns("data", "__pycache__", "*.pyc"))
    shutil.copytree(bundle_data, pkg / "data")
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default="all")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--backbone-cache", default=None,
                    help="PRONUNCIADA_CACHE holding only the backbone for --check (default: current cache)")
    ap.add_argument("--table", default=None)
    a = ap.parse_args(argv)
    langs = LANGS if a.langs == "all" else a.langs.split(",")
    out = Path(a.out).expanduser()
    data_dir = ROOT / "pronunciada" / "data"
    lic = licences()
    from pronunciada.acoustic import cache_root, download_size

    mdir = Path(cache_root()) / "models" / "zipa-small-crctc-500k"
    bb = {"int8": download_size("zipa-small-crctc-500k", "int8"), "fp32": download_size("zipa-small-crctc-500k", "fp32"),
          "tokens": (mdir / "tokens.txt").stat().st_size}
    table, sets = [], {}
    tmp = Path(tempfile.mkdtemp(prefix="bundles_"))
    for lang in langs:
        items = items_for(lang, a.n, tmp)
        rep = run_child(ROOT, lang, items, tmp)
        files = data_files(rep, data_dir)
        bad = [f for f in files if f.startswith(("lex/", "cjk/"))]
        if bad:
            raise SystemExit(f"{lang}: front-end data opened by score_phones: {bad}")
        sets[lang] = (files, items, rep)
    shared = set.intersection(*(set(v[0]) for v in sets.values())) if len(sets) == len(LANGS) else set()
    for lang, (files, items, rep) in sets.items():
        b = out / lang
        if b.exists():
            shutil.rmtree(b)
        bd = b / "pronunciada" / "data"
        rows = []
        for f in files:
            (bd / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(data_dir / f, bd / f)
            e = lic.get(f, {})
            rows.append({"file": f, "bytes": (data_dir / f).stat().st_size, "sha256": sha256(data_dir / f),
                         "licence": e.get("licence", "UNLISTED"), "source": e.get("source", ""),
                         "share_alike_question": bool(e.get("share_alike_question", False)),
                         "shared_by_all_languages": f in shared, "why": _why(f)})
        total = sum(r["bytes"] for r in rows)
        man = {"lang": lang, "files": rows, "total_bytes": total,
               "excluded": "lexicons and other front-end data (data/lex, data/cjk), coaching-only files",
               "traced_on": [it["id"] for it in items], "peak_rss_bytes_trace": rep["peak_rss_bytes"]}
        (b / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        row = {"lang": lang, "files": len(rows), "bundle_bytes": total,
               "backbone_int8_plus_bundle": bb["int8"] + bb["tokens"] + total,
               "backbone_fp32_plus_bundle": bb["fp32"] + bb["tokens"] + total,
               "unlisted_licence": [r["file"] for r in rows if r["licence"] == "UNLISTED"],
               "share_alike_question": [r["file"] for r in rows if r["share_alike_question"]]}
        if a.check:
            pk = package_copy(tmp / f"pkg_{lang}", bd)
            rep2 = run_child(pk, lang, items, tmp, cache=a.backbone_cache)
            same = json.dumps(rep2["results"], sort_keys=True) == json.dumps(rep["results"], sort_keys=True)
            outside = [p for p in rep2["opened"] if "/pronunciada/" in p and "/data/" in p
                       and not str(Path(p).resolve()).startswith(str(pk.resolve()))]
            row.update(check_identical=same, check_clips=len(items), check_outside_opens=outside[:5])
        table.append(row)
        print(f"{lang}: {row['files']:3d} files {total / 1e6:6.2f} MB | + int8 backbone "
              f"{row['backbone_int8_plus_bundle'] / 1e6:6.1f} MB | + fp32 {row['backbone_fp32_plus_bundle'] / 1e6:6.1f} MB"
              + (f" | bundle-only run identical: {row['check_identical']}" if a.check else ""), flush=True)
    res = {"backbone_bytes": bb, "languages": table, "shared_files": sorted(shared),
           "note": "sizes of raw files (no compression); backbone = model file + tokens.txt"}
    if a.table:
        Path(a.table).parent.mkdir(parents=True, exist_ok=True)
        Path(a.table).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    ok = all(r.get("check_identical", True) for r in table)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
