"""Command line.

    pronunciada check AUDIO "TEXT" [--lang en] [--coach] [--strictness 0.5] [--json] [--no-color]
    pronunciada score AUDIO "TEXT" --lang en [--strictness 0.5] [--json] [--coach]   (detailed table)
    pronunciada langs [--json]
    pronunciada serve [--host 127.0.0.1] [--port 8765] [--azure-compat]
    pronunciada mcp
    pronunciada --version
(`python -m pronunciada ...` is the same.)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys


def _fmt_p(p) -> str:
    return "  –  " if p is None else f"{p:5.2f}"


def table(res) -> str:
    L = [f"{res.lang}  strictness {res.strictness:g}   sentence {res.prob:.2f}   "
         f"fluency {_fmt_p(res.fluency).strip()}   completeness {_fmt_p(res.completeness).strip()}", ""]
    L.append(f"{'word':<18} {'prob':>5} {'sounds':>6} {'stress':>6} {'tone':>5}  {'time':>11}  sounds (prob, heard)")
    for w in res.words:
        snd = "  ".join(f"{p.ipa}:{p.prob:.2f}" + (f"→{p.heard or '∅'}" if p.heard is not None else "")
                        for p in w.phones)
        flag = " (omitted)" if w.omitted else ""
        L.append(f"{(w.text + flag)[:18]:<18} {w.prob:5.2f} {_fmt_p(w.sounds_prob):>6} "
                 f"{_fmt_p(w.stress_prob):>6} {_fmt_p(w.tone_prob):>5}  {w.start:5.2f}-{w.end:5.2f}  {snd}")
    if res.warnings:
        L += ["", "warnings: " + "; ".join(res.warnings)]
    L += ["", "timing ms: " + ", ".join(f"{k} {v:.1f}" for k, v in res.timing_ms.items())]
    return "\n".join(L)


def _common(s) -> None:
    s.add_argument("--strictness", type=float, default=0.5)
    _execution_options(s)
    s.add_argument("--json", action="store_true", help="print the SentenceResult as JSON")
    s.add_argument("--max-tips", type=int, default=3)
    s.add_argument("--sensitivity", default="balanced", choices=("careful", "balanced", "eager"))


def _execution_options(s) -> None:
    s.add_argument("--precision", default="fp32", choices=("fp32", "fp16", "int8"))
    s.add_argument("--threads", type=int, default=None, help="ONNX Runtime threads per model")
    s.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="acoustic model device")


def _parser() -> argparse.ArgumentParser:
    from . import __version__

    ap = argparse.ArgumentParser(prog="pronunciada", description="on-device pronunciation checking")
    ap.add_argument("--version", action="version", version=f"pronunciada {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="score one recording and print a compact report")
    c.add_argument("audio")
    c.add_argument("text")
    c.add_argument("--lang", default="en", help="language code (default en); see `pronunciada langs`")
    _common(c)
    c.add_argument("--coach", action="store_true",
                   help="also list every coaching item (report) / include coaching in --json")
    c.add_argument("--no-color", action="store_true", help="no ANSI color (also: NO_COLOR, non-TTY)")
    c.add_argument("--width", type=int, default=None, help="wrap width (default: terminal, max 100)")

    s = sub.add_parser("score", help="score one recording against its prompt text (detailed table)")
    s.add_argument("audio")
    s.add_argument("text")
    s.add_argument("--lang", required=True)
    _common(s)
    s.add_argument("--coach", action="store_true",
                   help="add coaching items and default tips (pronunciada.coach, pronunciada.tips)")

    lg = sub.add_parser("langs", help="list supported languages and the extras each needs")
    lg.add_argument("--json", action="store_true")

    sv = sub.add_parser("serve", help="local HTTP scoring server (no authentication: development use)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--azure-compat", action="store_true",
                    help="also answer the Azure short-audio REST path with Azure-shaped JSON")
    sv.add_argument("--max-mb", type=float, default=20.0, help="request body limit in MB (default 20)")
    _execution_options(sv)

    sub.add_parser("mcp", help="Model Context Protocol server on stdio (needs the `mcp` package)")
    return ap


def _score(a):
    from .scorer import Scorer
    from .text import normalize_lang

    sc = Scorer(normalize_lang(a.lang), precision=a.precision, threads=a.threads, device=a.device)
    return sc.score(a.audio, a.text, strictness=a.strictness)


def cmd_score(a) -> int:
    try:
        res = _score(a)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    items = tp = None
    if a.coach:
        from .tips import tips

        items = res.coaching(sensitivity=a.sensitivity)
        tp = [{k: v for k, v in t.items() if k != "item"} for t in tips(items, a.max_tips, res)]
    if a.json:
        d = dataclasses.asdict(res)
        if a.coach:
            from .coach import SCHEMA_VERSION

            d["coaching"] = {"schema_version": SCHEMA_VERSION, "items": items, "tips": tp}
        print(json.dumps(d, ensure_ascii=False, indent=1))
    else:
        print(table(res))
        if res.audio is not None:
            iss = ", ".join(res.audio["issues"]) or "none"
            print(f"audio: issues {iss}" + ("  (retry recommended)" if res.audio["retry_recommended"] else ""))
        if a.coach:
            print("\ncoaching:")
            for it in items:
                h = "?" if it["heard"] is None else (it["heard"] or "∅")
                print(f"  {it['kind']:<12} {(it['word'] or '')[:16]:<16} {it['expected']!s:>4} -> {h!s:<4} "
                      f"err {it['error_prob'] if it['error_prob'] is not None else '-'}  conf {it['confidence'] if it['confidence'] is not None else '-'}  "
                      f"{','.join(it['diff_keys'])}")
            print("tips:")
            for t in tp:
                print(f"  [{t['tip_key']}] {t['text']}")
    return 0


def cmd_check(a) -> int:
    import os

    from . import report

    if not os.path.isfile(a.audio):
        print(f"error: audio file not found: {a.audio}", file=sys.stderr)
        return 2
    try:
        res = _score(a)
    except (ValueError, FileNotFoundError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:              # soundfile: unreadable / unsupported audio file
        print(f"error: cannot score {a.audio!r}: {e}", file=sys.stderr)
        return 2
    cb = report.coaching_block(res, a.sensitivity, a.max_tips)
    if a.json:
        print(json.dumps(report.payload(res, coach=a.coach, coaching=cb), ensure_ascii=False, indent=1))
        return 0
    color = report.want_color(sys.stdout, a.no_color)
    sys.stdout.write(report.render(res, cb, color=color, width=a.width, show_items=a.coach))
    return 0


def cmd_langs(a) -> int:
    from .languages import languages

    L = languages()
    if a.json:
        print(json.dumps(L, ensure_ascii=False, indent=1))
        return 0
    print(f"{'code':<5} {'language':<17} {'needs':<22} status")
    for x in L:
        need = f"pronunciada[{x['extra']}]" if x["extra"] else "core"
        status = "ready" if x["ready"] else "missing " + ", ".join(x["missing"])
        print(f"{x['code']:<5} {x['name']:<17} {need:<22} {status}")
    return 0


def cmd_serve(a) -> int:
    from .server import serve

    try:
        return serve(a.host, a.port, azure_compat=a.azure_compat, max_bytes=int(a.max_mb * 1e6),
                     threads=a.threads, precision=a.precision, device=a.device)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def cmd_mcp(a) -> int:
    from .mcp_server import main as mcp_main

    return mcp_main()


def main(argv=None) -> int:
    a = _parser().parse_args(argv)
    return {"score": cmd_score, "check": cmd_check, "langs": cmd_langs, "serve": cmd_serve,
            "mcp": cmd_mcp}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
