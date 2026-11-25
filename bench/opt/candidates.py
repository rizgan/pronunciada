"""Named candidates (bench/opt/candidates.json) through parity, so the log table has one entry each.

    python -m bench.opt.candidates parity <name> [...] --code <rev> [--threads 2]
    python -m bench.opt.candidates spec <name> --code <rev>          # the spec, for gate / profile

Parity is against the default configuration of the candidate's precision on the same code. The
result goes to $PRONUNCIADA_CACHE/opt/results/<name>.parity.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import LANGS, load_spec, opt_root

FILE = Path(__file__).resolve().parent / "candidates.json"


def spec(name: str, code: str) -> dict:
    d = json.loads(FILE.read_text())[name]
    from .common import PRECISION_FILES, default_backbone

    # {M}/<download> = the plain downloaded file of that precision (restored from the cache's one file)
    for prec, f in PRECISION_FILES.items():
        if f"{{M}}/{f}" in json.dumps(d):
            d = json.loads(json.dumps(d).replace(f"{{M}}/{f}", str(default_backbone(prec))))
    s = json.loads(json.dumps(d).replace("{C}", str(opt_root() / "candidates")))
    return load_spec({"name": name, "code": code, **s})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("parity", "spec"))
    ap.add_argument("names", nargs="+")
    ap.add_argument("--code", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "spec":
        print(json.dumps(spec(a.names[0], a.code)))
        return
    from .parity import run

    out = opt_root() / "results"
    out.mkdir(parents=True, exist_ok=True)
    for n in a.names:
        c = spec(n, a.code)
        ref = load_spec({"name": f"default-{c['precision']}", "code": a.code, "precision": c["precision"]})
        r = run(ref, c, list(LANGS), a.n, a.threads)
        (out / f"{n}.parity.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
        print(f"== {n}: {r['verdict']}", flush=True)


if __name__ == "__main__":
    main()
