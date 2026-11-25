"""Which failing rows of a gate result depend on a fixed score threshold, and which are rankings.

    python -m bench.opt.row_kinds <gate result json>

Fixed threshold: the row counts scores above / below a constant cut-off fitted on fp32 outputs
(native false alarms below 0.6, natives accepted at strictness 0.5 / P >= 0.5, coaching tips and
precision at the shipped sensitivity thresholds, length native word acceptance). A small model whose
scores are shifted moves these rows even if it orders items exactly as fp32 does; refitting the
calibration on the small model's outputs would address that.
Ranking: the row depends only on the order of scores (Pearson / AUC, hit rate at the threshold set
by the run's own 5 % native quantile, "right word" = the lowest-scored word).
"""
from __future__ import annotations

import json
import sys
from collections import Counter


def kind(key: str, row_kind: str) -> str:
    area, lang, measure = key.split("|", 2)
    if row_kind in ("pcc", "detection_auc"):
        return "ranking"
    if row_kind in ("native_fa", "native_accept"):
        return "fixed threshold"
    if "AUC" in measure or "hit@5" in measure or "hit5" in measure or "caught" in measure or "right word" in measure:
        return "ranking"
    if area.startswith("coaching"):
        return "fixed threshold"
    return "ranking"


def main() -> None:
    d = json.loads(open(sys.argv[1]).read())
    kinds = {m["key"]: m["kind"] for m in d["moved"]}
    rows = {}
    for f in d["fails"]:
        k = f["key"]
        rows.setdefault(k, set()).add("budget" if f.get("budget") else "listed")
    by = Counter()
    lines = []
    for k, tags in sorted(rows.items()):
        kd = kind(k, kinds.get(k, ""))
        rule = "budget" if "budget" in tags else "strict only"
        by[(kd, rule)] += 1
        lines.append(f"{kd:16s} {rule:12s} {k}")
    print("\n".join(lines))
    print(f"\n{len(rows)} failing rows ({len(d['fails'])} failing checks):")
    for (kd, rule), n in sorted(by.items()):
        print(f"  {kd:16s} {rule:12s} {n}")
    print(json.dumps({f"{a} / {b}": n for (a, b), n in by.items()}))


if __name__ == "__main__":
    main()
