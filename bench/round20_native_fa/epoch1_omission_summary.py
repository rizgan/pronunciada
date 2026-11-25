"""Compare two epoch1_omission runs with paired speaker-cluster intervals."""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

def load(path):
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    keyed = {(r["id"], r["kind"]): r for r in rows}
    assert len(keyed) == len(rows)
    return keyed

def flags(row, kind=None):
    return {item["word_index"] for item in row["items"]
            if item["word_index"] is not None and (kind is None or item["kind"] == kind)}


def content(row, i):
    return {(item["kind"], item["phone_index"], item["expected"], item["heard"])
            for item in row["items"] if item["word_index"] == i}


def span(word):
    return min(phone[1] for phone in word["phones"]), max(phone[2] for phone in word["phones"])




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    base, candidate = map(load, (args.baseline, args.candidate))
    assert base.keys() == candidate.keys()
    metrics = defaultdict(list)
    times = defaultdict(list)
    examples = []
    for key, b in base.items():
        c = candidate[key]
        assert (b["prompt"], b["pos"], b["speaker"]) == (c["prompt"], c["pos"], c["speaker"])
        kind = b["kind"]
        pos = b["pos"]
        labels = b["labels"]["words"]
        bflag, cflag = flags(b), flags(c)

        def add(name, old, new):
            metrics[name].append((b["speaker"], old, new))

        times[kind].append((b["timing"]["total"], c["timing"]["total"], c["timing"].get("omission_recheck", 0)))
        if bflag != cflag:
            examples.append((b["id"], kind, b["prompt"], sorted(bflag), sorted(cflag)))
        for i, (bw, cw) in enumerate(zip(b["words"], c["words"], strict=True)):
            if i:
                old = span(bw)[0] + .050 < span(b["words"][i - 1])[1]
                new = span(cw)[0] + .050 < span(c["words"][i - 1])[1]
                add(f"{kind}/overlap_gt_50ms", old, new)
            target = (kind == "insert" and i == pos) or (kind == "real" and i in pos)
            if target:
                add(f"{kind}/missing_word", i in flags(b, "missing_word"), i in flags(c, "missing_word"))
                add(f"{kind}/omitted", bw["omitted"], cw["omitted"])
            name = None
            if kind == "near" and i == pos:
                name = "near/target"
            elif kind == "insert" and abs(i - pos) == 1:
                idx = i - int(i > pos)
                if labels[idx]["mispronunciations"]:
                    name = "insert/annotated_neighbor_error"
                elif labels[idx]["accuracy"] == 10 and idx not in flags(base[(b["id"], "control")]):
                    name = "insert/matched_correct_neighbor"
            elif kind == "real" and not target and any(abs(i - j) == 1 for j in pos):
                if labels[i]["mispronunciations"]:
                    name = "real/annotated_neighbor_error"
                elif labels[i]["accuracy"] == 10:
                    name = "real/annotated_good_neighbor"
            elif kind == "control" and labels[i]["mispronunciations"]:
                name = "control/annotated_error"
            elif kind == "control" and labels[i]["accuracy"] == 10:
                name = "control/annotated_good"
            if name:
                if name.endswith("/annotated_neighbor_error"):
                    add(f"{name}/content_changed", False, content(b, i) != content(c, i))
                add(f"{name}/coaching", i in bflag, i in cflag)
                add(f"{name}/below_0.6", bw["p"] < .6, cw["p"] < .6)

    print(f"{len(base)} scored prompts, {len({r['speaker'] for r in base.values()})} speakers")
    for kind, rows in times.items():
        print(kind, len(rows), "median ms (baseline, candidate, recheck):",
              [round(statistics.median(x),3) for x in zip(*rows)])
    for name, rows in sorted(metrics.items()):
        groups = defaultdict(lambda: [0, 0])
        for speaker, before, after in rows:
            groups[speaker][0] += int(after)-int(before)
            groups[speaker][1] += 1
        rng = random.Random(20261125)
        clusters = list(groups.values())
        samples = []
        for _ in range(2000):
            sel = rng.choices(clusters, k=len(clusters))
            samples.append(100 * sum(delta for delta, n in sel) / sum(n for delta, n in sel))
        samples.sort()
        print(name, sum(x[1] for x in rows), "->", sum(x[2] for x in rows), "/", len(rows),
              "speakers", len(clusters), "removed", sum(old and not new for _, old, new in rows),
              "added", sum(new and not old for _, old, new in rows),
              "delta 95% bootstrap pp", [round(samples[i], 2) for i in (50, 1949)])
    print("Example coaching word-index sets", *examples[:10], sep="\n")



if __name__ == "__main__":
    main()
