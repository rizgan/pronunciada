"""Collect paired English omission evidence from locally cached SpeechOcean762.

Run this file against each checkout using PYTHONPATH to choose the source tree. The output
contains manifest IDs, prompts, annotations, coaching items and timing, with no audio bytes.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pronunciada import Scorer
from pronunciada.coach import coaching

SEED = 20261125
NEAR = {
    "BEAR": "BEER", "CALL": "COAL", "IT": "EAT", "CAT": "CUT", "CAN": "CANE",
    "HE": "HEY", "SHE": "SHY", "SEE": "SAY", "DOG": "DIG", "IN": "ON",
    "PLAY": "PLATE", "DAY": "DIE", "WE": "WAY", "RED": "RAY", "DID": "DEAD",
    "GO": "GOAT", "ME": "MAY", "BIG": "BAG", "HAD": "HEAD", "IS": "AS",
    "LIKE": "LAKE", "AND": "END", "HIS": "HAS", "MAN": "MEN", "BACK": "BECK",
    "BED": "BAD", "BAT": "BET", "SIT": "SEAT", "LITTLE": "LIT",
}
INSERTIONS = ["APPLE", "WINDOW", "PENCIL", "CHAIR", "MUSIC", "BEFORE"]


def read_jsonl(path):
    with path.open(encoding="utf-8") as inp:
        return [json.loads(line) for line in inp]


def all_deleted(word):
    """Require one explicit deletion at every zero-based canonical phone index."""
    phones = word["phones"]
    deleted = {mark["index"] for mark in word["mispronunciations"]
               if mark["pronounced_phone"] == "<DEL>"
               and 0 <= mark["index"] < len(phones)
               and phones[mark["index"]].rstrip("012") == mark["canonical_phone"]}
    return bool(phones) and deleted == set(range(len(phones)))


def prompts(row, rng, *, real, varied):
    words = row["text"].split()
    if real:
        return {"real": (words, [i for i, w in enumerate(row["labels"]["words"]) if all_deleted(w)])}
    j = rng.randrange(1, len(words))
    extra = "APPLE" if "APPLE" not in words else "WINDOW"
    if varied:
        extra = next(word for word in rng.sample(INSERTIONS, len(INSERTIONS)) if word not in words)
    near_indices = [i for i, word in enumerate(words) if word in NEAR]
    k = rng.choice(near_indices) if near_indices else None
    if row["id"] == "so762-000010011":
        j = 2
    out = {"control": (words, None), "insert": (words[:j] + [extra] + words[j:], j)}
    if k is not None:
        out["near"] = (words[:k] + [NEAR[words[k]]] + words[k + 1:], k)
    return out


def word_result(word):
    return {"text": word.text, "p": word.prob, "present": word.present_prob, "omitted": word.omitted,
            "phones": [(phone.prob, phone.start, phone.end) for phone in word.phones]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("train", "test"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--real", action="store_true", help="score only clips with all-phone deletion labels")
    parser.add_argument("--exclude-speakers-from", type=Path,
                        help="exclude speakers appearing in this earlier JSONL run")
    parser.add_argument("--root", type=Path,
                        default=Path.home() / ".cache/pronunciada/data/speechocean762")
    args = parser.parse_args()

    def manifest(split):
        return read_jsonl(args.root / "parts" / f"{split}.jsonl")

    excluded = {row["speaker"] for row in manifest("train")} if args.split == "test" else set()
    if args.exclude_speakers_from:
        excluded.update(row["speaker"] for row in read_jsonl(args.exclude_speakers_from))
    rows = [r for r in manifest(args.split)
            if r["speaker"] not in excluded
            and len(r["text"].split()) >= 3
            and r["text"].split() == [w["text"] for w in r["labels"]["words"]]
            and (not args.real or any(all_deleted(w) for w in r["labels"]["words"]))]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    if args.split == "train":
        rows.sort(key=lambda r: r["id"] != "so762-000010011")
    scorer = Scorer("en", threads=2, precision="int8")
    with args.output.open("w", encoding="utf-8") as output:
        for row in rows[:args.n]:
            for kind, (words, pos) in prompts(row, rng, real=args.real,
                                              varied=bool(args.exclude_speakers_from)).items():
                result = scorer.score(row["audio"], " ".join(words))
                scored_words = [word_result(word) for word in result.words]
                item = {"id": row["id"], "speaker": row["speaker"], "kind": kind, "prompt": words,
                        "pos": pos, "labels": row["labels"], "timing": result.timing_ms,
                        "words": scored_words, "items": coaching(result, sensitivity="balanced")}
                print(json.dumps(item, ensure_ascii=False), file=output, flush=True)


if __name__ == "__main__":
    main()
