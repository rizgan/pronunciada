"""Measure the public English coaching baseline on official speechocean762 train."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from tempfile import NamedTemporaryFile

from bench.data import load
from bench.data.common import cache_root
from bench.pipeline.labels import align_reference, arpa_to_ours
from pronunciada import Scorer, __version__, coach
from pronunciada._overrides import adapter_file, backbone_file
from pronunciada.acoustic import PRECISION_FILES, ensure_model_file

ROOT = Path(__file__).resolve().parents[2]
DATASET = "speechocean762"
MODEL = "zipa-small-crctc-500k"
COMMAND = "python -m bench.coach.en_precision baseline --split train"


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_rows(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update((encode(row) + "\n").encode("utf-8"))
    return digest.hexdigest()


def measured_positions(ref, words, positions, items):
    """Map local positions to expert keys and retain losses on both sides."""
    phones = [[arpa_to_ours(p) for p in w["phones"]] for w in words]
    alignment, _ = align_reference(ref, [w["text"] for w in words], phones)
    marks = {(it["word_index"], it["phone_index"]) for it in items
             if it["kind"] in {"substitution", "missing"}}
    if marks - set(positions):
        raise ValueError(f"Marks outside the reference: {marks - set(positions)}")
    matched = {}
    losses = []
    for wi, (word, mapped) in enumerate(zip(words, phones, strict=True)):
        scores = word["phones_accuracy"]
        for pi, symbol in enumerate(mapped):
            score = scores[pi] if pi < len(scores) else None
            if score is None or not math.isfinite(score):
                reason = "missing_expert_score"
            elif symbol is None:
                reason = "unmapped_arpabet"
            else:
                reason = "alignment_gap"
                matched[wi, pi] = score
            losses.append({"expert_position": (wi, pi), "reason": reason,
                           "arpa": word["phones"][pi]})
        if len(scores) > len(mapped):
            raise ValueError(f"Word {wi} has more scores than phones")

    labels = {i: (key, matched[key]) for i, key in alignment.items() if key in matched}
    used = {key for key, _ in labels.values()}
    if len(used) != len(labels):
        raise ValueError("Alignment reused an expert key")
    losses = [loss for loss in losses if loss["expert_position"] not in used]
    reference_losses = {
        i: ("missing_expert_score" if i in alignment else "alignment_gap")
        for i in range(len(positions)) if i not in labels
    }
    return labels, marks, losses, reference_losses


def evaluate_record(rec, audio, result):
    ref = result._pronunciada["ref"]
    inputs = coach.sound_inputs(result)
    positions = [(int(w), int(k)) for w, k in zip(inputs["word"], inputs["k"], strict=True)]
    items = result.coaching(sensitivity="balanced")
    labels, marks, expert_losses, reference_losses = measured_positions(
        ref, rec["labels"]["words"], positions, items)
    raw = set(coach.flagged_indices(inputs, "en", "balanced"))
    probabilities = coach.error_prob(inputs, lang="en")
    omitted = {item["word_index"] for item in items if item["kind"] == "missing_word"}
    phones = []
    for i, (position, expected, probability) in enumerate(
            zip(positions, inputs["expected"], probabilities, strict=True)):
        expert_position, score = labels.get(i, (None, None))
        phones.append({"position": position, "expert_position": expert_position,
                       "score": score, "incorrect": None if score is None else score < 2,
                       "unmatched_reason": reference_losses.get(i),
                       "marked": position in marks, "raw_flag": i in raw,
                       "omitted_word": position[0] in omitted, "error_prob": float(probability),
                       "expected": expected, "sound_class": coach.sound_class(expected)})
    return {"id": rec["id"], "speaker": rec["speaker"], "text": rec["text"], "audio": audio,
            "phones": phones, "expert_losses": expert_losses}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def provenance(scorer, run_dir):
    paths = git("ls-files", "-z", "--cached", "--others", "--exclude-standard",
                "--", "pronunciada", "bench/coach/en_precision.py",
                "bench/data", "bench/pipeline/labels.py").decode().split("\0")
    files = {name: file_hash(ROOT / name) for name in sorted(set(paths))
             if name and (ROOT / name).is_file()}
    patch = git("diff", "--binary", "HEAD", "--", "pronunciada")
    patch_hash = hashlib.sha256(patch).hexdigest()
    run_dir.mkdir(parents=True, exist_ok=True)
    patch_path = run_dir / f"inference-{patch_hash}.patch"
    patch_path.write_bytes(patch)
    backbone = backbone_file() or ensure_model_file(MODEL, PRECISION_FILES["fp32"])
    external = {"backbone": backbone, "tokens": ensure_model_file(MODEL, "tokens.txt"),
                "session_model": scorer.am.session._model_path}
    adapter = adapter_file("en")
    if adapter:
        external["adapter_override"] = adapter
    return {
        "git_sha": git("rev-parse", "HEAD").decode().strip(),
        "inference_patch_sha256": patch_hash,
        "inference_patch": patch_path.relative_to(cache_root().resolve()).as_posix(),
        "files": files, "coach_model_sha256": files["pronunciada/data/coach/model.json"],
        "external_files": {key: {"name": Path(path).name, "sha256": file_hash(path)}
                           for key, path in external.items()},
        "settings": {"lang": "en", "strictness": 0.5, "sensitivity": "balanced",
                     "precision": "fp32", "model": MODEL,
                     "providers": scorer.am.session.get_providers()},
        "overrides": {key: value for key, value in sorted(os.environ.items())
                      if key.startswith("PRONUNCIADA_") and key != "PRONUNCIADA_CACHE"},
        "versions": {"pronunciada": __version__, "python": platform.python_version(),
                     **{dist.metadata["Name"]: dist.version for dist in
                        sorted(importlib.metadata.distributions(), key=lambda d: d.name.lower())}},
    }


def summarize(records):
    counts = {"items": Counter(), "raw": Counter()}
    classes = defaultdict(Counter)
    examples = {}
    keys = []
    coverage = Counter()
    losses = Counter()
    reference_losses = Counter()
    for rec in records:
        coverage["expert_unmatched"] += len(rec["expert_losses"])
        losses.update(loss["reason"] for loss in rec["expert_losses"])
        for phone in rec["phones"]:
            coverage["reference_phones"] += 1
            if phone["expert_position"] is None:
                coverage["reference_unmatched"] += 1
                reference_losses[phone["unmatched_reason"]] += 1
                coverage["items_without_expert_label"] += phone["marked"]
                coverage["raw_flags_without_expert_label"] += phone["raw_flag"]
                continue
            keys.append((rec["id"], *phone["expert_position"]))
            coverage["matched"] += 1
            coverage["matched_in_omitted_words"] += phone["omitted_word"]
            for name, flagged in (("items", phone["marked"]), ("raw", phone["raw_flag"])):
                outcome = ("T" if flagged == phone["incorrect"] else "F") + ("P" if flagged else "N")
                counts[name][outcome] += 1
                if name == "items":
                    classes[phone["sound_class"]][outcome] += 1
                    if outcome in {"FP", "TP"} and outcome not in examples:
                        examples[outcome] = {
                            **{k: rec[k] for k in ("id", "speaker", "text", "audio")},
                            **phone,
                        }
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate expert keys")
    metrics = {}
    for name, c in counts.items():
        tp, fp, fn, tn = (c[k] for k in ("TP", "FP", "FN", "TN"))
        if tp + fp + fn + tn != len(keys):
            raise ValueError(f"{name} counts do not cover all expert keys")
        metrics[name] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                         "precision": tp / (tp + fp) if tp + fp else None,
                         "recall": tp / (tp + fn) if tp + fn else None}
    return {"records": len(records), "speakers": len({r["speaker"] for r in records}),
            "coverage": coverage, "expert_loss_reasons": losses, "metrics": metrics,
            "reference_loss_reasons": reference_losses,
            "sound_classes": {k: {"phones": sum(c.values()),
                                  **{m: c[m] for m in ("TP", "FP", "FN", "TN")}}
                              for k, c in classes.items()},
            "examples": examples, "key_sha256": digest_rows(sorted(keys))}


def atomic_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                            dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            for row in rows:
                stream.write(row + "\n")
        except BaseException:
            stream.close()
            temp.unlink()
            raise
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def baseline(split):
    records = sorted(load(DATASET, split), key=lambda rec: rec["id"])
    if len(records) != 2500 or len({rec["id"] for rec in records}) != 2500:
        raise ValueError(f"Expected 2500 distinct train records, found {len(records)} rows")
    cache = cache_root().resolve()
    run_dir = cache / "bench" / "en_precision" / split
    scorer = Scorer("en", model=MODEL, precision="fp32", strictness=0.5)
    source = provenance(scorer, run_dir)
    evaluated = []
    inputs = []
    for index, rec in enumerate(records, 1):
        audio = Path(rec["audio"]).resolve()
        try:
            relative = audio.relative_to(cache).as_posix()
            inputs.append({**rec, "audio": relative, "audio_sha256": file_hash(audio)})
            result = scorer.score(audio, rec["text"], strictness=0.5)
            evaluated.append(evaluate_record(rec, relative, result))
        except Exception as exc:
            raise RuntimeError(f"Record {index}/{len(records)} ({rec['id']}): {exc}") from exc
        if index % 100 == 0:
            print(f"{index}/{len(records)}", flush=True)
    report = summarize(evaluated)
    data_path = run_dir / "baseline.jsonl"
    previous = data_path.with_name("baseline.previous.jsonl")
    if data_path.exists():
        # Preserve the last completed run for acceptance comparisons.
        atomic_rows(previous, data_path.read_text(encoding="utf-8").splitlines())
    atomic_rows(data_path, map(encode, evaluated))
    report.update({"command": COMMAND, "dataset": DATASET, "split": split,
                   "input_sha256": digest_rows(inputs), "source": source,
                   "processing_errors": 0, "full_records": data_path.relative_to(cache).as_posix(),
                   "full_records_sha256": file_hash(data_path)})
    path = ROOT / "results" / "tables" / f"en_precision_{split}.json"
    atomic_rows(path, [json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)])
    print(f"{path.relative_to(ROOT)}: {encode(report['metrics'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    baseline_parser = commands.add_parser("baseline")
    baseline_parser.add_argument("--split", choices=["train"], required=True)
    args = parser.parse_args()
    baseline(args.split)


if __name__ == "__main__":
    main()
