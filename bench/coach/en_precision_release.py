"""Compare the frozen English coaching rule with its recorded production baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bench.coach.en_precision import (
    DATASET, MODEL, ROOT, atomic_rows, digest_rows, encode, evaluate_record, file_hash,
    git, provenance, summarize,
)
from bench.coach.en_precision_fit import BASELINE, REPORT, paired_intervals
from bench.coach.tips_r19 import _so762_wrong
from bench.data import load
from bench.data.common import cache_root
from pronunciada import Scorer, coach
from pronunciada.report import coaching_block
from pronunciada.tips import PRONUNCIATION_KINDS, tips

MODEL_PATH = "pronunciada/data/coach/model.json"
COACH_PATH = "pronunciada/coach.py"
OUTPUT = ROOT / "results/tables/en_precision_release.json"


def checked_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def base_file(source, name):
    blob = git("show", f"{source['git_sha']}:{name}")
    for data in (blob, blob.replace(b"\n", b"\r\n")):
        if hashlib.sha256(data).hexdigest() == source["files"][name]:
            return data
    raise ValueError(f"Frozen base hash mismatch: {name}")


def verify_config(scorer, run_dir):
    train = checked_json(BASELINE)["source"]
    frozen = checked_json(REPORT)
    rule = frozen["rule"]
    candidate_hash = hashlib.sha256(
        json.dumps(rule, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if frozen["decision"] != "frozen" or candidate_hash != frozen["candidate_sha256"]:
        raise ValueError("Expected the frozen train candidate")
    if any(train[k] != v for k, v in frozen["base"].items()):
        raise ValueError("Candidate and baseline provenance disagree")
    if file_hash(cache_root() / train["inference_patch"]) != train["inference_patch_sha256"]:
        raise ValueError("Baseline inference patch hash mismatch")
    source = provenance(scorer, run_dir)
    for key in ("external_files", "settings", "overrides"):
        if source[key] != train[key]:
            raise ValueError(f"Baseline provenance mismatch: {key}")
    if any(source["versions"].get(k) != v for k, v in train["versions"].items()):
        raise ValueError("Baseline package versions differ")
    changed = {name for name in source["files"].keys() | train["files"].keys()
               if name.startswith("pronunciada/") and source["files"].get(name) != train["files"].get(name)}
    if changed != {MODEL_PATH, COACH_PATH}:
        raise ValueError(f"Unexpected inference changes from the baseline: {changed}")
    base_file(train, COACH_PATH)
    base = json.loads(base_file(train, MODEL_PATH))
    sound = base["by_lang"]["en"]["sound"]
    if sound["balanced"] != frozen["baseline_threshold"]:
        raise ValueError("Frozen baseline threshold mismatch")
    baseline_model = json.loads(base_file(train, MODEL_PATH))
    sound["balanced"] = {k: rule[k] for k in ("fallback", "thresholds")}
    candidate_model = coach.model()
    if candidate_model != base:
        raise ValueError("Installed coach model differs from the frozen rule")
    integration = git("diff", "--binary", train["git_sha"], "--", COACH_PATH, MODEL_PATH)
    integration_hash = hashlib.sha256(integration).hexdigest()
    integration_path = run_dir / f"candidate-{integration_hash}.patch"
    integration_path.write_bytes(integration)
    return (baseline_model, candidate_model), {
        "source": source, "base": frozen["base"], "candidate_sha256": candidate_hash,
        "integration_patch": integration_path.relative_to(cache_root()).as_posix(),
        "integration_patch_sha256": integration_hash,
        "release_measurer_sha256": file_hash(__file__),
    }


def measure(rec, relative, result, wrong, model):
    with patch.object(coach, "model", return_value=model):
        block = coaching_block(result)
        row = evaluate_record(rec, relative, result, block["items"])
    # The public payload drops item references. Verify that the same ranked tips
    # retain their source positions for the round-19 precision and coverage measures.
    selected = tips(block["items"], 3, result)
    if [{k: v for k, v in tip.items() if k != "item"} for tip in selected] != block["tips"]:
        raise ValueError("Tip provenance differs from public coaching tips")
    wrong_positions = {tuple(row["phones"][i]["position"]) for i in wrong}
    wrong_words = {w for w, _ in wrong_positions}
    pronunciation = [
        {"kind": item["kind"], "word_index": item["word_index"], "phone_index": item["phone_index"],
         "right": (item["word_index"] in wrong_words if item["phone_index"] is None else
                   (item["word_index"], item["phone_index"]) in wrong_positions)}
        for item in (tip["item"] for tip in selected) if item["kind"] in PRONUNCIATION_KINDS
    ]
    return {**row, "has_expert_error": bool(wrong), "pronunciation_tips": pronunciation,
            "public_coaching": block}


def tip_metrics(rows):
    positives = [r for r in rows if r["has_expert_error"]]
    clean = [r for r in rows if not r["has_expert_error"]]
    coverage = sum(bool(r["pronunciation_tips"]) for r in positives)
    on_clean = sum(bool(r["pronunciation_tips"]) for r in clean)
    tips = [t for r in rows for t in r["pronunciation_tips"]]
    right = sum(t["right"] for t in tips)
    return {
        "precision": {"numerator": right, "denominator": len(tips), "value": right / len(tips) if tips else None},
        "error_utterance_coverage": {"numerator": coverage, "denominator": len(positives),
                                     "value": coverage / len(positives) if positives else None},
        "clean_utterance_rate": {"numerator": on_clean, "denominator": len(clean),
                                 "value": on_clean / len(clean) if clean else None},
    }


def comparison(baseline, candidate):
    paired = []
    for a, b in zip(baseline, candidate, strict=True):
        for key in ("id", "speaker", "text", "audio", "expert_losses"):
            if a[key] != b[key]:
                raise ValueError(f"Pair disagrees on {key}: {a['id']}")
        for p, q in zip(a["phones"], b["phones"], strict=True):
            if any(p[k] != q[k] for k in p.keys() - {"raw_flag", "marked"}):
                raise ValueError(f"Paired evidence differs: {a['id']} {p['position']}")
            if p["expert_position"] is not None:
                paired.append((p["incorrect"], p["marked"], q["marked"], a["speaker"]))
    y, base, cand, speakers = (np.asarray(column) for column in zip(*paired))
    ci = paired_intervals(y, base, cand, speakers)
    reports = [summarize(rows) for rows in (baseline, candidate)]
    a, b = (r["metrics"]["items"] for r in reports)
    for summary, rows in zip(reports, (baseline, candidate), strict=True):
        summary["tips"] = tip_metrics(rows)
        for metrics in summary["metrics"].values():
            metrics["precision_denominator"] = metrics["TP"] + metrics["FP"]
            metrics["recall_denominator"] = metrics["TP"] + metrics["FN"]
    d = {key: (b[key] - a[key]) * 100 for key in ("precision", "recall")}
    criteria = {"identical_keys": reports[0]["key_sha256"] == reports[1]["key_sha256"],
                "fp_decreases": b["FP"] < a["FP"], "tp_preserved": b["TP"] >= a["TP"],
                "precision_ci_lower_above_zero": ci["precision"][0] > 0,
                "recall_delta_nonnegative": d["recall"] >= 0}
    return {
        "baseline": reports[0], "candidate": reports[1], "criteria": criteria,
        "decision": "accepted" if all(criteria.values()) else "rejected",
        "delta_pp": d, "paired_95_ci_pp": {k: [100 * x for x in v] for k, v in ci.items()},
        "precision_target_at_least_3pp": d["precision"] >= 3,
        "bootstrap": {"unit": "speaker", "paired_resamples": 1000, "seed": 0},
        "class_deltas": {
            cls: {k: reports[1]["sound_classes"][cls][k] - v for k, v in counts.items()}
            for cls, counts in reports[0]["sound_classes"].items()},
    }


def release(split):
    if OUTPUT.exists():
        raise FileExistsError(f"Official comparison already recorded in {OUTPUT}")
    cache = cache_root().resolve()
    run_dir = cache / "bench/en_precision/release"
    scorer = Scorer("en", model=MODEL, precision="fp32", strictness=0.5)
    models, source = verify_config(scorer, run_dir)
    records = sorted(load(DATASET, split), key=lambda rec: rec["id"])
    if len(records) != 2500 or len({rec["id"] for rec in records}) != 2500:
        raise ValueError(f"Expected 2500 unique test records, found {len(records)} rows")
    if len({rec["speaker"] for rec in records}) != 125:
        raise ValueError("Expected 125 test speakers")
    evaluated = {"baseline": [], "candidate": []}
    inputs = []
    for index, rec in enumerate(records, 1):
        try:
            audio = Path(rec["audio"]).resolve()
            relative = audio.relative_to(cache).as_posix()
            inputs.append({**rec, "audio": relative, "audio_sha256": file_hash(audio)})
            result = scorer.score(audio, rec["text"], strictness=0.5)
            wrong = _so762_wrong(result, rec)
            for rows, model in zip(evaluated.values(), models, strict=True):
                rows.append(measure(rec, relative, result, wrong, model))
        except Exception as exc:
            raise RuntimeError(f"Record {index}/{len(records)} ({rec['id']}): {exc}") from exc
        if index % 100 == 0:
            print(f"{index}/{len(records)}", flush=True)
    report = comparison(*evaluated.values())
    for name, rows in evaluated.items():
        path = run_dir / f"{name}-test.jsonl"
        atomic_rows(path, map(encode, rows))
        report[name].update({"full_records": path.relative_to(cache).as_posix(),
                             "full_records_sha256": file_hash(path)})
    input_path = run_dir / "inputs-test.jsonl"
    atomic_rows(input_path, map(encode, inputs))
    report.update({"input_manifest": input_path.relative_to(cache).as_posix(),
                   "dataset": DATASET, "split": split, "input_sha256": digest_rows(inputs),
                   "processing_errors": 0, **source,
                   "command": "python -m bench.coach.en_precision release --split test"})
    atomic_rows(OUTPUT, [json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)])
    print(encode({k: report[k] for k in ("decision", "criteria", "delta_pp", "paired_95_ci_pp")}))
