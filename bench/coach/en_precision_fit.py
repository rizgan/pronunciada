"""Evaluate and freeze a phoneme-class correction using official train only."""
from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

from bench.calib.common import calib_cache
from bench.coach.en_precision import ROOT, atomic_rows, file_hash, summarize
from bench.coach.fit_v2 import fit_logistic
from bench.pipeline.features import config
from pronunciada import coach

EPS = 1e-6
REPORT = ROOT / "results/tables/en_precision_candidate.json"
BASELINE = ROOT / "results/tables/en_precision_train.json"


def safe_logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def select_threshold(scores, y, base_tp, base_fp):
    if not len(scores):
        return None
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
    tp = np.cumsum(y[order])[ends]
    fp = ends + 1 - tp
    valid = np.flatnonzero((tp >= base_tp) & (fp < base_fp))
    if not valid.size:
        return None
    i = max(valid, key=lambda j: (tp[j] / (tp[j] + fp[j]), tp[j], s[ends[j]]))
    return float(s[ends[i]])


def shifted_threshold(p, delta, threshold):
    """Export the inclusive boundary, correcting floating-point round-trip at train ties."""
    boundary = threshold - delta
    if threshold <= safe_logit(0) + delta:
        return 0.0
    if threshold > safe_logit(1) + delta:
        return float(np.nextafter(1.0, np.inf))
    result = float(expit(boundary))
    included = safe_logit(p) + delta >= threshold
    if np.any(included):
        result = min(result, float(p[included].min()))
    if np.any(~included):
        result = max(result, float(np.nextafter(p[~included].max(), np.inf)))
    if not np.array_equal(p >= result, included):
        raise ValueError("Exported threshold differs from the inclusive logit rule")
    return result


def masks(p, classes, eligible, rule):
    cuts = np.array([rule["thresholds"].get(c, rule["fallback"]) for c in classes])
    return eligible & (p >= cuts)


def tune(p, classes, y, eligible, base):
    z = safe_logit(p)
    offsets = {}
    corrected = z.copy()
    for cls in np.unique(classes[eligible]):
        idx = eligible & (classes == cls)
        zi, yi = z[idx], y[idx]
        delta = brentq(lambda d: float((expit(zi + d) - yi).sum() + d), -50, 50)
        offsets[cls] = delta
        corrected[classes == cls] += delta
    threshold = select_threshold(corrected[eligible], y[eligible],
                                 int((base & y).sum()), int((base & ~y).sum()))
    if threshold is None:
        return None
    return {"logit_threshold": threshold, "offsets": offsets, "fallback": float(expit(threshold)),
            "thresholds": {cls: shifted_threshold(p[classes == cls], d, threshold)
                           for cls, d in offsets.items()}}


def probabilities(F, y, tr, te):
    return coach._logistic(fit_logistic(F[tr], y[tr]), F[te])


def confusion(y, flagged):
    tp = int((y & flagged).sum())
    fp = int((~y & flagged).sum())
    fn = int(y.sum()) - tp
    tn = len(y) - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.}


def paired_intervals(y, base, candidate, speaker):
    counts = []
    for s in np.unique(speaker):
        idx = speaker == s
        a, b = (confusion(y[idx], p[idx]) for p in (base, candidate))
        counts.append([a["tp"], a["fp"], b["tp"], b["fp"], a["tp"] + a["fn"]])
    counts = np.asarray(counts)
    rng = np.random.default_rng(0)
    a, af, b, bf, positives = counts[rng.integers(len(counts), size=(1000, len(counts)))].sum(axis=1).T
    delta = {"precision": b / np.maximum(b + bf, 1) - a / np.maximum(a + af, 1),
             "recall": (b - a) / np.maximum(positives, 1)}
    return {key: np.quantile(values, [0.025, 0.975]).tolist() for key, values in delta.items()}


def load_inputs():
    import pandas as pd

    from bench.calib.sound_emb import proto_path
    from bench.coach.dump import load_dump, out_dir
    from bench.coach.en_precision import MODEL
    from bench.data.common import cache_root
    from bench.pipeline.features import feat_dir
    from pronunciada import Scorer

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline_path = cache_root() / baseline["full_records"]
    if file_hash(baseline_path) != baseline["full_records_sha256"]:
        raise ValueError("Baseline cache differs from the accepted report")
    source = baseline["source"]
    for name, digest in source["files"].items():
        if name.startswith("pronunciada/") and file_hash(ROOT / name) != digest:
            raise ValueError(f"Inference input has changed: {name}")
    scorer = Scorer("en", model=MODEL, precision="fp32", strictness=0.5)
    if scorer.am.session.get_providers() != source["settings"]["providers"]:
        raise ValueError("Acoustic execution providers differ from the accepted baseline")
    model_path = scorer.am.session._model_path
    if any(file_hash(Path(model_path).with_name(entry["name"])) != entry["sha256"]
           for entry in source["external_files"].values()):
        raise ValueError("Acoustic model files differ from the accepted baseline")
    if coach.model()["by_lang"]["en"]["sound_model"]["features"] != list(coach.SOUND_FEATURES):
        raise ValueError("The shipped coaching head has different features")

    detail_path = calib_cache() / "sound_oof.parquet"
    detail_meta = json.loads(detail_path.with_suffix(".json").read_text(encoding="utf-8"))
    detail = pd.read_parquet(detail_path).set_index(["utt", "phone_index"], verify_integrity=True)
    inputs = (feat_dir() / "speechocean762.parquet", proto_path("en"),
              proto_path("en").with_name("so762_train_vectors.pkl"))
    if (detail_meta["schema"] != 3 or detail_meta["sha256"] != file_hash(detail_path)
            or detail_meta["feature_config"] != config("en")
            or any(version(name) != installed for name, installed in detail_meta["versions"].items())
            or detail_meta["inputs"] != {p.name: file_hash(p) for p in inputs}
            or not all(file_hash(ROOT / name) == sha for name, sha in detail_meta["shipped"].items())):
        raise ValueError("Incompatible detail OOF cache")
    if detail.groupby("speaker").fold.nunique().ne(1).any() or set(detail.fold) != set(range(5)):
        raise ValueError("Invalid detail fold assignments")

    records = [json.loads(row) for row in baseline_path.read_text(encoding="utf-8").splitlines()]
    if summarize(records)["key_sha256"] != baseline["key_sha256"]:
        raise ValueError("Baseline key mismatch")
    dump = load_dump("so762-train")
    if set(dump) != {r["id"] for r in records} or len(records) != 2500:
        raise ValueError("Expected identical coverage of 2500 train records")
    fold_range = range(5)
    prefixes = {(outer, None): f"E_outer{outer}" for outer in fold_range}
    prefixes.update({(outer, inner): f"E_outer{outer}_inner{inner}"
                     for outer in fold_range for inner in fold_range if inner != outer})
    matrices = {pair: [] for pair in prefixes}
    labels, classes, speakers, eligible, positions, fold_ids = [], [], [], [], [], []
    reference_count = 0
    for r in records:
        uid = r["id"]
        phones = r["phones"]
        X = dict(dump[uid]["sounds"])
        d = detail.loc[uid]
        if (not np.array_equal(d.index, np.arange(len(phones))) or not d.speaker.eq(r["speaker"]).all()
                or d.phone.tolist() != X["expected"] or X["expected"] != [p["expected"] for p in phones]
                or list(zip(X["word"], X["k"])) != [tuple(p["position"]) for p in phones]):
            raise ValueError(f"Incompatible feature keys for {uid}")
        y = np.array([np.nan if p["score"] is None else p["score"] for p in phones])
        if not np.array_equal(y, d.phone_score, equal_nan=True):
            raise ValueError(f"Feature labels mismatch for {uid}")
        omitted = [w["present_prob"] is not None and w["present_prob"] < coach.gate_params()["missing_word_prob"]
                   for w in dump[uid]["words"]]
        can_flag = ~np.asarray(omitted)[X["word"]]
        if not np.array_equal(can_flag, [not p["omitted_word"] for p in phones]):
            raise ValueError(f"Omission mask mismatch for {uid}")
        reference_count += len(phones)
        valid = np.isfinite(y)
        for pair, prefix in prefixes.items():
            zl, zs = d[prefix + "_z_len"].to_numpy(), d[prefix + "_z_str"].to_numpy()
            if not np.isfinite(zl).all() or not np.isfinite(zs).all():
                raise ValueError(f"Nonfinite detail logits for {uid}, folds {pair}")
            matrices[pair].append(coach.feature_matrix(dict(X, zl=zl, zs=zs))[valid])
        fold_ids.extend(d.fold.to_numpy()[valid])
        labels.extend(y[valid] < 2)
        eligible.extend(can_flag[valid])
        classes.extend(coach.sound_class(p["expected"]) for p, keep in zip(phones, valid) if keep)
        speakers.extend([r["speaker"]] * valid.sum())
        positions.extend((r, p) for p, keep in zip(phones, valid) if keep)

    if (len(detail) != reference_count or len(labels) != baseline["coverage"]["matched"]
            or not np.isfinite(detail[["E_z_len", "E_z_str"]].to_numpy()).all()):
        raise ValueError("Incomplete detail or coaching coverage")
    detail_meta["coach_dump_sha256"] = file_hash(out_dir() / "dump-so762-train.pkl")
    return ({pair: np.concatenate(blocks) for pair, blocks in matrices.items()}, np.asarray(labels), np.asarray(classes),
            np.asarray(speakers), np.asarray(fold_ids), np.asarray(eligible), positions, baseline, detail_meta)


def control_examples(positions, rule):
    # Phase 1 public scores are used only here, after freezing the rule.
    examples = {}
    for r, p in positions:
        if not p["marked"]:
            continue
        threshold = rule["thresholds"].get(p["sound_class"], rule["fallback"])
        stays = p["error_prob"] >= threshold
        if (not p["incorrect"] and not stays) or (p["incorrect"] and stays):
            key = "TP" if stays else "FP"
            examples.setdefault(key, {**{k: r[k] for k in ("id", "text", "audio")},
                                      "position": p["position"], "expected": p["expected"],
                                      "score": p["score"], "candidate_marked": stays})
        if len(examples) == 2:
            return examples
    return examples


def verify_examples():
    from bench.coach.en_precision import MODEL, evaluate_record
    from bench.data import load
    from bench.data.common import cache_root
    from pronunciada import Scorer

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rule = report["rule"]
    train = {r["id"]: r for r in load("speechocean762", "train")}
    scorer = Scorer("en", model=MODEL, precision="fp32", strictness=0.5)
    for name, example in report["examples"].items():
        rec = train[example["id"]]
        result = scorer.score(cache_root() / example["audio"], rec["text"])
        measured = evaluate_record(rec, example["audio"], result)
        phone = next(p for p in measured["phones"] if list(p["position"]) == example["position"])
        candidate = bool(masks(np.asarray([phone["error_prob"]]), [phone["sound_class"]],
                               np.asarray([not phone["omitted_word"]]), rule)[0])
        if not phone["marked"] or candidate != example["candidate_marked"]:
            raise ValueError(f"Public baseline or bench decision mismatch for {name}")
        print(json.dumps({"example": name, "id": example["id"], "position": example["position"],
                          "baseline_marked": phone["marked"], "candidate_marked": candidate}))


def evaluate(matrices, y, classes, groups, fold_ids, eligible, threshold):
    base = np.zeros(len(y), bool)
    candidate = base.copy()
    all_scores = np.empty(len(y))
    folds = []
    for fold in np.unique(fold_ids):
        held = fold_ids == fold
        tr, te = np.flatnonzero(~held), np.flatnonzero(held)
        if not len(te) or np.intersect1d(groups[tr], groups[te]).size:
            raise ValueError("Invalid outer split by speaker")
        p = np.full(len(tr), np.nan)
        for inner in np.unique(fold_ids[tr]):
            val = fold_ids[tr] == inner
            p[val] = probabilities(matrices[fold, inner], y, tr[~val], tr[val])
        rule = tune(p, classes[tr], y[tr], eligible[tr], eligible[tr] & (p >= threshold))
        all_scores[te] = probabilities(matrices[fold, None], y, tr, te)
        base[te] = eligible[te] & (all_scores[te] >= threshold)
        candidate[te] = base[te] if rule is None else masks(all_scores[te], classes[te], eligible[te], rule)
        folds.append({"fold": int(fold), "held_speakers": sorted(set(groups[te])), "has_rule": rule is not None,
                      "baseline": confusion(y[te], base[te]), "candidate": confusion(y[te], candidate[te])})
    return all_scores, base, candidate, folds


def fit():
    matrices, y, classes, groups, fold_ids, eligible, positions, baseline, detail = load_inputs()
    mdl = coach.model()
    threshold = coach.sound_threshold(mdl, "en", "balanced", True)
    if threshold != coach.sound_threshold(mdl, "en", "balanced", False):
        raise ValueError("Expected a single shipped English threshold")
    all_scores, base, candidate, folds = evaluate(matrices, y, classes, groups, fold_ids, eligible, threshold)
    a, b = confusion(y, base), confusion(y, candidate)
    passed = b["fp"] < a["fp"] and b["tp"] >= a["tp"] and all(f["has_rule"] for f in folds)
    final = tune(all_scores, classes, y, eligible, base) if passed else None
    source = baseline["source"]
    reason = ("frozen" if final else "outer_comparison_failed" if not passed else "no_full_train_threshold")
    report = {"split": "train", "decision": "frozen" if final else "rejected", "reason": reason, "folds": folds,
              "command": "python -m bench.coach.en_precision fit --split train",
              "records": baseline["records"], "speakers": baseline["speakers"], "labelled_phones": len(y),
              "baseline_threshold": threshold, "baseline": a, "candidate": b,
              "delta": {k: b[k] - a[k] for k in ("precision", "recall")},
              "paired_95_ci": paired_intervals(y, base, candidate, groups),
              "outer_gate_passed": passed, "detail_oof": detail, "key_sha256": baseline["key_sha256"],
              "input_sha256": baseline["input_sha256"],
              "base": {k: source[k] for k in ("git_sha", "inference_patch_sha256", "coach_model_sha256")},
              "omitted_labelled_phones": int((~eligible).sum()),
              "bootstrap": {"unit": "speaker", "paired_resamples": 1000, "seed": 0},
              "previously_fitted": [
                  "The ZIPA backbone and FLEURS-train English adapter were reused from the phase 1 scorer.",
                  "The omission gate was fitted on FLEURS validation with synthetic omissions.",
                  "Packaged FLEURS-en train prototypes were reused. The embedding projection and trees were refitted.",
                  "Sound_E hyperparameters, feature vocabulary, calibration families and the balanced cut "
                  "are historical selections on speechocean762 train.",
              ]}
    if final:
        report["rule"] = final
        report["candidate_sha256"] = hashlib.sha256(
            json.dumps(final, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        report["examples"] = control_examples(positions, final)
        if set(report["examples"]) != {"FP", "TP"}:
            raise ValueError("Frozen rule did not provide both required train examples")
    atomic_rows(REPORT, [json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)])
    print(json.dumps({k: report[k] for k in ("decision", "baseline", "candidate", "delta", "paired_95_ci")}))
