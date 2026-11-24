"""Prepare train-only inputs for the English phoneme-class experiment."""
from __future__ import annotations

import json
from functools import lru_cache
from importlib.metadata import version

import numpy as np

from bench.calib import common, train_sound_emb
from bench.calib.train_sound import KINDS
from bench.coach.en_precision import DATASET, ROOT, file_hash
from bench.pipeline import features as pipeline
from bench.pipeline.common import load_lp, lp_path
from pronunciada import calibrate, features


def prepare_table():
    import pandas as pd

    cfg = pipeline.config("en")
    rows = []
    for i, rec in enumerate(pipeline.records(DATASET, split="train"), 1):
        if not lp_path(DATASET, rec["id"]).exists():
            load_lp(DATASET, rec)
        phones, _, _ = pipeline.so762_utt((dict(rec, _dataset=DATASET), "en", cfg))
        if not phones:
            raise ValueError(f"No feature rows for {rec['id']}")
        rows.extend(phones)
        if i % 500 == 0:
            print(f"Train table: {i}", flush=True)
    pipeline._write(pd.DataFrame(rows), pipeline.feat_dir() / f"{DATASET}.parquet")


def prepare_prototypes():
    from bench.calib.sound_emb import proto_path

    path = ROOT / "pronunciada/data/calib/sound_E_emb.npz"
    with np.load(path) as shipped:
        np.savez(proto_path("en"), **{k: shipped["proto_" + k] for k in ("mu", "sd", "P", "phones")})


def checked_folds(groups):
    folds = common.speaker_folds(groups)
    visits = np.zeros(len(groups), dtype=int)
    for tr, te in folds:
        if np.intersect1d(groups[tr], groups[te]).size:
            raise ValueError("A speaker appears in both training and validation")
        visits[te] += 1
    if not np.all(visits == 1):
        raise ValueError("Fold validation does not cover each row once")
    return folds


def detail_oof():
    from bench.calib.sound_emb import proto_path
    from bench.data import load

    df, table, labels, groups, _, vectors = train_sound_emb.load()
    train = load(DATASET, "train")
    speakers = {r["id"]: r["speaker"] for r in train}
    if (len(speakers) != 2500 or len(set(speakers.values())) != 125
            or set(df.utt) != set(speakers) or not df.utt.map(speakers).eq(df.speaker).all()
            or df.duplicated(["utt", "phone_index"]).any() or not np.isfinite(vectors).all()):
        raise ValueError("Incomplete or incompatible train features/vectors")

    model_name = calibrate.config()["sound_detail_model"]["en"]
    model = calibrate.load_level(model_name)
    spec = features.SOUND_SETS[train_sound_emb.SPEC]
    if model_name != "sound_E" or model.num != list(spec["num"]) + train_sound_emb.EXTRA:
        raise ValueError("Shipped English detail model has different inputs")
    if model.cat != list(spec["cat"]) or model.encoding != "mixed":
        raise ValueError("Unexpected English detail encoding")
    cat_mask = features.categorical_mask(model.num, model.cat, model.vocab, model.encoding)
    params = common.load_json("calib_sound.json")["params"]
    splits = checked_folds(groups)
    fold_ids = np.full(len(df), -1)
    for fold, (_, te) in enumerate(splits):
        fold_ids[te] = fold
    ok_vectors = np.isfinite(vectors[:, 0])

    @lru_cache(maxsize=None)
    def held_logits(excluded):
        held = np.isin(fold_ids, excluded)
        tr, te = np.flatnonzero(~held), np.flatnonzero(held)
        projection = train_sound_emb.projection_for(vectors, labels["reg"], groups, tr, te, ok_vectors)
        x = model.matrix(dict(table, emb_proj=projection))
        logits = np.full((len(df), 2), np.nan)
        for j, head in enumerate(("len", "str")):
            y = labels[head]
            labelled = tr[np.isfinite(y[tr])]
            fitted = common.fit(KINDS[head], params[head], x[labelled], y[labelled], cat_mask)
            logits[te, j] = common.raw(fitted, x[te])
        print(f"Sound_E raw logits fitted with folds {excluded} held out", flush=True)
        return logits

    def calibrated(excluded, target):
        available = sorted(set(range(len(splits))) - set(excluded))
        # Reuse logits from the next level of exclusion to fit each calibration map.
        # For nested exclusions use one calibration fold to limit the number of fits.
        if len(excluded) > 1:
            available = available[:1]
        cal_parts = [splits[k][1] for k in available]
        cal_logits = np.concatenate([
            held_logits(tuple(sorted((*excluded, k))))[rows]
            for k, rows in zip(available, cal_parts, strict=True)
        ])
        cal_rows = np.concatenate(cal_parts)
        raw = held_logits(excluded)[target]
        logits = np.empty_like(raw)
        for j, head in enumerate(("len", "str")):
            y = labels[head][cal_rows]
            valid = np.isfinite(y)
            cal = common.fit_calib(cal_logits[valid, j], y[valid], model.heads[head].calib.kind)
            p = common.apply_calib(cal, raw[:, j])
            p = np.clip(p, 1e-6, 1 - 1e-6)
            logits[:, j] = np.log(p / (1 - p))
        logits[:, 1] = np.minimum(logits[:, 0], logits[:, 1])
        return logits

    def add_logits(frame, prefix, logits):
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite detail OOF logits")
        frame[prefix + "_z_len"], frame[prefix + "_z_str"] = logits.T

    logits = np.empty((len(df), 2))
    out = df[["utt", "phone_index", "speaker", "phone", "phone_score"]].copy()
    out["fold"] = fold_ids
    for fold, (_, te) in enumerate(splits):
        logits[te] = calibrated((fold,), te)
    add_logits(out, "E", logits)
    for outer, (_, te) in enumerate(splits):
        nested = np.empty_like(logits)
        nested[te] = logits[te]
        for inner, (_, val) in enumerate(splits):
            if inner != outer:
                nested[val] = calibrated(tuple(sorted((outer, inner))), val)
        add_logits(out, f"E_outer{outer}", nested)
        for inner in sorted(set(range(len(splits))) - {outer}):
            inner_features = nested.copy()
            for k, (_, train_rows) in enumerate(splits):
                if k not in (outer, inner):
                    inner_features[train_rows] = calibrated(tuple(sorted((outer, inner, k))), train_rows)
            add_logits(out, f"E_outer{outer}_inner{inner}", inner_features)
    path = common.calib_cache() / "sound_oof.parquet"
    pipeline._write(out, path)
    inputs = (pipeline.feat_dir() / f"{DATASET}.parquet", proto_path("en"),
              proto_path("en").with_name("so762_train_vectors.pkl"))
    metadata = {"schema": 3, "detail_model": model_name, "feature_config": pipeline.config("en"),
                "calibration": {h: model.heads[h].calib.kind for h in ("len", "str")},
                "rows": len(out), "labelled_rows": int(df.phone_score.count()),
                "speakers": len(set(groups)), "sha256": file_hash(path),
                "versions": {name: version(name) for name in (
                    "numpy", "scipy", "scikit-learn", "pandas", "pyarrow")},
                "inputs": {p.name: file_hash(p) for p in inputs},
                "shipped": {name: file_hash(ROOT / name) for name in (
                    "pronunciada/data/calib/sound_E.npz", "pronunciada/data/calib/sound_E_emb.npz",
                    "results/tables/calib_sound.json")}}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{path}: {len(out)} rows")
