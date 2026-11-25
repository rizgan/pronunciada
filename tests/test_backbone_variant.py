"""One backbone file per precision: the all-outputs variant serves every set of extra encoder
outputs and replaces the download in the model cache."""
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from pronunciada import acoustic as A

MODEL = "zipa-small-crctc-500k"
UNION = "model.int8.emb-enc0-enc1-enc2-enc3-enc4-enc5-final.onnx"


def _real_dir():
    return Path(os.environ.get("PRONUNCIADA_CACHE") or Path.home() / ".cache" / "pronunciada") / "models" / MODEL


def _fake_cache(tmp_path, real=False, copy=False):
    d = tmp_path / "models" / MODEL
    d.mkdir(parents=True)
    if real:
        src = _real_dir()
        for f in ("model.int8.onnx", "tokens.txt"):
            if not (src / f).is_file() and f == "model.int8.onnx" and A.union_valid(str(src / UNION)):
                # the cache holds only the all-outputs file: restore the download into the test folder
                A.restore_download(MODEL, "int8", str(src.parent.parent), dest=str(d / f))
                continue
            if not (src / f).is_file():
                pytest.skip("model not in the cache")
            if copy:
                shutil.copyfile(src / f, d / f)
            else:
                (d / f).symlink_to(src / f)
    else:
        (d / "model.int8.onnx").write_bytes(b"x")
        (d / "tokens.txt").write_text("<blk> 0\n")
    return d


def _fake_build(monkeypatch):
    calls = []

    def fake_write(src, dest, names):
        calls.append(list(names))
        Path(dest).write_bytes(Path(src).read_bytes() + b"y")

    monkeypatch.setattr(A, "write_with_graph_outputs", fake_write)
    monkeypatch.setattr(A, "_probe_outputs", lambda ort, path: [np.zeros(3, np.float32)])
    monkeypatch.setattr(A, "_download", lambda url, dest: (_ for _ in ()).throw(AssertionError("no download")))
    return calls


def test_every_output_set_maps_to_one_file(monkeypatch, tmp_path):
    pytest.importorskip("onnxruntime")
    monkeypatch.delenv("PRONUNCIADA_BACKBONE", raising=False)
    monkeypatch.delenv("PRONUNCIADA_KEEP_DOWNLOAD", raising=False)
    d = _fake_cache(tmp_path)
    calls = _fake_build(monkeypatch)
    # a set outside the known stacks still gets a file of its own (built next to the download)
    other = A.embedding_model_path(MODEL, "int8", str(tmp_path), {"x": "/some/other_output"})
    assert other == str(d / "model.int8.emb-x.onnx") and calls[-1] == ["/some/other_output"]
    os.remove(other)
    paths = {A.embedding_model_path(MODEL, "int8", str(tmp_path), A.embed_outputs_for(sel))
             for sel in (True, "all", ["enc0"], ["enc2", "enc4"])}
    paths.add(A.embedding_model_path(MODEL, "int8", str(tmp_path)))
    assert paths == {str(d / UNION)}
    assert calls[1:] == [list(A.ALL_LAYER_OUTPUTS.values())]       # built once, with every output
    # one model-sized file remains: the download is gone, the sidecar records both files
    assert sorted(os.listdir(d)) == sorted(["tokens.txt", UNION, UNION + ".json"])
    meta = json.loads((d / (UNION + ".json")).read_text())
    assert meta["source_size"] == 1 and meta["size"] == 2
    assert meta["source_sha256"] == hashlib.sha256(b"x").hexdigest()
    # not downloaded again while the variant is valid; a broken variant falls back to the download
    assert A.backbone_model_file(MODEL, "int8", str(tmp_path)) == str(d / UNION) and len(calls) == 2
    (d / UNION).write_bytes(b"truncated")
    assert not A.union_valid(str(d / UNION))
    with pytest.raises(AssertionError, match="no download"):
        A.backbone_model_file(MODEL, "int8", str(tmp_path))


def test_leftover_names():
    names = ["model.onnx", "model.int8.onnx", "model.fp16.onnx", "tokens.txt", "notes.onnx", "model.onnx.part",
             "model.int8.emb.onnx", "model.int8.emb-enc1.onnx", UNION, UNION + ".json",
             "model.int8.emb-enc0-enc1-enc2-enc3-enc4-enc5-final.ort1.32.0-arm64-b6694ac0.onnx",
             "model.int8.emb-enc0-enc1-enc2-enc3-enc4-enc5-final.ort1.31.1-arm64-b6694ac0.onnx",
             "model.emb.onnx", "model.emb-enc0-enc1-enc2-enc3-enc4-enc5-final.onnx", "mymodel.int8.emb.onnx",
             "model.int8.emb.onnx.bak", "model.int8.emb-enc1.onnx.123.part", "other.int8.onnx"]
    opt = "model.int8.emb-enc0-enc1-enc2-enc3-enc4-enc5-final.ort1.32.0-arm64-b6694ac0.onnx"
    off = A.leftover_files(names, "int8", preopt=False)
    assert off == sorted(["model.int8.onnx", "model.int8.emb.onnx", "model.int8.emb-enc1.onnx",
                          opt, opt.replace("ort1.32.0", "ort1.31.1")])
    on = A.leftover_files(names, "int8", preopt=True, preopt_name=opt)
    assert on == sorted(set(off) - {opt})
    fp32 = A.leftover_files(names, "fp32", preopt=False)
    assert fp32 == ["model.emb.onnx", "model.onnx"]          # never another precision's files


def test_leftovers_removed_only_in_a_private_folder(monkeypatch, tmp_path):
    pytest.importorskip("onnxruntime")
    monkeypatch.delenv("PRONUNCIADA_BACKBONE", raising=False)
    monkeypatch.delenv("PRONUNCIADA_KEEP_DOWNLOAD", raising=False)
    monkeypatch.delenv("PRONUNCIADA_PREOPT", raising=False)
    _fake_build(monkeypatch)
    old = ["model.int8.emb.onnx", "model.int8.emb-enc1.onnx",
           "model.int8.emb-enc0-enc1-enc2-enc3-enc4-enc5-final.ort1.31.1-arm64-b6694ac0.onnx"]
    keep = ["model.onnx", "model.emb.onnx", "notes.onnx", "model.int8.onnx.part", "tokens.txt"]
    # a shared cache mirrored through a symbolic link: nothing is removed there
    shared = tmp_path / "shared" / "models" / MODEL
    shared.mkdir(parents=True)
    for f in old + keep + ["model.int8.onnx"]:
        (shared / f).write_bytes(b"x")
    mirror = tmp_path / "mirror"
    (mirror / "models").mkdir(parents=True)
    (mirror / "models" / MODEL).symlink_to(shared)
    A.backbone_model_file(MODEL, "int8", str(mirror))
    assert {"model.int8.onnx", *old} <= set(os.listdir(shared))
    # a private folder: the download and old copies go, nothing else
    d = _fake_cache(tmp_path / "own")
    for f in old + keep:
        (d / f).write_bytes(b"x")
    assert A.stale_model_variants(MODEL, str(tmp_path / "own")) == []         # no variant yet
    A.backbone_model_file(MODEL, "int8", str(tmp_path / "own"))
    assert sorted(os.listdir(d)) == sorted(keep + [UNION, UNION + ".json"])
    # PRONUNCIADA_KEEP_DOWNLOAD=1 keeps the download
    d2 = _fake_cache(tmp_path / "keep")
    monkeypatch.setenv("PRONUNCIADA_KEEP_DOWNLOAD", "1")
    A.backbone_model_file(MODEL, "int8", str(tmp_path / "keep"))
    assert "model.int8.onnx" in os.listdir(d2)


@pytest.mark.model
def test_single_file_cache_and_byte_exact_restore(monkeypatch, tmp_path):
    """First load: the variant is built and checked, the download removed; later loads use the
    variant only; the download is restored byte for byte when a tool asks for it."""
    monkeypatch.delenv("PRONUNCIADA_BACKBONE", raising=False)
    monkeypatch.delenv("PRONUNCIADA_PREOPT", raising=False)
    monkeypatch.delenv("PRONUNCIADA_KEEP_DOWNLOAD", raising=False)
    d = _fake_cache(tmp_path, real=True, copy=True)
    sha = hashlib.sha256((d / "model.int8.onnx").read_bytes()).hexdigest()
    (d / "model.int8.emb.onnx").write_bytes(b"left over by an earlier version")
    x = np.random.default_rng(3).standard_normal((300, 80)).astype(np.float32) - 5
    m = A.AcousticModel(MODEL, "int8", threads=1, cache_dir=str(tmp_path), embeddings="all")
    assert sorted(os.listdir(d)) == sorted(["tokens.txt", UNION, UNION + ".json"])
    lp, e = m.posteriors_from_features(x, return_embeddings=True)
    plain = A.AcousticModel(MODEL, "int8", threads=1, cache_dir=str(tmp_path))
    assert np.array_equal(lp, plain.posteriors_from_features(x))
    assert sorted(os.listdir(d)) == sorted(["tokens.txt", UNION, UNION + ".json"])
    p = A.ensure_model_file(MODEL, "model.int8.onnx", str(tmp_path))
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha


@pytest.mark.model
def test_union_file_outputs_bit_identical(monkeypatch, tmp_path):
    """Round-5 outputs, every-stack outputs and the plain model's log-probs, from the single file."""
    monkeypatch.delenv("PRONUNCIADA_BACKBONE", raising=False)
    monkeypatch.setenv("PRONUNCIADA_PREOPT", "0")
    d = _fake_cache(tmp_path, real=True)
    x = np.random.default_rng(1).standard_normal((420, 80)).astype(np.float32) - 5
    # the download's own numbers, before the single file replaces it
    import onnxruntime as ort

    ref = ort.InferenceSession(str(d / "model.int8.onnx"), providers=["CPUExecutionProvider"])
    lp_dl = ref.run(["log_probs"], {"x": x[None], "x_lens": np.array([len(x)], np.int64)})[0][0]
    del ref
    raw = A.AcousticModel(MODEL, "int8", threads=1, cache_dir=str(tmp_path))
    r5 = A.AcousticModel(MODEL, "int8", threads=1, cache_dir=str(tmp_path), embeddings=True)
    al = A.AcousticModel(MODEL, "int8", threads=1, cache_dir=str(tmp_path), embeddings="all")
    assert r5.embeddings and al.embeddings
    assert list(r5.embed_outputs) == list(A.EMBED_OUTPUTS) and list(al.embed_outputs) == list(A.ALL_LAYER_OUTPUTS)
    assert sorted(f for f in os.listdir(d) if ".emb" in f and f.endswith(".onnx")) == [UNION]
    lp0 = raw.posteriors_from_features(x)
    lp5, e5 = r5.posteriors_from_features(x, return_embeddings=True)
    lpa, ea = al.posteriors_from_features(x, return_embeddings=True)
    assert np.array_equal(raw._fix(lp_dl), lp0)
    assert np.array_equal(lp0, lp5) and np.array_equal(lp0, lpa)
    assert set(e5) == set(A.EMBED_OUTPUTS)
    for k in e5:
        assert np.array_equal(e5[k], ea[k])
    assert np.array_equal(r5.posteriors_batch([np.zeros(16000, np.float32)])[0],
                          raw.posteriors_batch([np.zeros(16000, np.float32)])[0])


@pytest.mark.model
def test_preoptimized_graph_bit_identical(monkeypatch, tmp_path):
    """PRONUNCIADA_PREOPT=1: the saved optimized graph (first load builds it, later loads read it with
    the optimizer off) gives the same numbers as a session optimized at load; it is a second file."""
    monkeypatch.delenv("PRONUNCIADA_BACKBONE", raising=False)
    monkeypatch.delenv("PRONUNCIADA_ORT_OPTIONS", raising=False)
    d = _fake_cache(tmp_path, real=True)
    x = np.random.default_rng(2).standard_normal((380, 80)).astype(np.float32) - 5
    monkeypatch.delenv("PRONUNCIADA_PREOPT", raising=False)
    ref = A.AcousticModel(MODEL, "int8", threads=1, cache_dir=str(tmp_path), embeddings="all")
    monkeypatch.setenv("PRONUNCIADA_PREOPT", "1")
    built = A.AcousticModel(MODEL, "int8", threads=1, cache_dir=str(tmp_path), embeddings="all")
    opt = os.path.basename(A.preoptimized_model_path(str(d / UNION)))
    assert sorted(f for f in os.listdir(d) if ".emb" in f and f.endswith(".onnx")) == sorted([UNION, opt])
    loaded = A.AcousticModel(MODEL, "int8", threads=2, cache_dir=str(tmp_path), embeddings=True)
    r5 = A.AcousticModel(MODEL, "int8", threads=1, cache_dir=str(tmp_path), embeddings=True)
    lp, e = ref.posteriors_from_features(x, return_embeddings=True)
    for m in (built, loaded, r5):
        assert m.embeddings
        lp2, e2 = m.posteriors_from_features(x, return_embeddings=True)
        assert np.array_equal(lp, lp2)
        for k in e2:
            assert np.array_equal(e[k], e2[k])
    assert set(r5.embed_outputs) == set(A.EMBED_OUTPUTS)
    # an unreadable optimized graph is rebuilt
    (d / opt).write_bytes(b"broken")
    again = A.AcousticModel(MODEL, "int8", threads=1, cache_dir=str(tmp_path), embeddings="all")
    assert np.array_equal(lp, again.posteriors_from_features(x)) and (d / opt).stat().st_size > 1000
    # with the switch off again, the saved graph is a leftover
    monkeypatch.setenv("PRONUNCIADA_PREOPT", "0")
    assert str(d / opt) in A.stale_model_variants(MODEL, str(tmp_path))


def test_preopt_name_carries_runtime_and_cpu(monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    p = A.preoptimized_model_path("/c/m/model.int8.emb-x.onnx")
    assert p.startswith("/c/m/model.int8.emb-x.ort" + ort.__version__ + "-") and p.endswith(".onnx")
    assert A._cpu_fingerprint() in p and len(A._cpu_fingerprint()) == 8
    monkeypatch.setattr(A, "_cpu_fingerprint", lambda: "deadbeef")
    assert A.preoptimized_model_path("/c/m/model.int8.emb-x.onnx") != p   # another CPU: another file
    assert A._leftover_re().match(os.path.basename(p))


def test_fp16_union_declares_half_precision_outputs(monkeypatch, tmp_path):
    pytest.importorskip("onnxruntime")
    monkeypatch.delenv("PRONUNCIADA_BACKBONE", raising=False)
    monkeypatch.setenv("PRONUNCIADA_KEEP_DOWNLOAD", "1")
    monkeypatch.setattr(A, "_probe_outputs", lambda ort, path: [np.zeros(3, np.float32)])
    folder = tmp_path / "models" / MODEL
    folder.mkdir(parents=True)
    source = folder / "model.fp16.onnx"
    source.write_bytes(A._ld(7, b""))
    old_variant = folder / "model.fp16.emb-enc0-enc1-enc2-enc3-enc4-enc5-final.onnx"
    old_variant.write_bytes(A._ld(7, b"".join(A._ld(12, A._value_info(n)) for n in A.ALL_LAYER_OUTPUTS.values())))

    variant = Path(A.embedding_model_path(MODEL, "fp16", str(tmp_path)))
    assert variant != old_variant and A.union_valid(str(variant))
    half_outputs = b"".join(A._ld(12, A._value_info(n, 10)) for n in A.ALL_LAYER_OUTPUTS.values())
    assert variant.read_bytes() == A._ld(7, half_outputs)
    assert source.is_file()
