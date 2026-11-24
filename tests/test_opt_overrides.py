"""pronunciada/_overrides.py: no-op when unset, applied when set (no model download needed)."""
import json
from pathlib import Path

import pytest

from pronunciada import _overrides as O
from pronunciada.adapter import Adapter

VARS = ("PRONUNCIADA_BACKBONE", "PRONUNCIADA_ADAPTER_DIR", "PRONUNCIADA_ORT_OPTIONS")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for v in VARS:
        monkeypatch.delenv(v, raising=False)


def _so_state(so):
    keys = ("graph_optimization_level", "execution_mode", "enable_mem_pattern", "enable_cpu_mem_arena",
            "intra_op_num_threads", "inter_op_num_threads", "log_severity_level", "optimized_model_filepath")
    return {k: str(getattr(so, k)) for k in keys}


def test_unset_is_a_no_op():
    ort = pytest.importorskip("onnxruntime")
    assert O.backbone_file() is None
    assert O.adapter_file("en") is None
    assert O.ort_options() is None
    so = ort.SessionOptions()
    before = _so_state(so)
    assert O.apply_ort_options(so, "backbone") is so
    assert _so_state(so) == before
    assert Adapter.path("en").as_posix().endswith("pronunciada/data/adapters/en.npz")


def test_backbone_override(monkeypatch, tmp_path):
    f = tmp_path / "cand.onnx"
    f.write_bytes(b"x")
    monkeypatch.setenv("PRONUNCIADA_BACKBONE", str(f))
    assert O.backbone_file() == str(f)
    monkeypatch.setenv("PRONUNCIADA_BACKBONE", str(tmp_path / "missing.onnx"))
    with pytest.raises(FileNotFoundError):
        O.backbone_file()


def test_backbone_override_reaches_the_model_paths(monkeypatch, tmp_path):
    from pronunciada import acoustic as A

    f = tmp_path / "cand.onnx"
    f.write_bytes(b"not a model")
    monkeypatch.setenv("PRONUNCIADA_BACKBONE", str(f))
    seen = {}

    def fake_write(src, dest, names, **kwargs):
        seen["src"] = src
        Path(dest).write_bytes(b"y")

    monkeypatch.setattr(A, "write_with_graph_outputs", fake_write)
    dest = A.embedding_model_path("zipa-small-crctc-500k", "int8", cache_dir=str(tmp_path / "nocache"))
    assert seen["src"] == str(f) and dest == str(tmp_path / "cand.emb-enc0-enc1-enc2-enc3-enc4-enc5-final-metadata-int8.onnx")


def test_adapter_dir_override(monkeypatch, tmp_path):
    (tmp_path / "de.npz").write_bytes(b"")
    monkeypatch.setenv("PRONUNCIADA_ADAPTER_DIR", str(tmp_path))
    assert Adapter.path("de") == tmp_path / "de.npz"
    assert Adapter.path("en").as_posix().endswith("pronunciada/data/adapters/en.npz")   # no file: packaged
    monkeypatch.setenv("PRONUNCIADA_ADAPTER_DIR", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError):
        Adapter.path("de")


def test_ort_options(monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    monkeypatch.setenv("PRONUNCIADA_ORT_OPTIONS", json.dumps(
        {"graph_optimization_level": "ORT_ENABLE_BASIC", "enable_mem_pattern": True,
         "config": {"session.intra_op.allow_spinning": "0"}, "sessions": ["backbone"]}))
    so = O.apply_ort_options(ort.SessionOptions(), "backbone")
    assert so.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_BASIC and so.enable_mem_pattern
    assert so.get_session_config_entry("session.intra_op.allow_spinning") == "0"
    other = ort.SessionOptions()
    before = _so_state(other)
    O.apply_ort_options(other, "prosody")                  # not listed in "sessions"
    assert _so_state(other) == before
    monkeypatch.setenv("PRONUNCIADA_ORT_OPTIONS", json.dumps({"no_such_option": 1}))
    with pytest.raises(ValueError):
        O.apply_ort_options(ort.SessionOptions(), "backbone")
    monkeypatch.setenv("PRONUNCIADA_ORT_OPTIONS", json.dumps({"sessions": ["gpu"]}))
    with pytest.raises(ValueError):
        O.ort_options()


def test_spec_env_mapping(tmp_path):
    from bench.opt.common import child_env, load_spec, override_env

    s = load_spec({"name": "x", "precision": "int8"})
    assert override_env(s) == {}
    assert override_env(s, suite=True)["PRONUNCIADA_BACKBONE"].endswith("model.int8.onnx")
    assert override_env(load_spec({"name": "y", "precision": "fp32"}), suite=True) == {}
    s = load_spec({"name": "z", "ort_options": {"enable_mem_pattern": True}, "adapter_dir": str(tmp_path)})
    e = child_env(s, tmp_path, threads=2)
    assert json.loads(e["PRONUNCIADA_ORT_OPTIONS"]) == {"enable_mem_pattern": True}
    assert e["PRONUNCIADA_ADAPTER_DIR"] == str(tmp_path) and e["OMP_NUM_THREADS"] == "2"
    assert not any(k in e for k in ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "LINGOLIX_API_KEY"))
    with pytest.raises(ValueError):
        load_spec({"name": "bad name"})
    with pytest.raises(ValueError):
        load_spec({"name": "ok", "backbon": "typo"})


def test_renamed_backbone_uses_declared_activation_types(tmp_path, monkeypatch):
    from pronunciada import acoustic as A

    # A minimal GraphProto with two intermediate tensors of different types.
    values = [A._value_info("half", 10), A._value_info("single", 1)]
    graph = b"".join(A._ld(13, value) for value in values)
    src = tmp_path / "candidate.onnx"
    src.write_bytes(A._ld(7, graph))
    monkeypatch.setenv("PRONUNCIADA_BACKBONE", str(src))
    variant = A.embedding_model_path("unused", "fp32", outputs={"a": "half", "b": "single"})
    expected = graph + b"".join(A._ld(12, value) for value in values)
    assert Path(variant).read_bytes() == A._ld(7, expected)
