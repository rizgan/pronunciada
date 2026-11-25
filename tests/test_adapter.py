import os
import time

import numpy as np
import pytest

from pronunciada.adapter import Adapter, _log_softmax, _windows

V = 127


def _lp(T=50, seed=0):
    rng = np.random.default_rng(seed)
    return _log_softmax(rng.normal(0, 3, (T, V)).astype(np.float32))


def _zero(kind, H=8, K=5):
    base = {"kind": np.array(kind), "floor": np.array(-20.0), "scale": np.array(10.0)}
    if kind == "linear":
        base.update(W=np.zeros((V, V), np.float32), b=np.zeros(V, np.float32))
    elif kind == "ctx":
        base.update(W=np.zeros((K, V, V), np.float32), b=np.zeros(V, np.float32))
    else:
        base.update(W1=np.zeros((K, V, H), np.float32), b1=np.zeros(H, np.float32),
                    W2=np.zeros((K, H, H), np.float32), b2=np.zeros(H, np.float32),
                    Wo=np.zeros((H, V), np.float32), bo=np.zeros(V, np.float32))
    return base


@pytest.mark.parametrize("kind", ["linear", "ctx", "conv"])
def test_zero_adapter_is_identity(kind, tmp_path):
    p = tmp_path / "a.npz"
    np.savez(p, **_zero(kind))
    a = Adapter.from_file(p)
    lp = _lp()
    out = a(lp)
    assert out.shape == lp.shape and out.dtype == np.float32
    np.testing.assert_allclose(out, lp, atol=1e-5)
    np.testing.assert_allclose(np.exp(out).sum(1), 1.0, atol=1e-4)


def test_ctx_matches_explicit_sum():
    rng = np.random.default_rng(1)
    params = _zero("ctx")
    params["W"] = rng.normal(0, 0.01, (5, V, V)).astype(np.float32)
    a = Adapter(params)
    lp = _lp(20)
    x = np.maximum(lp, -20.0) / 10.0
    T = len(x)
    f = np.zeros_like(x)
    for t in range(T):
        for k in range(5):
            f[t] += x[min(max(t + k - 2, 0), T - 1)] @ params["W"][k]
    np.testing.assert_allclose(a.delta(lp), f, atol=1e-4)


def test_windows_edge_replicated():
    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    w = _windows(x, 3)
    np.testing.assert_array_equal(w[0], [0, 1, 0, 1, 2, 3])
    np.testing.assert_array_equal(w[2], [2, 3, 4, 5, 4, 5])


def test_empty_input():
    a = Adapter(_zero("linear"))
    assert a(np.zeros((0, V), np.float32)).shape == (0, V)


def test_missing_language_has_no_adapter():
    assert Adapter.load("xx") is None


@pytest.mark.parametrize("lang", ["en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko"])
def test_shipped_adapters_fast_and_normalised(lang):
    a = Adapter.load(lang)
    if a is None:
        pytest.skip(f"no adapter shipped for {lang}")
    lp = _lp(500)                         # 10 s of audio at 50 frames/s
    a(lp)
    best = min(_time(a, lp) for _ in range(5))
    out = a(lp)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(np.exp(out).sum(1), 1.0, atol=1e-3)
    limit = 20e-3 if os.environ.get("CI") else 3e-3      # shared CI runners are several times slower than a laptop
    assert best < limit, f"{lang}: {best * 1e3:.2f} ms"


def _time(a, lp):
    t = time.perf_counter()
    a(lp)
    return time.perf_counter() - t


def test_runtime_has_no_training_deps():
    import subprocess
    import sys

    code = ("import sys, pronunciada.adapter, pronunciada.variants; "
            "bad = [m for m in ('torch', 'sklearn') if m in sys.modules]; print(bad); sys.exit(bool(bad))")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
