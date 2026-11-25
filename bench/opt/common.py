"""Shared pieces of bench/opt: configuration specs, source trees, child environments, clip lists.

A configuration ("spec") is a JSON object, given inline or as a file path:

    {"name": "int8-default",          # label used in file names
     "code": "HEAD",                  # a git revision, "worktree" (the working tree as it is, with
                                      #   uncommitted changes), or the path of a source tree
     "precision": "int8",             # Scorer precision (int8 = product default; fp32 = scoreboard)
     "backbone": "/path/model.onnx",  # optional: PRONUNCIADA_BACKBONE
     "adapter_dir": "/path/dir",      # optional: PRONUNCIADA_ADAPTER_DIR (<lang>.npz files)
     "ort_options": {...}}            # optional: PRONUNCIADA_ORT_OPTIONS

Every child process gets PYTHONPATH = the spec's source tree, so the code under test is what runs,
and never gets Azure or Lingolix credentials.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import io
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LANGS = ("en", "de", "es", "fr", "it", "pt", "ru", "ja", "ko", "zh")
SECRET_VARS = ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "AZURE_SPEECH_ENDPOINT", "LINGOLIX_API_KEY")
SECRET_PATTERN = re.compile(r"(_API_KEY|_SPEECH_KEY|_SECRET|_SECRET_KEY|_TOKEN|_PASSWORD)$")   # any other service
THREAD_VARS =("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
MODEL = "zipa-small-crctc-500k"
PRECISION_FILES = {"fp32": "model.onnx", "fp16": "model.fp16.onnx", "int8": "model.int8.onnx"}
CLIPS_FILE = Path(__file__).resolve().parent / "clips.json"


def cache_root() -> Path:
    base = os.environ.get("PRONUNCIADA_CACHE")
    return Path(base).expanduser() if base else Path.home() / ".cache" / "pronunciada"


def opt_root() -> Path:
    return cache_root() / "opt"


def model_dir() -> Path:
    return cache_root() / "models" / MODEL


def default_backbone(precision: str) -> Path:
    """The plain downloaded backbone file of a precision (what candidates are built from and the int8
    suite runs as PRONUNCIADA_BACKBONE). The model cache keeps one all-outputs file per precision and
    removes the download; the package's own loader restores the download from it, byte for byte with
    its SHA-256 checked (no network), and PRONUNCIADA_KEEP_DOWNLOAD=1 (set for this process and every
    child) keeps it in place while the harness uses it."""
    os.environ["PRONUNCIADA_KEEP_DOWNLOAD"] = "1"
    plain = model_dir() / PRECISION_FILES[precision]
    if plain.exists():
        return plain
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))       # the build scripts run as plain scripts in the tools venv
    from pronunciada.acoustic import ensure_model_file

    return Path(ensure_model_file(MODEL, PRECISION_FILES[precision], str(cache_root())))


# ----------------------------------------------------------------------------- specs

def load_spec(s: str | dict | None, default_name: str = "config") -> dict:
    """Spec from a dict, an inline JSON string, a JSON file path, or None (= HEAD, int8)."""
    if s is None:
        spec = {}
    elif isinstance(s, dict):
        spec = dict(s)
    elif s.strip().startswith("{"):
        spec = json.loads(s)
    else:
        spec = json.loads(Path(s).expanduser().read_text())
    spec.setdefault("code", "HEAD")
    spec.setdefault("precision", "int8")
    spec.setdefault("name", default_name)
    if spec["precision"] not in PRECISION_FILES:
        raise ValueError(f"precision must be one of {sorted(PRECISION_FILES)}")
    unknown = set(spec) - {"name", "code", "precision", "backbone", "adapter_dir", "ort_options", "note"}
    if unknown:
        raise ValueError(f"unknown spec keys: {sorted(unknown)}")
    for k in ("backbone", "adapter_dir"):
        if spec.get(k):
            spec[k] = str(Path(spec[k]).expanduser().resolve())
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", spec["name"]):
        raise ValueError(f"spec name must be a plain file-name word: {spec['name']!r}")
    return spec


def is_default(spec: dict) -> bool:
    """True when the spec changes nothing but code and precision (no override variables)."""
    return not (spec.get("backbone") or spec.get("adapter_dir") or spec.get("ort_options"))


def override_env(spec: dict, suite: bool = False) -> dict:
    """PRONUNCIADA_* variables for a spec. With suite=True the precision is applied through
    PRONUNCIADA_BACKBONE as well, because the benchmark scripts build Scorer(lang) with the default
    precision (fp32): the int8 suite is the same scripts with the int8 file as the backbone."""
    env = {}
    bb = spec.get("backbone")
    if not bb and suite and spec["precision"] != "fp32":
        bb = str(default_backbone(spec["precision"]))
    if bb:
        env["PRONUNCIADA_BACKBONE"] = bb
    if spec.get("adapter_dir"):
        env["PRONUNCIADA_ADAPTER_DIR"] = spec["adapter_dir"]
    if spec.get("ort_options"):
        env["PRONUNCIADA_ORT_OPTIONS"] = json.dumps(spec["ort_options"], sort_keys=True)
    return env


def child_env(spec: dict, tree: Path, threads: int | None = None, suite: bool = False, extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in SECRET_VARS and not SECRET_PATTERN.search(k)
           and not k.startswith(("PRONUNCIADA_BACKBONE", "PRONUNCIADA_ADAPTER_DIR", "PRONUNCIADA_ORT_OPTIONS"))}
    env["PYTHONPATH"] = str(tree)
    env["PYTHONHASHSEED"] = "0"
    env["PRONUNCIADA_KEEP_DOWNLOAD"] = "1"      # the plain backbone files stay usable (default_backbone)
    if threads is not None:
        for v in THREAD_VARS:
            env[v] = str(threads)
    env.update(override_env(spec, suite=suite))
    env.update(extra or {})
    return env


def spec_hash(spec: dict, tree: Path) -> str:
    """Identity of what a spec runs: the tree's content id + every override (file contents hashed)."""
    h = hashlib.sha1()
    h.update(tree_id(tree).encode())
    h.update(spec["precision"].encode())
    for k in ("backbone",):
        if spec.get(k):
            h.update(file_sha1(Path(spec[k])).encode())
    if spec.get("adapter_dir"):
        for p in sorted(Path(spec["adapter_dir"]).glob("*.npz")):
            h.update(p.name.encode() + file_sha1(p).encode())
    if spec.get("ort_options"):
        h.update(json.dumps(spec["ort_options"], sort_keys=True).encode())
    return h.hexdigest()[:16]


def file_sha1(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- source trees

def git(*args, cwd: Path = REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _tracked_and_new(root: Path) -> list[str]:
    out = git("ls-files", "-z", "-c", "-o", "--exclude-standard", cwd=root)
    return sorted({p for p in out.split("\0") if p and (root / p).is_file()})


def tree_id(tree: Path) -> str:
    m = tree / ".opt_tree_id"
    if m.exists():
        return m.read_text().strip()
    return "path:" + str(tree.resolve())


def materialize(code: str, dest_root: Path | None = None) -> Path:
    """A pristine source tree for `code` (cached under opt/trees/), never the live worktree:
    a git revision is extracted with `git archive`; "worktree" copies every tracked and untracked
    (not ignored) file as it is now; a path is used as given."""
    dest_root = dest_root or opt_root() / "trees"
    if code not in ("worktree",) and Path(code).expanduser().is_dir():
        return Path(code).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    if code == "worktree":
        files = _tracked_and_new(REPO)
        h = hashlib.sha1()
        for f in files:
            h.update(f.encode() + b"\0" + file_sha1(REPO / f).encode())
        tid = "wt-" + h.hexdigest()[:12]
        dest = dest_root / tid
        if not (dest / ".opt_tree_id").exists():
            tmp = dest_root / f".{tid}.{os.getpid()}.tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            for f in files:
                (tmp / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO / f, tmp / f)
            (tmp / ".opt_tree_id").write_text(tid + "\n")
            shutil.rmtree(dest, ignore_errors=True)
            os.replace(tmp, dest)
        return dest
    sha = git("rev-parse", "--verify", code + "^{commit}")
    tid = sha[:12]
    dest = dest_root / tid
    if not (dest / ".opt_tree_id").exists():
        tmp = dest_root / f".{tid}.{os.getpid()}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        data = subprocess.run(["git", "archive", "--format=tar", sha], cwd=REPO, check=True,
                              capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            t.extractall(tmp, filter="data")
        (tmp / ".opt_tree_id").write_text(sha + "\n")
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(tmp, dest)
    return dest


def copy_tree(src: Path, dest: Path) -> Path:
    """A private, writable copy of a source tree (benchmark scripts write into results/)."""
    if dest.exists():
        return dest
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp, symlinks=True)
    os.replace(tmp, dest)
    return dest


# ----------------------------------------------------------------------------- machine

def free_pct() -> int:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return int(m.group(1)) if m else 100


def machine() -> dict:
    def sysctl(k):
        return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()

    import platform

    import numpy
    try:
        import onnxruntime
        ort_v = onnxruntime.__version__
    except ImportError:
        ort_v = None
    return {"cpu": sysctl("machdep.cpu.brand_string") or platform.processor(), "ncpu": os.cpu_count(),
            "mem_gb": round(int(sysctl("hw.memsize") or 0) / 2**30, 1), "python": platform.python_version(),
            "numpy": numpy.__version__, "onnxruntime": ort_v, "platform": platform.platform(),
            "load_avg_1min": os.getloadavg()[0]}


# ----------------------------------------------------------------------------- clips

def clip_list() -> dict:
    return json.loads(CLIPS_FILE.read_text())


def practice_clips(langs=LANGS, n: int | None = None) -> list[dict]:
    """The committed practice clips (FLEURS train, 20 per language): [{id, lang, audio, text, dur}]."""
    from bench.data import load

    ids = clip_list()["practice"]["ids"]
    out = []
    for lang in langs:
        want = ids[lang][:n] if n else ids[lang]
        by = {r["id"]: r for r in load("fleurs", split="train", lang=lang)}
        missing = [i for i in want if i not in by]
        if missing:
            raise FileNotFoundError(f"{len(missing)} practice clips of {lang} not in the FLEURS manifest")
        out += [{"id": i, "lang": lang, "audio": by[i]["audio"], "text": by[i]["text"], "dur": by[i]["dur"]}
                for i in want]
    return out


def so762_clips(n: int | None = None) -> list[dict]:
    """The 200 speechocean762 test clips bench/ondevice.py and the round-20 cost runs time (first 200
    in manifest order; speed only, never a quality number)."""
    from bench.data import load

    ids = clip_list()["so762_200"]["ids"][:n] if n else clip_list()["so762_200"]["ids"]
    by = {r["id"]: r for r in load("speechocean762", split="test")}
    return [{"id": i, "lang": "en", "audio": by[i]["audio"], "text": by[i]["text"], "dur": by[i]["dur"]} for i in ids]


def make_clip_list(per_lang: int = 20, seed: int = 20261123) -> dict:
    """How clips.json was made (run once; the ids are committed): per language, `per_lang` FLEURS
    TRAIN recordings drawn with numpy default_rng(seed) from the manifest sorted by id; plus the
    first 200 speechocean762 test recordings in manifest order (bench/ondevice.py's clips)."""
    import numpy as np

    from bench.data import load

    prac = {}
    for lang in LANGS:
        recs = sorted(load("fleurs", split="train", lang=lang), key=lambda r: r["id"])
        rng = np.random.default_rng(seed + LANGS.index(lang))
        pick = sorted(rng.choice(len(recs), per_lang, replace=False).tolist())
        prac[lang] = [recs[i]["id"] for i in pick]
    so = [r["id"] for r in load("speechocean762", split="test")[:200]]
    return {"practice": {"source": "FLEURS train split (bench.data.load('fleurs', split='train'))",
                         "rule": f"{per_lang} per language, numpy default_rng({seed} + language index) over "
                                 "the manifest sorted by id", "ids": prac},
            "so762_200": {"source": "speechocean762 test, first 200 in manifest order (= bench.ondevice.clips(200))",
                          "use": "latency and memory only; never a quality number", "ids": so}}
