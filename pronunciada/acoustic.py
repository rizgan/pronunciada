"""ZIPA ONNX wrapper: audio -> frame log-posteriors over the packaged token list.

Frame timing (measured on zipa-small-crctc-500k): with T fbank frames (10 ms, snip_edges=False,
i.e. T = (num_samples + 80) // 160) the model emits T' = (T - 7) // 2 frames (needs T >= 9).
Output frame j depends mostly on fbank frames 2j .. 2j+7, centre 2j + 3.5, i.e. centre time
0.02 j + 0.04 s. `frame_to_time(j)` = 0.02 j + 0.03 is the START of the 20 ms interval of
frame j, so a span [s, e) of frames covers [frame_to_time(s), frame_to_time(e)).
"""
from __future__ import annotations

import functools
import os

import numpy as np

from ._overrides import apply_ort_options, backbone_file, ort_options
from .audio import fbank, load_audio
from .text.ipa import load_tokens

HF_URL = "https://huggingface.co/anyspeech/{name}/resolve/main/{file}"
PRECISION_FILES = {"fp32": "model.onnx", "fp16": "model.fp16.onnx", "int8": "model.int8.onnx"}
NON_SPEECH = 4                  # ids 0..3: <blk> <sos/eos> <unk> ▁
MIN_FBANK_FRAMES = 9            # model needs >= 9 input frames to emit one output frame
SILENCE_FEAT = float(np.log(np.finfo(np.float32).eps))   # log-mel value of digital silence

# Intermediate encoder activations exposed as extra graph outputs (round 5, sentence "delivery"
# features, pronunciada.delivery). Names are the same in the fp32, fp16 and int8 exports of
# zipa-small-crctc-500k; shapes [T', 1, C] for the stacks and [1, T', C] for the final output
# (the input of the CTC layer), all on the 20 ms output grid.
EMBED_OUTPUTS = {
    "enc1": "/encoder/1/out_combiner/Add_output_0",      # stack 1 (early), 256-d
    "enc3": "/encoder/3/out_combiner/Add_output_0",      # stack 3 (middle), 512-d
    "enc5": "/encoder/5/out_combiner/Add_output_0",      # stack 5 (late), 256-d
    "final": "/Transpose_1_output_0",                    # encoder output before the CTC layer, 512-d
}
# Round 19: the output of every encoder stack (layer fusion for the English sentence head).
# Stack 0 runs at the full rate and has no out_combiner; its output is its last layer's bypass.
# Same [T', 1, C] layout as the stacks above.
ALL_LAYER_OUTPUTS = {
    "enc0": "/encoder/0/layers.1/bypass/Add_output_0",
    "enc1": EMBED_OUTPUTS["enc1"],
    "enc2": "/encoder/2/out_combiner/Add_output_0",
    "enc3": EMBED_OUTPUTS["enc3"],
    "enc4": "/encoder/4/out_combiner/Add_output_0",
    "enc5": EMBED_OUTPUTS["enc5"],
    "final": EMBED_OUTPUTS["final"],
}


def embed_outputs_for(layers) -> dict:
    """{key: tensor name} for a layer selection: True = EMBED_OUTPUTS (round 5), "all" =
    ALL_LAYER_OUTPUTS, or an iterable of keys of ALL_LAYER_OUTPUTS (EMBED_OUTPUTS are always
    included, since earlier rounds' inputs read them)."""
    if layers is True:
        return dict(EMBED_OUTPUTS)
    if layers == "all":
        return dict(ALL_LAYER_OUTPUTS)
    keys = set(layers) | set(EMBED_OUTPUTS)
    return {k: v for k, v in ALL_LAYER_OUTPUTS.items() if k in keys}


# --------------------------------------------------------------------------- protobuf surgery
# Adding graph outputs needs no `onnx` package: a repeated protobuf field may be extended by
# appending its encoding to the message, so the ModelProto bytes are edited in place
# (ModelProto.graph = field 7, GraphProto.output = field 12, ValueInfoProto.name = field 1,
# ValueInfoProto.type = 2 -> TypeProto.tensor_type = 1 -> {elem_type = 1 (float), shape = 2 with
# three symbolic dims}).

def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _ld(field: int, payload: bytes) -> bytes:
    """Length-delimited protobuf field."""
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def _value_info(name: str, element_type: int = 1) -> bytes:
    shape = b"".join(_ld(1, _ld(2, f"{name}_d{k}".encode())) for k in range(3))
    tensor = _varint(1 << 3) + _varint(element_type) + _ld(2, shape)
    return _ld(1, name.encode("utf-8")) + _ld(2, _ld(1, tensor))


def _read_varint(buf, i: int) -> tuple[int, int]:
    shift = n = 0
    while True:
        b = buf[i]
        i += 1
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            return n, i
        shift += 7


def add_graph_outputs(model_bytes: bytes, names) -> bytes:
    """Serialized ONNX ModelProto -> the same model with `names` (existing tensor names)
    appended to graph.output as float tensors of unspecified shape."""
    buf = memoryview(model_bytes)
    i = 0
    while i < len(buf):
        start = i
        key, i = _read_varint(buf, i)
        field, wt = key >> 3, key & 7
        if wt == 0:
            _, i = _read_varint(buf, i)
        elif wt == 1:
            i += 8
        elif wt == 5:
            i += 4
        elif wt == 2:
            ln, j = _read_varint(buf, i)
            if field == 7:
                extra = b"".join(_ld(12, _value_info(n)) for n in names)
                payload = bytes(buf[j:j + ln]) + extra
                return bytes(buf[:start]) + _ld(7, payload) + bytes(buf[j + ln:])
            i = j + ln
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")
    raise ValueError("no graph in the ONNX model")


def _metadata_fields(buf, start, end):
    """Yield length-delimited protobuf fields without copying the model weights."""
    while start < end:
        key, start = _read_varint(buf, start)
        wire = key & 7
        if wire == 2:
            size, start = _read_varint(buf, start)
            if start + size > end:
                raise ValueError("truncated ONNX metadata")
            yield key >> 3, start, start + size
            start += size
        elif wire == 0:
            _, start = _read_varint(buf, start)
        elif wire in (1, 5):
            start += 8 if wire == 1 else 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")


def _override_outputs(src_path, start, end, names, fallback_type):
    """Reuse graph.value_info when an override declares its own activation types."""
    import mmap

    wanted = {name.encode(): name for name in names}
    found = {}
    with open(src_path, "rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        for field, i, j in _metadata_fields(buf, start, end):
            if field != 13:  # GraphProto.value_info
                continue
            fields = {k: (a, b) for k, a, b in _metadata_fields(buf, i, j)}
            if 1 in fields and 2 in fields:
                a, b = fields[1]
                name = wanted.get(buf[a:b])
                if name is not None:
                    found[name] = buf[i:j]
    return b"".join(_ld(12, found.get(n) or _value_info(n, fallback_type)) for n in names)


def _copy_range(src, dst, start: int, end: int, chunk: int = 1 << 22) -> None:
    src.seek(start)
    left = end - start
    while left > 0:
        b = src.read(min(chunk, left))
        if not b:
            raise ValueError("truncated ONNX model")
        dst.write(b)
        left -= len(b)


def write_with_graph_outputs(src_path: str, dest_path: str, names, *, element_type: int = 1, use_metadata: bool = False) -> None:
    """add_graph_outputs from file to file, streamed in 4 MB chunks: the same bytes, but without
    holding the model in memory (three copies of it at once otherwise: ~1 GB for fp32)."""
    size = os.path.getsize(src_path)
    with open(src_path, "rb") as f:
        i = 0
        while i < size:
            f.seek(i)
            head = f.read(32)
            key, j = _read_varint(head, 0)
            field, wt = key >> 3, key & 7
            if wt == 0:
                _, j = _read_varint(head, j)
                i += j
            elif wt == 1:
                i += j + 8
            elif wt == 5:
                i += j + 4
            elif wt == 2:
                ln, k = _read_varint(head, j)
                if field == 7:
                    if use_metadata:
                        extra = _override_outputs(src_path, i + k, i + k + ln, names, element_type)
                    else:
                        extra = b"".join(_ld(12, _value_info(n, element_type)) for n in names)
                    with open(dest_path, "wb") as out:
                        _copy_range(f, out, 0, i)
                        out.write(_varint((7 << 3) | 2) + _varint(ln + len(extra)))
                        _copy_range(f, out, i + k, i + k + ln)
                        out.write(extra)
                        _copy_range(f, out, i + k + ln, size)
                    return
                i += k + ln
            else:
                raise ValueError(f"unsupported protobuf wire type {wt}")
    raise ValueError("no graph in the ONNX model")


def _variant_path(src: str, outputs: dict) -> str:
    tag = "" if list(outputs) == list(EMBED_OUTPUTS) else "-" + "-".join(outputs)
    if src.endswith(".fp16.onnx"):
        tag = "-fp16" + tag  # Do not reuse older variants that declared float32 outputs.
    return src[:-len(".onnx")] + f".emb{tag}.onnx"


def embedding_model_path(name: str, precision: str, cache_dir: str | None = None,
                         outputs: dict | None = None) -> str:
    """Path of the model variant that carries `outputs` (default EMBED_OUTPUTS) as extra outputs.

    Every subset of ALL_LAYER_OUTPUTS (which is every set the scorer asks for) is served by ONE
    file, `<model>.emb-enc0-enc1-enc2-enc3-enc4-enc5-final.onnx`, that carries all of them; the
    session then fetches only the outputs asked for. Exposing a tensor as a graph output does not
    change what ONNX Runtime computes: log-probs and every activation are bit-identical to those
    of a file with fewer extra outputs (or none), fp32 and int8, and an output that is not fetched
    costs no measurable time. In the model cache that file is the only model-sized file of its
    precision (`backbone_model_file`). Only a set outside ALL_LAYER_OUTPUTS gets a file of its own,
    built next to the download (with PRONUNCIADA_BACKBONE: next to that file)."""
    outputs = dict(EMBED_OUTPUTS) if outputs is None else outputs
    if backbone_file() is None and set(outputs.values()) <= set(ALL_LAYER_OUTPUTS.values()):
        return backbone_model_file(name, precision, cache_dir)
    src = backbone_file() or ensure_model_file(name, PRECISION_FILES[precision], cache_dir)
    if set(outputs.values()) <= set(ALL_LAYER_OUTPUTS.values()):
        outputs = dict(ALL_LAYER_OUTPUTS)
    override = backbone_file() is not None
    dest = _variant_path(src, outputs)
    if override:
        dest = dest[:-5] + f"-metadata-{precision}.onnx"
    if not os.path.exists(dest) or os.path.getmtime(dest) < os.path.getmtime(src):
        part = f"{dest}.{os.getpid()}.part"
        try:
            if override:
                write_with_graph_outputs(src, part, list(outputs.values()), use_metadata=True,
                                         element_type=10 if precision == "fp16" else 1)
            elif precision == "fp16":
                write_with_graph_outputs(src, part, list(outputs.values()), element_type=10)  # ONNX FLOAT16
            else:
                write_with_graph_outputs(src, part, list(outputs.values()))
        except BaseException:
            if os.path.exists(part):
                os.remove(part)
            raise
        os.replace(part, dest)
    return dest


# --------------------------------------------------------------------------- one file per precision
# The model cache keeps ONE model-sized file per precision: the all-outputs variant, a superset of
# the download (the same bytes, with the extra encoder outputs declared in the graph). A small
# sidecar (`<variant>.json`) records the sizes and SHA-256 of the variant and of the download it was
# built from. Once the variant is built and verified (both files load and give the same outputs on
# a probe input), the download is removed; it is not downloaded again while the variant is present
# and valid, and `ensure_model_file` restores it byte for byte from the variant (checked against the
# recorded SHA-256) for a tool that asks for the plain file. A missing or invalid variant falls back
# to the download (downloading it again if needed). PRONUNCIADA_KEEP_DOWNLOAD=1 keeps the download.
# Nothing is removed from a model folder that is reached through a symbolic link (a shared cache
# mirrored into a private one), and only files with this module's own names are ever removed.

_SIDECAR_VERSION = 1
_PROBE_FRAMES = 160


def union_model_path(name: str, precision: str, cache_dir: str | None = None) -> str:
    """Where the all-outputs variant of a precision lives in the model cache."""
    d = os.path.join(cache_root(cache_dir), "models", name)
    return _variant_path(os.path.join(d, PRECISION_FILES[precision]), ALL_LAYER_OUTPUTS)


def _sidecar(path: str) -> str:
    return path + ".json"


def _read_sidecar(path: str) -> dict | None:
    import json

    try:
        with open(_sidecar(path), encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or meta.get("version") != _SIDECAR_VERSION:
        return None
    return meta


def union_valid(path: str) -> bool:
    """The variant exists, has a sidecar of this format and the recorded size (cheap check, every
    load; the SHA-256 is checked when the file is built and when the download is restored)."""
    meta = _read_sidecar(path)
    try:
        return meta is not None and os.path.getsize(path) == int(meta["size"]) \
            and list(meta.get("outputs") or ()) == list(ALL_LAYER_OUTPUTS.values())
    except OSError:
        return False


def _sha256(path: str, chunk: int = 1 << 22) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def _probe_outputs(ort, path: str) -> list:
    """Log-probs of a fixed probe input (one thread, optimizer default): the load check."""
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.enable_mem_pattern = False
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    x = (np.random.default_rng(20261123).standard_normal((1, _PROBE_FRAMES, 80)).astype(np.float32) - 5.0)
    feeds = {}
    for i in sess.get_inputs():
        feeds[i.name] = x if len(i.shape) == 3 else np.array([_PROBE_FRAMES], dtype=np.int64)
    out = sess.run(["log_probs"], feeds)
    del sess
    return out


def _private_folder(d: str, root: str) -> bool:
    """False when the model folder, or a folder between it and the cache root, is a symbolic link
    (an overlay of a shared cache): nothing is removed there."""
    p = os.path.abspath(d)
    stop = os.path.abspath(root)
    while True:
        if os.path.islink(p):
            return False
        if p == stop or os.path.dirname(p) == p:
            return True
        p = os.path.dirname(p)


def build_union(name: str, precision: str, cache_dir: str | None = None) -> str:
    """Build the all-outputs variant from the download (streamed: the model is never held in memory),
    check it, write its sidecar; then remove the download (see the block comment above)."""
    import json

    import onnxruntime as ort

    src = ensure_model_file(name, PRECISION_FILES[precision], cache_dir)
    dest = union_model_path(name, precision, cache_dir)
    part = f"{dest}.{os.getpid()}.part"
    try:
        if precision == "fp16":
            write_with_graph_outputs(src, part, list(ALL_LAYER_OUTPUTS.values()), element_type=10)
        else:
            write_with_graph_outputs(src, part, list(ALL_LAYER_OUTPUTS.values()))
        a = _probe_outputs(ort, src)
        b = _probe_outputs(ort, part)
        if len(a) != len(b) or not all(np.array_equal(x, y) for x, y in zip(a, b)):
            raise ValueError("the all-outputs variant does not reproduce the model's outputs")
        meta = {"version": _SIDECAR_VERSION, "model": name, "precision": precision,
                "source": PRECISION_FILES[precision], "source_size": os.path.getsize(src),
                "source_sha256": _sha256(src), "outputs": list(ALL_LAYER_OUTPUTS.values()),
                "size": os.path.getsize(part), "sha256": _sha256(part)}
        os.replace(part, dest)
    except BaseException:
        if os.path.exists(part):
            os.remove(part)
        raise
    tmp = f"{_sidecar(dest)}.{os.getpid()}.part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp, _sidecar(dest))
    return dest


def model_cached(name: str = "zipa-small-crctc-500k", precision: str = "fp32", cache_dir: str | None = None) -> bool:
    """The backbone of a precision is in the model cache (the download or the all-outputs file
    that replaces it), so loading it needs no network."""
    d = os.path.join(cache_root(cache_dir), "models", name)
    return os.path.exists(os.path.join(d, PRECISION_FILES[precision])) or union_valid(
        union_model_path(name, precision, cache_dir))


def download_size(name: str, precision: str, cache_dir: str | None = None) -> int:
    """Size in bytes of the downloadable backbone file of a precision: the file in the model cache,
    or, once it was replaced by the all-outputs variant, the size recorded in the sidecar."""
    d = os.path.join(cache_root(cache_dir), "models", name)
    plain = os.path.join(d, PRECISION_FILES[precision])
    if os.path.exists(plain):
        return os.path.getsize(plain)
    meta = _read_sidecar(union_model_path(name, precision, cache_dir))
    if meta is None:
        raise FileNotFoundError(plain)
    return int(meta["source_size"])


def backbone_model_file(name: str, precision: str, cache_dir: str | None = None) -> str:
    """The one backbone file of a precision in the model cache (all extra outputs declared): built on
    first use, reused afterwards; leftovers of earlier versions are removed once it is in place."""
    dest = union_model_path(name, precision, cache_dir)
    if not union_valid(dest):
        dest = build_union(name, precision, cache_dir)
    remove_leftovers(name, precision, cache_dir)
    return dest


def restore_download(name: str, precision: str, cache_dir: str | None = None, dest: str | None = None) -> str:
    """Write the plain download back from the all-outputs variant (the inverse of
    write_with_graph_outputs, streamed) and check it against the recorded SHA-256; `dest`
    defaults to its place in the model cache."""
    union = union_model_path(name, precision, cache_dir)
    meta = _read_sidecar(union)
    if meta is None or not union_valid(union):
        raise FileNotFoundError(union)
    dest = dest or os.path.join(os.path.dirname(union), PRECISION_FILES[precision])
    extra = b"".join(_ld(12, _value_info(n)) for n in meta["outputs"])
    part = f"{dest}.{os.getpid()}.part"
    size = os.path.getsize(union)
    try:
        with open(union, "rb") as f:
            i = 0
            while i < size:
                f.seek(i)
                head = f.read(32)
                key, j = _read_varint(head, 0)
                field, wt = key >> 3, key & 7
                if wt == 0:
                    _, j = _read_varint(head, j)
                    i += j
                elif wt == 1:
                    i += j + 8
                elif wt == 5:
                    i += j + 4
                elif wt == 2:
                    ln, k = _read_varint(head, j)
                    if field == 7:
                        with open(part, "wb") as out:
                            _copy_range(f, out, 0, i)
                            out.write(_varint((7 << 3) | 2) + _varint(ln - len(extra)))
                            _copy_range(f, out, i + k, i + k + ln - len(extra))
                            f.seek(i + k + ln - len(extra))
                            if f.read(len(extra)) != extra:
                                raise ValueError("the variant does not end its graph with the recorded outputs")
                            _copy_range(f, out, i + k + ln, size)
                        break
                    i += k + ln
                else:
                    raise ValueError(f"unsupported protobuf wire type {wt}")
        if _sha256(part) != meta["source_sha256"]:
            raise ValueError("restored model does not match the recorded SHA-256")
        os.replace(part, dest)
    except BaseException:
        if os.path.exists(part):
            os.remove(part)
        raise
    return dest


_LEFTOVER = None


def _leftover_re():
    import re

    global _LEFTOVER
    if _LEFTOVER is None:
        stems = "|".join(re.escape(f[:-len(".onnx")]) for f in PRECISION_FILES.values())
        # <stem>.emb[-keys][.ort<version>-<machine>-<cpu>].onnx
        _LEFTOVER = re.compile(rf"^(?P<stem>{stems})\.emb(?P<keys>-[A-Za-z0-9-]+)?"
                               rf"(?P<opt>\.ort[0-9][0-9A-Za-z.+]*-[A-Za-z0-9_]+-[0-9a-f]{{8}})?\.onnx$")
    return _LEFTOVER


def leftover_files(names, precision: str, preopt: bool, preopt_name: str | None = None) -> list[str]:
    """Which of the file names `names` (a model folder's listing) are leftovers once the
    all-outputs variant of `precision` is in place: the plain download, copies with other output
    sets (`<model>.emb.onnx`, `<model>.emb-<keys>.onnx`), and saved optimized graphs except the
    current one when PRONUNCIADA_PREOPT=1. Pure name matching; nothing else is ever listed."""
    plain = PRECISION_FILES[precision]
    stem = plain[:-len(".onnx")]
    union = _variant_path(plain, ALL_LAYER_OUTPUTS)
    out = []
    for f in names:
        if f == plain:
            out.append(f)
            continue
        m = _leftover_re().match(f)
        if not m or m.group("stem") != stem or f == union:
            continue
        if m.group("opt") and preopt and f == preopt_name:
            continue
        out.append(f)
    return sorted(out)


def remove_leftovers(name: str, precision: str, cache_dir: str | None = None) -> list[str]:
    """Remove `leftover_files` of a precision from the model folder (only when its all-outputs
    variant is valid and the folder is not reached through a symbolic link)."""
    root = cache_root(cache_dir)
    union = union_model_path(name, precision, cache_dir)
    d = os.path.dirname(union)
    if not union_valid(union) or not _private_folder(d, root):
        return []
    preopt = preopt_enabled()
    opt_name = os.path.basename(preoptimized_model_path(union)) if preopt else None
    removed = []
    for f in leftover_files(os.listdir(d), precision, preopt, opt_name):
        p = os.path.join(d, f)
        if os.path.islink(p) or not os.path.isfile(p):
            continue
        if f == PRECISION_FILES[precision] and os.environ.get("PRONUNCIADA_KEEP_DOWNLOAD", "") == "1":
            continue
        try:
            os.remove(p)
            removed.append(p)
        except OSError:
            pass
    return removed


@functools.lru_cache(maxsize=1)
def _cpu_fingerprint() -> str:
    """Short hash of what identifies the CPU's instruction set: the optimizer's output can depend
    on it (on x86, e.g., the vector width picks the blocked layout), so a cache copied to another
    machine (a container image, a synced home directory) builds its own graph."""
    import hashlib
    import platform

    parts = [platform.machine(), platform.system()]           # (platform.processor() forks uname)
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith(("flags", "Features", "model name", "CPU part")):
                    parts.append(line.strip())
                    if len(parts) > 8:
                        break
    except OSError:
        pass
    if platform.system() == "Darwin":                         # no /proc: ask sysctl directly
        try:
            import ctypes

            libc = ctypes.CDLL(None)
            for key in (b"machdep.cpu.brand_string", b"machdep.cpu.features", b"machdep.cpu.leaf7_features"):
                size = ctypes.c_size_t(0)
                if libc.sysctlbyname(key, None, ctypes.byref(size), None, 0) == 0 and size.value:
                    buf = ctypes.create_string_buffer(size.value)
                    if libc.sysctlbyname(key, buf, ctypes.byref(size), None, 0) == 0:
                        parts.append(buf.value.decode("ascii", "replace"))
        except (OSError, AttributeError):
            pass
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:8]


def _preopt_tag() -> str:
    """ONNX Runtime version, machine and CPU fingerprint: an optimized graph is only reused by
    the same runtime on the same kind of CPU (the optimizer's output may use hardware-specific
    kernels)."""
    import platform

    import onnxruntime as ort

    return f"ort{ort.__version__}-{platform.machine() or 'cpu'}-{_cpu_fingerprint()}"


def preoptimized_model_path(src_variant: str) -> str:
    """`<variant>.ort<version>-<machine>.onnx`: the variant's graph after ONNX Runtime's own
    load-time optimization (level ORT_ENABLE_ALL, the default), saved by ONNX Runtime."""
    return src_variant[:-len(".onnx")] + f".{_preopt_tag()}.onnx"


def preopt_enabled() -> bool:
    """PRONUNCIADA_PREOPT=1 saves ONNX Runtime's optimized graph of the backbone and loads it on
    later runs (session creation int8 ~0.55 -> ~0.11 s, fp32 ~0.6 -> ~0.25 s). Off by default: the
    saved graph is a second model-sized file per precision (67 MB int8, 248 MB fp32)."""
    return os.environ.get("PRONUNCIADA_PREOPT", "0") == "1"


def _preoptimized_session(ort, name: str, precision: str, cache_dir, so, providers):
    """Backbone session with every extra output, loaded from the saved optimized graph with the
    optimizer off; built on first use from the all-outputs variant, which stays (it is the cache's
    backbone file). ONNX Runtime runs the same optimized graph either way: log-probs and activations
    are bit-identical to a session optimized at load."""
    union = backbone_model_file(name, precision, cache_dir)
    dest = preoptimized_model_path(union)
    if os.path.exists(dest) and os.path.getmtime(dest) >= os.path.getmtime(union):
        level = so.graph_optimization_level
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        try:
            return ort.InferenceSession(dest, so, providers=providers)
        except Exception:  # noqa: BLE001  (unreadable / other build: rebuilt below)
            so.graph_optimization_level = level
    part = f"{dest}.{os.getpid()}.part"
    so.optimized_model_filepath = part
    try:
        sess = ort.InferenceSession(union, so, providers=providers)
    finally:
        so.optimized_model_filepath = ""
    if os.path.exists(part):
        os.replace(part, dest)
    return sess


def stale_model_variants(name: str = "zipa-small-crctc-500k", cache_dir: str | None = None) -> list[str]:
    """Model-sized files in the model cache that this version no longer reads, for every precision
    whose all-outputs variant is in place (`leftover_files`): the plain download, copies with other
    output sets written by earlier versions (`<model>.emb.onnx`, `<model>.emb-<keys>.onnx`) and
    saved optimized graphs (all of them, or all but the current one with PRONUNCIADA_PREOPT=1).
    `backbone_model_file` removes them automatically for its precision; this lists them."""
    d = os.path.join(cache_root(cache_dir), "models", name)
    if not os.path.isdir(d):
        return []
    names = os.listdir(d)
    out = []
    preopt = preopt_enabled()
    for prec in PRECISION_FILES:
        union = union_model_path(name, prec, cache_dir)
        if not union_valid(union):
            continue
        opt = os.path.basename(preoptimized_model_path(union)) if preopt else None
        out += [os.path.join(d, f) for f in leftover_files(names, prec, preopt, opt)]
    return sorted(out)


def remove_stale_model_variants(name: str = "zipa-small-crctc-500k", cache_dir: str | None = None) -> list[str]:
    """Delete the files of `stale_model_variants` (same safeguards as `remove_leftovers`)."""
    out = []
    for prec in PRECISION_FILES:
        out += remove_leftovers(name, prec, cache_dir)
    return sorted(out)


def cache_root(cache_dir: str | None = None) -> str:
    return cache_dir or os.environ.get("PRONUNCIADA_CACHE") or os.path.expanduser("~/.cache/pronunciada")


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    import urllib.request          # only when downloading: ~14 ms of import time otherwise

    req = urllib.request.Request(url, headers={"User-Agent": "pronunciada"})
    with urllib.request.urlopen(req) as r, open(part, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(part, dest)


def ensure_model_file(name: str, file: str, cache_dir: str | None = None) -> str:
    """A file of the model's Hugging Face repository in the model cache, downloaded if missing. A
    backbone file whose all-outputs variant is in place is restored from it instead (no download)."""
    path = os.path.join(cache_root(cache_dir), "models", name, file)
    if not os.path.exists(path):
        prec = next((p for p, f in PRECISION_FILES.items() if f == file), None)
        if prec is not None and union_valid(union_model_path(name, prec, cache_dir)):
            try:
                return restore_download(name, prec, cache_dir)
            except (OSError, ValueError):
                pass
        _download(HF_URL.format(name=name, file=file), path)
    return path


class AcousticModel:
    frames_per_input = 2            # fbank frames per output frame
    fbank_shift = 0.01
    frame_shift = 0.02              # seconds per output frame
    time_offset = 0.03              # start time of output frame 0 (centre at 0.04 s)

    def __init__(self, name: str = "zipa-small-crctc-500k", precision: str = "fp32",
                 threads: int | None = None, providers: list | None = None,
                 cache_dir: str | None = None, embeddings=False):
        import onnxruntime as ort

        if precision not in PRECISION_FILES:
            raise ValueError(f"precision must be one of {sorted(PRECISION_FILES)}")
        providers = providers or ["CPUExecutionProvider"]
        use_cuda = providers[0] == "CUDAExecutionProvider"
        if use_cuda:
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise ValueError("CUDA requires the onnxruntime-gpu package")
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
        self.name, self.precision = name, precision
        model_path = backbone_file()
        if model_path is None:
            try:
                model_path = backbone_model_file(name, precision, cache_dir)   # the one file of the cache
            except Exception:  # noqa: BLE001  (cannot build / write it: use the download as before)
                model_path = ensure_model_file(name, PRECISION_FILES[precision], cache_dir)
        self.embeddings = False
        self.embed_outputs = embed_outputs_for(embeddings) if embeddings else {}
        tok_path = ensure_model_file(name, "tokens.txt", cache_dir)

        ort.set_default_logger_severity(3)

        def options():
            so = ort.SessionOptions()
            so.log_severity_level = 3
            # The memory-pattern planner keeps one preallocated block per input shape it has
            # seen, and every clip length is a new shape: about +20 MB after a few calls, growing
            # with use. Without it the outputs are bit-identical and a run is ~0.5 ms slower.
            so.enable_mem_pattern = False
            if threads:
                so.intra_op_num_threads = int(threads)
                so.inter_op_num_threads = 1
            apply_ort_options(so, "backbone")
            return so

        self.session = None
        # saved optimized graph: only for the default configuration (no PRONUNCIADA_BACKBONE /
        # PRONUNCIADA_ORT_OPTIONS override, CPU provider); anything going wrong falls back below
        if embeddings and preopt_enabled() and backbone_file() is None and ort_options() is None \
                and list(providers) == ["CPUExecutionProvider"] \
                and set(self.embed_outputs.values()) <= set(ALL_LAYER_OUTPUTS.values()):
            try:
                self.session = _preoptimized_session(ort, name, precision, cache_dir, options(), providers)
                self.embeddings = True
            except Exception:  # noqa: BLE001
                self.session = None
        if self.session is None:
            if embeddings:
                try:
                    model_path = embedding_model_path(name, precision, cache_dir, self.embed_outputs)
                    self.embeddings = True
                except (OSError, ValueError):
                    pass        # fall back to the plain model: no embeddings, same log-probs
            self.session = ort.InferenceSession(model_path, options(), providers=providers)
        if use_cuda:
            if "CUDAExecutionProvider" not in self.session.get_providers():
                raise ValueError("ONNX Runtime could not initialize CUDA. Check the CUDA and cuDNN libraries.")
            self.session.disable_fallback()

        self.tokens = load_tokens()
        with open(tok_path, encoding="utf-8") as f:
            model_tokens = [ln.split()[0] for ln in f if ln.strip()]
        self.model_tokens = tuple(model_tokens)
        if self.model_tokens == self.tokens:
            self._remap = None
        else:
            pos = {t: i for i, t in enumerate(model_tokens)}
            # column k of the output = model column remap[k], or -1 (-> -inf) if missing
            self._remap = np.array([pos.get(t, -1) for t in self.tokens], dtype=np.int64)
        if self.embeddings:
            outs = {o.name for o in self.session.get_outputs()}
            if not all(v in outs for v in self.embed_outputs.values()):
                self.embeddings = False
        if not self.embeddings:
            self.embed_outputs = {}
        self._out_names = ["log_probs", "log_probs_len"] + list(self.embed_outputs.values())

    # ------------------------------------------------------------------ timing
    @staticmethod
    def num_output_frames(num_fbank_frames: int) -> int:
        return max((max(int(num_fbank_frames), MIN_FBANK_FRAMES) - 7) // 2, 0)

    def frame_to_time(self, t):
        """Start time (s) of output frame t (works on scalars and arrays)."""
        if isinstance(t, (list, tuple)):
            t = np.asarray(t)
        return t * self.frame_shift + self.time_offset

    # ------------------------------------------------------------------ inference
    def _features(self, wav) -> np.ndarray:
        if not isinstance(wav, np.ndarray) or wav.dtype != np.float32 or wav.ndim != 1:
            wav = load_audio(wav)
        x = fbank(wav)
        if len(x) < MIN_FBANK_FRAMES:
            pad = np.full((MIN_FBANK_FRAMES - len(x), x.shape[1] if x.size else 80), SILENCE_FEAT,
                          dtype=np.float32)
            x = np.concatenate([x.reshape(-1, 80), pad], axis=0)
        return x

    def _fix(self, lp: np.ndarray) -> np.ndarray:
        lp = lp.astype(np.float32, copy=False)
        if self._remap is None:
            return lp
        out = lp[:, np.maximum(self._remap, 0)]
        out[:, self._remap < 0] = -np.inf
        return out

    def posteriors_from_features(self, x: np.ndarray, return_embeddings: bool = False):
        """[T', 127] log-posteriors; with return_embeddings also {layer: [T', C] float32}
        (None when the model was loaded without embeddings)."""
        res = self.session.run(self._out_names if return_embeddings else self._out_names[:2],
                               {"x": x[None].astype(np.float32),
                                "x_lens": np.array([len(x)], dtype=np.int64)})
        n = int(res[1][0])
        lp = self._fix(res[0][0, :n])
        if not return_embeddings:
            return lp
        if not self.embeddings:
            return lp, None
        emb = {}
        for k, v in zip(self.embed_outputs, res[2:]):
            v = v[:, 0] if v.shape[1] == 1 and v.shape[0] != 1 else v[0]
            emb[k] = np.ascontiguousarray(v[:n], dtype=np.float32)
        return lp, emb

    def posteriors(self, wav) -> np.ndarray:
        """[T', 127] float32 log-posteriors in packaged token order."""
        return self.posteriors_from_features(self._features(wav))

    def posteriors_batch(self, wavs, max_pad_ratio: float = 1.0) -> list[np.ndarray]:
        """Length-bucketed batch inference.

        Padding is NOT neutral for this export: padding an utterance by even one frame changes
        its log-probs by up to ~0.1 (and ~1-2 when padded 4x), argmax unchanged. So by default
        (max_pad_ratio=1.0) only utterances with identical fbank length share a batch and the
        results equal the unbatched ones (to ~1e-5). `max_pad_ratio > 1` opts in to padded
        batching (faster for many similar-length clips, slightly different numbers)."""
        feats = [self._features(w) for w in wavs]
        order = sorted(range(len(feats)), key=lambda i: len(feats[i]))
        out: list[np.ndarray | None] = [None] * len(feats)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and len(feats[order[j]]) <= max_pad_ratio * len(feats[order[i]]):
                j += 1
            group = order[i:j]
            tmax = max(len(feats[k]) for k in group)
            x = np.full((len(group), tmax, 80), SILENCE_FEAT, dtype=np.float32)
            lens = np.zeros(len(group), dtype=np.int64)
            for b, k in enumerate(group):
                x[b, : len(feats[k])] = feats[k]
                lens[b] = len(feats[k])
            lp, n = self.session.run(self._out_names[:2], {"x": x, "x_lens": lens})
            for b, k in enumerate(group):
                out[k] = self._fix(lp[b, : int(n[b])])
            i = j
        return out  # type: ignore[return-value]

    # ------------------------------------------------------------------ decoding
    def greedy(self, lp: np.ndarray) -> list[tuple[str, int, int]]:
        return greedy_tokens(lp, self.tokens)


def greedy_tokens(lp: np.ndarray, tokens=None) -> list[tuple[str, int, int]]:
    """Collapsed greedy decode -> [(token, start_frame, end_frame_exclusive)]; ids 0-3 are
    non-speech."""
    tokens = tokens or load_tokens()
    if len(lp) == 0:
        return []
    ids = np.asarray(lp).argmax(-1)
    change = np.flatnonzero(np.diff(ids)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(ids)]])
    return [(tokens[ids[s]], int(s), int(e)) for s, e in zip(starts, ends) if ids[s] >= NON_SPEECH]
