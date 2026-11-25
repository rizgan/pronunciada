"""Build backbone candidates outside the repository (~/.cache/pronunciada/opt/candidates/<name>/).

Runs in the tools virtualenv (onnx, onnxslim, onnxruntime; it does not import pronunciada):
    ~/.cache/pronunciada/opt/venv_tools/bin/python bench/opt/build_candidates.py <name> [...]
    ~/.cache/pronunciada/opt/venv_tools/bin/python bench/opt/build_candidates.py list

Every candidate keeps the intermediate tensors the Scorer reads as extra outputs
(pronunciada.acoustic.ALL_LAYER_OUTPUTS): they are added as graph outputs before the graph is
rewritten, so no optimizer can fuse them away or rename them, and removed again afterwards; the
Scorer appends them at load time exactly as it does for the downloaded model. Each build writes
<dir>/model.onnx and <dir>/build.json (source file, sha1, tool versions, settings, sizes).

Candidates:
  slim-fp32 / slim-int8        onnxslim (default passes) of model.onnx / model.int8.onnx
  ortopt-basic-fp32/-int8      ONNX Runtime offline optimization, ORT_ENABLE_BASIC, saved model
  ortopt-ext-fp32/-int8        the same with ORT_ENABLE_EXTENDED
  prep-fp32                    onnxruntime.quantization quant_pre_process (shape inference, ONNX
                               and ORT basic optimisation: constant folding, fusion), no quantization
  f16store-fp32                fp32 graph with every large float initializer stored as float16 and a
                               Cast back to float32 in front of its users (CPU runs fp32; ~half size)
  int8-keep-<ops>              dynamic int8 (QUInt8 weights, as the shipped file?) with the listed op
                               types left in fp32 (e.g. int8-keep-Conv)
  nbits4-fp32                  4-bit MatMulNBits weights (block 32) from the fp32 model
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

CACHE = Path(os.environ.get("PRONUNCIADA_CACHE") or Path.home() / ".cache" / "pronunciada")
MODELS = CACHE / "models" / "zipa-small-crctc-500k"
OUT = CACHE / "opt" / "candidates"
KEEP = ["/encoder/0/layers.1/bypass/Add_output_0", "/encoder/1/out_combiner/Add_output_0",
        "/encoder/2/out_combiner/Add_output_0", "/encoder/3/out_combiner/Add_output_0",
        "/encoder/4/out_combiner/Add_output_0", "/encoder/5/out_combiner/Add_output_0", "/Transpose_1_output_0"]


def sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def _src(prec: str) -> Path:
    from common import default_backbone

    return default_backbone(prec)


def with_outputs(m):
    import onnx

    have = {o.name for o in m.graph.output}
    for n in KEEP:
        if n not in have:
            m.graph.output.append(onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, None))
    return m


def strip_outputs(m):
    keep = [o for o in m.graph.output if o.name not in KEEP]
    del m.graph.output[:]
    m.graph.output.extend(keep)
    return m


def check_names(m) -> list[str]:
    produced = {o for n in m.graph.node for o in n.output}
    return [n for n in KEEP if n not in produced]


def save(m, d: Path) -> Path:
    import onnx

    d.mkdir(parents=True, exist_ok=True)
    p = d / "model.onnx"
    onnx.save(m, str(p))
    return p


def build(name: str) -> dict:
    import onnx
    import onnxruntime as ort

    d = OUT / name
    prec = "int8" if name.endswith("int8") or name.startswith("int8") else "fp32"
    src = _src(prec)
    t0 = time.time()
    info = {"name": name, "source": src.name, "source_sha1": sha1(src), "source_bytes": src.stat().st_size,
            "onnx": onnx.__version__, "onnxruntime": ort.__version__, "date": time.strftime("%Y-%m-%d %H:%M")}
    if name.startswith("slim-"):
        import onnxslim

        info["onnxslim"] = onnxslim.__version__
        m = with_outputs(onnx.load(str(src)))
        m = onnxslim.slim(m)
        m = strip_outputs(m)
        info["settings"] = "onnxslim.slim(model) default passes"
        p = save(m, d)
    elif name.startswith("ortopt-"):
        level = {"basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                 "ext": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED}[name.split("-")[1]]
        tmp = d / "with_outputs.onnx"
        d.mkdir(parents=True, exist_ok=True)
        onnx.save(with_outputs(onnx.load(str(src))), str(tmp))
        so = ort.SessionOptions()
        so.graph_optimization_level = level
        so.optimized_model_filepath = str(d / "opt_with_outputs.onnx")
        ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"])
        m = strip_outputs(onnx.load(str(d / "opt_with_outputs.onnx")))
        tmp.unlink()
        (d / "opt_with_outputs.onnx").unlink()
        info["settings"] = f"InferenceSession(optimized_model_filepath, {level})"
        p = save(m, d)
    elif name == "prep-fp32":
        from onnxruntime.quantization.shape_inference import quant_pre_process

        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "with_outputs.onnx"
        onnx.save(with_outputs(onnx.load(str(src))), str(tmp))
        quant_pre_process(str(tmp), str(d / "prep.onnx"), skip_symbolic_shape=True)
        m = strip_outputs(onnx.load(str(d / "prep.onnx")))
        tmp.unlink()
        (d / "prep.onnx").unlink()
        info["settings"] = "quant_pre_process(skip_symbolic_shape=True): onnx shape inference + ORT basic optimisation"
        p = save(m, d)
    elif name == "f16store-fp32":
        import numpy as np
        from onnx import numpy_helper

        m = onnx.load(str(src))
        g = m.graph
        new_init, casts, n_conv = [], [], 0
        for t in g.initializer:
            a = numpy_helper.to_array(t)
            if a.dtype == np.float32 and a.size >= 256:
                h = numpy_helper.from_array(a.astype(np.float16), t.name + "__f16")
                new_init.append(h)
                casts.append(onnx.helper.make_node("Cast", [h.name], [t.name], to=onnx.TensorProto.FLOAT,
                                                   name=t.name + "__cast"))
                n_conv += 1
            else:
                new_init.append(t)
        del g.initializer[:]
        g.initializer.extend(new_init)
        nodes = list(g.node)
        del g.node[:]
        g.node.extend(casts + nodes)
        info["settings"] = f"{n_conv} float32 initializers (>= 256 values) stored as float16 + Cast to float32"
        p = save(m, d)
    elif name.startswith("int8-keep-"):
        from onnxruntime.quantization import QuantType, quantize_dynamic

        keep_ops = name[len("int8-keep-"):].split("+")
        src = _src("fp32")
        info["source"], info["source_sha1"] = src.name, sha1(src)
        m0 = onnx.load(str(src))
        ops = sorted({n.op_type for n in m0.graph.node if n.op_type in ("MatMul", "Gemm", "Conv")})
        quant_ops = [o for o in ops if o not in keep_ops]
        d.mkdir(parents=True, exist_ok=True)
        quantize_dynamic(str(src), str(d / "model.onnx"), op_types_to_quantize=quant_ops,
                         weight_type=QuantType.QUInt8)
        info["settings"] = f"quantize_dynamic(op_types_to_quantize={quant_ops}, weight_type=QUInt8); fp32 kept: {keep_ops}"
        p = d / "model.onnx"
    elif name.startswith("int8x-"):
        # the shipped int8 file is quantize_dynamic on MatMul with signed int8 weights (290 of 370
        # MatMuls carry weights); here the MatMuls whose node name starts with one of the given
        # prefixes stay fp32 ("embed" = /encoder_embed/, "enc0" = /encoder/0/, "out" = the CTC head)
        from onnxruntime.quantization import QuantType, quantize_dynamic

        pref = {"embed": "/encoder_embed/", "enc0": "/encoder/0/", "enc1": "/encoder/1/", "out": "/ctc_output/", "none": "\0"}
        keys = name[len("int8x-"):].split("+")
        src = _src("fp32")
        info["source"], info["source_sha1"] = src.name, sha1(src)
        m0 = onnx.load(str(src))
        excl = [n.name for n in m0.graph.node if n.op_type == "MatMul"
                and any(n.name.startswith(pref[k]) or (k == "out" and "ctc" in n.name.lower()) for k in keys)]
        d.mkdir(parents=True, exist_ok=True)
        quantize_dynamic(str(src), str(d / "model.onnx"), op_types_to_quantize=["MatMul"],
                         weight_type=QuantType.QInt8, nodes_to_exclude=excl)
        info["settings"] = f"quantize_dynamic(MatMul, QInt8), {len(excl)} MatMul nodes kept fp32 ({keys})"
        info["excluded_nodes"] = excl
        p = d / "model.onnx"
    elif name == "nbits4-fp32":
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

        m = onnx.load(str(src))
        q = MatMulNBitsQuantizer(m, block_size=32, is_symmetric=True)
        q.process()
        info["settings"] = "MatMulNBitsQuantizer(block_size=32, is_symmetric=True), default accuracy level"
        p = save(q.model.model, d)
    else:
        raise SystemExit(f"unknown candidate {name}")
    mm = onnx.load(str(p), load_external_data=False)
    info["missing_kept_tensors"] = check_names(mm)
    info["bytes"] = p.stat().st_size
    info["sha1"] = sha1(p)
    info["ops"] = sorted({n.op_type for n in mm.graph.node})
    info["build_s"] = round(time.time() - t0, 1)
    (d / "build.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("name", "bytes", "source_bytes", "missing_kept_tensors", "build_s")}))
    return info


if __name__ == "__main__":
    for n in sys.argv[1:]:
        build(n)
