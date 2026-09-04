#!/usr/bin/env python3
"""
独立验证 mcTriton kernel 的脚本（不依赖 Poseidon 服务）。

用法一（三个文件）:
    python tools/run_kernel.py --reference examples/vector_add/input.py \
                               --kernel /path/to/kernel_code.py \
                               --inputs /path/to/inputs_code.py \
                               --rtol 0.02 --atol 0.02 --iters 50

用法二（LLM 按 skill 协议输出的 JSON 文件）:
    python tools/run_kernel.py --json /path/to/generated.json

运行前请确保已安装 MetaX torch/triton，并设置:
    export MACA_PATH=/opt/maca
    export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:/opt/maca/ompi/lib:$LD_LIBRARY_PATH
    export TRITON_METAX_ENABLE_TORCH_REDUCTION_ORDER=1
"""
import argparse
import json
import sys
import traceback

import torch
import triton  # noqa: F401  (供 exec 出的 kernel 代码使用)
import triton.language as tl  # noqa: F401

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _load_code(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _exec_into(name, code, g):
    code = code.split('if __name__ == "__main__":')[0]  # 防御性截断
    exec(compile(code, f"<{name}>", "exec"), g)


def _as_list(x):
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, list):
        return x
    return [x]


def _metrics(ref, out, rtol, atol):
    ref_f = ref.detach().float()
    out_f = out.detach().float()
    if list(ref_f.shape) != list(out_f.shape):
        return {"shape_ref": list(ref.shape), "shape_out": list(out.shape),
                "error": "shape 不一致", "allclose": False}
    flat_r = ref_f.reshape(-1)
    flat_o = out_f.reshape(-1)
    abs_err = (flat_r - flat_o).abs()
    denom = flat_r.abs() + 1e-6
    cosine = None
    if flat_r.numel() >= 2 and ref_f.is_floating_point():
        cosine = float(torch.nn.functional.cosine_similarity(flat_r, flat_o, dim=0))
    return {
        "shape": list(ref.shape),
        "dtype": str(ref.dtype).replace("torch.", ""),
        "max_abs_err": float(abs_err.max()),
        "max_rel_err": float((abs_err / denom).max()),
        "mean_abs_err": float(abs_err.mean()),
        "allclose": bool(torch.allclose(ref_f, out_f, rtol=rtol, atol=atol, equal_nan=True)),
        "cosine_similarity": cosine,
    }


def _bench(fn, args, kwargs, warmup, iters):
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn(*args, **kwargs)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return float(times[len(times) // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", help="torch 参考实现文件（定义 torch_fn）")
    ap.add_argument("--kernel", help="kernel 代码文件（定义 run_kernel）")
    ap.add_argument("--inputs", help="输入构造文件（定义 make_inputs）")
    ap.add_argument("--json", help="LLM 按协议输出的 JSON 文件（含 kernel_code/inputs_code/rtol/atol）")
    ap.add_argument("--rtol", type=float, default=0.02)
    ap.add_argument("--atol", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    g = {"torch": torch, "triton": triton, "tl": tl}
    if args.json:
        with open(args.json, "r", encoding="utf-8") as f:
            gen = json.load(f)
        _exec_into("kernel", gen["kernel_code"], g)
        _exec_into("inputs", gen["inputs_code"], g)
        rtol, atol = float(gen.get("rtol", args.rtol)), float(gen.get("atol", args.atol))
    else:
        if not (args.reference and args.kernel and args.inputs):
            ap.error("需要 --reference/--kernel/--inputs 或 --json")
        _exec_into("kernel", _load_code(args.kernel), g)
        _exec_into("inputs", _load_code(args.inputs), g)
        rtol, atol = args.rtol, args.atol
    _exec_into("reference", _load_code(args.reference) if args.reference else "def torch_fn(*a, **k): raise NotImplementedError", g)

    print(f"device: {DEVICE}  torch {torch.__version__}  triton {triton.__version__}")
    args_in, kwargs_in = g["make_inputs"](DEVICE)

    ref = g["torch_fn"](*args_in, **kwargs_in)
    torch.cuda.synchronize()
    out = g["run_kernel"](*args_in, **kwargs_in)
    torch.cuda.synchronize()

    print(f"输入: {[(a.shape, a.dtype) for a in args_in if isinstance(a, torch.Tensor)]}")
    refs, outs = _as_list(ref), _as_list(out)
    for i, (r, o) in enumerate(zip(refs, outs)):
        m = _metrics(r, o, rtol, atol)
        print(f"输出#{i} shape={m['shape']} {m['dtype']} "
              f"max_abs={m['max_abs_err']:.3e} max_rel={m['max_rel_err']:.3e} "
              f"mean_abs={m['mean_abs_err']:.3e} cosine={m['cosine_similarity']} allclose={m['allclose']}")
        if not m["allclose"]:
            print("精度校验未通过", file=sys.stderr)
            sys.exit(1)

    torch_ms = _bench(g["torch_fn"], args_in, kwargs_in, args.warmup, args.iters)
    triton_ms = _bench(g["run_kernel"], args_in, kwargs_in, args.warmup, args.iters)
    print(f"torch: {torch_ms:.4f} ms   triton: {triton_ms:.4f} ms   加速比: {torch_ms / triton_ms:.2f}×")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
