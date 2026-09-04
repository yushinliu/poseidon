import json
import os
import sys
import traceback

import torch
import triton
import triton.language as tl


def _get_active_torch_device():
    # triton 3.6（MetaX 移植版）提供该接口；3.0 无此接口（driver.active 为 LazyProxy），退回 torch 查询
    try:
        return triton.runtime.driver.active.get_active_torch_device()
    except AttributeError:
        return torch.device("cuda", torch.cuda.current_device())


DEVICE = _get_active_torch_device()

RTOL = float(os.environ.get("POSEIDON_RTOL", "0.02"))
ATOL = float(os.environ.get("POSEIDON_ATOL", "0.02"))
WARMUP = int(os.environ.get("POSEIDON_WARMUP", "5"))
ITERS = int(os.environ.get("POSEIDON_ITERS", "50"))

# ===POSEIDON_REF_BEGIN===
# ===POSEIDON_REF_END===

# ===POSEIDON_KERNEL_BEGIN===
# ===POSEIDON_KERNEL_END===

# ===POSEIDON_INPUTS_BEGIN===
# ===POSEIDON_INPUTS_END===


def _emit(result):
    print("###POSEIDON_RESULT###" + json.dumps(result, ensure_ascii=False), flush=True)


def _info(msg):
    print(f"[harness] {msg}", flush=True)


def _fail(phase, exc):
    _emit({
        "ok": False,
        "phase": phase,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    })
    sys.exit(1)


def _as_list(x):
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, list):
        return x
    return [x]


def _metrics(ref, out):
    ref_f = ref.detach().float()
    out_f = out.detach().float()
    if list(ref_f.shape) != list(out_f.shape):
        return {
            "shape_ref": list(ref.shape),
            "shape_out": list(out.shape),
            "dtype": str(ref.dtype).replace("torch.", ""),
            "error": "shape 不一致",
            "allclose": False,
        }
    flat_r = ref_f.reshape(-1)
    flat_o = out_f.reshape(-1)
    abs_err = (flat_r - flat_o).abs()
    max_abs = float(abs_err.max())
    denom = flat_r.abs() + 1e-6
    max_rel = float((abs_err / denom).max())
    mean_abs = float(abs_err.mean())
    ok = bool(torch.allclose(ref_f, out_f, rtol=RTOL, atol=ATOL, equal_nan=True))
    cosine = None
    if flat_r.numel() >= 2 and ref_f.is_floating_point():
        cosine = float(torch.nn.functional.cosine_similarity(flat_r, flat_o, dim=0))
    return {
        "shape": list(ref.shape),
        "dtype": str(ref.dtype).replace("torch.", ""),
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "mean_abs_err": mean_abs,
        "allclose": ok,
        "cosine_similarity": cosine,
    }


def _bench(fn, args, kwargs, label):
    _info(f"{label} 基准测试中（warmup={WARMUP}，iters={ITERS}）...")
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        starts[i].record()
        fn(*args, **kwargs)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    _info(f"{label} 基准测试完成（中位数 {times[len(times) // 2]:.4f} ms）")
    return float(times[len(times) // 2])


def main():
    _info(f"环境就绪：{DEVICE} | torch {torch.__version__} | triton {triton.__version__}")

    try:
        if "torch_fn" not in globals():
            raise RuntimeError("用户代码中未定义 torch_fn(*args, **kwargs)")
        if "run_kernel" not in globals():
            raise RuntimeError("生成代码中未定义 run_kernel(*args, **kwargs)")
        if "make_inputs" not in globals():
            raise RuntimeError("生成代码中未定义 make_inputs(device)")

        _info("构造测试输入 ...")
        args, kwargs = make_inputs(DEVICE)
        _info("测试输入构造完成")

        _info("运行 torch 参考实现 ...")
        ref = torch_fn(*args, **kwargs)
        torch.cuda.synchronize()
        _info("torch 参考实现运行完成")
    except Exception as e:
        _fail("reference", e)

    try:
        _info("首次运行 mcTriton kernel（含 JIT 编译，可能耗时数十秒）...")
        out = run_kernel(*args, **kwargs)
        torch.cuda.synchronize()
        _info("kernel 运行完成")
    except Exception as e:
        _fail("kernel", e)

    # ---- accuracy ----
    try:
        _info("精度校验中 ...")
        refs = _as_list(ref)
        outs = _as_list(out)
        if len(refs) != len(outs):
            raise RuntimeError(f"输出数量不一致: torch_fn 返回 {len(refs)} 个，run_kernel 返回 {len(outs)} 个")
        acc_items = [_metrics(r, o) for r, o in zip(refs, outs)]
    except Exception as e:
        _fail("accuracy", e)

    passed = all(m.get("allclose", False) for m in acc_items)
    result = {
        "ok": passed,
        "accuracy": {
            "passed": passed,
            "rtol": RTOL,
            "atol": ATOL,
            "outputs": acc_items,
        },
    }
    if not passed:
        result["phase"] = "accuracy"
        result["error"] = "精度校验未通过（allclose=False），请对比 outputs 中各指标修复数值差异"
        _emit(result)
        sys.exit(1)
    _info("精度校验通过")

    # ---- performance ----
    try:
        torch_ms = _bench(torch_fn, args, kwargs, "torch")
        triton_ms = _bench(run_kernel, args, kwargs, "triton")
        result["performance"] = {
            "torch_ms": torch_ms,
            "triton_ms": triton_ms,
            "speedup": (torch_ms / triton_ms) if triton_ms > 0 else None,
            "warmup": WARMUP,
            "iters": ITERS,
        }
    except Exception as e:
        _fail("benchmark", e)

    _emit(result)


if __name__ == "__main__":
    main()
