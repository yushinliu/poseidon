# Poseidon 精度校验与性能测试协议

模型生成的 `kernel_code` / `inputs_code` 会被注入以下测试框架（平台生成，无需模型输出）。
生成的代码只需遵守本文件描述的**调用约定**。

## 调用约定

1. 用户参考实现：`def torch_fn(*args, **kwargs)` 返回 Tensor 或 Tensor 元组。
2. `def make_inputs(device)` 返回 `(args, kwargs)`；同一组输入同时喂给 `torch_fn` 与 `run_kernel`。
3. `def run_kernel(*args, **kwargs)` 返回 Tensor 或 Tensor 元组，其结构 / shape / dtype 必须与 `torch_fn` 返回值一致。

## 测试流程（平台自动执行）

1. **正确性（先于性能）**
   - `ref = torch_fn(*args, **kwargs)`，`out = run_kernel(*args, **kwargs)`，各同步一次。
   - 逐输出计算（在 float32 下比较）：
     - `max_abs_err`（最大绝对误差）
     - `max_rel_err`（最大相对误差，分母加 1e-6 防除零）
     - `mean_abs_err`（平均绝对误差）
     - `allclose(rtol, atol)` 判定（由模型在 JSON 中给出的 rtol/atol）
     - `cosine_similarity`（展平后余弦相似度，非浮点输出跳过）
   - 任一输出 `allclose` 为 False ⇒ 精度失败，平台将差异数值反馈给模型要求修复。
2. **性能**
   - 对 `torch_fn` 与 `run_kernel` 分别计时：warmup 后跑 N 轮，用 `torch.cuda.Event` 记录每轮耗时，取**中位数**（单位 ms）。
   - 输出 `torch_ms`、`triton_ms`、`speedup = torch_ms / triton_ms`。
3. 失败反馈：若编译/运行抛异常，平台把阶段名 + stderr + traceback 反馈给模型，模型按修复协议重新输出完整 JSON。

## 模型侧注意事项

- `run_kernel` 内部自行计算 grid（`triton.cdiv` 等）；不要依赖全局变量跨调用。
- 输出 Tensor 用 `torch.empty_like(...)` 分配后，**每个元素都必须被写**（带 mask 写满），否则出现未初始化数据导致精度误判。
- 输入 Tensor 可能带任意 stride；若需要连续内存，用 `x = x.contiguous()`（在 `run_kernel` 内处理）。
- 保持确定性：不要用随机数填充输出。
