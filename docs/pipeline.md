# 流水线协议说明

## 1. LLM 输出协议

模型以 `skills/mctriton-kernel-generation/` 为 system prompt，被要求**只输出一个 JSON 对象**：

```json
{
  "analysis": "一句话方案说明（中文）",
  "kernel_code": "kernel 定义 + run_kernel(*args, **kwargs) 启动函数（无 import、无测试代码）",
  "inputs_code": "def make_inputs(device) -> (args, kwargs)",
  "rtol": 0.02,
  "atol": 0.02
}
```

- 通过 DeepSeek 的 `response_format: {type: "json_object"}` 请求 JSON 输出；
  解析失败时会依次尝试 markdown 围栏与首个平衡 `{...}` 块（`server/src/llm/extractor.js`）。
- `kernel_code` / `inputs_code` 中若出现 `if __name__ == "__main__"` 块会被防御性截断。

## 2. 测试 harness

模型输出被注入 [`server/templates/harness.py.tpl`](../server/templates/harness.py.tpl)：

```
[固定头部: torch/triton 导入 + 环境变量读取]
  → 用户 torch 参考实现（torch_fn）
  → 生成的 kernel 代码（run_kernel）
  → 生成的输入构造（make_inputs）
[固定尾部: 精度校验 + 性能测试 + 结果输出]
```

运行方式（构建机上）：

```sh
export MACA_PATH=<sdk>
export LD_LIBRARY_PATH=<sdk>/lib:<sdk>/mxgpu_llvm/lib:<sdk>/ompi/lib:$LD_LIBRARY_PATH
export TRITON_CACHE_DIR=<job>/cache
export TRITON_METAX_ENABLE_TORCH_REDUCTION_ORDER=1
export TRITON_PRINT_AUTOTUNING=1                          # autotune 调优过程打印进度（保持看门狗活性）
export TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS=1        # autotune 结果持久化（3.0 构建支持；3.6 忽略）
export TRITON_AUTOTUNE_CONFIG_PATH=<job>/autotune_configs
python -u harness.py
```

### autotune 调参

- skill 要求：存在可调参数（BLOCK 尺寸、num_warps、num_stages、pipeline、scenario）的 kernel **必须**使用 `@triton.autotune`
  （参考沐曦用户指南"功能支持"章节的 MetaX `triton.Config` 扩展）。
- 调优发生在 `run_kernel` 首次调用（即精度校验阶段），日志中可见调优进度；
  之后基准测试直接使用最佳配置，计时不受调优影响。
- 每次尝试前清空的 `cache` 是编译缓存；autotune 配置持久化路径（`autotune_configs`）不清空，重试可复用调优结果。

### 精度校验

对 `torch_fn` 与 `run_kernel` 的每个输出（float32 下比较）：

- `max_abs_err` / `max_rel_err` / `mean_abs_err`
- `allclose(rtol, atol, equal_nan=True)`
- `cosine_similarity`（展平，仅浮点输出）

### 性能测试

- warmup 后跑 `iters` 轮，`torch.cuda.Event` 计时，取**中位数**；
- 输出 `torch_ms`、`triton_ms`、`speedup = torch_ms / triton_ms`。

## 3. 结果 schema（stdout 标记行）

harness 打印一行 `###POSEIDON_RESULT###<json>`，后端解析为：

```json
{
  "ok": true,
  "accuracy": {
    "passed": true, "rtol": 0.02, "atol": 0.02,
    "outputs": [{"shape": [..], "dtype": "float32", "max_abs_err": 1e-6,
                 "max_rel_err": 1e-6, "mean_abs_err": 1e-7,
                 "allclose": true, "cosine_similarity": 0.99999}]
  },
  "performance": {"torch_ms": 1.234, "triton_ms": 0.567, "speedup": 2.18,
                  "warmup": 5, "iters": 50}
}
```

失败时：

```json
{"ok": false, "phase": "reference|kernel|accuracy|benchmark", "error": "...", "traceback": "..."}
```

## 4. 自动修复循环

```
for attempt in 0 .. max_retries:
    调用 LLM（对话上下文保留：system + 任务 + 历次生成与失败反馈）
    → 校验 JSON 协议 → 组装 harness → 上传构建机 → 运行（流式日志回传）
    → ok          ⇒ 作业成功，展示精度 + 性能
    → 失败(phase) ⇒ 将 phase + error + traceback + stderr/stdout 尾部 反馈给 LLM，重试
超出轮次 ⇒ 作业失败（页面展示最后一次生成的代码与错误）
```

- 阶段 `reference` 失败说明**用户输入的 torch 代码本身无法运行**（或模型构造的输入不合法），
  页面会提示检查输入代码，同时仍允许模型修复 `inputs_code`。
- 单次运行超时由 `build_machine.run_timeout_ms`（默认 15 分钟）控制，超时同样触发修复循环。

## 5. 任务生命周期与持久化

```
queued → running → success | failed
```

- 事件流（LLM 回复、构建机 stdout/stderr、阶段切换）实时写入任务对象，前端 1.5s 轮询增量渲染；
- 任务结束后持久化到 `jobs/<id>/`（job.json、generated_kernel.py、inputs.py、reference_torch.py、run.log）；
- 构建机侧作业目录：`~/ws/poseidon/runs/<id>/`（harness.py + 独立 TRITON 缓存）。
