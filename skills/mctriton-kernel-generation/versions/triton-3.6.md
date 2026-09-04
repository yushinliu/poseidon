# triton 3.6（MetaX 移植版）版本差异

适用 whl 版本：`3.7.1.3-dsv4`（triton 3.6.0 + torch 2.10.0）。以下为与基础 SKILL.md 的差异与已验证事实。

## 可用且推荐

- `triton.runtime.driver.active.get_active_torch_device()`：本版本可用（3.0 无此接口），返回 `cuda:0`。
- **`tl.dot(a, b, acc, input_precision="ieee")`**：本平台实测通过。
  默认情况下 MetaX 的 `tl.dot` 可能使用降精度模式（类似 TF32），与 torch 的 fp32 矩阵乘产生约 1e-1 量级误差；
  指定 `input_precision="ieee"` 后误差降到 1e-4 以下，与 torch 对齐。**矩阵乘类 kernel 默认带上该参数。**
- MetaX `triton.Config` 扩展：`pipeline`（`"basic"`/`"cpasync"`/空串）与 `scenario`（`"flashattn-fwd"`/`"flashattn-bwd"`/`"mla"`/`"unroll"`/`"roll"`/`"unprefetch"`/`"fullstage"`/`"storeCoalesce"`，可 `";"` 组合）实测可用，**写在 kwargs 字典内**（`triton.Config({..., "pipeline": "cpasync"}, num_warps=..., num_stages=...)`），可用于矩阵乘/flashattn 类 kernel 并配合 autotune 搜索最优组合。
- `num_warps`（1/2/4/8）、`num_stages`（1~5）常规可用。
- autotune：`TRITON_PRINT_AUTOTUNING=1` 会打印调优进度（本机构建确认）；**本机安装的 3.6 构建不支持 autotune 结果持久化**（无 `TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS`），勿依赖跨进程复用调优结果。

## 不可用 / 避免

- `tl.load` / `tl.store` 的 `boundary_check` 参数已移除（上游 3.4 起）：一律用 `mask=...` + `other=...`。
- `tl.math` 模块已移除：数学函数直接用 `tl.exp` / `tl.rsqrt` 等顶层命名空间。
- 上游 3.3+ 新增算子（`tl.split` / `tl.join` / `tl.gather` / `tl.interleave` / `tl.make_tensor_descriptor` 等）
  在 MetaX 3.6 上的兼容性**未验证**，不要使用。
- fp8 类型（`tl.float8e4nv` 等）不要使用。

## 配套 torch 2.10

- torch 2.10 的算子作为参考实现无限制；但生成代码中仍只允许"单 kernel"（见 SKILL.md 协议）。
