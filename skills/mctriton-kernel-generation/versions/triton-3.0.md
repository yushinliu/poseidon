# triton 3.0（MetaX 移植版）版本差异

适用 whl 版本：`3.7.2.0`（triton 3.0.0 + torch 2.8.0）。以下为与基础 SKILL.md 的差异与注意事项。

## 与 3.6 的关键差异

- **运行时 driver API 不同（本平台实测）**：`triton.runtime.driver.active.get_active_torch_device()` **不存在**
  （`driver.active` 是 LazyProxy，无该接口）。获取设备用 `torch.device("cuda", torch.cuda.current_device())`。
  生成的 `inputs_code` 中不要调用 `get_active_torch_device`，设备由平台以 `device` 参数传入，直接使用即可。
- **`tl.dot` 不支持 `input_precision` 参数**（上游 3.2+ 才引入）：
  - 不要写 `input_precision="ieee"`，否则直接编译报错；
  - 需要 fp32 精度对齐时：确保 `a`、`b` 在进入 `tl.dot` 前已 `to(tl.float32)`，累加器为 fp32；
    若仍与 torch 有 1e-1 量级差异，说明后端矩阵乘走降精度路径——优先在修复循环中调大 rtol/atol 并在 analysis 中说明，
    或改用手写外积累加（性能差但精度可控）。
- `boundary_check` 参数仍可用，但**建议统一用 mask 写法**，保持与 3.6 的代码风格一致、跨版本可迁移。
- `tl.math` 模块仍存在（如 `tl.math.exp`），但**建议直接用 `tl.exp`** 等顶层函数（两版通用）。

## 未验证项（保守处理）

- MetaX `triton.Config` 扩展（`pipeline` / `scenario` kwargs）在本版本的 MetaX 构建上**未验证**：默认不要使用，
  优先用 `num_warps` / `num_stages` 的常规配置；平台反馈错误后再调整。

## 配套 torch 2.8

- 参考实现中不要依赖 torch 2.9/2.10 新增的算子或参数（如 `torch.nn.functional.scaled_dot_product_attention` 的新增参数行为差异）。
