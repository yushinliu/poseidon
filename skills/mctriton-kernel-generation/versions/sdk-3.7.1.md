# SDK 小版本 3.7.1 注意事项

对应 whl 版本目录：`3.7.1.3-dsv4`。

## 软件栈

- torch：`2.10.0+metax3.7.1.3.dsv4`（cp312）
- triton：`3.6.0+metax3.7.1.3.dsv4`（MetaX 移植版，见 `triton-3.6.md`）
- Python 3.12

## 说明

- `dsv4` 后缀表示这是面向 DeepSeek v4 场景的定制构建；算子行为以该构建实测为准。
- 平台已注入 `TRITON_METAX_ENABLE_TORCH_REDUCTION_ORDER=1`，规约顺序与 torch 对齐，无需在 kernel 中特殊处理。
- 该组合是平台**默认推荐**环境，本仓库 skills 中的"实测通过"结论默认在此组合上得出。
