# SDK 小版本 3.7.2 注意事项

对应 whl 版本目录：`3.7.2.0`。

## 软件栈

- torch：`2.8.0+metax3.7.2.0`（cp312）
- triton：`3.0.0+metax3.7.2.0`（MetaX 移植版，见 `triton-3.0.md`）
- Python 3.12

## 说明

- 该组合使用 triton 3.0 大版本，**务必同时遵守 `triton-3.0.md` 中的 API 限制**（无 `input_precision`、保守 Config 等）。
- torch 为 2.8：参考实现避免依赖 2.9+ 新特性。
- 生成代码风格建议与 3.6 保持一致（mask 写法、`tl.*` 顶层函数），便于跨版本维护。
