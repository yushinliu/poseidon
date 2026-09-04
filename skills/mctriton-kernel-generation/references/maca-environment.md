# MetaX / MACA 运行环境事实

本文件描述 Poseidon 构建/测试机的真实环境。kernel 代码**无需**处理这些环境变量（平台已注入），
但生成代码必须与下述软件栈版本兼容。

## 硬件

- GPU：MetaX C500（1 卡），驱动 Kernel Mode Driver 3.9.0
- 主机：x86_64 Linux（Ubuntu 22.04，内核 5.15）
- 管理工具：`mx-smi`（类似 nvidia-smi）

## SDK（MACA）

- `MACA_PATH=/opt/maca`（当前指向 `/opt/maca-20260318`，Version: `20260318.1063`）
- 另有独立版本目录 `/opt/maca-3.7.0`、`/opt/maca/maca-3.7.0`
- SDK 目录结构：`bin/ lib/ lib64/ include/ mxgpu_llvm/ ompi/ samples/ share/`

## 编译/运行必需环境变量（平台注入）

```sh
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:/opt/maca/ompi/lib:$LD_LIBRARY_PATH
export TRITON_CACHE_DIR=<job>/cache
export TRITON_METAX_ENABLE_TORCH_REDUCTION_ORDER=1
```

说明：
- 不设置 `LD_LIBRARY_PATH` 时，`import torch` 会报 `libmxomp.so: cannot open shared object file`。
- `TRITON_METAX_ENABLE_TORCH_REDUCTION_ORDER=1` 使 Triton 规约顺序与 torch 对齐，减小数值差异。

## Python 软件栈

| whl 版本目录 | torch | triton | Python |
|---|---|---|---|
| `3.7.1.3-dsv4` | 2.10.0+metax3.7.1.3.dsv4 | 3.6.0（MetaX fork） | 3.12 |
| `3.7.2.0` | 2.8.0+metax3.7.2.0 | 3.0.0+metax3.7.2.0 | 3.12 |

- 设备名：MetaX torch 将设备注册为 `cuda`（`triton.runtime.driver.active.get_active_torch_device()` → `cuda:0`）。
- 显存同步：`torch.cuda.synchronize()` 有效。
- 构建机自带 conda（miniforge3）环境 `yuliu`（Python 3.12.13，torch 2.10 + triton 3.6）。

## 约束（生成代码务必遵守）

1. `@triton.jit` 函数必须定义在真实 Python 文件中（stdin / REPL 会报 `could not get source code`）。平台总是把生成代码写入 `.py` 文件后执行。
2. 单个 kernel 编译可能耗时数秒到几十秒，作业整体有超时限制，避免无意义的超长 autotune 空间。
3. 每次作业在独立目录运行（独立的 `TRITON_CACHE_DIR`），不要依赖跨作业的缓存。
