# 示例：torch 参考实现

本目录提供几个可直接粘贴到 Poseidon Web UI 的 torch 参考实现示例，
也可配合 `tools/run_kernel.py` 独立使用。

| 示例 | 文件 | 说明 |
|---|---|---|
| 向量加法 | `vector_add/input.py` | 最简单的逐元素算子，首次试用推荐 |
| Softmax | `softmax/input.py` | 规约类算子（max/sum/exp） |
| 矩阵乘 | `matmul/input.py` | `tl.dot` 类重计算算子 |
| GELU | `gelu/input.py` | 激活函数 |
| LayerNorm | `layernorm/input.py` | 带参数的规约算子 |
| FlashAttention 前向 | `flashattn_fwd/input.py` | 无因果掩码的标准注意力前向（softmax + 两次矩阵乘），难度高 |
| FlashAttention 反向 | `flashattn_bwd/input.py` | 返回 dq/dk/dv 三路梯度，难度高；受"单 kernel 铁律"约束，模型可能需要多轮修复 |

> 注意：flash 注意力类参考实现受平台"一个 torch_fn 只对应一个 triton kernel"约束，属于高难度生成任务，
> 模型可能多轮修复后仍无法满足精度；可适当放宽 rtol/atol 或调大最大修复轮次。

## 在 Web UI 中使用

1. 打开 Poseidon 页面，选择 GPU/SDK/whl/模型；
2. 把 `input.py` 的内容粘贴到 "torch 参考实现"；
3. 在 "输入说明" 中填入对应 `inputs_hint.txt` 的内容（或留空让模型自己设计）；
4. 点击 Run。

## 独立使用（构建机上）

```bash
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:/opt/maca/ompi/lib:$LD_LIBRARY_PATH

python tools/run_kernel.py \
  --reference examples/vector_add/input.py \
  --kernel  <模型生成的 kernel_code 文件> \
  --inputs  <模型生成的 inputs_code 文件>
```
