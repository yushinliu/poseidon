# mcTriton Python API 速查（MetaX 移植版 Triton）

以下 API 均已在本平台构建机（MetaX C500 + SDK 20260318.1063 + triton 3.0/3.6）验证可用。
模块名与上游 Triton 一致：`import triton` / `import triton.language as tl`。

## 顶层 triton

| API | 说明 |
|---|---|
| `@triton.jit` | 装饰 Python 函数为 JIT kernel |
| `@triton.autotune(configs=[...], key=[...])` | 按 key（标量参数名）自动选择最优 config |
| `triton.Config(kwargs, num_warps=4, num_stages=2)` | 启动配置；MetaX 扩展 kwargs：`pipeline: "cpasync"`、`scenario: "storeCoalesce"` |
| `triton.cdiv(a, b)` | 向上取整除 |
| `triton.next_power_of_2(n)` | 下一个 2 的幂 |
| `triton.runtime.driver.active.get_active_torch_device()` | 获取当前设备（MetaX 返回 `cuda:0`） |
| `kernel[(grid,)](args...)` | 启动 kernel，grid 为 1~3 维 tuple |

## triton.language 核心

### 索引与形状
| API | 说明 |
|---|---|
| `tl.program_id(axis)` | 当前 block 在 axis 维的编号 |
| `tl.num_programs(axis)` | 该维 block 总数 |
| `tl.arange(start, end)` | 一维整数序列（编译期常量范围） |
| `tl.zeros(shape, dtype)` / `tl.full(shape, value, dtype)` | 常量 tensor |
| `tl.reshape` / `tl.view` / `tl.trans` / `tl.expand_dims` / `tl.broadcast_to` | 形状操作 |
| `tl.range(a, b, step)` | 编译期循环 range |

### 内存
| API | 说明 |
|---|---|
| `tl.load(ptr, mask=None, other=None, cache_modifier="")` | 全局内存加载；mask 为 bool tensor，other 为越界填充值 |
| `tl.store(ptr, value, mask=None)` | 全局内存写回 |
| `tl.atomic_add(ptr, val, mask=None)` | 原子加（多 block 写同一位置时使用） |
| `tl.atomic_max(ptr, val, mask=None)` | 原子 max |

### 算术与数学
`+ - * / // % **`、比较 `== != < <= > >=`、逻辑 `& | ^ ~`（逐元素）。
`tl.` 命名空间：`exp, exp2, log, log2, sqrt, rsqrt, abs, sin, cos, tan, tanh, sigmoid, softmax(x, axis=...)`、
`floor, ceil, fma(x,y,z), min, max, clamp(x, lo, hi)`。

### 规约
`tl.sum(x, axis=None, keep_dims=False)`、`tl.max`、`tl.min`、`tl.argmax`。
MetaX 平台已设置 `TRITON_METAX_ENABLE_TORCH_REDUCTION_ORDER=1`，规约顺序与 torch 对齐。

### 类型
`tl.float32, tl.float16, tl.bfloat16, tl.int8, tl.int16, tl.int32, tl.int64, tl.uint8, tl.float64(避免)`。
转换：`x.to(tl.float32)`、`tl.cast(x, dtype)`。比较运算 `x > 0` 返回 `tl.int1`。

### 矩阵乘
`tl.dot(a, b, acc=None)`：a 形状 (M, K)、b 形状 (K, N)，输出 (M, N)。
要求 M、N、K ≥ 16（2 的幂更稳）；dtype 支持 fp32 / fp16。累加器用 fp32。

### 控制流
- 编译期常量可用 `if`；运行期条件用 `tl.where(cond, a, b)`。
- 循环：`for i in range(...)` 使用编译期边界，或 `for i in tl.range(...)`；运行期次数用 `tl.static_range`（3.x 中 `range` 配合 constexpr 边界亦可）。

## 不支持 / 应避免（MetaX 与上游差异）

- `tl.make_tensor_descriptor`（TMA / Hopper 专属）：不可用。
- `triton.language.extra`（CUDA libdevice）：不可用，数学函数用 `tl.` 前缀。
- `tl.inline_asm_elementwise` / PTX 内联：不可用。
- fp8 相关（`tl.float8e4nv` 等）：C500 上谨慎/默认不支持，不要使用。
- `tl.split` / `tl.join` / `tl.gather` / `tl.interleave`：新版 API，兼容性未验证，优先用基础算子组合代替。

## 参考实现出处

- 上游 API 文档：<https://triton-lang.org/main/python-api/triton.language.html>（以本文件标注的差异为准）
- MetaX mcTriton 仓库：<https://github.com/MetaX-MACA/mcTriton>（分支 3.0）
