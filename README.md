# Poseidon 🔱

基于沐曦（MetaX）GPU 的 **mcTriton kernel 自动生成与性能测试 Web 平台**。

用户在 Web 页面输入一段 **torch 参考实现**，选择 MACA SDK 版本、whl 包版本与目标 GPU（如沐曦 C500），
点击 **Run** 后，平台调用大模型（DeepSeek API，默认模型 `deepseek-v4-pro`）生成等价的 mcTriton kernel，
在构建机上完成 **精度校验**（与 torch 实现逐元素对比）与 **性能测试**，并把结果（精度指标、延迟、加速比）与
生成的 kernel 代码展示在页面上。精度不通过时平台会自动把错误反馈给模型进行修复重试。

## 整体流程

```
用户输入 torch 参考实现
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ Poseidon 后端 (Node.js)                                       │
│  ① LLM 生成：system prompt = skills/mctriton-kernel-generation │
│     （SKILL.md + references/），调用 DeepSeek API 输出 JSON    │
│  ② 组装测试脚本：kernel + 输入构造 + 精度/性能 harness          │
│  ③ SSH → 构建机（10.2.118.21），注入 MACA 环境变量运行          │
│  ④ 解析结果；失败则把错误反馈给 LLM 自动修复（可配置轮次）       │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
Web UI：任务日志、精度指标表、torch vs triton 性能对比（加速比）
```

## 目录结构

```
poseidon/
├── server/                     # Node.js 后端（Express + ssh2）
│   ├── src/
│   │   ├── index.js            # 入口：Web 服务 + REST API
│   │   ├── config.js           # 配置加载（config.yaml + 环境变量覆盖）
│   │   ├── jobs.js             # 作业队列（GPU 独占，串行执行）
│   │   ├── llm/                # DeepSeek 客户端 + JSON 协议解析
│   │   ├── runner/             # SSH 会话、MACA 环境注入、harness 组装
│   │   ├── pipeline/           # 生成 → 精度 → 性能 → 自动修复 主流程
│   │   └── api/                # REST 路由 + 构建机目录发现
│   └── templates/harness.py.tpl# 注入用户/生成代码的测试模板（精度+性能）
├── web/                        # 前端（原生 HTML/CSS/JS，中文界面）
├── skills/
│   └── mctriton-kernel-generation/   # ★ 大模型生成 triton kernel 的核心技能
│       ├── SKILL.md                   # 主技能（角色、JSON 协议、kernel 写法、修复协议）
│       └── references/                # API 速查 / MACA 环境事实 / 测试协议
├── tools/run_kernel.py         # 脱离平台独立运行 kernel 的验证脚本
├── examples/                   # 示例 torch 实现（vector_add / softmax / matmul / gelu / layernorm）
├── docs/                       # 部署与协议文档
├── config.example.yaml         # 配置模板（复制为 config.yaml，不入库）
└── README.md
```

## 快速开始

### 0. 前提

- 本机（或部署机）安装 Node.js ≥ 20；
- 能通过 SSH（密码）访问构建机 `10.2.118.21`（默认账号 `yuliu`）；
- 有可用的 DeepSeek API Key。

### 1. 安装依赖

```bash
cd server
npm install
```

### 2. 配置

```bash
# 仓库根目录
cp config.example.yaml config.yaml
# 编辑 config.yaml：填入 llm.api_key（必填）、build_machine.password（必填）
```

所有配置项也可用环境变量覆盖：`POSEIDON_LLM_API_KEY`、`POSEIDON_SSH_HOST`、
`POSEIDON_SSH_USER`、`POSEIDON_SSH_PASSWORD`、`POSEIDON_PORT` 等。

### 3. 启动

```bash
cd server
npm start
# 打开 http://localhost:8000
```

### 4. 使用

1. 左侧选择 **GPU 设备 / MACA SDK / whl 包 / 生成模型**，设置精度容差 rtol/atol 与最大修复轮次；
2. 在 "torch 参考实现" 输入框粘贴代码（必须定义 `def torch_fn(*args, **kwargs)`），可选填"输入说明"；
3. 点击 **▶ 生成并测试（Run）**；
4. 右侧实时显示：LLM 生成 → 构建机编译运行日志 → 精度指标表 → torch vs triton 性能对比（加速比）→ 生成的 kernel 代码。

## 构建机环境（默认）

| 项目 | 值 |
|---|---|
| 主机 | `yuliu@10.2.118.21` |
| GPU | MetaX C500（`mx-smi` 查看） |
| SDK | `/opt/maca`（Version 20260318.1063，另有 `/opt/maca-3.7.0`） |
| whl 包目录 | `~/ws/poseidon/whl/{3.7.1.3-dsv4, 3.7.2.0}/wheel`（torch/triton 等） |
| 运行环境变量 | `MACA_PATH=/opt/maca`，`LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:/opt/maca/ompi/lib`，`TRITON_METAX_ENABLE_TORCH_REDUCTION_ORDER=1` |

平台会自动为每个 whl 版本准备 Python 环境：优先使用 `build_machine.python_overrides` 中指定的解释器，
否则在 `~/ws/poseidon/venvs/<版本>` 自动创建 venv 并安装对应的 torch/triton wheel（首次运行需要几分钟）。

## API 摘要

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 服务健康状态 |
| GET | `/api/catalog` | 从构建机发现 GPU/SDK/whl/模型列表 |
| POST | `/api/jobs` | 提交生成任务 `{torch_code, whl, gpu, sdk, model, rtol, atol, max_retries, ...}` |
| GET | `/api/jobs` | 任务历史 |
| GET | `/api/jobs/:id` | 任务详情（事件日志 + 结果） |

任务结果结构（`result` 字段）见 [docs/pipeline.md](docs/pipeline.md)。

## 关于 skill

仓库的核心资产是 [`skills/mctriton-kernel-generation`](skills/mctriton-kernel-generation/)——
一份指导大模型生成 **沐曦 mcTriton kernel** 的完整技能：

- `SKILL.md`：角色定义、严格 JSON 输出协议、mcTriton 语言要点（含 MetaX 与上游 Triton 的差异）、
  常见 torch→Triton 映射模板、精度/性能准则、错误修复协议；
- `references/mctriton-api.md`：在 MetaX C500 上验证过的 API 速查表；
- `references/maca-environment.md`：构建机真实环境事实（SDK、环境变量、约束）；
- `references/accuracy-benchmark-protocol.md`：平台测试框架对生成代码的调用约定。

Poseidon 运行时把整个目录拼接为 system prompt。它也可以独立使用：
把 `SKILL.md` + `references/*.md` 作为 system prompt 直接提问，
模型按协议输出的 JSON 可用 [`tools/run_kernel.py`](tools/run_kernel.py) 在装有 MetaX torch/triton 的机器上验证。

## 文档

- [docs/deployment.md](docs/deployment.md) — 部署到构建机/其他机器、systemd、常见问题
- [docs/pipeline.md](docs/pipeline.md) — LLM JSON 协议、harness 说明、结果 schema、自动修复流程
