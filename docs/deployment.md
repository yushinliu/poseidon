# 部署指南

## 部署位置选择

Poseidon 后端是纯 Node.js 程序，可以部署在任意能通过 SSH 访问构建机的机器上，也可以直接部署在构建机本机
（此时把 `build_machine.host` 配成 `127.0.0.1` 即可，只要本机 sshd 可用）。

| 方案 | 说明 |
|---|---|
| A. 部署在构建机本机 | 延迟最低；需要构建机安装 Node.js（见下） |
| B. 部署在其他机器 | 通过 SSH 远程驱动构建机；把 `build_machine.host` 指向构建机 |

> 文中 `<构建机>`、`<用户>`、`<路径>` 均为占位符，实际值只写在本地 `config.yaml`（不入库）。

## 方案 A：部署到构建机本机

### 1. 安装 Node.js ≥ 20

```bash
# 方式一：nvm
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc && nvm install 20

# 方式二：直接下载二进制
# https://nodejs.org/dist/v20.x/ 下载 linux-x64 二进制，解压后把 bin 加入 PATH
```

### 2. 拉取代码并配置

```bash
git clone <仓库地址> ~/ws/poseidon/server-app
cd ~/ws/poseidon/server-app
cp config.example.yaml config.yaml
vim config.yaml   # 填 llm.api_key；build_machine.host 改 127.0.0.1；填用户名/密码/python 路径
cd server && npm install --omit=dev
```

### 3. 以服务方式运行（systemd）

```ini
# /etc/systemd/system/poseidon.service
[Unit]
Description=Poseidon mcTriton kernel generation platform
After=network.target

[Service]
User=<用户>
WorkingDirectory=<安装路径>/server
Environment=POSEIDON_LLM_API_KEY=sk-xxx
Environment=POSEIDON_SSH_PASSWORD=xxx
ExecStart=/usr/bin/node src/index.js
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now poseidon
curl http://127.0.0.1:8000/api/health
```

### 4. 防火墙

```bash
sudo ufw allow 8000/tcp   # 按需放行，供内网用户访问 http://<构建机>:8000
```

## 方案 B：远程部署

在部署机上装好 Node.js 后照常 `npm install && npm start`，配置中把
`build_machine.host` 指向构建机地址。注意部署机需能访问构建机 22 端口。

## 安全说明

- `config.yaml` 含明文凭据（LLM Key、SSH 密码），已被 `.gitignore` 排除，**切勿提交**；
  生产环境建议改用环境变量注入（`POSEIDON_LLM_API_KEY` / `POSEIDON_SSH_PASSWORD`）。
- 推送到远端仓库的文件（`config.example.yaml`、本文档等）不包含任何真实服务器地址、用户名或密码。
- v0.1 未内置用户认证，请仅在内网部署。

## 常见问题

1. **`import torch` 报 `libmxomp.so: cannot open shared object file`**
   → 平台已自动注入 `LD_LIBRARY_PATH`；若手动运行，先执行
   `export LD_LIBRARY_PATH=$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:$LD_LIBRARY_PATH`。

2. **作业卡在"创建 venv"**
   → 首次使用某个 whl 版本需要安装 torch（约 2GB），等待数分钟属正常。
   平台采用**离线安装策略**：venv 复用 `base_python` 所在环境的 site-packages（构建机通常无外网 pip 源），
   本版本的 torch/triton 用本地 wheel `--no-index --no-deps` 安装，不依赖任何 pip 源；
   若构建机可访问 pip 源，也可在 `build_machine.pip_index_url` 配置内网源。

3. **LLM 一直返回格式错误**
   → 确认模型为 `deepseek-v4-pro`（或 `deepseek-v4-flash`）；推理模型输出需较大 `max_tokens`（默认 16384）。

4. **GPU 被占用/作业排队**
   → 平台默认串行执行（`jobs.max_concurrent: 1`），多用户同时提交会排队，属预期行为。
