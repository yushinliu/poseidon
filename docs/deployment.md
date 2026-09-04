# 部署指南

## 部署位置选择

Poseidon 后端是纯 Node.js 程序，可以部署在任意能通过 SSH 访问构建机的机器上，也可以直接部署在构建机
`10.2.118.21` 本机（此时 SSH 走 `127.0.0.1` 或保持默认配置即可，只要本机 sshd 可用）。

| 方案 | 说明 |
|---|---|
| A. 部署在构建机本机 | 延迟最低；需要构建机安装 Node.js（见下） |
| B. 部署在其他机器 | 通过 SSH 远程驱动构建机；把 `build_machine.host` 指向构建机 |

## 方案 A：部署到 10.2.118.21

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
git clone https://github.com/yushinliu/poseidon.git ~/ws/poseidon/server-app
cd ~/ws/poseidon/server-app
cp config.example.yaml config.yaml
vim config.yaml   # 填 llm.api_key；build_machine.host 可改 127.0.0.1
cd server && npm install --omit=dev
```

### 3. 以服务方式运行（systemd）

```ini
# /etc/systemd/system/poseidon.service
[Unit]
Description=Poseidon mcTriton kernel generation platform
After=network.target

[Service]
User=yuliu
WorkingDirectory=/home/yuliu/ws/poseidon/server-app/server
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
sudo ufw allow 8000/tcp   # 按需放行，供内网用户访问 http://10.2.118.21:8000
```

## 方案 B：远程部署

在部署机上装好 Node.js 后照常 `npm install && npm start`，配置中保持
`build_machine.host: 10.2.118.21`（默认）。注意部署机需能访问构建机 22 端口。

## 安全说明

- `config.yaml` 含明文凭据（LLM Key、SSH 密码），已被 `.gitignore` 排除，**切勿提交**；
  生产环境建议改用环境变量注入（`POSEIDON_LLM_API_KEY` / `POSEIDON_SSH_PASSWORD`）。
- v0.1 未内置用户认证，请仅在内网部署。

## 常见问题

1. **`import torch` 报 `libmxomp.so: cannot open shared object file`**
   → 平台已自动注入 `LD_LIBRARY_PATH`；若手动运行，先执行
   `export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:/opt/maca/ompi/lib:$LD_LIBRARY_PATH`。

2. **作业卡在"创建 venv"**
   → 首次使用某个 whl 版本需要安装 torch（约 2GB），等待数分钟属正常；
   若 pip 无法访问公网，在 `build_machine.pip_index_url` 配置内网 pip 源（如 `http://mirrors.aliyun.com/pypi/simple`）。

3. **LLM 一直返回格式错误**
   → 确认模型为 `deepseek-v4-pro`（或 `deepseek-v4-flash`）；推理模型输出需较大 `max_tokens`（默认 16384）。

4. **GPU 被占用/作业排队**
   → 平台默认串行执行（`jobs.max_concurrent: 1`），多用户同时提交会排队，属预期行为。
