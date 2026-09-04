import { Client } from 'ssh2';
import path from 'node:path';

/** 基于 ssh2 的轻量 SSH 会话封装（连接/执行/上传/读取）。 */
export class SshSession {
  constructor(opts) {
    this.opts = {
      host: opts.host,
      port: opts.port || 22,
      username: opts.username,
      password: opts.password,
      readyTimeout: opts.connect_timeout_ms || 20000,
      keepaliveInterval: 10000,
    };
    this.client = new Client();
    this.sftp = null;
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.client.once('ready', () => resolve(this));
      this.client.once('error', (err) => reject(new Error(`SSH 连接失败 ${this.opts.host}:${this.opts.port} - ${err.message}`)));
      try {
        this.client.connect(this.opts);
      } catch (e) {
        reject(e);
      }
    });
  }

  close() {
    try { this.client.end(); } catch { /* ignore */ }
  }

  /** 执行命令，返回 { code, stdout, stderr }。 */
  exec(cmd, { timeoutMs = 600000 } = {}) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`命令执行超时 (${Math.round(timeoutMs / 1000)}s): ${cmd.slice(0, 120)}`));
      }, timeoutMs);
      this.client.exec(cmd, (err, stream) => {
        if (err) { clearTimeout(timer); reject(new Error(`exec 失败: ${err.message}`)); return; }
        let stdout = '';
        let stderr = '';
        stream.on('data', (d) => { stdout += d; });
        stream.stderr.on('data', (d) => { stderr += d; });
        stream.on('close', (code, signal) => {
          clearTimeout(timer);
          resolve({ code, signal, stdout, stderr });
        });
        stream.on('error', (e) => { clearTimeout(timer); reject(new Error(`流错误: ${e.message}`)); });
      });
    });
  }

  /**
   * 执行命令并流式回调输出。
   * - timeoutMs: 总超时（超时前会先强杀远端进程再 reject）
   * - stallTimeoutMs: 无输出停滞超时（同样先强杀再 reject，用于"卡住自动重跑"）
   * - isCancelled: () => boolean，为 true 时强杀远端进程并 reject(CancelledError)
   * - pidFile: 远端 pid 文件路径（由包装命令写入，用于精准强杀）
   */
  execStream(cmd, { timeoutMs = 600000, stallTimeoutMs = 0, isCancelled = () => false, pidFile = null, onData = () => {} } = {}) {
    return new Promise((resolve, reject) => {
      let settled = false;
      let lastActivity = Date.now();

      const killRemote = () => {
        if (!pidFile) return;
        // [h]arness.py 括号技巧：模式不会匹配 kill 命令自身
        const killCmd = `pkill -9 -P "$(cat '${pidFile}' 2>/dev/null)" 2>/dev/null; kill -9 "$(cat '${pidFile}' 2>/dev/null)" 2>/dev/null; pkill -9 -f '[h]arness.py' 2>/dev/null; echo KILLED`;
        this.client.exec(killCmd, () => { /* 尽力而为 */ });
      };

      const settleReject = (err) => {
        if (settled) return;
        settled = true;
        killRemote();
        clearInterval(watchdog);
        clearTimeout(timer);
        reject(err);
      };

      const timer = setTimeout(() => {
        settleReject(new Error(`命令执行超时 (${Math.round(timeoutMs / 1000)}s)，已终止远端进程`));
      }, timeoutMs);

      const watchdog = setInterval(() => {
        if (settled) return;
        if (isCancelled()) {
          settleReject(new CancelledError('任务已中断，已终止远端进程'));
          return;
        }
        if (stallTimeoutMs > 0 && Date.now() - lastActivity > stallTimeoutMs) {
          settleReject(new Error(`执行停滞超过 ${Math.round(stallTimeoutMs / 1000)}s 无输出，已终止远端进程并准备自动重试`));
        }
      }, 1000);

      this.client.exec(cmd, (err, stream) => {
        if (err) { settleReject(new Error(`exec 失败: ${err.message}`)); return; }
        let stdout = '';
        let stderr = '';
        stream.on('data', (d) => {
          lastActivity = Date.now();
          stdout += d;
          onData(d.toString());
        });
        stream.stderr.on('data', (d) => {
          lastActivity = Date.now();
          stderr += d;
          onData(d.toString());
        });
        stream.on('close', (code, signal) => {
          if (settled) return;
          settled = true;
          clearInterval(watchdog);
          clearTimeout(timer);
          resolve({ code, signal, stdout, stderr });
        });
        stream.on('error', (e) => {
          settleReject(new Error(`流错误: ${e.message}`));
        });
      });
    });
  }

  _ensureSftp() {
    if (this.sftp) return Promise.resolve(this.sftp);
    return new Promise((resolve, reject) => {
      this.client.sftp((err, sftp) => {
        if (err) { reject(new Error(`SFTP 失败: ${err.message}`)); return; }
        this.sftp = sftp;
        resolve(sftp);
      });
    });
  }

  _mkdirP(sftp, dir) {
    return new Promise((resolve) => {
      const parts = dir.split('/').filter(Boolean);
      let cur = dir.startsWith('/') ? '/' : '';
      const step = (i) => {
        if (i >= parts.length) return resolve();
        cur = cur === '/' ? `/${parts[i]}` : `${cur}/${parts[i]}`;
        sftp.mkdir(cur, (err) => {
          step(i + 1); // EEXIST 视为已存在
        });
      };
      step(0);
    });
  }

  /** 上传字符串内容到远程文件（自动创建父目录）。 */
  async uploadFile(remotePath, content) {
    const sftp = await this._ensureSftp();
    await this._mkdirP(sftp, path.posix.dirname(remotePath));
    return new Promise((resolve, reject) => {
      sftp.writeFile(remotePath, content, (err) => (err ? reject(new Error(`上传失败 ${remotePath}: ${err.message}`)) : resolve()));
    });
  }

  /** 读取远程文件内容。 */
  async readFile(remotePath) {
    const sftp = await this._ensureSftp();
    return new Promise((resolve, reject) => {
      sftp.readFile(remotePath, (err, data) => (err ? reject(new Error(`读取失败 ${remotePath}: ${err.message}`)) : resolve(data.toString('utf8'))));
    });
  }
}

export class CancelledError extends Error {
  constructor(message) {
    super(message);
    this.name = 'CancelledError';
    this.cancelled = true;
  }
}

/** 单引号转义，用于 shell 命令拼接。 */
export function shq(s) {
  return `'${String(s).replace(/'/g, `'\\''`)}'`;
}
