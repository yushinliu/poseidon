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
        stream.on('close', (code) => {
          clearTimeout(timer);
          resolve({ code, stdout, stderr });
        });
        stream.on('error', (e) => { clearTimeout(timer); reject(new Error(`流错误: ${e.message}`)); });
      });
    });
  }

  /** 执行命令并流式回调输出（用于实时日志）。返回 { code, stdout, stderr }。 */
  execStream(cmd, { timeoutMs = 600000, onData = () => {} } = {}) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`命令执行超时 (${Math.round(timeoutMs / 1000)}s)`));
      }, timeoutMs);
      this.client.exec(cmd, (err, stream) => {
        if (err) { clearTimeout(timer); reject(new Error(`exec 失败: ${err.message}`)); return; }
        let stdout = '';
        let stderr = '';
        stream.on('data', (d) => { stdout += d; onData(d.toString()); });
        stream.stderr.on('data', (d) => { stderr += d; onData(d.toString()); });
        stream.on('close', (code) => {
          clearTimeout(timer);
          resolve({ code, stdout, stderr });
        });
        stream.on('error', (e) => { clearTimeout(timer); reject(new Error(`流错误: ${e.message}`)); });
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
    return new Promise((resolve, reject) => {
      const parts = dir.split('/').filter(Boolean);
      let cur = dir.startsWith('/') ? '/' : '';
      const step = (i) => {
        if (i >= parts.length) return resolve();
        cur = cur === '/' ? `/${parts[i]}` : `${cur}/${parts[i]}`;
        sftp.mkdir(cur, (err) => {
          // EEXIST(4) 或 SFTP 状态 4 视为已存在
          step(i + 1);
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

/** 单引号转义，用于 shell 命令拼接。 */
export function shq(s) {
  return `'${String(s).replace(/'/g, `'\\''`)}'`;
}
