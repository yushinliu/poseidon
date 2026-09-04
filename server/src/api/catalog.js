import { config } from '../config.js';
import { SshSession } from '../runner/ssh-client.js';

const DEFAULT_DEVICES = [{ id: 'C500', name: 'MetaX C500' }];
const DEFAULT_MODELS = ['deepseek-v4-pro', 'deepseek-v4-flash', 'deepseek-chat'];

let cache = null;
let cacheTime = 0;
const TTL_MS = 120000;

/**
 * 从构建机发现：GPU 型号、SDK 目录/版本、whl 版本集合。
 * 失败时回退到缓存或静态默认值。
 */
export async function getCatalog() {
  const now = Date.now();
  if (cache && now - cacheTime < TTL_MS) return cache;

  const cfg = config.build_machine;
  const cat = {
    source: 'build_machine',
    host: cfg.host,
    fetched_at: new Date().toISOString(),
    devices: [...DEFAULT_DEVICES],
    sdks: [],
    whls: [],
    models: DEFAULT_MODELS,
  };

  try {
    const ssh = new SshSession(cfg);
    await ssh.connect();

    // 1) GPU 型号（mx-smi 表格中的 "MetaX C500"）
    try {
      const r = await ssh.exec('mx-smi 2>/dev/null | grep -oE "MetaX[[:space:]]+[A-Za-z0-9.-]+" | sort -u', { timeoutMs: 30000 });
      const names = [...new Set(r.stdout.split('\n').map((s) => s.trim()).filter(Boolean))]
        .filter((n) => /^MetaX\s+[A-Z]\d+/.test(n)); // 排除 "MetaX System Management..." 表头
      if (names.length) {
        cat.devices = names.map((n) => ({ id: n.replace('MetaX', '').trim(), name: n }));
      }
    } catch { /* 保持默认 */ }

    // 2) SDK 目录与版本
    for (const base of cfg.sdk_dirs) {
      try {
        const r = await ssh.exec(`ls -d ${base} 2>/dev/null && cat ${base}/Version.txt 2>/dev/null | head -1`, { timeoutMs: 30000 });
        const version = r.stdout.split('\n').map((s) => s.trim()).find((s) => s.startsWith('Version')) || '';
        cat.sdks.push({ path: base, version: version.replace('Version:', '').trim() || 'unknown' });
      } catch { /* 跳过 */ }
    }

    // 3) whl 版本集合（whl_dir 下的子目录），并解析 torch/triton wheel 文件名
    try {
      const r = await ssh.exec(`ls ${cfg.whl_dir} 2>/dev/null`, { timeoutMs: 30000 });
      const sets = r.stdout.split('\n').map((s) => s.trim()).filter(Boolean);
      for (const s of sets) {
        if (s.startsWith('.')) continue;
        const info = await ssh.exec(
          `ls ${cfg.whl_dir}/${s}/wheel/torch-*.whl ${cfg.whl_dir}/${s}/wheel/triton-*.whl 2>/dev/null | xargs -n1 basename 2>/dev/null`,
          { timeoutMs: 30000 },
        );
        const files = info.stdout.split('\n').map((x) => x.trim()).filter(Boolean);
        cat.whls.push({
          id: s,
          name: s,
          torch: files.find((f) => f.startsWith('torch-')) || '',
          triton: files.find((f) => f.startsWith('triton-')) || '',
        });
      }
    } catch { /* 无 whl 信息 */ }

    ssh.close();
  } catch (e) {
    cat.source = 'fallback';
    cat.error = e.message;
    if (cache) return cache;
    // 首次失败也给出静态默认
    cat.sdks = cfg.sdk_dirs.map((p) => ({ path: p, version: 'unknown' }));
  }

  cache = cat;
  cacheTime = Date.now();
  return cat;
}
