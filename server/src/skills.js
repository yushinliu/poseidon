import fs from 'node:fs';
import path from 'node:path';
import { REPO_ROOT, config } from './config.js';

const cache = new Map();

/** 由 kernel 类型 id 解析 skill 目录（默认 mctriton-kernel-generation）。 */
function skillDirOf(kernelType) {
  const kt = (config.kernel_types || []).find((k) => k.id === kernelType);
  return path.join(REPO_ROOT, 'skills', (kt?.skill || 'mctriton-kernel-generation'));
}

function readManifest(dir) {
  try {
    return JSON.parse(fs.readFileSync(path.join(dir, 'versions', 'manifest.json'), 'utf8'));
  } catch {
    return {};
  }
}

/**
 * 解析 whl 版本对应的 skill 版本标识。
 * 优先取目录发现（catalog）中的 wheel 文件名，其次取 manifest.json，最后从 whl 目录名本身推断。
 * @param {string} whl whl 版本目录名（如 "3.7.1.3-dsv4"）
 * @param {{torch?: string, triton?: string}|null} catalogEntry catalog.whls 中对应条目
 * @param {string} [kernelType]
 * @returns {{triton_major: string|null, sdk_minor: string|null}}
 */
export function resolveSkillVersions(whl, catalogEntry = null, kernelType = 'mctriton') {
  const m = readManifest(skillDirOf(kernelType));
  const base = m[whl] || {};
  let tritonMajor = base.triton_major || null;
  let sdkMinor = base.sdk_minor || null;

  // wheel 文件名（如 triton-3.6.0+metax3.7.1.3.dsv4-...whl）是权威来源
  const t = String(catalogEntry?.triton || '').match(/triton-(\d+)\.(\d+)/);
  if (t) tritonMajor = `${t[1]}.${t[2]}`;
  const s = String(whl).match(/^(\d+\.\d+\.\d+)/);
  if (s) sdkMinor = s[1];

  // 兜底：按 whl 目录名前缀在 manifest 中查找
  if (!tritonMajor || !sdkMinor) {
    for (const [k, v] of Object.entries(m)) {
      if (String(whl).startsWith(k.split('-')[0])) {
        if (!tritonMajor) tritonMajor = v.triton_major;
        if (!sdkMinor) sdkMinor = v.sdk_minor;
        break;
      }
    }
  }
  return { triton_major: tritonMajor, sdk_minor: sdkMinor };
}

function readIfExists(p) {
  return fs.existsSync(p) ? fs.readFileSync(p, 'utf8') : null;
}

/**
 * 加载拼装后的 skill system prompt：
 * SKILL.md → references/*.md → versions/triton-<major>.md → versions/sdk-<minor>.md（存在才追加）。
 * @param {{triton_major?: string|null, sdk_minor?: string|null}} [versions]
 * @param {string} [kernelType]
 * @returns {{prompt: string, parts: string[]}}
 */
export function loadSkillPrompt(versions = {}, kernelType = 'mctriton') {
  const skillDir = skillDirOf(kernelType);
  const refDir = path.join(skillDir, 'references');
  const versionsDir = path.join(skillDir, 'versions');
  const key = `${kernelType}|${versions?.triton_major || ''}|${versions?.sdk_minor || ''}`;
  if (cache.has(key)) return cache.get(key);

  const parts = [];
  const used = [];
  const main = readIfExists(path.join(skillDir, 'SKILL.md'));
  if (main) { parts.push(main); used.push('SKILL.md'); }
  if (fs.existsSync(refDir)) {
    for (const f of fs.readdirSync(refDir).sort()) {
      if (f.endsWith('.md')) {
        const c = readIfExists(path.join(refDir, f));
        if (c) { parts.push(`\n\n---\n# 参考资料: ${f}\n---\n\n${c}`); used.push(`references/${f}`); }
      }
    }
  }
  for (const f of [`triton-${versions?.triton_major}.md`, `sdk-${versions?.sdk_minor}.md`]) {
    if (!f.endsWith('.md') || f === 'triton-null.md' || f === 'sdk-null.md') continue;
    const c = readIfExists(path.join(versionsDir, f));
    if (c) { parts.push(`\n\n---\n# 版本专属: ${f}\n---\n\n${c}`); used.push(`versions/${f}`); }
  }

  const out = { prompt: parts.join('\n'), parts: used };
  cache.set(key, out);
  return out;
}
