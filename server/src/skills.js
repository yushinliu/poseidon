import fs from 'node:fs';
import path from 'node:path';
import { REPO_ROOT } from './config.js';

const SKILL_DIR = path.join(REPO_ROOT, 'skills', 'mctriton-kernel-generation');
const REF_DIR = path.join(SKILL_DIR, 'references');
const VERSIONS_DIR = path.join(SKILL_DIR, 'versions');

let manifest = null;
const cache = new Map();

function readManifest() {
  if (manifest) return manifest;
  try {
    manifest = JSON.parse(fs.readFileSync(path.join(VERSIONS_DIR, 'manifest.json'), 'utf8'));
  } catch {
    manifest = {};
  }
  return manifest;
}

/**
 * 解析 whl 版本对应的 skill 版本标识。
 * 优先取目录发现（catalog）中的 wheel 文件名，其次取 manifest.json，最后从 whl 目录名本身推断。
 * @param {string} whl whl 版本目录名（如 "3.7.1.3-dsv4"）
 * @param {{torch?: string, triton?: string}|null} catalogEntry catalog.whls 中对应条目
 * @returns {{triton_major: string|null, sdk_minor: string|null}}
 */
export function resolveSkillVersions(whl, catalogEntry = null) {
  const m = readManifest();
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
 * @returns {{prompt: string, parts: string[]}}
 */
export function loadSkillPrompt(versions = {}) {
  const key = `${versions?.triton_major || ''}|${versions?.sdk_minor || ''}`;
  if (cache.has(key)) return cache.get(key);

  const parts = [];
  const used = [];
  const main = readIfExists(path.join(SKILL_DIR, 'SKILL.md'));
  if (main) { parts.push(main); used.push('SKILL.md'); }
  if (fs.existsSync(REF_DIR)) {
    for (const f of fs.readdirSync(REF_DIR).sort()) {
      if (f.endsWith('.md')) {
        const c = readIfExists(path.join(REF_DIR, f));
        if (c) { parts.push(`\n\n---\n# 参考资料: ${f}\n---\n\n${c}`); used.push(`references/${f}`); }
      }
    }
  }
  for (const f of [`triton-${versions?.triton_major}.md`, `sdk-${versions?.sdk_minor}.md`]) {
    if (!f.endsWith('.md') || f === 'triton-null.md' || f === 'sdk-null.md') continue;
    const c = readIfExists(path.join(VERSIONS_DIR, f));
    if (c) { parts.push(`\n\n---\n# 版本专属: ${f}\n---\n\n${c}`); used.push(`versions/${f}`); }
  }

  const out = { prompt: parts.join('\n'), parts: used };
  cache.set(key, out);
  return out;
}
