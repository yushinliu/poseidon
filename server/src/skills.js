import fs from 'node:fs';
import path from 'node:path';
import { REPO_ROOT } from './config.js';

const SKILL_DIR = path.join(REPO_ROOT, 'skills', 'mctriton-kernel-generation');

let cached = null;

/** 加载 skill（SKILL.md + references/*.md），拼接为 system prompt。 */
export function loadSkillPrompt() {
  if (cached) return cached;
  const parts = [];
  const main = path.join(SKILL_DIR, 'SKILL.md');
  if (fs.existsSync(main)) parts.push(fs.readFileSync(main, 'utf8'));
  const refDir = path.join(SKILL_DIR, 'references');
  if (fs.existsSync(refDir)) {
    for (const f of fs.readdirSync(refDir).sort()) {
      if (f.endsWith('.md')) {
        parts.push(`\n\n---\n# 参考资料: ${f}\n---\n\n` + fs.readFileSync(path.join(refDir, f), 'utf8'));
      }
    }
  }
  cached = parts.join('\n');
  return cached;
}
