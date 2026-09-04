import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TPL_PATH = path.join(__dirname, '..', '..', 'templates', 'harness.py.tpl');

const MARKERS = {
  ref: { begin: '# ===POSEIDON_REF_BEGIN===', end: '# ===POSEIDON_REF_END===' },
  kernel: { begin: '# ===POSEIDON_KERNEL_BEGIN===', end: '# ===POSEIDON_KERNEL_END===' },
  inputs: { begin: '# ===POSEIDON_INPUTS_BEGIN===', end: '# ===POSEIDON_INPUTS_END===' },
};

/** 把用户 torch 代码与模型生成的 kernel/inputs 代码注入测试模板。 */
export function buildHarness({ userCode, kernelCode, inputsCode }) {
  const tpl = fs.readFileSync(TPL_PATH, 'utf8');
  let out = insertBetween(tpl, MARKERS.ref, userCode);
  out = insertBetween(out, MARKERS.kernel, stripMainBlock(kernelCode));
  out = insertBetween(out, MARKERS.inputs, stripMainBlock(inputsCode));
  return out;
}

function insertBetween(tpl, { begin, end }, code) {
  const b = tpl.indexOf(begin);
  const e = tpl.indexOf(end);
  if (b < 0 || e < 0 || e < b) throw new Error('harness 模板标记缺失');
  const body = String(code)
    .replace(/\r\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  return tpl.slice(0, b + begin.length) + '\n' + body + '\n' + tpl.slice(e);
}

/** 防御性去掉生成代码中可能存在的 `if __name__ == "__main__"` 块。 */
export function stripMainBlock(code) {
  const m = String(code).match(/^if\s+__name__\s*==\s*["']__main__["']\s*:\s*$/m);
  return m ? String(code).slice(0, m.index) : String(code);
}

/** 从 stdout 中解析 ###POSEIDON_RESULT### 行。 */
export function parseResult(stdout) {
  const marker = '###POSEIDON_RESULT###';
  const idx = stdout.lastIndexOf(marker);
  if (idx < 0) return null;
  const line = stdout.slice(idx + marker.length).split('\n')[0].trim();
  try {
    return JSON.parse(line);
  } catch {
    return null;
  }
}
