import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import yaml from 'js-yaml';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..', '..');

const DEFAULTS = {
  server: { host: '0.0.0.0', port: 8000 },
  // 可输出的 kernel 类型（Web 运行配置下拉框；skill 字段对应 skills/ 目录名）
  kernel_types: [
    { id: 'mctriton', name: 'mcTriton', skill: 'mctriton-kernel-generation' },
  ],
  llm: {
    base_url: 'https://api.deepseek.com',
    api_key: '',
    model: 'deepseek-v4-pro',
    max_tokens: 16384,
    temperature: 0,
    timeout_ms: 600000,
    max_retries: 3,
  },
  build_machine: {
    host: '127.0.0.1',
    port: 22,
    username: '',
    password: '',
    workdir: '~/ws/poseidon/runs',
    whl_dir: '~/ws/poseidon/whl',
    sdk_dirs: ['/opt/maca'],
    base_python: '',
    venvs_dir: '~/ws/poseidon/venvs',
    pip_index_url: '',
    python_overrides: {},
    run_timeout_ms: 900000,
    stall_timeout_ms: 240000,
    connect_timeout_ms: 20000,
  },
  benchmark: { warmup: 5, iters: 50, default_rtol: 0.02, default_atol: 0.02 },
  jobs: { max_concurrent: 1, keep_dir: true },
};

function loadConfigFile() {
  const p = path.join(REPO_ROOT, 'config.yaml');
  if (!fs.existsSync(p)) return {};
  try {
    return yaml.load(fs.readFileSync(p, 'utf8')) || {};
  } catch (e) {
    console.error(`[config] 解析 config.yaml 失败: ${e.message}`);
    return {};
  }
}

function deepMerge(base, over) {
  const out = { ...base };
  for (const [k, v] of Object.entries(over || {})) {
    if (v && typeof v === 'object' && !Array.isArray(v) && base[k] && typeof base[k] === 'object' && !Array.isArray(base[k])) {
      out[k] = deepMerge(base[k], v);
    } else {
      out[k] = v;
    }
  }
  return out;
}

function envOverrides(cfg) {
  const get = (name) => process.env[name];
  const c = structuredClone(cfg);
  if (get('POSEIDON_PORT')) c.server.port = Number(get('POSEIDON_PORT'));
  if (get('POSEIDON_LLM_BASE_URL')) c.llm.base_url = get('POSEIDON_LLM_BASE_URL');
  if (get('POSEIDON_LLM_API_KEY')) c.llm.api_key = get('POSEIDON_LLM_API_KEY');
  if (get('POSEIDON_LLM_MODEL')) c.llm.model = get('POSEIDON_LLM_MODEL');
  if (get('POSEIDON_SSH_HOST')) c.build_machine.host = get('POSEIDON_SSH_HOST');
  if (get('POSEIDON_SSH_PORT')) c.build_machine.port = Number(get('POSEIDON_SSH_PORT'));
  if (get('POSEIDON_SSH_USER')) c.build_machine.username = get('POSEIDON_SSH_USER');
  if (get('POSEIDON_SSH_PASSWORD')) c.build_machine.password = get('POSEIDON_SSH_PASSWORD');
  return c;
}

export const config = envOverrides(deepMerge(DEFAULTS, loadConfigFile()));
export { REPO_ROOT };
