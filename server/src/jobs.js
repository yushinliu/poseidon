import { randomUUID } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { config, REPO_ROOT } from './config.js';
import { runJob } from './pipeline/orchestrator.js';

const jobs = new Map();
let queue = [];
let running = false;

export function createJob(input) {
  const job = {
    id: randomUUID().slice(0, 8),
    created_at: Date.now(),
    status: 'queued',
    torch_code: input.torch_code,
    inputs_hint: input.inputs_hint || '',
    gpu: input.gpu || 'C500',
    sdk: input.sdk || '/opt/maca',
    whl: input.whl,
    model: input.model,
    rtol: Number(input.rtol) || config.benchmark.default_rtol,
    atol: Number(input.atol) || config.benchmark.default_atol,
    warmup: Number(input.warmup) || config.benchmark.warmup,
    iters: Number(input.iters) || config.benchmark.iters,
    maxRetries: Number.isFinite(Number(input.max_retries)) ? Math.min(Math.max(Number(input.max_retries), 0), 10) : config.llm.max_retries,
    events: [],
    attempts: [],
    result: null,
    generated: null,
    error: null,
    runLog: null,
  };
  jobs.set(job.id, job);
  queue.push(job.id);
  setImmediate(pump);
  return job;
}

export function getJob(id) {
  return jobs.get(id) || null;
}

/** 中断任务：排队中 → 直接出队取消；运行中 → 置取消标记（运行流程会强杀远端进程并结束）。 */
export function cancelJob(id) {
  const job = jobs.get(id);
  if (!job) return { ok: false, error: '作业不存在' };
  if (job.status === 'success' || job.status === 'failed' || job.status === 'cancelled') {
    return { ok: false, error: `作业已结束（${job.status}），无需中断` };
  }
  job.cancelled = true;
  job.events.push({ t: Date.now(), type: 'warn', message: '收到中断请求，正在终止任务...' });
  if (job.status === 'queued') {
    const idx = queue.indexOf(id);
    if (idx >= 0) queue.splice(idx, 1);
    job.status = 'cancelled';
    job.finished_at = Date.now();
    persistJob(job);
  }
  return { ok: true, status: job.status };
}

export function listJobs() {
  return [...jobs.values()]
    .map(publicView)
    .sort((a, b) => b.created_at - a.created_at);
}

function publicView(job) {
  const { torch_code, ...rest } = job;
  return { ...rest, torch_code_len: torch_code.length };
}

async function pump() {
  if (running) return;
  let id = queue.shift();
  while (id) {
    const j = jobs.get(id);
    if (j && !j.cancelled) break;
    id = queue.shift();
  }
  if (!id) return;
  const job = jobs.get(id);
  if (!job) { setImmediate(pump); return; }
  running = true;
  job.status = 'running';
  job.started_at = Date.now();
  try {
    await runJob(job);
  } catch (e) {
    job.status = 'failed';
    job.error = e.message;
    job.events.push({ t: Date.now(), type: 'error', message: `作业异常: ${e.message}` });
  } finally {
    job.finished_at = Date.now();
    running = false;
    persistJob(job);
    setImmediate(pump);
  }
}

function persistJob(job) {
  try {
    const dir = path.join(REPO_ROOT, 'jobs', job.id);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'job.json'), JSON.stringify(job, null, 2));
    if (job.generated?.kernel_code) fs.writeFileSync(path.join(dir, 'generated_kernel.py'), job.generated.kernel_code);
    if (job.generated?.inputs_code) fs.writeFileSync(path.join(dir, 'inputs.py'), job.generated.inputs_code);
    fs.writeFileSync(path.join(dir, 'reference_torch.py'), job.torch_code);
    if (job.runLog) fs.writeFileSync(path.join(dir, 'run.log'), job.runLog);
  } catch (e) {
    console.error(`[jobs] 持久化失败: ${e.message}`);
  }
}

/** 进程退出时把内存中的作业摘要写到磁盘。 */
export function flushAll() {
  for (const job of jobs.values()) {
    if (job.status === 'running' || job.status === 'queued') continue;
    persistJob(job);
  }
}
