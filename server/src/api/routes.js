import { Router } from 'express';
import { config } from '../config.js';
import { getCatalog } from './catalog.js';
import { createJob, getJob, listJobs, cancelJob } from '../jobs.js';
import { loadSkillPrompt } from '../skills.js';

export const api = Router();

api.get('/health', (_req, res) => {
  res.json({
    ok: true,
    app: 'poseidon',
    version: '0.1.0',
    model: config.llm.model,
    llm_configured: Boolean(config.llm.api_key),
    build_machine: { host: config.build_machine.host, username: config.build_machine.username },
    skill_chars: loadSkillPrompt().length,
  });
});

api.get('/catalog', async (_req, res) => {
  try {
    res.json(await getCatalog());
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

api.post('/jobs', (req, res) => {
  const b = req.body || {};
  if (typeof b.torch_code !== 'string' || !b.torch_code.trim()) {
    return res.status(400).json({ error: '缺少 torch_code（torch 参考实现）' });
  }
  if (!b.whl) {
    return res.status(400).json({ error: '缺少 whl（请选择 whl 包版本）' });
  }
  const job = createJob(b);
  res.status(201).json({ job_id: job.id, status: job.status });
});

api.get('/jobs', (_req, res) => {
  res.json({ jobs: listJobs() });
});

api.get('/jobs/:id', (req, res) => {
  const job = getJob(req.params.id);
  if (!job) return res.status(404).json({ error: '作业不存在' });
  const { torch_code, ...rest } = job;
  res.json({ ...rest, torch_code_len: torch_code.length });
});

api.post('/jobs/:id/cancel', (req, res) => {
  const r = cancelJob(req.params.id);
  if (!r.ok) return res.status(400).json(r);
  res.json(r);
});
