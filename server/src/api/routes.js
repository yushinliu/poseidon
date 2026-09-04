import { Router } from 'express';
import { config } from '../config.js';
import { getCatalog } from './catalog.js';
import { createJob, getJob, listJobs, cancelJob } from '../jobs.js';
import { loadSkillPrompt, resolveSkillVersions } from '../skills.js';

export const api = Router();

api.get('/health', (_req, res) => {
  res.json({
    ok: true,
    app: 'poseidon',
    version: '0.1.0',
    model: config.llm.model,
    llm_configured: Boolean(config.llm.api_key),
    build_machine: { host: config.build_machine.host, username: config.build_machine.username },
    skill_chars: loadSkillPrompt().prompt.length,
  });
});

api.get('/catalog', async (_req, res) => {
  try {
    res.json(await getCatalog());
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

api.post('/jobs', async (req, res) => {
  const b = req.body || {};
  if (typeof b.torch_code !== 'string' || !b.torch_code.trim()) {
    return res.status(400).json({ error: '缺少 torch_code（torch 参考实现）' });
  }
  if (!b.whl) {
    return res.status(400).json({ error: '缺少 whl（请选择 whl 包版本）' });
  }
  const kernelType = b.kernel_type || 'mctriton';
  const kt = (config.kernel_types || []).find((k) => k.id === kernelType);
  if (!kt) {
    return res.status(400).json({ error: `不支持的 kernel 类型: ${kernelType}` });
  }
  // 解析该 whl 版本对应的 skill 版本（triton 大版本 + SDK 小版本）
  let skillVersions = resolveSkillVersions(b.whl, null, kernelType);
  try {
    const cat = await getCatalog();
    const entry = cat?.whls?.find((w) => w.id === b.whl);
    if (entry) skillVersions = resolveSkillVersions(b.whl, entry, kernelType);
  } catch { /* 目录发现失败时用 manifest/目录名推断 */ }
  const job = createJob({ ...b, kernel_type: kernelType, kernel_type_name: kt.name, skill_versions: skillVersions });
  res.status(201).json({ job_id: job.id, status: job.status });
});

api.get('/jobs', (_req, res) => {
  res.json({ jobs: listJobs() });
});

api.get('/jobs/:id', (req, res) => {
  const job = getJob(req.params.id);
  if (!job) return res.status(404).json({ error: '作业不存在' });
  // 详情接口返回完整信息（含用户输入的 torch_fn，供结果页展示）
  res.json(job);
});

api.post('/jobs/:id/cancel', (req, res) => {
  const r = cancelJob(req.params.id);
  if (!r.ok) return res.status(400).json(r);
  res.json(r);
});
