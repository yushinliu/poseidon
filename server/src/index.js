import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { config, REPO_ROOT } from './config.js';
import { api } from './api/routes.js';
import { flushAll } from './jobs.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB_DIR = path.join(REPO_ROOT, 'web');

const app = express();
app.use(express.json({ limit: '8mb' }));
app.use('/api', api);
app.use(express.static(WEB_DIR));

app.get('/', (_req, res) => res.sendFile(path.join(WEB_DIR, 'index.html')));

const { host, port } = config.server;
app.listen(port, host, () => {
  console.log('='.repeat(60));
  console.log('  🔱 Poseidon - 沐曦 mcTriton Kernel 自动生成平台');
  console.log(`  Web UI:   http://${host === '0.0.0.0' ? 'localhost' : host}:${port}`);
  console.log(`  构建机:   ${config.build_machine.host} (${config.build_machine.username})`);
  console.log(`  默认模型: ${config.llm.model}`);
  console.log('='.repeat(60));
});

process.on('SIGINT', () => { flushAll(); process.exit(0); });
process.on('SIGTERM', () => { flushAll(); process.exit(0); });
