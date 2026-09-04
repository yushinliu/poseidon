/* Poseidon 前端逻辑 */
const $ = (id) => document.getElementById(id);

const EXAMPLES = {
  vector_add: {
    code: `def torch_fn(x, y):
    return x + y
`,
    hint: 'x: float32[1024, 4096]，y: float32[1024, 4096]',
  },
  softmax: {
    code: `def torch_fn(x):
    return torch.softmax(x, dim=-1)
`,
    hint: 'x: float32[4096, 1024]',
  },
  matmul: {
    code: `def torch_fn(a, b):
    return a @ b
`,
    hint: 'a: float32[1024, 1024]，b: float32[1024, 1024]',
  },
  gelu: {
    code: `def torch_fn(x):
    return torch.nn.functional.gelu(x)
`,
    hint: 'x: float32[1024, 4096]',
  },
  layernorm: {
    code: `def torch_fn(x, weight, bias):
    return torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias, eps=1e-5)
`,
    hint: 'x: float32[4096, 512]，weight: float32[512]，bias: float32[512]',
  },
  flashattn_fwd: {
    code: `def torch_fn(q, k, v):
    scale = q.shape[-1] ** -0.5
    s = (q @ k.transpose(-2, -1)) * scale
    p = torch.softmax(s, dim=-1)
    return p @ v
`,
    hint: 'q: float32[2, 4, 512, 64]，k: float32[2, 4, 512, 64]，v: float32[2, 4, 512, 64]（未加因果掩码，head_dim=64）',
  },
  flashattn_bwd: {
    code: `def torch_fn(q, k, v, do):
    scale = q.shape[-1] ** -0.5
    s = (q @ k.transpose(-2, -1)) * scale
    p = torch.softmax(s, dim=-1)
    dp = do @ v.transpose(-2, -1)
    dv = p.transpose(-2, -1) @ do
    ds = p * (dp - (dp * p).sum(dim=-1, keepdim=True))
    dq = (ds * scale) @ k
    dk = (ds * scale).transpose(-2, -1) @ q
    return dq, dk, dv
`,
    hint: 'q/k/v: float32[2, 4, 512, 64]，do: float32[2, 4, 512, 64]（上游梯度；未加因果掩码，head_dim=64）',
  },
};

let catalog = null;
let currentJobId = null;
let pollTimer = null;
let seenEvents = 0;

// ---------- 初始化 ----------
async function init() {
  loadPrefs();
  $('sel-example').addEventListener('change', onExample);
  $('btn-run').addEventListener('click', onRun);
  $('btn-cancel').addEventListener('click', onCancel);
  addEventListener('beforeunload', savePrefs);

  try {
    const h = await fetch('/api/health').then((r) => r.json());
    $('health').textContent = `服务正常 · 模型 ${h.model} · 构建机 ${h.build_machine.host}`;
    $('health').style.color = '#3fb950';
  } catch {
    $('health').textContent = '后端不可用';
    $('health').style.color = '#f85149';
  }

  try {
    catalog = await fetch('/api/catalog').then((r) => r.json());
    if (catalog.error) $('health').textContent += `（构建机目录发现失败，使用默认值：${catalog.error}）`;
    populateSelects();
  } catch {
    $('health').textContent += '（catalog 加载失败）';
  }

  refreshHistory();
}

function populateSelects() {
  const fill = (sel, items, { valueKey = 'id', labelFn = (it) => (typeof it === 'string' ? it : it.name) } = {}) => {
    sel.innerHTML = '';
    for (const it of items) {
      const o = document.createElement('option');
      o.value = typeof it === 'string' ? it : it[valueKey];
      o.textContent = labelFn(it);
      sel.appendChild(o);
    }
  };

  fill($('sel-ktype'), catalog.kernel_types || [{ id: 'mctriton', name: 'mcTriton' }]);
  fill($('sel-gpu'), catalog.devices);
  fill($('sel-sdk'), catalog.sdks, { valueKey: 'path', labelFn: (s) => `${s.path}（${s.version}）` });
  fill($('sel-whl'), catalog.whls, { labelFn: (w) => w.torch ? `${w.id}（${w.torch.replace('.whl', '').replace('torch-', 'torch ')} / ${w.triton.replace('.whl', '').replace('triton-', 'triton ')}）` : w.id });
  fill($('sel-model'), catalog.models, { labelFn: (m) => m });

  const prefs = JSON.parse(localStorage.getItem('poseidon-prefs') || '{}');
  if (prefs.ktype && $('sel-ktype').querySelector(`option[value="${prefs.ktype}"]`)) $('sel-ktype').value = prefs.ktype;
  if (prefs.gpu && $('sel-gpu').querySelector(`option[value="${prefs.gpu}"]`)) $('sel-gpu').value = prefs.gpu;
  if (prefs.sdk && $('sel-sdk').querySelector(`option[value="${prefs.sdk}"]`)) $('sel-sdk').value = prefs.sdk;
  if (prefs.whl && $('sel-whl').querySelector(`option[value="${prefs.whl}"]`)) $('sel-whl').value = prefs.whl;
  if (prefs.model && $('sel-model').querySelector(`option[value="${prefs.model}"]`)) $('sel-model').value = prefs.model;
}

function savePrefs() {
  localStorage.setItem('poseidon-prefs', JSON.stringify({
    ktype: $('sel-ktype').value,
    gpu: $('sel-gpu').value,
    sdk: $('sel-sdk').value,
    whl: $('sel-whl').value,
    model: $('sel-model').value,
    torch: $('torch-code').value,
    hint: $('inputs-hint').value,
    rtol: $('inp-rtol').value,
    atol: $('inp-atol').value,
    retries: $('inp-retries').value,
    iters: $('inp-iters').value,
  }));
}
function loadPrefs() {
  const p = JSON.parse(localStorage.getItem('poseidon-prefs') || '{}');
  if (p.torch) $('torch-code').value = p.torch;
  if (p.hint) $('inputs-hint').value = p.hint;
  if (p.rtol) $('inp-rtol').value = p.rtol;
  if (p.atol) $('inp-atol').value = p.atol;
  if (p.retries) $('inp-retries').value = p.retries;
  if (p.iters) $('inp-iters').value = p.iters;
}

function onExample() {
  const ex = EXAMPLES[$('sel-example').value];
  if (!ex) return;
  $('torch-code').value = ex.code;
  $('inputs-hint').value = ex.hint;
}

// ---------- 任务 ----------
async function onRun() {
  const torchCode = $('torch-code').value.trim();
  if (!torchCode) { alert('请先输入 torch 参考实现'); return; }
  if (!/def\s+torch_fn\s*\(/.test(torchCode)) { alert('torch 代码必须定义 def torch_fn(*args, **kwargs)'); return; }
  savePrefs();

  $('btn-run').disabled = true;
  $('btn-run').textContent = '提交中…';
  try {
    const res = await fetch('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        torch_code: torchCode,
        inputs_hint: $('inputs-hint').value.trim(),
        kernel_type: $('sel-ktype').value,
        gpu: $('sel-gpu').value,
        sdk: $('sel-sdk').value,
        whl: $('sel-whl').value,
        model: $('sel-model').value,
        rtol: parseFloat($('inp-rtol').value) || 0.02,
        atol: parseFloat($('inp-atol').value) || 0.02,
        max_retries: parseInt($('inp-retries').value, 10),
        iters: parseInt($('inp-iters').value, 10) || 50,
      }),
    });
    const data = await res.json();
    if (!res.ok) { alert(`提交失败：${data.error || res.status}`); return; }
    currentJobId = data.job_id;
    seenEvents = 0;
    $('job-card').hidden = false;
    $('job-title').textContent = `任务 #${currentJobId}`;
    $('job-log').innerHTML = '';
    $('job-result').innerHTML = '';
    setStatus('queued');
    startPolling();
  } catch (e) {
    alert(`提交失败：${e.message}`);
  } finally {
    $('btn-run').disabled = false;
    $('btn-run').textContent = '▶ 生成并测试（Run）';
  }
}

function startPolling() {
  stopPolling();
  pollTimer = setInterval(poll, 1500);
  poll();
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

async function poll() {
  if (!currentJobId) return;
  try {
    const job = await fetch(`/api/jobs/${currentJobId}`).then((r) => r.json());
    renderJob(job);
    if (job.status === 'success' || job.status === 'failed' || job.status === 'cancelled') {
      stopPolling();
      refreshHistory();
    }
  } catch { /* 网络抖动忽略 */ }
}

function setStatus(s) {
  const el = $('job-status');
  const map = { queued: '排队中', running: '运行中', success: '成功', failed: '失败', cancelled: '已中断' };
  el.textContent = map[s] || s;
  el.className = `badge ${s}`;
  $('btn-cancel').hidden = !(s === 'queued' || s === 'running');
}

async function onCancel() {
  if (!currentJobId) return;
  const btn = $('btn-cancel');
  btn.disabled = true;
  btn.textContent = '中断中…';
  try {
    const r = await fetch(`/api/jobs/${currentJobId}/cancel`, { method: 'POST' });
    const data = await r.json();
    if (!r.ok) { alert(`中断失败：${data.error || r.status}`); btn.disabled = false; btn.textContent = '⏹ 中断任务'; return; }
    poll();
  } catch (e) {
    alert(`中断失败：${e.message}`);
    btn.disabled = false;
    btn.textContent = '⏹ 中断任务';
  }
}

function renderJob(job) {
  setStatus(job.status);
  $('job-title').textContent = `任务 #${job.id}`;

  const log = $('job-log');
  const newEvents = job.events.slice(seenEvents);
  seenEvents = job.events.length;
  for (const ev of newEvents) {
    const line = document.createElement('span');
    line.className = 'line';
    const time = new Date(ev.t).toLocaleTimeString('zh-CN', { hour12: false });
    line.innerHTML = `<span class="t">${time}</span><span class="${ev.type}">${escapeHtml(ev.message)}</span>`;
    log.appendChild(line);
  }
  // 只保留最近 400 行，避免 DOM 过大导致卡顿/叠字
  while (log.children.length > 400) log.removeChild(log.firstChild);
  if (newEvents.length) log.scrollTop = log.scrollHeight;

  if (job.status === 'success') renderSuccess(job);
  else if (job.status === 'failed') renderFailed(job);
}

function renderSuccess(job) {
  const r = job.result;
  if (!r) return;
  const box = $('job-result');
  box.innerHTML = '';

  if (job.generated?.analysis) {
    const d = document.createElement('div');
    d.className = 'analysis';
    d.textContent = `🤖 模型分析：${job.generated.analysis}`;
    box.appendChild(d);
  }

  // 展示用户输入的 torch_fn 参考实现
  if (job.torch_code) {
    addCodeBlock(box, '输入 torch_fn 参考实现', job.torch_code, true);
  }

  if (r.performance) {
    const p = r.performance;
    const grid = document.createElement('div');
    grid.className = 'result-grid';
    grid.innerHTML = `
      <div class="result-box"><h3>torch 参考（中位数）</h3><div class="big-num">${p.torch_ms.toFixed(3)}<span class="unit">ms</span></div></div>
      <div class="result-box"><h3>mcTriton kernel（中位数）</h3><div class="big-num">${p.triton_ms.toFixed(3)}<span class="unit">ms</span></div></div>
      <div class="result-box"><h3>加速比</h3><div class="big-num">${p.speedup.toFixed(2)}<span class="unit">×</span></div></div>
      <div class="result-box"><h3>测试参数</h3><div style="color:var(--dim)">warmup=${p.warmup} · iters=${p.iters} · ${escapeHtml(job.kernel_type_name || job.kernel_type || 'mcTriton')} / ${job.gpu} / ${job.whl}</div></div>`;
    box.appendChild(grid);
  }

  const acc = r.accuracy;
  if (acc?.outputs?.length) {
    const title = document.createElement('h3');
    title.textContent = `精度校验 ${acc.passed ? '✓ 通过' : '✗ 未通过'}（rtol=${acc.rtol}, atol=${acc.atol}）`;
    title.style.color = acc.passed ? 'var(--ok)' : 'var(--err)';
    box.appendChild(title);
    const table = document.createElement('table');
    table.innerHTML = `<tr><th>输出</th><th>shape</th><th>dtype</th><th>max_abs_err</th><th>max_rel_err</th><th>mean_abs_err</th><th>cosine</th><th>allclose</th></tr>` +
      acc.outputs.map((o, i) => `<tr>
        <td>#${i}</td>
        <td>${escapeHtml(JSON.stringify(o.shape || o.shape_ref))}</td>
        <td>${escapeHtml(String(o.dtype || ''))}</td>
        <td>${fmt(o.max_abs_err)}</td>
        <td>${fmt(o.max_rel_err)}</td>
        <td>${fmt(o.mean_abs_err)}</td>
        <td>${fmt(o.cosine_similarity)}</td>
        <td class="${o.allclose ? 'ok-yes' : 'ok-no'}">${o.allclose ? '✓' : '✗'}</td>
      </tr>`).join('');
    box.appendChild(table);
  }

  if (job.generated) {
    addCodeBlock(box, '生成 kernel 代码', job.generated.kernel_code);
    addCodeBlock(box, '测试输入代码', job.generated.inputs_code, true);
  }
}

function renderFailed(job) {
  const box = $('job-result');
  box.innerHTML = '';
  const d = document.createElement('div');
  d.className = 'analysis';
  d.style.color = 'var(--err)';
  d.textContent = `✗ ${job.error || '作业失败'}`;
  box.appendChild(d);
  if (job.generated?.kernel_code) {
    addCodeBlock(box, '最后一次生成的 kernel 代码', job.generated.kernel_code, true);
  }
}

function addCodeBlock(parent, label, code, collapsed = false) {
  const wrap = document.createElement('div');
  wrap.style.marginTop = '10px';
  const head = document.createElement('span');
  head.className = 'details-toggle';
  head.textContent = collapsed ? `▸ ${label}` : `▾ ${label}`;
  const pre = document.createElement('code-block');
  pre.textContent = code;
  pre.hidden = collapsed;
  const copy = document.createElement('button');
  copy.className = 'copy-btn';
  copy.textContent = '复制';
  copy.addEventListener('click', () => {
    navigator.clipboard.writeText(code).then(() => { copy.textContent = '已复制'; setTimeout(() => { copy.textContent = '复制'; }, 1200); });
  });
  pre.appendChild(copy);
  head.addEventListener('click', () => {
    pre.hidden = !pre.hidden;
    head.textContent = `${pre.hidden ? '▸' : '▾'} ${label}`;
  });
  wrap.appendChild(head);
  wrap.appendChild(pre);
  parent.appendChild(wrap);
}

// ---------- 历史 ----------
async function refreshHistory() {
  try {
    const data = await fetch('/api/jobs').then((r) => r.json());
    const box = $('job-history');
    box.innerHTML = '';
    if (!data.jobs.length) {
      box.innerHTML = '<div class="history-empty">暂无任务</div>';
      return;
    }
    for (const job of data.jobs) {
      const item = document.createElement('div');
      item.className = 'history-item';
      const time = new Date(job.created_at).toLocaleString('zh-CN', { hour12: false });
      const summary = job.result?.performance
        ? `加速 ${job.result.performance.speedup.toFixed(2)}×`
        : (job.error ? job.error.slice(0, 60) : '');
      item.innerHTML = `<span class="dot ${job.status}"></span>
        <span>#${job.id}</span>
        <span class="meta">${job.status} · ${time}${summary ? ' · ' + escapeHtml(summary) : ''}</span>`;
      item.addEventListener('click', () => {
        currentJobId = job.id;
        seenEvents = 0;
        $('job-card').hidden = false;
        $('job-title').textContent = `任务 #${job.id}`;
        $('job-log').innerHTML = '';
        $('job-result').innerHTML = '';
        poll();
        if (job.status === 'queued' || job.status === 'running') startPolling();
      });
      box.appendChild(item);
    }
  } catch { /* ignore */ }
}

// ---------- 工具 ----------
function fmt(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') {
    if (Math.abs(v) >= 100 || (v !== 0 && Math.abs(v) < 1e-4)) return v.toExponential(3);
    return v.toFixed(6);
  }
  return String(v);
}
function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

init();
