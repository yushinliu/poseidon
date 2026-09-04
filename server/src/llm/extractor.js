/**
 * 从 LLM 回复中稳健地提取 JSON 对象。
 * 优先直接 JSON.parse；失败则尝试提取 ```json 围栏 / 首个平衡的 {...} 块。
 */
export function extractJson(text) {
  if (!text) throw new Error('LLM 返回空内容');
  let s = String(text).trim();

  // 1) 直接解析
  try { return JSON.parse(s); } catch { /* continue */ }

  // 2) markdown 围栏
  const fence = s.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fence) {
    try { return JSON.parse(fence[1].trim()); } catch { /* continue */ }
  }

  // 3) 首个平衡 { ... } 块
  const start = s.indexOf('{');
  if (start >= 0) {
    let depth = 0;
    let inStr = false;
    let esc = false;
    for (let i = start; i < s.length; i++) {
      const ch = s[i];
      if (inStr) {
        if (esc) esc = false;
        else if (ch === '\\') esc = true;
        else if (ch === '"') inStr = false;
        continue;
      }
      if (ch === '"') { inStr = true; continue; }
      if (ch === '{') depth++;
      else if (ch === '}') {
        depth--;
        if (depth === 0) {
          try { return JSON.parse(s.slice(start, i + 1)); } catch { break; }
        }
      }
    }
  }
  throw new Error(`无法从 LLM 回复中解析 JSON，回复开头: ${s.slice(0, 300)}`);
}

/** 校验生成结果 JSON 的结构（含单 kernel 约束）。 */
export function validateGeneration(gen) {
  if (!gen || typeof gen !== 'object') throw new Error('生成结果不是 JSON 对象');
  if (typeof gen.kernel_code !== 'string' || !gen.kernel_code.trim()) throw new Error('缺少 kernel_code 字段');
  if (typeof gen.inputs_code !== 'string' || !gen.inputs_code.trim()) throw new Error('缺少 inputs_code 字段');
  if (!/def\s+run_kernel\s*\(/.test(gen.kernel_code)) throw new Error('kernel_code 中缺少 run_kernel 函数');
  if (!/def\s+make_inputs\s*\(/.test(gen.inputs_code)) throw new Error('inputs_code 中缺少 make_inputs 函数');

  // 一个 torch_fn 只允许对应一个 triton kernel
  const jitCount = (gen.kernel_code.match(/@triton\.jit\b/g) || []).length;
  if (jitCount === 0) throw new Error('kernel_code 中未找到 @triton.jit 内核');
  if (jitCount > 1) throw new Error(`一个 torch_fn 只能对应一个 triton kernel（检测到 ${jitCount} 个 @triton.jit），请合并为单个内核`);

  // 禁止 kernel + torch 函数混用：kernel_code 中不得出现 torch 计算调用
  const torchCompute = /torch\.(matmul|mm|bmm|addmm|sum|mean|std|var|softmax|gelu|relu|silu|sigmoid|add|sub|mul|div|exp|log|pow|sqrt|rsqrt|tanh|max|min|abs|sin|cos|clamp|cat|stack|squeeze|unsqueeze|permute|transpose|flip|roll)\s*\(|torch\.nn\.functional\.|import\s+torch\.nn\b/.test(gen.kernel_code);
  if (torchCompute) throw new Error('kernel_code 中不允许使用 torch 计算函数（一个 torch_fn 必须只对应一个 triton kernel，禁止 kernel + torch 函数混用）');

  return {
    analysis: String(gen.analysis || '').slice(0, 2000),
    kernel_code: gen.kernel_code,
    inputs_code: gen.inputs_code,
    rtol: Number.isFinite(Number(gen.rtol)) ? Number(gen.rtol) : 0.02,
    atol: Number.isFinite(Number(gen.atol)) ? Number(gen.atol) : 0.02,
  };
}
