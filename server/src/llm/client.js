import { config } from '../config.js';

/** DeepSeek OpenAI 兼容客户端（Node 内置 fetch，支持流式输出与中断）。 */
export class LLMClient {
  constructor(opts = {}) {
    this.baseUrl = (opts.base_url || config.llm.base_url).replace(/\/+$/, '');
    this.apiKey = opts.api_key || config.llm.api_key;
    this.model = opts.model || config.llm.model;
    this.maxTokens = opts.max_tokens ?? config.llm.max_tokens;
    this.temperature = opts.temperature ?? config.llm.temperature;
    this.timeoutMs = opts.timeout_ms ?? config.llm.timeout_ms;
  }

  /**
   * 发起一次对话补全（流式）。
   * @param {Array<{role:string, content:string}>} messages
   * @param {{json?: boolean, isCancelled?: () => boolean, onProgress?: (p: {contentLen:number, reasoningLen:number}) => void}} [options]
   * @returns {Promise<{content: string, reasoning: string, usage: object}>}
   */
  async chat(messages, options = {}) {
    if (!this.apiKey) throw new Error('未配置 LLM API Key（config.yaml 的 llm.api_key 或环境变量 POSEIDON_LLM_API_KEY）');
    const body = {
      model: this.model,
      messages,
      max_tokens: this.maxTokens,
      temperature: this.temperature,
      stream: true,
    };
    if (options.json) body.response_format = { type: 'json_object' };

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    let cancelWatch = null;
    if (options.isCancelled) {
      cancelWatch = setInterval(() => {
        if (options.isCancelled()) controller.abort();
      }, 1000);
    }

    try {
      const res = await fetch(`${this.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!res.ok) {
        const text = await res.text();
        let data = null;
        try { data = JSON.parse(text); } catch { /* ignore */ }
        const errMsg = data?.error?.message || data?.error || `HTTP ${res.status}`;
        if (controller.signal.aborted && options.isCancelled?.()) throw new Error(`LLM 调用已中断`);
        throw new Error(`LLM 调用失败 (${this.model}): ${errMsg}`);
      }
      if (!res.body) throw new Error('LLM 响应无内容流');

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let content = '';
      let reasoning = '';
      let usage = {};
      let buf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, idx).trim();
          buf = buf.slice(idx + 1);
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (payload === '[DONE]') continue;
          try {
            const chunk = JSON.parse(payload);
            const delta = chunk.choices?.[0]?.delta || {};
            if (delta.content) content += delta.content;
            if (delta.reasoning_content) reasoning += delta.reasoning_content;
            if (chunk.usage) usage = chunk.usage;
            options.onProgress?.({ contentLen: content.length, reasoningLen: reasoning.length });
          } catch { /* 忽略无法解析的 chunk */ }
        }
      }
      return { content, reasoning, usage };
    } catch (e) {
      if (e.name === 'AbortError') {
        if (options.isCancelled?.()) throw new Error('LLM 调用已中断');
        throw new Error(`LLM 调用超时 (${Math.round(this.timeoutMs / 1000)}s)`);
      }
      throw e;
    } finally {
      clearTimeout(timer);
      if (cancelWatch) clearInterval(cancelWatch);
    }
  }
}
