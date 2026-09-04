import { config } from '../config.js';

/** DeepSeek OpenAI 兼容客户端（Node 内置 fetch）。 */
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
   * 发起一次对话补全。
   * @param {Array<{role:string, content:string}>} messages
   * @param {{json?: boolean}} [options] json=true 时使用 json_object 输出模式
   * @returns {Promise<{content: string, reasoning: string, usage: object}>}
   */
  async chat(messages, options = {}) {
    if (!this.apiKey) throw new Error('未配置 LLM API Key（config.yaml 的 llm.api_key 或环境变量 POSEIDON_LLM_API_KEY）');
    const body = {
      model: this.model,
      messages,
      max_tokens: this.maxTokens,
      temperature: this.temperature,
      stream: false,
    };
    if (options.json) body.response_format = { type: 'json_object' };

    const url = `${this.baseUrl}/chat/completions`;
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeoutMs),
    });

    const text = await res.text();
    let data;
    try { data = JSON.parse(text); } catch { data = null; }
    if (!res.ok) {
      const errMsg = data?.error?.message || data?.error || `HTTP ${res.status}`;
      throw new Error(`LLM 调用失败 (${this.model}): ${errMsg}`);
    }
    const msg = data?.choices?.[0]?.message || {};
    return {
      content: msg.content ?? '',
      reasoning: msg.reasoning_content ?? '',
      usage: data?.usage ?? {},
    };
  }
}
