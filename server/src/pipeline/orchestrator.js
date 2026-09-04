import { config } from '../config.js';
import { loadSkillPrompt } from '../skills.js';
import { LLMClient } from '../llm/client.js';
import { extractJson, validateGeneration } from '../llm/extractor.js';
import { SshSession } from '../runner/ssh-client.js';
import { ensurePython, buildRunCommand, benchmarkDefaults, jobRemoteDir } from '../runner/env.js';
import { buildHarness, parseResult } from '../runner/harness.js';

/** 构建首次生成的 user 消息。 */
export function buildTaskPrompt(job) {
  return [
    '请为以下 torch 参考实现生成等价的 mcTriton kernel，并设计测试输入。',
    '',
    '【目标环境】',
    `- GPU: ${job.gpu || 'MetaX C500'}`,
    `- MACA SDK: ${job.sdk}`,
    `- whl 版本: ${job.whl}`,
    `- 精度要求: rtol=${job.rtol}, atol=${job.atol}`,
    '',
    `【输入说明】${job.inputs_hint ? '\n' + job.inputs_hint : '（未提供，由你设计合理的测试输入）'}`,
    '',
    '【torch 参考实现】',
    '```python',
    job.torch_code,
    '```',
    '',
    '按协议输出 JSON。',
  ].join('\n');
}

/** 构建失败修复的 user 消息。 */
export function buildFixPrompt(phase, errorText) {
  return [
    `上一次生成未通过【${phase}】阶段，请修复后重新输出完整 JSON（不要输出 diff 或解释）。`,
    '',
    '【失败信息】',
    errorText,
  ].join('\n');
}

/**
 * 执行一个作业：连接构建机 → 生成 kernel → 精度校验 → 性能测试。
 * - 失败自动修复重试（LLM 反馈闭环）；
 * - 无输出停滞（stall）或超时会先强杀远端进程（释放 GPU）再自动重试；
 * - 支持用户中断（job.cancelled）。
 * @param {object} job 作业对象（含 events 数组、状态字段）
 */
export async function runJob(job) {
  const cfg = config.build_machine;
  const bench = benchmarkDefaults();
  const emit = (type, message, extra = {}) => {
    job.events.push({ t: Date.now(), type, message: String(message), ...extra });
    if (job.events.length > 800) job.events.splice(0, job.events.length - 800);
  };
  const isCancelled = () => Boolean(job.cancelled);
  const checkCancelled = () => {
    if (!job.cancelled) return;
    job.status = 'cancelled';
    emit('warn', '任务已中断');
  };

  const ssh = new SshSession(cfg);
  try {
    emit('info', `连接构建机 ${cfg.host}:${cfg.port} ...`);
    await ssh.connect();
    emit('info', '构建机连接成功');
    if (job.cancelled) { checkCancelled(); return; }

    const workdir = jobRemoteDir(job.id);
    await ssh.exec(`mkdir -p '${workdir}'`);

    const { python } = await ensurePython(ssh, job.whl, emit, isCancelled);
    if (job.cancelled) { checkCancelled(); return; }

    const messages = [
      { role: 'system', content: loadSkillPrompt() },
      { role: 'user', content: buildTaskPrompt(job) },
    ];

    const llm = new LLMClient({ model: job.model });

    for (let attempt = 0; attempt <= job.maxRetries; attempt++) {
      if (job.cancelled) { checkCancelled(); return; }
      job.attempts.push({ attempt: attempt + 1, phase: 'llm' });
      emit('phase', attempt === 0 ? '① LLM 生成 mcTriton kernel ...' : `① 第 ${attempt} 次修复：LLM 重新生成 ...`);

      let lastProgress = 0;
      const resp = await llm.chat(messages, {
        json: true,
        isCancelled,
        onProgress: (p) => {
          const now = Date.now();
          if (now - lastProgress > 10000) {
            lastProgress = now;
            emit('info', `LLM 生成中…（已输出 ${p.contentLen} 字符代码，推理 ${p.reasoningLen} 字符）`);
          }
        },
      });
      if (job.cancelled) { checkCancelled(); return; }
      emit('llm', `模型 ${job.model} 已回复（${resp.usage?.total_tokens ?? '?'} tokens，推理 ${resp.usage?.completion_tokens_details?.reasoning_tokens ?? 0} tokens）`, {
        tokens: resp.usage?.total_tokens,
        reasoningTokens: resp.usage?.completion_tokens_details?.reasoning_tokens ?? 0,
      });

      let gen;
      try {
        gen = validateGeneration(extractJson(resp.content));
      } catch (e) {
        emit('error', `LLM 回复格式不符合协议: ${e.message}`);
        if (job.cancelled) { checkCancelled(); return; }
        if (attempt >= job.maxRetries) {
          job.status = 'failed';
          job.error = e.message;
          emit('error', '已达最大修复次数，作业失败');
          return;
        }
        messages.push(
          { role: 'assistant', content: resp.content },
          { role: 'user', content: buildFixPrompt('protocol', `回复无法解析为协议 JSON。\n${e.message}\n\n请严格按协议只输出一个 JSON 对象。`) },
        );
        continue;
      }

      if (gen.analysis) emit('info', `模型分析：${gen.analysis}`);
      job.generated = {
        attempt: attempt + 1,
        analysis: gen.analysis,
        kernel_code: gen.kernel_code,
        inputs_code: gen.inputs_code,
      };
      messages.push({ role: 'assistant', content: resp.content });

      // 组装 harness 并上传
      const script = buildHarness({
        userCode: job.torch_code,
        kernelCode: gen.kernel_code,
        inputsCode: gen.inputs_code,
      });
      await ssh.uploadFile(`${workdir}/harness.py`, script);
      await ssh.uploadFile(`${workdir}/generated_kernel.py`, gen.kernel_code);
      await ssh.uploadFile(`${workdir}/reference_torch.py`, job.torch_code);

      const cmd = buildRunCommand({
        python,
        workdir,
        sdkDir: job.sdk,
        rtol: gen.rtol,
        atol: gen.atol,
        warmup: job.warmup ?? bench.warmup,
        iters: job.iters ?? bench.iters,
      });
      // 每次尝试前清空编译缓存，避免上一次异常（崩溃/挂起）残留的脏缓存导致复现
      await ssh.exec(`rm -rf ${workdir}/cache`).catch(() => {});
      emit('phase', `② 构建机编译并运行（${job.whl} @ ${job.gpu}，第 ${attempt + 1} 次尝试）...`);
      emit('info', `无输出停滞超过 ${Math.round((cfg.stall_timeout_ms ?? 240000) / 1000)}s 将自动终止并重试`);
      job.attempts[job.attempts.length - 1].phase = 'run';

      let run;
      try {
        run = await ssh.execStream(cmd, {
          timeoutMs: cfg.run_timeout_ms,
          stallTimeoutMs: cfg.stall_timeout_ms ?? 240000,
          isCancelled,
          pidFile: `${workdir}/run.pid`,
          onData: (t) => emit('log', t),
        });
      } catch (e) {
        if (job.cancelled) { checkCancelled(); return; }
        emit('error', `运行异常: ${e.message}`);
        if (attempt >= job.maxRetries) {
          job.status = 'failed';
          job.error = e.message;
          return;
        }
        messages.push({ role: 'user', content: buildFixPrompt('run', `${e.message}\n（可能原因：kernel 死循环/死锁、编译挂起、共享内存超限；远端进程已被终止，请修复后重试）`) });
        continue;
      }
      if (job.cancelled) { checkCancelled(); return; }

      const result = parseResult(run.stdout);
      if (result && result.ok) {
        job.status = 'success';
        job.result = result;
        job.runLog = run.stdout + run.stderr;
        const perf = result.performance;
        emit('result', perf
          ? `✓ 精度通过。性能：torch ${perf.torch_ms.toFixed(3)} ms vs triton ${perf.triton_ms.toFixed(3)} ms，加速比 ${perf.speedup.toFixed(2)}×`
          : '✓ 精度通过。');
        return;
      }

      // 失败处理
      const phase = result?.phase || 'run';
      let errMsg;
      if (result?.error) {
        errMsg = result.error;
      } else if (run.code === null || run.code === undefined) {
        errMsg = `进程被信号终止${run.signal ? ` (${run.signal})` : ''}（内核崩溃/段错误/被杀死）`;
      } else if (run.code !== 0) {
        errMsg = `进程退出码 ${run.code}`;
      } else {
        errMsg = '无法解析测试结果';
      }
      const accDetail = result?.accuracy?.outputs
        ? `\n[精度指标]\n${JSON.stringify(result.accuracy.outputs, null, 2)}`
        : '';
      const detail = [
        `阶段: ${phase}`,
        `错误: ${errMsg}`,
        accDetail,
        result?.traceback ? `\n${result.traceback}` : '',
        run.stderr ? `\n[stderr]\n${run.stderr}` : '',
        run.stdout ? `\n[stdout 尾部]\n${run.stdout.slice(-4000)}` : '',
      ].join('\n').slice(0, 12000);
      emit('error', `✗ 失败于【${phase}】阶段：${errMsg}`);
      if (phase === 'reference') {
        emit('warn', '参考实现本身运行失败——请检查输入的 torch 代码是否完整可运行。');
      }
      if (job.cancelled) { checkCancelled(); return; }
      if (attempt >= job.maxRetries) {
        job.status = 'failed';
        job.error = `最终失败于 ${phase} 阶段: ${errMsg}`;
        job.runLog = run.stdout + run.stderr;
        emit('error', '已达最大修复次数，作业失败');
        return;
      }
      messages.push({ role: 'user', content: buildFixPrompt(phase, detail) });
    }
  } catch (e) {
    if (job.cancelled) {
      job.status = 'cancelled';
      emit('warn', '任务已中断');
    } else {
      job.status = 'failed';
      job.error = e.message;
      emit('error', `作业异常终止: ${e.message}`);
    }
  } finally {
    ssh.close();
  }
}
