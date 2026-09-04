import { config } from '../config.js';
import { shq } from './ssh-client.js';

/** 展开构建机上的 ~ 路径（远程 home 由 username 决定，这里约定为 /home/<user>）。 */
export function expandRemote(user, p) {
  if (typeof p !== 'string') return p;
  if (p === '~') return `/home/${user}`;
  if (p.startsWith('~/')) return `/home/${user}/${p.slice(2)}`;
  return p;
}

/**
 * 确定某个 whl 版本使用的 Python 解释器。
 * - 若配置了 python_overrides[whl]：直接用该解释器（如已有的 conda 环境）。
 * - 否则在 venvs_dir/<whl> 创建 venv 并安装该版本的 torch/triton wheel。
 * 返回 { python, kind: 'override'|'venv' }。
 */
export async function ensurePython(ssh, whl, emit, isCancelled = () => false, sdkDir = null) {
  const cfg = config.build_machine;
  const user = cfg.username;
  const override = cfg.python_overrides?.[whl];
  if (override) {
    const p = expandRemote(user, override);
    emit('info', `使用解释器: ${override}`);
    const r = await ssh.exec(`${shq(p)} --version`);
    if (r.code !== 0) throw new Error(`解释器不可用: ${override}\n${r.stderr}`);
    return { python: p, kind: 'override' };
  }

  const venvDir = expandRemote(user, `${cfg.venvs_dir}/${sanitize(whl)}`);
  const marker = `${venvDir}/.poseidon-ready`;
  const check = await ssh.exec(`test -f ${shq(marker)} && echo READY || echo MISSING`);
  if (check.stdout.includes('READY')) {
    emit('info', `venv 已就绪: ${venvDir}`);
    return { python: `${venvDir}/bin/python`, kind: 'venv' };
  }

  emit('phase', `首次使用 whl 版本 ${whl}：创建 venv 并安装 torch/triton（可能需要几分钟）...`);
  const basePython = expandRemote(user, cfg.base_python);
  const wheelDir = `${expandRemote(user, cfg.whl_dir)}/${whl}/wheel`;
  const sdk = sdkDir && sdkDir !== 'auto' ? sdkDir : (cfg.sdk_dirs?.[0] || '/opt/maca');

  // venv 复用 base_python 所在环境的 site-packages（构建机通常无外网 pip 源），
  // 依赖（numpy/sympy/jinja2/fsspec/filelock/typing_extensions 等）来自该系统环境；
  // 本版本 torch/triton 用本地 wheel 离线安装（--no-index --no-deps），venv 内的包优先于系统包。
  const steps = [
    `test -x ${shq(`${venvDir}/bin/python`)} || (rm -rf ${shq(venvDir)} && ${shq(basePython)} -m venv --system-site-packages ${shq(venvDir)})`,
    `${shq(`${venvDir}/bin/pip`)} install --no-cache-dir --no-index --no-deps ${shq(wheelDir)}/torch-*.whl ${shq(wheelDir)}/triton-*.whl`,
    `export MACA_PATH=${shq(sdk)}; export LD_LIBRARY_PATH=${shq(`${sdk}/lib:${sdk}/mxgpu_llvm/lib:${sdk}/ompi/lib`)}:$LD_LIBRARY_PATH; ${shq(`${venvDir}/bin/python`)} -c "import torch, triton; print('venv OK torch', torch.__version__, 'triton', triton.__version__)"`,
    `touch ${shq(marker)}`,
  ];
  for (const cmd of steps) {
    if (isCancelled()) throw new Error('任务已中断');
    emit('log', `$ ${cmd}`);
    const r = await ssh.execStream(cmd, { timeoutMs: 1800000, isCancelled, onData: (t) => emit('log', t) });
    if (r.code !== 0) {
      throw new Error(`venv 安装失败 (${whl})，退出码 ${r.code}。请检查 whl 目录 ${wheelDir} 与 pip 源配置。`);
    }
  }
  emit('info', `venv 安装完成: ${venvDir}`);
  return { python: `${venvDir}/bin/python`, kind: 'venv' };
}

/** 构建运行命令（注入 MACA 环境变量；后台化并记录 pid，便于看门狗/中断时强杀远端进程）。 */
export function buildRunCommand({ python, workdir, sdkDir, rtol, atol, warmup, iters }) {
  const cfg = config.build_machine;
  const sdk = sdkDir && sdkDir !== 'auto' ? sdkDir : '/opt/maca';
  const libPath = `${sdk}/lib:${sdk}/mxgpu_llvm/lib:${sdk}/ompi/lib`;
  const env = [
    `export MACA_PATH=${shq(sdk)}`,
    `export LD_LIBRARY_PATH=${shq(libPath)}:$LD_LIBRARY_PATH`,
    `export TRITON_CACHE_DIR=${shq(`${workdir}/cache`)}`,
    `export TRITON_METAX_ENABLE_TORCH_REDUCTION_ORDER=1`,
    // autotune：调优过程打印进度（保持看门狗活性）+ 结果持久化（跳过重复调优；3.0 支持，3.6 构建忽略该变量）
    `export TRITON_PRINT_AUTOTUNING=1`,
    `export TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS=1`,
    `export TRITON_AUTOTUNE_CONFIG_PATH=${shq(`${workdir}/autotune_configs`)}`,
    `export POSEIDON_RTOL=${rtol}`,
    `export POSEIDON_ATOL=${atol}`,
    `export POSEIDON_WARMUP=${warmup}`,
    `export POSEIDON_ITERS=${iters}`,
  ].join(' && ');
  // ( cd && export ... && exec python ) &  整体在单个子 shell 中后台运行；
  // exec 使 python 直接取代该子 shell，因此 $!（写入 run.pid）就是 python 进程本身，
  // 看门狗/中断时 kill $(cat run.pid) 可精准命中。wait $! 使本命令随 python 退出而结束。
  return `( cd ${shq(workdir)} && ${env} && exec ${shq(python)} -u harness.py ) & echo $! > ${shq(`${workdir}/run.pid`)}; wait $!`;
}

function sanitize(name) {
  return String(name).replace(/[^a-zA-Z0-9._-]/g, '_');
}

/** 默认基准参数（从配置读取）。 */
export function benchmarkDefaults() {
  return {
    warmup: config.benchmark.warmup,
    iters: config.benchmark.iters,
    default_rtol: config.benchmark.default_rtol,
    default_atol: config.benchmark.default_atol,
  };
}

/** 作业在构建机上的目录。 */
export function jobRemoteDir(jobId) {
  const cfg = config.build_machine;
  return `${expandRemote(cfg.username, cfg.workdir)}/${sanitize(jobId)}`;
}
