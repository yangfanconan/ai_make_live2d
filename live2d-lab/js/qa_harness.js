/* Live2D QA Harness — 无 UI 自治扫描页（QA Director 的浏览器执行端）
 *
 * 三种扫描模式（由 plan.json 驱动）：
 *   static      逐 pose 复位 → 写参 → settle → 截帧（极值姿势/表情）
 *   trajectory  按时间线逐帧插值写参，10Hz + 源帧时刻截帧（多参数耦合穿帮）
 *   motion      真实播放已注册的 motion3.json，定时截帧（开放格式产物验收）
 *
 * 与 capture_server.py 协作：GET plan / POST frame / POST done|error。
 * 截帧纪律：preserveDrawingBuffer + 显式 render；加载后等 5s 纹理上传；
 * 每帧先全量复位 default 再覆写目标参数，防串态。
 */
(function () {
  'use strict';

  const CAP = new URLSearchParams(location.search).get('cap') || 'http://127.0.0.1:8900';
  const JOB = new URLSearchParams(location.search).get('job');
  const hud = document.getElementById('hud');
  const big = document.getElementById('big');
  const stageEl = document.getElementById('stage');

  const log = (msg) => { hud.textContent = msg; console.log('[harness]', msg); };
  const fail = (msg) => {
    hud.classList.add('err'); hud.textContent = 'ERROR: ' + msg;
    console.error('[harness]', msg);
  };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function postJSON(path, obj, retry) {
    for (let i = 0; i <= (retry || 0); i++) {
      try {
        const resp = await fetch(CAP + path, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(obj),
        });
        if (resp.ok) return await resp.json();
        throw new Error('HTTP ' + resp.status);
      } catch (e) {
        if (i === (retry || 0)) throw e;
        await sleep(300);
      }
    }
  }

  // Cubism 标准参数量程（cdi3 不给量程时的兜底，与 arkit_map.STANDARD_RANGES 对齐）
  const STD = {
    ParamAngleX: [-30, 30, 0], ParamAngleY: [-30, 30, 0], ParamAngleZ: [-30, 30, 0],
    ParamBodyAngleX: [-10, 10, 0], ParamBodyAngleY: [-10, 10, 0], ParamBodyAngleZ: [-10, 10, 0],
    ParamEyeLOpen: [0, 1, 1], ParamEyeROpen: [0, 1, 1],
    ParamEyeLSmile: [0, 1, 0], ParamEyeRSmile: [0, 1, 0],
    ParamEyeBallX: [-1, 1, 0], ParamEyeBallY: [-1, 1, 0],
    ParamBrowLY: [-1, 1, 0], ParamBrowRY: [-1, 1, 0],
    ParamMouthOpenY: [0, 1, 0], ParamMouthForm: [-1, 1, 0],
    ParamBreath: [0, 1, 0],
  };

  let app = null, model = null, modelJson = null, murlNoQuery = null;
  let RW = 900, RH = 700;       // 固定渲染缓冲尺寸（不依赖标签页是否可见/布局尺寸）
  let ptable = new Map();      // id -> {min,max,def,index}
  let heldParams = null;       // 非 null 时每帧写入（static/trajectory）
  const captured = [];
  const dropped = new Set();
  const meshDumps = [];

  function buildParamTable() {
    const core = model.internalModel.coreModel;
    ptable = new Map();
    // classic Cubism Core API：getParameterCount() + 按索引单数取值器；
    // id 列表无公开访问器，用该固定 vendor 构建里的 _parameterIds（framework 同样依赖它）
    const n = core.getParameterCount();
    const ids = core._parameterIds || [];
    const call = (fn, fb) => {
      try { const v = fn(); return v == null || Number.isNaN(v) ? fb : v; }
      catch (e) { return fb; }
    };
    for (let i = 0; i < n; i++) {
      const id = ids[i];
      if (!id) continue;
      const std = STD[id] || [-1, 1, 0];
      ptable.set(id, {
        index: i,
        min: call(() => core.getParameterMinimumValue(i), std[0]),
        max: call(() => core.getParameterMaximumValue(i), std[1]),
        def: call(() => core.getParameterDefaultValue(i), std[2]),
      });
    }
  }

  // 写入当前 heldParams：先全量复位 default 再覆写（截帧前与每帧 ticker 都调，保证最后写入者是我们）
  function applyHeld() {
    if (heldParams === null) return;
    const core = model.internalModel.coreModel;
    ptable.forEach((p, id) => core.setParameterValueById(id, p.def));
    for (const [id, v] of Object.entries(heldParams)) {
      const cv = clampv(id, v);
      if (cv !== null) core.setParameterValueById(id, cv);
    }
  }

  function layoutModel() {
    const w = RW, h = RH;
    const oh = model.internalModel.originalHeight || 1200;
    const ow = model.internalModel.originalWidth || 900;
    const s = (h * 0.92) / oh;
    model.scale.set(s);
    model.x = (w - ow * s) / 2;
    model.y = (h - oh * s) / 2 + h * 0.015;
  }

  // 固定尺寸离屏式画布：即便自动化浏览器标签 offscreen / 视口 0 尺寸也能正常截帧
  function initRenderer(w, h) {
    RW = w; RH = h;
    const a = new PIXI.Application({
      width: w, height: h, backgroundAlpha: 0, antialias: true,
      autoDensity: false, preserveDrawingBuffer: true,
    });
    a.view.style.cssText = 'width:100vw;height:100vh;display:block;';
    stageEl.appendChild(a.view);
    return a;
  }

  const clampv = (id, v) => {
    const p = ptable.get(id);
    if (!p) { dropped.add(id); return null; }
    return Math.max(p.min, Math.min(p.max, v));
  };

  async function holdPose(params, settleMs) {
    heldParams = params || {};
    await sleep(settleMs || 600);
  }

  async function captureFrame(name, meta, forceUpdate = true) {
    applyHeld();
    // 写后读回：确认值真的进了 core（诊断用）
    const readback = {};
    if (heldParams) {
      for (const id of Object.keys(heldParams)) {
        try { readback[id] = model.internalModel.coreModel.getParameterValueById(id); }
        catch (e) { readback[id] = null; }
      }
    }
    // 不在此处手动 core.update()：framework 的 update 循环包含
    // saveParameters→pose/focus→core.update→loadParameters，手动二次求值输入不等价。
    // 参数通过 'beforeModelUpdate' 事件钩子在 framework 自己的 core.update 前注入，
    // 这里只需等待一帧让 shared ticker 求值后手动取画。
    app.renderer.render(app.stage);
    const data = app.view.toDataURL('image/png');
    if (!data || data === 'data:,' || data.length < 256) {
      throw new Error(`截帧为空（canvas ${RW}x${RH}，dataURL ${data.length} 字符）：${name}`);
    }
    await postJSON(`/job/${JOB}/frame`, { name, data, params: meta || null, readback }, 3);
    captured.push(name);
  }

  function releaseDrive() { heldParams = null; }

  /* ---------- static ---------- */
  async function runStatic(poses, settleMs) {
    for (const pose of poses || []) {
      log(`static: ${pose.name}`);
      await holdPose(pose.params, settleMs);
      await captureFrame(pose.name, pose.params);
    }
    // 收尾再拍一张中性帧：首帧 rest 可能受加载后首拍时序影响，rest2 供判定器择优
    if ((poses || []).some((p) => p.name === 'rest')) {
      log('static: rest2 (warm)');
      await holdPose({}, settleMs);
      await captureFrame('rest2', {});
    }
  }

  /* ---------- mesh（keyform 顶点位移 ground-truth 采集） ---------- */
  // 直接读 Cubism Core 求值后的变形顶点（core Model.drawables.vertexPositions，
  // 即 csmGetDrawableVertexPositions 的活缓冲），与全默认参数基线之差即该姿势的位移场。
  // 底层 core Model 实例（pixi CubismModel 把原始 core Model 存在 _model 上）
  function coreModelObj() {
    try {
      const c = model.internalModel.coreModel;
      return c._model || ((typeof c.getModel === 'function') ? c.getModel() : null);
    } catch (e) { return null; }
  }

  // 诊断：确认访问器与画布信息（PixelsPerUnit 为正确的大小写属性名）
  function meshDiag() {
    const core = model.internalModel.coreModel;
    const coreObj = coreModelObj();
    const dr = coreObj && coreObj.drawables;
    const ci = coreObj && coreObj.canvasinfo;
    const out = {
      wrapperAccessor: typeof core.getDrawableVertexPositions,
      wrapperUpdate: typeof core.update,
      coreModel: coreObj ? 'yes' : 'no',
      drawables: dr ? {
        count: dr.count,
        isArray: Array.isArray(dr.vertexPositions),
        v0len: (dr.vertexPositions && dr.vertexPositions[0] && dr.vertexPositions[0].length) || 0,
      } : null,
      canvasinfo: ci ? { Width: ci.Width, Height: ci.Height, PixelsPerUnit: ci.PixelsPerUnit } : null,
      getDrawableCount: typeof core.getDrawableCount,
      paramReadback: null, posA: null,
    };
    try { out.paramReadback = core.getParameterValueById('ParamAngleX'); } catch (e) {}
    if (coreObj && dr) out.coreModelKeys = Object.keys(coreObj).slice(0, 16);
    const head = (p) => (p && p.length) ? Array.prototype.slice.call(p, 0, 4) : null;
    try { out.posA = head(core.getDrawableVertexPositions(0)); } catch (e) { out.posA = 'err:' + e; }
    return out;
  }

  function readDrawables() {
    const core = model.internalModel.coreModel;
    const coreObj = coreModelObj();
    const dr = coreObj && coreObj.drawables;
    const n = (dr && dr.count) ||
      ((typeof core.getDrawableCount === 'function') ? core.getDrawableCount() : 0);
    const ids = (dr && dr.ids) || core._drawableIds || [];
    const out = [];
    for (let i = 0; i < n; i++) {
      let pos = null, src = null;
      // 主路径：core Drawables 的活缓冲（csmGetDrawableVertexPositions，变形顶点）
      if (dr && dr.vertexPositions) {
        try { pos = dr.vertexPositions[i]; src = 'core'; } catch (e) { pos = null; }
      }
      if (!pos && typeof core.getDrawableVertexPositions === 'function') {
        try { pos = core.getDrawableVertexPositions(i); src = 'fw'; } catch (e) { pos = null; }
      }
      if (!pos || !pos.length) continue;
      // 必须拷贝：活缓冲区随 update 变化，持有引用会让 rest 基线一起变（差值恒 0）
      out.push({ i, id: ids[i] || ('drawable_' + i), pos: new Float32Array(pos), src });
    }
    return out;
  }

  // 核心级单步求值：复位默认 → 写目标参数 → core.update() → 读变形顶点。
  // 走的是 pixi CubismModel.update()（只调 core 的 updateModel + resetDynamicFlags，
  // 不含 framework 的 pose/motion/物理），且整段同步执行、无 await 间隙，
  // 因此后台标签的 rAF 节流不会让 ticker 停摆影响采集。
  function meshStep(params) {
    const core = model.internalModel.coreModel;
    ptable.forEach((p, id) => core.setParameterValueById(id, p.def));
    for (const [id, v] of Object.entries(params || {})) {
      const cv = clampv(id, v);
      if (cv !== null) core.setParameterValueById(id, cv);
    }
    core.update();
    return readDrawables();
  }

  async function runMesh(poses, settleMs) {
    const core = model.internalModel.coreModel;
    releaseDrive();                            // 不让 beforeModelUpdate 钩子抢写参数
    model.internalModel.motionManager.stopAllMotions();
    await sleep(settleMs || 300);
    const rest = meshStep({});                 // 全默认参数基线（已拷贝活缓冲）
    if (!rest.length) throw new Error('mesh: 无法读回顶点（core Model/drawables 不可达）');
    const probe = meshDiag();
    const ci = probe.canvasinfo || {};
    const oh = model.internalModel.originalHeight || 4500;
    const ow = model.internalModel.originalWidth || 2400;
    // 屏幕像素/模型单位 = PixelsPerUnit × 渲染缩放（layoutModel 的 s = RH*0.92/oh）
    const ppu = ci.PixelsPerUnit || (oh / (ci.Height || 1));
    const unitPx = ppu * ((RH * 0.92) / oh);
    // 门限按模型自身尺度取：原生坐标多为 ±1 量级，固定阈值会误杀全部网格
    let extent = 1e-6;
    for (const d of rest) for (let j = 0; j < d.pos.length; j++) {
      const a = Math.abs(d.pos[j]); if (a > extent) extent = a;
    }
    const thrN = extent * 1e-4;
    log(`mesh: 基线 ${rest.length} drawable(src=${rest[0] && rest[0].src}) extent=${extent.toFixed(4)} `
        + `ppu=${ppu.toFixed(1)} unit_px=${unitPx.toFixed(2)} canvas=${ci.Width}×${ci.Height} `
        + `probe=${JSON.stringify(probe).slice(0, 240)}`);
    for (const pose of poses || []) {
      const cur = meshStep(pose.params);       // 同步：写参 → core.update → 读顶点
      const rb = {};
      for (const id of Object.keys(pose.params || {})) {
        try { rb[id] = core.getParameterValueById(id); } catch (e) { rb[id] = null; }
      }
      const deltas = [];
      let maxAbs = 0;
      for (let k = 0; k < rest.length && k < cur.length; k++) {
        const r = rest[k], c = cur[k];
        if (r.pos.length !== c.pos.length) continue;
        const d = new Array(c.pos.length);
        let mx = 0;
        for (let j = 0; j < c.pos.length; j++) {
          const dv = c.pos[j] - r.pos[j];
          d[j] = Math.round(dv * 1e6) / 1e6;
          const a = Math.abs(dv);
          if (a > mx) mx = a;
        }
        if (mx < thrN) continue;                // 数值死网格丢弃
        if (mx > maxAbs) maxAbs = mx;
        deltas.push({ i: c.i, id: c.id, n: c.pos.length / 2, max: +mx.toFixed(6), d });
      }
      await postJSON(`/job/${JOB}/mesh`, {
        name: pose.name, params: pose.params, readback: rb,
        drawable_total: rest.length, mesh_total: deltas.length, src: rest[0] && rest[0].src,
        max_abs_native: +maxAbs.toFixed(6), max_abs_px: +(maxAbs * unitPx).toFixed(2),
        extent, unit_px: unitPx, pixels_per_unit: ppu,
        canvas: { w: ci.Width, h: ci.Height }, original_width: ow, original_height: oh,
        thr_native: thrN,
        probe: (meshDumps.length === 0 ? probe : null),
        drawables: deltas,
      }, 3);
      meshDumps.push(pose.name);
      log(`mesh: ${pose.name} → ${deltas.length}/${rest.length} drawable，`
          + `最大位移 ${(maxAbs * unitPx).toFixed(2)}px（原生 ${maxAbs.toFixed(5)}）`);
    }
  }

  // 表达式直接按开放格式 .exp3.json 施加（不经过 framework 播放，保证确定性）
  async function runExpressions(settleMs) {
    const list = (modelJson.FileReferences && modelJson.FileReferences.Expressions) || [];
    const modelBase = murlNoQuery.substring(0, murlNoQuery.lastIndexOf('/'));
    for (let i = 0; i < list.length; i++) {
      const ref = list[i];
      const nm = ref.Name || `F${i + 1}`;
      log(`expression: ${nm}`);
      const expJson = await (await fetch(`${modelBase}/${ref.File}?t=${Date.now()}`)).json();
      const params = {};
      for (const p of expJson.Parameters || []) {
        const info = ptable.get(p.Id);
        if (!info) { dropped.add(p.Id); continue; }
        let v;
        if (p.Blend === 'Add') v = info.def + (p.Value || 0);
        else if (p.Blend === 'Multiply') v = info.def * (p.Value || 1);
        else v = p.Value; // Overwrite
        const cv = clampv(p.Id, v);
        if (cv !== null) params[p.Id] = cv;
      }
      await holdPose(params, settleMs);
      await captureFrame('exp_' + nm, { expression: nm });
    }
  }

  /* ---------- trajectory ---------- */
  function interp(frames, t) {
    if (t <= frames[0].t) return frames[0].params;
    if (t >= frames[frames.length - 1].t) return frames[frames.length - 1].params;
    for (let i = 0; i < frames.length - 1; i++) {
      const a = frames[i], b = frames[i + 1];
      if (t >= a.t && t <= b.t) {
        const k = (t - a.t) / Math.max(b.t - a.t, 1e-6);
        const keys = new Set([...Object.keys(a.params), ...Object.keys(b.params)]);
        const out = {};
        keys.forEach((id) => { out[id] = (a.params[id] ?? 0) * (1 - k) + (b.params[id] ?? 0) * k; });
        return out;
      }
    }
    return frames[frames.length - 1].params;
  }

  async function runTrajectory(tr, settleMs) {
    const frames = tr.frames.sort((a, b) => a.t - b.t);
    // 前向填充：某帧缺写的参数继承之前最近关键帧的值（而非插值时当 0 滑落）
    let carry = {};
    for (const f of frames) { carry = { ...carry, ...f.params }; f.params = { ...carry }; }
    const dur = frames[frames.length - 1].t;
    // 采样时刻：10Hz 网格 + 源帧时刻（极值不漏）
    const times = new Set();
    for (let t = 0; t <= dur + 1e-6; t += 0.1) times.add(Math.round(t * 1000) / 1000);
    frames.forEach((f) => times.add(f.t));
    const schedule = [...times].sort((a, b) => a - b);
    const t0 = performance.now();
    for (const t of schedule) {
      const target = t0 + t * 1000;
      while (performance.now() < target) await sleep(10);
      heldParams = interp(frames, t);
      app.renderer.render(app.stage); // 立刻按该姿态渲染
      await captureFrame(`${tr.name}_t${String(t.toFixed(2)).replace('.', 'p')}`, heldParams);
    }
    await sleep(settleMs);
  }

  /* ---------- motion ---------- */
  async function runMotion(m, settleMs) {
    releaseDrive();
    model.internalModel.motionManager.stopAllMotions();
    const ptable0 = {};
    ptable.forEach((p, id) => { ptable0[id] = p.def; });
    heldParams = ptable0;                 // 未被曲线控制的参数保持中性
    await sleep(100);
    heldParams = null;                    // 交还 motionManager
    log(`motion: ${m.group}#${m.index} ${m.name || ''}`);
    await model.motion(m.group, m.index);
    const dur = (m.dur || 4) * 1000;
    const t0 = performance.now();
    let k = 0;
    while (performance.now() - t0 < dur + settleMs) {
      await sleep(200);
      await captureFrame(`${m.name || 'motion'}_${String(k).padStart(2, '0')}`,
                         { motion: `${m.group}#${m.index}`, k }, false);
      k++;
    }
  }

  /* ---------- 主流程 ---------- */
  async function main() {
    if (!JOB) { fail('缺少 ?job= 参数'); return; }
    log('拉取 plan: ' + JOB);
    const plan = await (await fetch(`${CAP}/job/${JOB}/plan.json`)).json();

    app = initRenderer(plan.width || 900, plan.height || 700);

    murlNoQuery = plan.modelUrl;
    const sep = plan.modelUrl.includes('?') ? '&' : '?';
    const murl = plan.modelUrl + `${sep}t=${Date.now()}`;
    log('加载模型: ' + murl);
    model = await PIXI.live2d.Live2DModel.from(murl, { autoInteract: false });
    app.stage.addChild(model);
    layoutModel();
    modelJson = await (await fetch(murl)).json();
    buildParamTable();
    log(`模型就绪：${ptable.size} 个运行时参数`);

    // 确定性：停 motion、关物理/自动眨眼/呼吸，一切驱动由本页控制
    model.internalModel.motionManager.stopAllMotions();
    // 关键：framework 会在每次更新后自动请求 Idle 组动作（含部件透明度切换，
    // 如抱臂），stopAllMotions 下一帧又会重启。清空 idle 组映射才能真正停住。
    // motion 模式显式 startMotion(指定组) 不受影响。
    try { model.internalModel.motionManager.groups.idle = null; } catch (e) {}
    try { model.internalModel.eyeBlink = null; } catch (e) {}
    try { model.internalModel.breath = null; } catch (e) {}
    // 参数注入点：framework 在自己的 core.update() 前一刻触发本事件，
    // 此时写入的参数会被本次求值渲染（避开 save/load 与双 ticker 时序坑）
    model.internalModel.on('beforeModelUpdate', applyHeld);
    if (!plan.physics) {
      model.internalModel.savedPhysics = model.internalModel.physics;
      model.internalModel.physics = null;
    }

    await sleep(plan.load_wait_ms || 5000); // 纹理上传 GPU 等待（黑画布高发窗口）
    const settle = plan.settle_ms || 600;

    // 预热归一化：以全默认参数驱动若干帧，消掉加载后首拍的残留姿态/物理瞬态
    heldParams = {};
    await sleep(800);

    if (plan.mode === 'mesh') {
      await runMesh(plan.poses, settle);
    } else {
      if (plan.mode === 'static' || plan.poses) await runStatic(plan.poses, settle);
      if (plan.expressions !== false && plan.mode !== 'trajectory' && plan.mode !== 'motion') {
        await runExpressions(settle);
      }
    }
    for (const tr of plan.trajectories || []) await runTrajectory(tr, settle);
    for (const m of plan.motions || []) await runMotion(m, settle);

    const runtimeParams = [...ptable.entries()].map(([id, p]) => (
      { id, min: p.min, max: p.max, def: p.def }));
    await postJSON(`/job/${JOB}/done`, {
      frames: captured,
      meshes: meshDumps,
      manifest: {
        runtime_params: runtimeParams,
        dropped_params: [...dropped],
        model: plan.model,
      },
    });
    log(`DONE — ${captured.length} 帧已回传`);
    big.style.display = 'flex';
  }

  main().catch(async (e) => {
    fail(e && e.stack || String(e));
    let probe = null;
    try {
      const c = model.internalModel.coreModel;
      const names = ['getParameterCount', 'getParameterIds', 'getParameterValues',
        'getParameterMinimumValues', 'getParameterMaximumValues', 'getParameterDefaultValues',
        'getParameterValueById', 'setParameterValueById', 'getParameterIndexById',
        'addParameterValueById', 'getParameterId', 'getModel', 'saveParameters', 'loadParameters'];
      probe = { types: {}, count: null, ownKeys: Object.keys(c).slice(0, 40),
                protoKeys: Object.getOwnPropertyNames(Object.getPrototypeOf(c)) };
      names.forEach((n) => { probe.types[n] = typeof c[n]; });
      if (typeof c.getParameterCount === 'function') probe.count = c.getParameterCount();
    } catch (pe) { probe = { probeError: String(pe) }; }
    try {
      await postJSON(`/job/${JOB}/error`, {
        error: String(e && e.message || e), stack: String(e && e.stack || ''),
        frames: captured, probe,
      });
    } catch (_) {}
  });

  window.__qaHarness = { get app() { return app; }, get model() { return model; } };
})();
