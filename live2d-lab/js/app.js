/* Live2D Lab —— 参数驱动 demo
 * 技术栈：PixiJS 7 + pixi-live2d-display 0.4 + Cubism Core（本地 vendor，无外网依赖）
 * 核心演示点：程序在每帧直接写入 Cubism 参数（setParameterValueById），
 * 与官方运行时的 motion / physics / lipsync 机制叠加，实现完全的程序化控制。
 */
(function () {
  'use strict';

  /* ---------- 模型清单（json 加时间戳 query，防浏览器缓存旧清单/贴图） ---------- */
  const MODELS = {
    Hiyori: 'assets/models/Hiyori/Hiyori.model3.json',
    'Hiyori-AI': 'assets/models/HiyoriAI/Hiyori.model3.json',
    Haru: 'assets/models/Haru/Haru.model3.json',
    'Haru-Gold': 'assets/models/HaruGold/Haru.model3.json',
    'Haru-GoldAI': 'assets/models/HaruGoldAI/Haru.model3.json',
    'Haru-AI': 'assets/models/HaruAI/Haru.model3.json',
    'HaruWineGold': 'assets/models/HaruWineGold/Haru.model3.json',
    'Hiyori-PinkGold': 'assets/models/HiyoriPinkGold/Hiyori.model3.json',
    'Hiyori-PurpleBlack': 'assets/models/HiyoriPurpleBlack/Hiyori.model3.json',
    Mark: 'assets/models/Mark/Mark.model3.json',
  };
  const modelUrl = (name) => MODELS[name] + '?t=' + Math.floor(Date.now() / 60000); // 分钟级

  /* ---------- 参数白名单：id / 中文名 / min / max / 默认 / 分组 ---------- */
  const PARAM_DEFS = [
    { id: 'ParamAngleX',    name: '头部左右',   min: -30, max: 30, def: 0, group: '头部 Head' },
    { id: 'ParamAngleY',    name: '头部上下',   min: -30, max: 30, def: 0, group: '头部 Head' },
    { id: 'ParamAngleZ',    name: '头部歪头',   min: -30, max: 30, def: 0, group: '头部 Head' },
    { id: 'ParamEyeLOpen',  name: '左眼开闭',   min: 0,   max: 1,  def: 1, group: '眼睛 Eye', step: 0.01 },
    { id: 'ParamEyeROpen',  name: '右眼开闭',   min: 0,   max: 1,  def: 1, group: '眼睛 Eye', step: 0.01 },
    { id: 'ParamEyeBallX',  name: '瞳孔左右',   min: -1,  max: 1,  def: 0, group: '眼睛 Eye', step: 0.01 },
    { id: 'ParamEyeBallY',  name: '瞳孔上下',   min: -1,  max: 1,  def: 0, group: '眼睛 Eye', step: 0.01 },
    { id: 'ParamEyeBrowY',  name: '眉毛上下',   min: -1,  max: 1,  def: 0, group: '眉毛 Brow', step: 0.01 },
    { id: 'ParamEyeBrowForm', name: '眉毛形态', min: -1,  max: 1,  def: 0, group: '眉毛 Brow', step: 0.01 },
    { id: 'ParamMouthOpenY', name: '嘴巴开合',  min: 0,   max: 1,  def: 0, group: '嘴 Mouth', step: 0.01 },
    { id: 'ParamMouthForm', name: '嘴型（笑/嘟）', min: -1, max: 1, def: 0, group: '嘴 Mouth', step: 0.01 },
    { id: 'ParamBodyAngleX', name: '身体左右',  min: -10, max: 10, def: 0, group: '身体 Body' },
    { id: 'ParamBodyAngleY', name: '身体上下',  min: -10, max: 10, def: 0, group: '身体 Body' },
    { id: 'ParamBodyAngleZ', name: '身体扭转',  min: -10, max: 10, def: 0, group: '身体 Body' },
    { id: 'ParamBreath',    name: '呼吸',       min: 0,   max: 1,  def: 0, group: '其他 Misc', step: 0.01 },
  ];

  /* ---------- 全局状态 ---------- */
  const state = {
    modelName: 'Hiyori',
    mode: 'auto',              // auto | manual
    blink: true,
    focus: true,
    speak: false,
    physics: true,
    params: {},                // id -> 当前滑块值
    activeIds: new Set(),      // 模型里实际存在的参数
  };

  let app = null;
  let model = null;
  let savedPhysics = null;
  let speakT = 0;
  let nextBlinkAt = 0;
  let blinkT0 = -1;

  const $ = (sel) => document.querySelector(sel);
  const stageEl = $('#stage');

  /* ---------- Pixi 应用 ---------- */
  app = new PIXI.Application({
    resizeTo: stageEl,
    backgroundAlpha: 0,
    antialias: true,
    autoDensity: true,
    resolution: Math.min(window.devicePixelRatio || 1, 2),
  });
  stageEl.appendChild(app.view);

  /* ---------- 工具 ---------- */
  let toastTimer = 0;
  function toast(msg) {
    const el = $('#toast');
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove('show'), 1800);
  }

  function showLoading(on) {
    $('#loading').classList.toggle('hidden', !on);
  }

  /* ---------- 模型布局 ---------- */
  function layoutModel() {
    if (!model) return;
    const w = stageEl.clientWidth;
    const h = stageEl.clientHeight;
    const oh = model.internalModel.originalHeight || 1200;
    const ow = model.internalModel.originalWidth || 900;
    const s = (h * 0.92) / oh;
    model.scale.set(s);
    model.x = (w - ow * s) / 2;
    model.y = (h - oh * s) / 2 + h * 0.015;
  }

  /* ---------- 每帧参数注入（注册晚于模型，故在库更新之后执行，写入即生效） ---------- */
  function applyParams(dtMs) {
    if (!model) return;
    const core = model.internalModel.coreModel;
    const dt = dtMs / 1000;

    // 1) 手动模式：把面板滑块值直接写入 Cubism 参数
    if (state.mode === 'manual') {
      state.activeIds.forEach((id) => {
        if (id in state.params) core.setParameterValueById(id, state.params[id]);
      });
    }

    // 2) 说话模拟：正弦包络驱动口型（叠加覆盖 MouthOpenY）
    if (state.speak) {
      speakT += dt;
      const env = 0.35 + 0.65 * Math.abs(Math.sin(speakT * 2.3));
      core.setParameterValueById('ParamMouthOpenY', Math.abs(Math.sin(speakT * 12)) * env);
      core.setParameterValueById('ParamMouthForm', 0.3);
    }

    // 3) 自动眨眼：三角波开合（叠加覆盖 EyeLOpen / EyeROpen）
    if (state.blink) {
      const now = performance.now();
      if (blinkT0 < 0 && now >= nextBlinkAt) blinkT0 = now;
      let open = 1;
      if (blinkT0 >= 0) {
        const p = (now - blinkT0) / 260;
        if (p >= 1) { blinkT0 = -1; nextBlinkAt = now + 2200 + Math.random() * 2800; }
        else open = 1 - Math.sin(p * Math.PI);
      }
      const baseL = state.mode === 'manual' ? (state.params.ParamEyeLOpen ?? 1) : 1;
      const baseR = state.mode === 'manual' ? (state.params.ParamEyeROpen ?? 1) : 1;
      core.setParameterValueById('ParamEyeLOpen', baseL * open);
      core.setParameterValueById('ParamEyeROpen', baseR * open);
    }
  }

  /* ---------- 加载模型 ---------- */
  async function loadModel(name) {
    showLoading(true);
    state.modelName = name;
    document.querySelectorAll('#model-row .btn').forEach((b) =>
      b.classList.toggle('active', b.dataset.model === name));

    if (model) { model.destroy(); model = null; savedPhysics = null; }
    speakT = 0; blinkT0 = -1; nextBlinkAt = performance.now() + 1500;

    const murl = modelUrl(name);
    model = await PIXI.live2d.Live2DModel.from(murl, { autoInteract: false });
    app.stage.addChild(model);
    layoutModel();

    // 直接解析 model3.json（不依赖库内部 settings 结构）
    state.modelJson = await (await fetch(murl)).json();
    const fr = (state.modelJson && state.modelJson.FileReferences) || {};

    // 读取模型实际拥有的参数（来自 cdi3.json DisplayInfo）
    state.activeIds = new Set(PARAM_DEFS.map((p) => p.id));
    try {
      if (fr.DisplayInfo) {
        const cdi = await (await fetch(new URL(fr.DisplayInfo, murl))).json();
        const ids = new Set((cdi.Parameters || []).map((p) => p.Id));
        // 保留白名单之外的判断依据：cdi 中存在的参数才显示
        state.activeIds = new Set(PARAM_DEFS.filter((p) => ids.has(p.id)).map((p) => p.id));
      }
    } catch (e) { /* cdi 缺失时退化为全量白名单 */ }

    buildParamPanel();
    buildMotionInfo();
    await buildExpressionRow();
    applyPhysicsOption();

    if (state.mode === 'auto') await model.motion('Idle');
    else model.internalModel.motionManager.stopAllMotions();

    showLoading(false);
    toast(`已加载 ${name}`);
  }

  /* ---------- 参数面板 ---------- */
  function buildParamPanel() {
    const box = $('#params');
    box.innerHTML = '';
    const groups = new Map();
    PARAM_DEFS.forEach((p) => {
      if (!state.activeIds.has(p.id)) return;
      if (!groups.has(p.group)) groups.set(p.group, []);
      groups.get(p.group).push(p);
    });

    const manual = state.mode === 'manual';
    $('#param-hint').textContent = manual ? '手动模式：滑块实时写入参数' : '自动模式下仅作预览，切到手动生效';

    groups.forEach((list, gname) => {
      const g = document.createElement('div');
      g.className = 'group';
      const gn = document.createElement('div');
      gn.className = 'group-name';
      gn.textContent = gname;
      g.appendChild(gn);

      list.forEach((p) => {
        if (!(p.id in state.params)) state.params[p.id] = p.def;
        const row = document.createElement('div');
        row.className = 'param-row';
        const step = p.step || (p.max - p.min) / 60;

        row.innerHTML =
          `<label title="${p.id}">${p.name}</label>` +
          `<input type="range" min="${p.min}" max="${p.max}" step="${step}" value="${state.params[p.id]}" data-id="${p.id}">` +
          `<span class="val">${Number(state.params[p.id]).toFixed(2)}</span>`;
        g.appendChild(row);
      });
      box.appendChild(g);
    });

    box.querySelectorAll('input[type=range]').forEach((input) => {
      input.addEventListener('input', () => {
        const id = input.dataset.id;
        state.params[id] = parseFloat(input.value);
        input.parentElement.querySelector('.val').textContent = Number(input.value).toFixed(2);
      });
    });
  }

  /* ---------- 动作 / 表情 ---------- */
  function buildMotionInfo() {
    const motions = (state.modelJson && state.modelJson.FileReferences && state.modelJson.FileReferences.Motions) || {};
    const parts = Object.entries(motions).map(([k, v]) => `${(v || []).length} ${k}`);
    $('#st-model').textContent = `${state.modelName} · ${parts.join(' / ') || '无动作'} · 参数 ${state.activeIds.size} 项可控`;
  }

  async function buildExpressionRow() {
    const row = $('#exp-row');
    const list = (state.modelJson && state.modelJson.FileReferences && state.modelJson.FileReferences.Expressions) || [];
    const sec = $('#sec-exp');
    row.innerHTML = '';
    if (!list.length) { sec.style.display = 'none'; return; }
    sec.style.display = '';
    list.forEach((exp, i) => {
      const btn = document.createElement('button');
      btn.className = 'btn';
      btn.textContent = (exp.Name || `表情${i + 1}`);
      btn.onclick = async () => { await model.expression(i); toast(`表情：${btn.textContent}`); };
      row.appendChild(btn);
    });
  }

  /* ---------- 选项 ---------- */
  function applyPhysicsOption() {
    if (!model) return;
    const im = model.internalModel;
    if (state.physics) {
      if (savedPhysics) im.physics = savedPhysics;
    } else {
      savedPhysics = im.physics || savedPhysics;
      im.physics = null;
    }
  }

  /* ---------- 事件绑定 ---------- */
  // 模型切换
  const modelRow = $('#model-row');
  Object.keys(MODELS).forEach((name) => {
    const btn = document.createElement('button');
    btn.className = 'btn';
    btn.dataset.model = name;
    btn.textContent = name;
    btn.onclick = () => { if (name !== state.modelName) loadModel(name); };
    modelRow.appendChild(btn);
  });

  // 模式
  $('#mode-auto').onclick = () => setMode('auto');
  $('#mode-manual').onclick = () => setMode('manual');
  function setMode(m) {
    state.mode = m;
    $('#mode-auto').classList.toggle('active', m === 'auto');
    $('#mode-manual').classList.toggle('active', m === 'manual');
    buildParamPanel();
    if (!model) return;
    if (m === 'manual') {
      model.internalModel.motionManager.stopAllMotions();
      toast('手动模式：参数由面板接管');
    } else {
      model.motion('Idle');
      toast('自动模式：Idle 循环播放');
    }
  }

  // 自动器开关
  $('#opt-blink').onchange = (e) => { state.blink = e.target.checked; };
  $('#opt-focus').onchange = (e) => { state.focus = e.target.checked; };
  $('#opt-speak').onchange = (e) => { state.speak = e.target.checked; speakT = 0; };
  $('#opt-physics').onchange = (e) => { state.physics = e.target.checked; applyPhysicsOption(); };

  // 动作按钮
  $('#btn-idle').onclick = () => model && model.motion('Idle');
  $('#btn-tap').onclick = () => model && model.motion('TapBody');

  // 重置
  $('#btn-reset').onclick = () => {
    PARAM_DEFS.forEach((p) => { state.params[p.id] = p.def; });
    document.querySelectorAll('#params input[type=range]').forEach((i) => {
      i.value = state.params[i.dataset.id];
      i.parentElement.querySelector('.val').textContent = Number(i.value).toFixed(2);
    });
    toast('参数已重置');
  };

  // 舞台交互：视线跟随 + 点击命中（坐标统一用相对舞台的 clientX/Y 计算）
  function stagePoint(e) {
    const r = stageEl.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }
  stageEl.addEventListener('pointermove', (e) => {
    if (model && state.focus) { const p = stagePoint(e); model.focus(p.x, p.y); }
  });
  // pointerdown 与 click 双监听（click 兜底：部分环境/自动化只派发 click）
  let lastTapAt = 0;
  function onStageTap(e) {
    const now = performance.now();
    if (now - lastTapAt < 300) return; // 防双触发
    if (!model) return;
    const p = stagePoint(e);
    const hits = model.hitTest(p.x, p.y) || [];
    if (hits.length) {
      lastTapAt = now;
      $('#st-hit').textContent = `命中：${hits.join(', ')}`;
      model.motion('TapBody');
      toast('命中身体 → 播放 TapBody');
    }
  }
  stageEl.addEventListener('pointerdown', onStageTap);
  stageEl.addEventListener('click', onStageTap);

  // 窗口缩放
  window.addEventListener('resize', layoutModel);

  // FPS 状态
  setInterval(() => {
    $('#st-fps').textContent = 'FPS ' + (app.ticker.FPS || 0).toFixed(0);
  }, 500);

  // 每帧注入（模型更新之后）
  app.ticker.add((dt) => applyParams(app.ticker.deltaMS));

  // 自动模式的 Idle 接力
  app.ticker.add(() => {
    if (!model || state.mode !== 'auto') return;
    const mm = model.internalModel.motionManager;
    if (mm.isFinished && mm.isFinished()) model.motion('Idle');
  });

  /* ---------- 调试导出（控制台可访问 window.__live2dLab） ---------- */
  window.__live2dLab = {
    get app() { return app; },
    get model() { return model; },
    get state() { return state; },
  };

  /* ---------- 启动 ---------- */
  loadModel('Hiyori').catch((err) => {
    showLoading(false);
    toast('加载失败：' + err.message);
    console.error(err);
  });
})();
