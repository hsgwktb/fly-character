/* ui_smoke.js — 无浏览器前端冒烟测试
 *
 * 为什么需要它：
 *   前端有一整类故障"不报错、只是某几块面板变空"。实例：push() 里把 s 写成 state，
 *   抛 ReferenceError，而异常发生在图表画完之后、事件渲染之前 —— 表现就是
 *   "图表在动，但对话/思维/事件流全是空框"，肉眼极难定位。
 *   这个脚本用假 DOM 真跑一次 push()，让异常直接暴露。
 *
 * 用法：node tools/ui_smoke.js        （在仓库根目录执行；失败返回非 0）
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(ROOT, 'ui.html'), 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) { console.error('ui.html 里找不到 <script>'); process.exit(1); }
const js = m[1];

/* ---------- 假 DOM ---------- */
class El {
  constructor(tag = 'div') {
    this.tagName = tag; this.className = ''; this.id = '';
    this.style = {}; this._children = []; this._text = '';
    this.clientWidth = 400; this.clientHeight = 110;
    this.scrollTop = 0; this.scrollHeight = 600;
    this.width = 400; this.height = 110; this.listeners = {};
  }
  get children() { return this._children; }
  get childElementCount() { return this._children.length; }
  get firstChild() { return this._children[0]; }
  get lastChild() { return this._children[this._children.length - 1]; }
  set textContent(v) { this._text = String(v); this._children = []; }
  get textContent() { return this._text; }
  set innerHTML(v) {
    this._html = String(v);
    const n = (this._html.match(/<div class="bar">/g) || []).length;
    this._children = [];
    for (let i = 0; i < n; i++) {
      const lab = new El('span'), trk = new El('span'), fil = new El('span'), num = new El('span');
      trk._children = [fil];
      const row = new El('div'); row._children = [lab, trk, num];
      this._children.push(row);
    }
  }
  get innerHTML() { return this._html || ''; }
  appendChild(c) { this._children.push(c); return c; }
  prepend(c) { this._children.unshift(c); return c; }
  removeChild(c) { const i = this._children.indexOf(c); if (i >= 0) this._children.splice(i, 1); return c; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  getContext() {
    const noop = () => {};
    return { setTransform: noop, clearRect: noop, beginPath: noop, moveTo: noop,
             lineTo: noop, stroke: noop, setLineDash: noop };
  }
}

const reg = {};
const mk = id => (reg[id] = Object.assign(new El(), {id}));
['s_t','s_up','s_fps','s_share','s_int','s_sex','s_ci','s_llm','s_gate','s_actn','s_evn',
 's_paramsrc','s_dialogue','c_drv','c_spk','c_rd','c_int','c_court','drv_bars','ro_bars',
 'act_bars','chat','think','log','cogkv','say','b_say','b_run','b_reset','b_export','controls',
 'b_gear','gearwin','gwt','gwb','b_gwclose','p_system','p_channel','p_reply','b_psave','b_preset',
 'p_saved','slots','p_file',
].forEach(mk);
// 滑块由 buildControls() 生成，假环境要预注册
['gain','steps','hab','lo','hi','mh','mt','mb','mi','ml','mc','soc','sr','kp','ada',
 'use','think','reply'].forEach(k => { mk('p_' + k); mk('v_' + k); });

const document = {
  getElementById: id => reg[id] || null,
  createElement: t => new El(t),
  querySelectorAll: () => [],
  activeElement: null,
  body: new El('body'),
};

/* ---------- 一份覆盖所有事件类型的合成状态 ---------- */
const DRIVES = ['hunger','thirst','heat','cold','dry','humid','lonely','sexual','fatigue','threat','boredom'];
const state = {
  t: 123.4, tick: 1234, fps: 10.0,
  d: Object.fromEntries(DRIVES.map(k => [k, 0.3])),
  ro: {eat: 12.5, flee: 0.0, approach: 33.3, explore: 5.1},
  interest: 0.42, social: 0.0, ci: 0.05, u_speak: 1.1, theta: 1.05,
  appr: {novelty: 0.9, threat: 0.1, control: 0.7}, appr_active: true, want: 'explore',
  llm: {think_mode: 'kwargs', busy: false, calls: 3, done: 3, stale: 0,
        rejected: 0, tokens: 120, last_ms: 1500, err: ''},
  n_act: 20, n_grounded: 13, actions: {eat: 5, explore: 9, rest: 3, flee: 3},
  affect: {valence: 0.2, arousal: 0.5},
  params: Object.fromEntries([...['gain','steps','hab','lo','hi','mh','mt','mb','mi','ml','mc','soc',
        'sr','kp','ada','use','think','reply','run'].map((k, i) => [k, i === 0 ? 0.65 : 0])]),
  seq: 6, reply_mode: 1, unanswered: 1,
  prompts: {system: '你是一只果蝇。{"thought": "", "say": ""}', channel: '只输出一个很短的中文句子。',
            reply: '你必须回一句 —— 用你果蝇的身份回它。'},
  templates: [null, {name: '模板B', system: 'S2{"thought":1}', channel: 'C2', reply: 'R2'},
              null, null, null],
  prompts_file: '/content/fly/prompts.json',
  says: [{t: 100, say: '饿。'}], thoughts: [{t: 100, thought: '想吃。'}],
  heard: [{t: 99, text: '你在干嘛？'}],
  events: [
    {seq: 1, t: 100.0, kind: 'hear',   text: '📣 听到: 你在干嘛？'},
    {seq: 2, t: 101.0, kind: 'think',  text: '… 强制回复：「你在干嘛？」'},
    {seq: 3, t: 102.0, kind: 'tht',    text: '那边有味道。'},
    {seq: 4, t: 102.5, kind: 'spk',    text: '「别烦我。」'},
    {seq: 5, t: 103.0, kind: 'act',    text: "explore  <span class='badge g'>连接组</span>"},
    {seq: 6, t: 104.0, kind: 'fb',     text: "rest  <span class='badge f'>回退</span>"},
  ],
};

/* ---------- 执行 ---------- */
const sandbox = {
  document, devicePixelRatio: 1, console,
  Blob: class {}, URL: {createObjectURL: () => 'blob:x', revokeObjectURL: () => {}},
  setTimeout: () => 0, clearTimeout: () => {}, fetch: () => new Promise(() => {}),
  addEventListener: () => {},
  Date, JSON, Math, Object, Array, String, Number, Error, RegExp,
};
sandbox.window = sandbox;
vm.createContext(sandbox);

let rc = 0;
try {
  vm.runInContext(js, sandbox, {filename: 'ui.html<script>'});
  console.log('✓ 脚本加载');
} catch (e) {
  console.error('✗ 脚本加载失败（整个页面会失效）:', e.message);
  rc = 1;
}

if (!rc) {
  try {
    sandbox.push(state);
    console.log('✓ push() 未抛异常');
  } catch (e) {
    console.error('✗ push() 抛异常 —— 这会让某几块面板静默变空:', e.message);
    const at = String(e.stack || '').split('\n')[1];
    if (at) console.error('   ', at.trim());
    rc = 1;
  }
  // 合成状态里 6 条事件的分流：hear+spk -> chat(2)，tht -> think(1)，think+act+fb -> log(3)
  const want = {chat: 2, think: 1, log: 3, drv_bars: 11, ro_bars: 4, act_bars: 4};
  const got = {chat: reg.chat.childElementCount, think: reg.think.childElementCount,
               log: reg.log.childElementCount, drv_bars: reg.drv_bars.childElementCount,
               ro_bars: reg.ro_bars.childElementCount, act_bars: reg.act_bars.childElementCount};
  for (const k of Object.keys(want)) {
    const ok = got[k] >= want[k];
    console.log((ok ? '✓' : '✗') + ' ' + k + ' 子节点 = ' + got[k] + '（期望 >= ' + want[k] + '）');
    if (!ok) rc = 1;
  }
  if (!(reg.cogkv.innerHTML || '').length) { console.error('✗ cogkv 未渲染'); rc = 1; }
  else console.log('✓ 认知面板已渲染');

  // 齿轮面板：三段落入输入框 + 5 个模板槽渲染 + 二次 push 不许覆盖输入框
  const psys = reg.p_system.value || '';
  const okSys = psys.indexOf('果蝇') >= 0 && (reg.p_channel.value || '').length > 0
             && (reg.p_reply.value || '').length > 0;
  console.log((okSys ? '✓' : '✗') + ' 三段提示词已灌入输入框（system=' +
              JSON.stringify(psys.slice(0, 12)) + '）');
  if (!okSys) rc = 1;

  const nslot = reg.slots.childElementCount;
  console.log((nslot === 5 ? '✓' : '✗') + ' 模板槽 = ' + nslot + '（期望 5）');
  if (nslot !== 5) rc = 1;
  const row1 = reg.slots.children[1];
  const nm1 = row1 && row1.children[1] ? row1.children[1].value : '';
  console.log((nm1 === '模板B' ? '✓' : '✗') + ' 模板名回填 = ' + JSON.stringify(nm1));
  if (nm1 !== '模板B') rc = 1;

  state.prompts.system = 'CHANGED-BY-POLL';
  sandbox.push(state);
  const stillSame = reg.p_system.value === psys;
  console.log((stillSame ? '✓' : '✗') + ' 二次 push 未覆盖输入框（防轮询抹掉正在输入的内容）');
  if (!stillSame) rc = 1;
}

console.log(rc ? '\n冒烟测试失败' : '\n冒烟测试通过');
process.exit(rc);
