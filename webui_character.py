#!/usr/bin/env python3
"""
webui_character.py — 果蝇数字角色的 WebUI 观测台后端

独立进程运行(不用 daemon 线程), 见《Colab-使用经验与技巧.md》§4.1。
- 角色: 真实 MaleCNS 连接组的识别接口驱动行为 + "想说话阈值"门控发声
- HTTP: /  (观测台页面)  /api/state  /api/control  /api/reset
- 端口 8000 (避开 Colab 自带的 8080), 交给 cloudflared 暴露
"""
import json, math, os, subprocess, threading, time, urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
import torch

from fly_llm import FlyLLM, _post as llm_post

OUT = os.environ.get("FLY_DATA", "/content/fly/normalized")
UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")
PORT = int(os.environ.get("FLY_UI_PORT", "8000"))
LLM = FlyLLM()          # 认知层；不可用时自动降级为模板发声


def _sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, executable="/bin/bash",
                              capture_output=True, text=True).stdout.strip()
    except Exception as e:
        return "ERR " + repr(e)


def diag() -> dict:
    """通过角色网页暴露 Colab 侧状态 —— 桥不稳时的"望远镜"。"""
    out = {"llm_stats": LLM.stats(), "key_len": len(LLM.key or ""), "url": LLM.url}
    for f in ("/content/llm_status.txt", "/content/llm_gate.json"):
        try:
            out[os.path.basename(f)] = open(f, encoding="utf-8").read().strip()[:500]
        except Exception:
            out[os.path.basename(f)] = None
    out["gpu"] = _sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader")
    out["procs"] = _sh("ps -eo pid,etime,cmd | grep -E '[l]lama-server|[w]ebui_character' | head -3")
    try:
        req = urllib.request.Request("http://127.0.0.1:8081/v1/models",
                                     headers={"Authorization": "Bearer " + (LLM.key or "")})
        with urllib.request.urlopen(req, timeout=10) as r:
            out["models"] = json.loads(r.read().decode())
    except Exception as e:
        out["models"] = "ERR " + repr(e)[:200]
    return out


def probe(prompt: str, max_tokens: int = 400, think: bool = False) -> dict:
    """经公网隧道直连测试认知层（含"关思考是否生效"）。"""
    payload = {"model": LLM.model,
               "messages": [{"role": "system", "content": "只输出一个很短的中文句子。"},
                            {"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.7}
    if not think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.time()
    try:
        data = llm_post(LLM.url + "/chat/completions", payload, LLM.key, 600)
    except Exception as e:
        return {"ok": False, "err": repr(e)[:300], "ms": int((time.time() - t0) * 1000)}
    m = data["choices"][0]["message"]
    return {"ok": True, "ms": int((time.time() - t0) * 1000),
            "content": (m.get("content") or "")[:600],
            "reasoning_chars": len(m.get("reasoning_content") or ""),
            "usage": data.get("usage")}

# ------------------------------------------------------------------ 连接组
meta = pd.read_feather(OUT + "/neurons.feather")
indptr = np.load(OUT + "/indptr.npy").astype(np.int32)
indices = np.load(OUT + "/indices.npy").astype(np.int32)
w = np.load(OUT + "/weights.npy"); sign = np.load(OUT + "/sign.npy")
N = len(meta)
dev = "cuda" if torch.cuda.is_available() else "cpu"
crow = torch.from_numpy(indptr).to(dev); col = torch.from_numpy(indices).to(dev)
sg = torch.from_numpy(sign).to(dev)
Wraw = torch.from_numpy(w).to(dev) * sg[col.long()]
ts = meta.type.astype(str).str.strip()

def sel(fn):
    return torch.from_numpy(np.flatnonzero(fn(ts).to_numpy()).astype(np.int64)).to(dev)

SENSE = {
    "sugar": sel(lambda s: s.str.match(r"^LB3[a-d]$")),
    "loom": torch.cat([sel(lambda s: s.str.startswith("LC4")),
                       sel(lambda s: s.str.startswith("LPLC2"))]),
    "walk_vis": sel(lambda s: s.str.match(r"^LC9$")),
    "turn_vis": sel(lambda s: s.str.match(r"^VS$")),
}
GROUND = {"eat": ("MN9", 55.0), "flee": ("DNp01", 280.0),
          "approach": ("DNp20", 120.0), "explore": ("DNp09", 120.0)}
MOTOR = {b: sel(lambda s, n=nm: s.eq(nm)) for b, (nm, _) in GROUND.items()}

V0 = -52.0; VR = -52.0; VTH = -45.0
T_MBR = 20.0; TAU_SYN = 5.0; T_RFC = 2.2; T_DLY = 1.8
W_SYN = 0.275; R_POI = 150.0; F_POI = 250.0
DT_MS = 1.0
DELAY = int(round(T_DLY / DT_MS))
DECAY_G = float(np.exp(-DT_MS / TAU_SYN))

V = torch.full((N,), V0, device=dev)
G = torch.zeros(N, device=dev)
REFR = torch.zeros(N, device=dev)
SPK = torch.zeros(N, device=dev)
BUF = [torch.zeros(N, device=dev) for _ in range(DELAY)]
RNG = torch.Generator(device=dev).manual_seed(12345)
A_DEFAULT = torch.sparse_csr_tensor(crow, col, Wraw * (W_SYN * 0.65), (N, N))
A = A_DEFAULT

def brain_step(sensory, n_steps):
    global V, G, REFR, SPK
    p = {k: (R_POI * v) * DT_MS / 1000.0 for k, v in sensory.items()}
    cnt = {b: 0 for b in MOTOR}
    for _ in range(n_steps):
        REFR = torch.clamp(REFR - DT_MS, min=0.0)
        act = REFR <= 0
        I = torch.sparse.mm(A, BUF.pop(0).view(-1, 1)).view(-1)
        BUF.append(SPK)
        G = torch.where(act, G * DECAY_G + I, G)
        V = torch.where(act, V + (DT_MS / T_MBR) * (V0 - V + G), V)
        for k, pp in p.items():
            if pp <= 0:
                continue
            idx = SENSE[k]
            h = torch.rand(len(idx), device=dev, generator=RNG) < pp
            if bool(h.any()):
                V[idx[h]] += W_SYN * F_POI
        s = act & (V > VTH)
        V = torch.where(s, torch.full_like(V, VR), V)
        G = torch.where(s, torch.zeros_like(G), G)
        REFR = torch.where(s, torch.full_like(REFR, T_RFC), REFR)
        SPK = s.float()
        for b in MOTOR:
            cnt[b] += int(s[MOTOR[b]].sum())
    sec = n_steps * DT_MS / 1000.0
    return {b: cnt[b] / max(len(MOTOR[b]), 1) / sec for b in MOTOR}

# ------------------------------------------------------------------ 角色层
BASE = ("hunger", "thirst", "heat", "cold", "dry", "humid", "threat")
BEH = {
    "eat": ({"hunger": 2.2}, 0.0, 1.00, 3.0, 2.0), "drink": ({"thirst": 2.2}, 0.0, 1.00, 2.0, 2.0),
    "cool": ({"heat": 2.4}, 0.05, 1.00, 3.0, 1.0), "warm": ({"cold": 2.4}, 0.05, 1.00, 3.0, 1.0),
    "seek_humid": ({"dry": 2.0}, 0.0, 1.05, 3.0, 2.0), "groom": ({"humid": 1.2}, 0.25, 1.05, 3.0, 3.0),
    "approach": ({"lonely": 2.2}, 0.10, 1.00, 3.0, 2.0), "court": ({"sexual": 2.2, "lonely": 0.6}, 0.0, 1.00, 4.0, 6.0),
    "explore": ({}, 0.70, 1.00, 4.0, 1.0), "rest": ({"fatigue": 1.8}, 0.10, 1.05, 6.0, 2.0),
    "flee": ({"threat": 4.5}, 0.0, 0.70, 0.6, 1.0),
}
PRIO = {"eat": 1.0, "drink": 1.0, "cool": 1.0, "warm": 1.0, "seek_humid": 1.0, "groom": 0.85,
        "approach": 0.95, "court": 0.95, "explore": 0.55, "rest": 1.0, "flee": 1.2}
TEMPL = {
    "hunger": ["饿。闻到烂水果的味道了。", "肚子空得发慌。"],
    "thirst": ["渴，舌头都干了。", "得找点水。"],
    "heat": ["太热了，翅膀发软。", "得躲到阴凉底下。"],
    "cold": ["冷得动不了。", "得挪到暖和点的地方。"],
    "dry": ["空气干得刺人。", "得找个潮一点的地方。"],
    "humid": ["湿气黏在翅膀上。", "浑身发潮，不舒服。"],
    "lonely": ["这么久都没有别的果蝇了。", "有点想凑过去。"],
    "sexual": ["空气里有股气味，坐不住。", "该去找它了。"],
    "fatigue": ["累了，想歇会儿。", "翅膀沉得抬不起来。"],
    "threat": ["有东西过来了，快跑！", "不对劲，得闪开！"],
    "boredom": ["什么都没有。到处都一样。", "闲得发慌。"],
    "calm": ["……风很轻。", "四下里没什么动静。"],
}
LOCK = threading.Lock()

class Char:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t = 0.0; self.tick = 0; self.day = 0.15
        self.body = dict(energy=0.70, hydration=0.70, temp=25.0, fatigue=0.10,
                         boredom=0.05, last_social=0.0, last_mating=0.0, activity=0.0)
        self.world = dict(food=1.0, water=1.0, flies_near=0, threat=0.0,
                          humidity=0.55, temp=25.0, novelty=1.0)
        self.aff = dict(valence=0.0, arousal=0.25, blocked=0.0)
        self.hab = {b: 0.0 for b in GROUND}
        self.cooldown = {b: 0.0 for b in BEH}
        self.urges = {b: 0.0 for b in BEH}
        self.ro = {b: 0.0 for b in GROUND}
        self.u_speak = 0.0; self.theta = 1.0; self.refr = 0.0; self.rum = 0.0
        self.reply_refr = 0.0         # 强制回复的独立不应期(短)
        self.speak_times = deque(); self.events = []; self.seq = 0
        self.court_log = deque()      # (起始时刻, 持续时长) -> 用于算求偶指数
        self.ci = 0.0                 # 求偶指数: 近 60s 处于求偶行为的时间占比
        self.social = 0.0             # 社交显著性(视野内有无同类)
        self.interest = 0.0           # 兴趣: 指向客体的相位性注意状态(不是"1-无聊")
        self.interest_boost = 0.0     # 显著事件对兴趣的瞬时抬升
        # ---- 认知层 ----
        self.external = deque(maxlen=20)      # 外部消息(当作刺激, 不是提示词)
        self.vocal_bump = 0.0                 # 外部消息带来的发声倾向(衰减)
        self.social_boost = 0.0               # "有人在场"的衰减量
        self.thoughts = deque(maxlen=24)      # 思维日志
        self.says = deque(maxlen=24)          # 台词日志
        self.memory = deque(maxlen=16)        # 供下次请求的短期记忆
        self.appr = {"novelty": 0.5, "threat": 0.5, "control": 0.5}
        self.appr_until = 0.0
        self.want = "none"
        self.want_until = 0.0
        self.llm_pending = 0
        self.llm_consumed = 0
        self.llm_discarded = 0
        self.n_act = 0; self.n_grounded = 0; self.fps = 0.0
        self.act_hist = {}            # 行为分布(消融对照要用)
        self.rng = np.random.default_rng(7)
        self.params = dict(gain=0.65, steps=15, hab=0.7, lo=0.10, hi=0.40,
                           mh=1.0, mt=1.0, mb=1.0, ml=1.0, mc=1.0, soc=0.0,
                           mi=1.0, sr=0.05, kp=2.0, ada=1.0,
                           use=1.0,      # 消融: 0 = LLM 输出只显示、不消费
                           think=0.0,    # 发声时是否允许模型思考(慢但更丰富)
                           reply=0.0,    # 1 = 强制回复我的输入(绕过冲动阈值)
                           run=1.0)

    def drives(self):
        b, wd, t = self.body, self.world, self.t
        return {
            "hunger": min(1, max(0, 1 - b["energy"])),
            "thirst": min(1, max(0, 1 - b["hydration"])),
            "heat": min(1, max(0, (b["temp"] - 27) / 8)),
            "cold": min(1, max(0, (22 - b["temp"]) / 8)),
            "dry": min(1, max(0, (0.40 - wd["humidity"]) / 0.40)),
            "humid": min(1, max(0, (wd["humidity"] - 0.75) / 0.25)),
            "lonely": min(1, max(0, (t - b["last_social"] - 25) / 120)),
            "sexual": min(1, max(0, (t - b["last_mating"] - 55) / 180)),
            "fatigue": min(1, max(0, b["fatigue"])),
            "threat": min(1, max(0, wd["threat"])),
            "boredom": min(1, max(0, b["boredom"])),
        }

    def ev(self, kind, text):
        self.seq += 1
        self.events.append({"seq": self.seq, "t": round(self.t, 1), "kind": kind, "text": text})
        if len(self.events) > 500:
            self.events = self.events[-200:]

    def courtship_index(self, win=60.0):
        """求偶指数 = 观察窗口内处于求偶行为的时间占比。

        沿用果蝇行为学里 CI(courtship index) 的口径: 是"时间占比"而不是加权和,
        因此可以直接解释、也可以直接和文献里的 CI 对照。
        """
        now = self.t
        while self.court_log and self.court_log[0][0] + self.court_log[0][1] < now - win:
            self.court_log.popleft()
        return min(1.0, sum(d for t0, d in self.court_log if t0 >= now - win) / win)

    # ---------------- 认知层 ----------------
    def build_packet(self, intent: str) -> dict:
        """喂给认知层的状态包。

        关键：给的是**全部数值与近况**，不是一个标签。
        旧实现是 `argmax(drives)` 得到一个词再查模板 —— 那样 LLM 无事可做。
        """
        d = self.drives()
        return {
            "t": round(self.t, 1),
            "drives": {k: round(v, 2) for k, v in d.items()},
            "strongest_drive": intent,
            "affect": {"valence": round(self.aff["valence"], 2),
                       "arousal": round(self.aff["arousal"], 2)},
            "interest": round(self.interest, 2),
            "courtship_index": round(self.ci, 2),
            "world": {"food": round(self.world["food"], 2),
                      "water": round(self.world["water"], 2),
                      "temp": round(self.world["temp"], 1),
                      "humidity": round(self.world["humidity"], 2),
                      "novelty": round(self.world["novelty"], 2),
                      "flies_near": self.world["flies_near"],
                      "threat": round(self.world["threat"], 2)},
            "connectome_readout_hz": {k: round(v, 1) for k, v in self.ro.items()},
            "recent_actions": [e["text"].split("<")[0].strip()
                               for e in self.events if e["kind"] in ("act", "fb")][-6:],
            "recent_thoughts": [x["thought"] for x in list(self.thoughts)[-3:]],
            "recent_says": [x["say"] for x in list(self.says)[-3:]],
            "heard_from_outside": [m["text"] for m in list(self.external)[-3:]],
        }

    def hear(self, text: str) -> None:
        """外部消息 = 刺激，不是提示词。

        它只抬高兴趣/唤醒/社交显著性/发声倾向 —— **说不说仍由阈值决定**。
        这样"由果蝇大脑主动驱动"没有被破坏：人可以引诱它开口，但不能命令它开口。
        """
        text = (text or "").strip()[:200]
        if not text:
            return
        self.external.append({"t": round(self.t, 1), "text": text,
                              "answered": False, "taken": False, "tries": 0,
                              "taken_at": -999.0})
        self.ev("hear", "📣 听到: %s" % text[:80])
        self.interest_boost = min(1.4, self.interest_boost + 0.9)
        self.aff["arousal"] = min(1.0, self.aff["arousal"] + 0.25)
        self.social_boost = 1.0
        self.vocal_bump = min(1.2, self.vocal_bump + 0.9)

    def apply_llm(self, r: dict, use_effect: float) -> None:
        tag = r.get("tag") or ""
        meta = r.get("meta") or {}
        ok = bool(r.get("ok"))
        say = (r.get("say") or "").strip()
        thought = (r.get("thought") or "").strip()

        # 强制回复：把结果对回到那条外部消息上。
        # 只有"成功且有台词"才算已回复；失败或空回复交回队列重试（tries 上限 3）。
        if tag == "reply" and meta.get("msg_t") is not None:
            m = next((x for x in self.external if x["t"] == meta["msg_t"]), None)
            if m is not None:
                if ok and say:
                    m["answered"] = True
                    m["taken"] = False
                else:
                    m["taken"] = False

        if not ok:
            self.ev("think", "(认知层失败: %s) <span class='badge f'>错误</span>"
                    % (r.get("err") or "")[:70])
            return
        if thought:
            self.thoughts.append({"t": round(self.t, 1), "thought": thought,
                                  "ms": r.get("ms", 0)})
            self.ev("tht", thought)      # 走事件流, 前端按 kind 路由到思维面板
        if say:
            self.says.append({"t": round(self.t, 1), "say": say})
            self.ev("spk", "「%s」 <span class='stat'>%dms%s%s</span>" % (
                say, r.get("ms", 0), " ·带思考" if r.get("reasoning_len") else "",
                " ·回复" if tag == "reply" else ""))
        if use_effect > 0.5:
            for k in ("novelty", "threat", "control"):
                v = r.get(k)
                if v is not None:
                    self.appr[k] = float(v)
            self.appr_until = self.t + 25.0
            if r.get("want") and r["want"] != "none":
                self.want = r["want"]
                self.want_until = self.t + 20.0
            self.llm_consumed += 1
        else:
            self.llm_discarded += 1        # 消融: 只显示、不消费
        self.rum = min(1.0, self.rum + 0.6)   # 想法回流

    def step(self, dt):
        P = self.params; b, wd, a = self.body, self.world, self.aff
        self.tick += 1
        self.day += dt / 120.0
        bt = 25.0 + 6.0 * math.sin(2 * math.pi * self.day)
        bh = 0.55 + 0.20 * math.sin(2 * math.pi * self.day + 1.2)
        wd["temp"] += dt * (-(wd["temp"] - bt) / 30.0) + 0.10 * math.sqrt(dt) * float(self.rng.standard_normal())
        wd["humidity"] += dt * (-(wd["humidity"] - bh) / 40.0) + 0.015 * math.sqrt(dt) * float(self.rng.standard_normal())
        wd["humidity"] = float(np.clip(wd["humidity"], 0.05, 0.98))
        wd["food"] = min(1.0, wd["food"] + dt * 0.0015)
        wd["water"] = min(1.0, wd["water"] + dt * 0.0020)
        wd["novelty"] = float(np.clip(wd["novelty"] - dt * 0.008, 0.0, 1.0))   # 环境熟悉化
        wd["threat"] = max(0.0, wd["threat"] - dt * 0.05)
        if self.rng.random() < dt * 0.030:
            wd["threat"] = min(1.0, wd["threat"] + 0.5 + 0.4 * float(self.rng.random()))
            self.interest_boost = min(1.0, self.interest_boost + 0.6)      # 威胁=显著事件
        if self.rng.random() < dt * 0.05:
            nf = int(np.clip(wd["flies_near"] + int(self.rng.integers(-1, 2)), 0, 4))
            if nf > wd["flies_near"]:
                self.interest_boost = min(1.0, self.interest_boost + 0.5)  # 同类出现=显著事件
            wd["flies_near"] = nf
        b["energy"] = max(0.0, b["energy"] - dt * (0.0050 + 0.0030 * b["activity"]))
        b["hydration"] = max(0.0, b["hydration"] - dt * 0.0045)
        b["fatigue"] = min(1.0, b["fatigue"] + dt * 0.0060 * (0.4 + b["activity"]))
        b["temp"] += dt / 20.0 * (wd["temp"] - b["temp"]) + dt * 0.15 * b["activity"]
        # ---- 兴趣(相位性, τ≈4s) 与 无聊(累积量, τ≈70s): 相对但不同 ----
        # 兴趣由"环境里有多少可看的"加上显著事件的瞬时抬升驱动;
        # 无聊则是慢积累的亏缺 —— 没兴趣时涨得快, 有兴趣时被缓解。
        base_i = 0.06 + 0.30 * float(np.clip(wd["novelty"], 0, 1))
        if self.t < self.appr_until and P["use"] > 0.5:
            # 消费 LLM 的情境评估: 它说"这地方很新鲜"就真的抬高兴趣基线
            base_i = 0.5 * base_i + 0.5 * (0.06 + 0.30 * self.appr.get("novelty", 0.5))
        self.interest += dt / 4.0 * (min(1.0, base_i + self.interest_boost) - self.interest)
        self.interest = float(np.clip(self.interest, 0.0, 1.0))
        self.interest_boost *= math.exp(-dt / 6.0)
        b["boredom"] = float(np.clip(
            b["boredom"] + dt * (0.016 * (1.0 - 0.6 * self.interest) * (1 - 0.5 * a["arousal"])
                                 - 0.004 * P["mi"] * self.interest), 0.0, 1.0))
        b["activity"] *= math.exp(-dt / 3.0)
        for k in self.hab:
            self.hab[k] *= math.exp(-dt / 20.0)
        self.vocal_bump *= math.exp(-dt / 20.0)      # 外部消息带来的发声倾向衰减
        self.social_boost *= math.exp(-dt / 15.0)    # "有人在场"衰减
        if self.reply_refr > 0:
            self.reply_refr = max(0.0, self.reply_refr - dt)
        for k in self.cooldown:
            if self.cooldown[k] > 0:
                self.cooldown[k] = max(0.0, self.cooldown[k] - dt)

        d = self.drives()
        g = P["lo"] + P["hi"]
        sugar = float(np.clip(wd["food"], 0, 1)) * (P["lo"] + P["hi"] * min(1.0, P["mh"] * d["hunger"]))
        loom = (P["lo"] + P["hi"] * min(1.0, P["mt"] * d["threat"])) if d["threat"] > 0.02 else 0.0
        social = 1.0 if wd["flies_near"] > 0 else 0.0
        social = max(social, float(P["soc"]), self.social_boost)
        self.social = social
        walk_vis = P["lo"] + P["hi"] * min(1.0, P["mb"] * d["boredom"])
        turn_vis = social * (P["lo"] + P["hi"] * min(1.0, P["ml"] * d["lonely"]))
        gi = 0.75 + 0.50 * self.interest      # 有兴趣时对同一刺激反应更强(注意力增益)
        self.ro = brain_step({"sugar": sugar * gi, "loom": loom * gi,
                              "walk_vis": walk_vis * gi, "turn_vis": turn_vis * gi},
                             int(P["steps"]))

        stress = max(d[k] for k in BASE)
        best, bs, bsrc = None, 0.0, ""
        for name, (wts, base, thr, dur, cd) in BEH.items():
            if self.cooldown[name] > 0:
                continue
            if name == "court" and social <= 0.05:
                continue        # 视野里没有同类就不求偶 —— CI 才能解释为"表达出来的"求偶
            drv = base + sum(wt * d[k] for k, wt in wts.items())
            if name == "court":        # 性欲->求偶 的倍率可调
                drv = base + wts["sexual"] * P["mc"] * d["sexual"] + wts["lonely"] * d["lonely"]
            if name in GROUND:
                term = (self.ro[name] / GROUND[name][1]) * (1.0 - P["hab"] * min(1.0, self.hab[name]))
                src = "connectome"
            else:
                term = 0.0; src = "fallback"
            u = drv + term
            sc = u * PRIO[name] * (1.0 + 1.5 * stress)
            if (name == self.want and self.t < self.want_until and P["use"] > 0.5):
                sc *= 1.6      # LLM 的"想做某事" = 软先验, 不是硬命令
            if sc > thr and sc > bs:
                best, bs, bsrc = name, sc, src
        if best:
            self.n_act += 1
            self.act_hist[best] = self.act_hist.get(best, 0) + 1
            if bsrc == "connectome":
                self.n_grounded += 1; self.hab[best] = min(1.0, self.hab[best] + 0.35)
                self.ev("act", "%s  <span class='badge g'>连接组</span>" % best)
            else:
                self.ev("fb", "%s  <span class='badge f'>回退</span>" % best)
            self.cooldown[best] = dur + cd
            before = dict(d)
            if best == "eat" and wd["food"] > 0.02:
                b["energy"] = min(1.0, b["energy"] + 0.15); wd["food"] -= 0.03
            elif best == "drink" and wd["water"] > 0.02:
                b["hydration"] = min(1.0, b["hydration"] + 0.12); wd["water"] -= 0.03
            elif best == "cool": b["temp"] -= 0.3
            elif best == "warm": b["temp"] += 0.3
            elif best == "seek_humid": wd["humidity"] = min(1.0, wd["humidity"] + 0.02)
            elif best == "groom": b["boredom"] = max(0.0, b["boredom"] - 0.03)
            elif best == "approach":
                if wd["flies_near"] > 0: b["last_social"] = self.t
                elif self.rng.random() < 0.25: wd["flies_near"] += 1
            elif best == "court":
                self.court_log.append((self.t, BEH["court"][3]))    # 记录求偶时长
                self.interest_boost = min(1.0, self.interest_boost + 0.8)  # 求偶=强显著事件
                if social > 0 or wd["flies_near"] > 0:
                    b["last_mating"] = self.t
            elif best == "explore":
                b["boredom"] = max(0.0, b["boredom"] - 0.10)
                wd["novelty"] = min(1.0, wd["novelty"] + 0.35)   # 探索发现新东西 -> 新颖度回升
                if self.rng.random() < 0.20:
                    wd["food"] = min(1.0, wd["food"] + 0.04)
                    self.interest_boost = min(1.0, self.interest_boost + 0.4)   # 发现食物=显著事件
                if self.rng.random() < 0.15:
                    wd["water"] = min(1.0, wd["water"] + 0.04)
                    self.interest_boost = min(1.0, self.interest_boost + 0.3)
            elif best == "rest": b["fatigue"] = max(0.0, b["fatigue"] - 0.06)
            elif best == "flee":
                wd["threat"] = max(0.0, wd["threat"] - 0.5)
                a["arousal"] = min(1.0, a["arousal"] + 0.2)
            b["activity"] = 0.6
            relief = sum(max(0.0, before.get(k, 0) - self.drives().get(k, 0)) for k in BEH[best][0])
            a["valence"] = float(np.clip(a["valence"] + 1.8 * relief, -1, 1))

        a["arousal"] += dt / 6.0 * ((0.18 + 0.55 * stress + 0.30 * abs(a["valence"])
                                     + 0.15 * self.interest
                                     + (0.20 * self.appr.get("threat", 0.5)
                                        if (self.t < self.appr_until and P["use"] > 0.5) else 0.0)
                                     ) - a["arousal"])
        a["arousal"] = float(np.clip(a["arousal"], 0, 1))
        a["valence"] *= math.exp(-dt / 30.0)
        if self.t < self.appr_until and P["use"] > 0.5:
            # 消费"控制感": 觉得自己应付不来 -> 效价被持续下压
            a["valence"] = float(np.clip(
                a["valence"] + dt * 0.35 * (self.appr.get("control", 0.5) - 0.5), -1, 1))
        self.rum *= math.exp(-dt / 8.0)

        # ---- 想说话阈值门控 ----
        sat = float(np.clip(1.0 - stress / 0.35, 0, 1))
        boost = 1.0 + 1.8 * sat
        w_bored = boost
        w_lonely = boost * (0.30 + 0.70 * d["boredom"])
        vocal = (0.30 + 0.80 * a["arousal"] + 0.70 * abs(a["valence"])
                 + 0.90 * w_lonely * d["lonely"] + 0.60 * d["sexual"]
                 + 0.90 * max(d[k] for k in BASE) + 0.80 * w_bored * d["boredom"]
                 + 0.35 * self.interest
                 + 0.60 * self.vocal_bump
                 + 0.50 * self.rum)
        self.u_speak += dt / 2.0 * (-self.u_speak + vocal)
        self.u_speak += 0.28 * math.sqrt(dt) * float(self.rng.standard_normal())
        self.u_speak = max(0.0, self.u_speak)
        soc = max(d["lonely"], d["sexual"])
        target = min(P["sr"] * (1.0 + 2.5 * stress + 1.5 * soc), 0.30)
        if P["ada"] > 0.5:
            while self.speak_times and self.t - self.speak_times[0] > 30.0:
                self.speak_times.popleft()
            rate = len(self.speak_times) / 30.0
            self.theta += dt * (P["kp"] * (rate - target) - (self.theta - 1.0) / 200.0)
            self.theta = float(np.clip(self.theta, 0.25, 6.0))
        else:
            self.theta = 1.0
        if self.refr > 0:
            self.refr = max(0.0, self.refr - dt)
        # ---- 强制回复模式：有新消息就绕过冲动阈值直接回 ----
        # 默认(主动模式)下消息只是刺激、不保证被回应; 打开这个开关后改成必须回。
        # 关键: 必须**先判断有没有待回复的消息**，并让自发发言给它让位 ——
        # 否则自发发言的 25s 不应期会把到消息的窗口全占掉，回复永远轮不上(实测踩到过)。
        pending = None
        if P.get("reply", 0.0) > 0.5:
            for m in self.external:
                if m["answered"]:
                    continue
                # taken 超过 60s 视为请求被队列丢弃(队列深度 1, latest-wins)，交回重试
                if m["taken"] and (self.t - m.get("taken_at", -999.0)) < 60.0:
                    continue
                if m["tries"] >= 3:
                    continue
                pending = m
                break

        if pending is None and self.refr <= 0 and self.u_speak > self.theta:
            strongest = max(d, key=lambda k: d[k])
            key = strongest if d[strongest] > 0.15 else "calm"
            if os.environ.get("FLY_USE_LLM", "1") == "1":
                # 触发条件仍然是"想说话的冲动越过阈值" —— LLM 只被内生动力唤起
                self.llm_pending += 1
                LLM.request(self.build_packet(key), tag="speak",
                            allow_think=bool(P.get("think", 0.0) > 0.5))
                self.ev("think", "… <span class='stat'>认知层生成中 (最强内驱力=%s)</span>" % key)
                self.refr = 25.0      # 不应期必须覆盖生成延迟, 否则请求会堆积
            else:
                txt = str(self.rng.choice(TEMPL.get(key, TEMPL["calm"])))
                self.ev("spk", "「%s」 <span class='badge f'>模板</span>" % txt)
                self.refr = 5.0
            self.speak_times.append(self.t)
            self.theta += 0.15
            self.u_speak = 0.0

        if pending is not None and self.reply_refr <= 0:
            pending["taken"] = True
            pending["taken_at"] = self.t
            pending["tries"] += 1
            self.llm_pending += 1
            pkt = self.build_packet("reply")
            pkt["reply_to"] = pending["text"]
            LLM.request(pkt, tag="reply",
                        allow_think=bool(P.get("think", 0.0) > 0.5),
                        meta={"msg_t": pending["t"], "msg_text": pending["text"]})
            self.ev("think", "… <span class='stat'>强制回复：「%s」</span>" % pending["text"][:30])
            self.reply_refr = 8.0

        r = LLM.poll()
        if r is not None:
            self.llm_pending = max(0, self.llm_pending - 1)
            self.apply_llm(r, float(P["use"]))
        self.ci = self.courtship_index()
        self.t += dt


CH = Char()

def snapshot(since):
    with LOCK:
        c = CH
        return {
            "t": round(c.t, 1), "tick": c.tick, "fps": round(c.fps, 1),
            "d": {k: round(v, 3) for k, v in c.drives().items()},
            "u_speak": round(c.u_speak, 3), "theta": round(c.theta, 3),
            "ci": round(c.ci, 3), "social": round(c.social, 3),
            "interest": round(c.interest, 3),
            "ro": {k: round(v, 2) for k, v in c.ro.items()},
            "n_act": c.n_act, "n_grounded": c.n_grounded,
            "actions": dict(c.act_hist),
            "affect": {"valence": round(c.aff["valence"], 3),
                       "arousal": round(c.aff["arousal"], 3)},
            "llm": LLM.stats(),
            "thoughts": list(c.thoughts)[-6:],
            "says": list(c.says)[-6:],
            "heard": list(c.external)[-4:],
            "appr": {k: round(v, 2) for k, v in c.appr.items()},
            "appr_active": c.t < c.appr_until,
            "want": c.want if c.t < c.want_until else "none",
            "llm_pending": c.llm_pending,
            "llm_consumed": c.llm_consumed,
            "llm_discarded": c.llm_discarded,
            "reply_mode": c.params.get("reply", 0),
            "unanswered": sum(1 for m in c.external if not m["answered"]),
            "params": dict(c.params),
            "seq": c.seq,
            "events": [e for e in c.events if e["seq"] > since][-40:],
        }

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/state"):
            since = 0
            if "since=" in self.path:
                try: since = int(self.path.split("since=")[1].split("&")[0])
                except Exception: since = 0
            self._send(200, json.dumps(snapshot(since)).encode("utf-8"))
        elif self.path in ("/", "/index.html"):
            with open(UI_PATH, "rb") as fh:
                self._send(200, fh.read(), "text/html; charset=utf-8")
        elif self.path.startswith("/api/diag"):
            self._send(200, json.dumps(diag(), ensure_ascii=False).encode("utf-8"))
        else:
            self._send(404, b"{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        try:
            data = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception:
            data = {}
        if self.path == "/api/control":
            with LOCK:
                for k, v in data.items():
                    if k in CH.params:
                        CH.params[k] = float(v)
                if "gain" in data:
                    global A
                    A = torch.sparse_csr_tensor(crow, col,
                        Wraw * (W_SYN * float(CH.params["gain"])), (N, N))
            self._send(200, b'{"ok":true}')
        elif self.path == "/api/reset":
            with LOCK:
                keep = dict(CH.params)      # 重置时要保留界面设的参数, 否则消融开关会被抹掉
                CH.__init__()
                CH.params.update(keep)
            self._send(200, b'{"ok":true}')
        elif self.path == "/api/say":
            txt = str(data.get("text", "")).strip()
            if not txt:
                # 不能静默返回 ok:true —— 用 curl -d 传中文时字节会被破坏、JSON 解析失败，
                # 表现就是"消息发出去了但它没收到"。必须让客户端看得见。
                self._send(400, json.dumps(
                    {"ok": False, "err": "empty_or_unparsable_body",
                     "hint": "中文请用 --data-binary @file.json 传"}, ensure_ascii=False).encode("utf-8"))
                return
            with LOCK:
                CH.hear(txt)
            self._send(200, b'{"ok":true}')
        elif self.path == "/api/probe":
            r = probe(str(data.get("prompt", "你现在很饿，说一句。")),
                      int(data.get("max_tokens", 400)),
                      bool(data.get("think", False)))
            self._send(200, json.dumps(r, ensure_ascii=False).encode("utf-8"))
        else:
            self._send(404, b"{}")

def loop():
    last = time.time()
    while True:
        t0 = time.time()
        with LOCK:
            run = CH.params.get("run", 1.0) > 0.5
        if run:
            with LOCK:
                CH.step(0.1)
        now = time.time()
        CH.fps = 0.9 * CH.fps + 0.1 * (1.0 / max(now - last, 1e-6))   # tick 率(目标 10 Hz)
        last = now
        time.sleep(max(0.0, 0.1 - (now - t0)))

if __name__ == "__main__":
    threading.Thread(target=loop, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print("webui on http://127.0.0.1:%d" % PORT, flush=True)
    srv.serve_forever()
