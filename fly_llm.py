"""fly_llm.py — 果蝇角色的 LLM 认知层

设计约束（来自实测：6 tok/s、单次生成 10–130 秒、-np 2）：
  1. 队列深度 1 + 截止时间：只保留最新请求，超时的结果直接作废，绝不堆积
  2. 结构化输出：一次调用同时产出 思维 / 台词 / 情境评估 / 行动意图
  3. 每个输出都必须被消费，否则就是装饰（见 appraisal/want 的消费方）
  4. 思考字段(reasoning_content)单独收走，作为"思维"而不是丢弃

不做的事：不在这里做轮询循环。调用方（角色主循环）决定何时请求。
"""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request

WANT_CHOICES = ["eat", "drink", "cool", "warm", "seek_humid", "groom",
                "approach", "court", "explore", "rest", "flee", "none"]


def _post(url: str, payload: dict, api_key: str, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + api_key})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def extract_json(text: str) -> dict | None:
    """从可能带散文的输出里抠出第一个 JSON 对象。"""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    for cand in (t[i:j + 1],):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # 退化路径：逐个字段正则抓
    out = {}
    for k in ("thought", "say"):
        m = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"' % k, t)
        if m:
            out[k] = m.group(1)
    return out or None


SYSTEM = (
    "你是一只果蝇。你不是助手，不要客套，不要解释自己在做什么。\n"
    "下面会给你你此刻的身体状态、环境、以及最近发生的事。\n"
    "只输出一个 JSON 对象，不要任何多余文字，格式：\n"
    '{"thought": "你脑子里闪过的念头(20字内)", '
    '"say": "你如果开口会说的一句很短的中文(25字内)", '
    '"appraisal": {"novelty": 0-1, "threat": 0-1, "control": 0-1}, '
    '"want": "eat|drink|cool|warm|seek_humid|groom|approach|court|explore|rest|flee|none"}\n'
    "thought 是你真实在想的东西（可以混乱、可以只有感受）；"
    "say 是你对外发出的一句话。若此刻并不想说话，say 就给空字符串。\n"
    "appraisal 是你对当前处境的主观评估：novelty=有多新鲜，threat=有多危险，control=你觉得自己能不能应付。"
)


class FlyLLM:
    """非阻塞 LLM 客户端：put 即返回，后台线程生成，poll 取最新未过期结果。"""

    def __init__(self, base_url=None, api_key=None, model="qwen3.8-27b-uncensored",
                 deadline=180.0, timeout=600.0):
        self.url = (base_url or os.environ.get("FLY_LLM_URL",
                    "http://127.0.0.1:8081/v1")).rstrip("/")
        self.key = api_key or os.environ.get("FLY_LLM_KEY", "") or self._read_key()
        self.model = model
        self.deadline = deadline
        self.timeout = timeout
        self.think_mode = self._detect_think_mode()
        self.q_in: queue.Queue = queue.Queue(maxsize=1)
        self.q_out: queue.Queue = queue.Queue()
        self.busy = False
        self.last_error = ""
        self.n_calls = 0
        self.n_done = 0
        self.n_stale = 0
        self.n_rejected = 0
        self.total_tokens = 0
        self.t_last_ms = 0
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    @staticmethod
    def _read_key() -> str:
        """部署脚本把 key 持久化在 /content/api_key.txt；环境变量没给就读它。"""
        for p in ("/content/api_key.txt", os.path.expanduser("~/api_key.txt")):
            try:
                k = open(p).read().strip()
                if k:
                    return k
            except Exception:
                pass
        return ""

    # ---------- 思考开关：按部署时的闸门验证结果自动选 ----------
    def _detect_think_mode(self) -> str:
        p = "/content/llm_gate.json"
        try:
            g = json.load(open(p)).get("gate", "")
        except Exception:
            g = ""
        if g == "C":
            return "kwargs"      # chat_template_kwargs.enable_thinking=False
        if g == "B":
            return "no_think"    # system prompt 里加 /no_think
        return "on"              # 关不掉：只能给足 max_tokens，并容忍慢

    # ---------- 请求 ----------
    def request(self, packet: dict, tag: str, allow_think: bool = False) -> bool:
        """投递一次请求。队列满则丢弃最旧的（latest-wins）。"""
        job = {"packet": packet, "tag": tag, "allow_think": allow_think,
               "t0": time.time(), "id": self.n_calls}
        self.n_calls += 1
        try:
            self.q_in.put_nowait(job)
        except queue.Full:
            try:
                self.q_in.get_nowait()
                self.n_rejected += 1
            except queue.Empty:
                pass
            try:
                self.q_in.put_nowait(job)
            except queue.Full:
                pass
        return True

    def poll(self):
        """取最新结果；过期的丢弃并计数。"""
        out = None
        while True:
            try:
                r = self.q_out.get_nowait()
            except queue.Empty:
                break
            if time.time() - r["t0"] > self.deadline:
                self.n_stale += 1
                continue
            out = r
        return out

    # ---------- 后台线程 ----------
    def _loop(self):
        while not self._stop:
            try:
                job = self.q_in.get(timeout=0.5)
            except queue.Empty:
                continue
            self.busy = True
            t0 = time.time()
            try:
                res = self._generate(job)
            except Exception as e:
                res = {"ok": False, "err": repr(e)[:200], "job": job}
                self.last_error = res["err"]
            res["t0"] = job["t0"]
            res["tag"] = job["tag"]
            res["ms"] = int((time.time() - t0) * 1000)
            self.t_last_ms = res["ms"]
            self.n_done += 1
            self.q_out.put(res)
            self.busy = False

    def _generate(self, job: dict) -> dict:
        packet = job["packet"]
        use_think = job["allow_think"] and self.think_mode != "kwargs"
        max_tok = 4096 if use_think else 400
        sysmsg = SYSTEM if (use_think or self.think_mode == "on") else SYSTEM + " /no_think"

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": sysmsg},
                         {"role": "user", "content": json.dumps(packet, ensure_ascii=False)}],
            "max_tokens": max_tok,
            "temperature": 0.85,
            "top_p": 0.9,
        }
        if not use_think:
            if self.think_mode == "kwargs":
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            elif self.think_mode == "no_think":
                pass                       # 已在 system prompt 里加了 /no_think
            else:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
        return self._post_and_parse(payload)

    def _post_and_parse(self, payload: dict) -> dict:
        data = _post(self.url + "/chat/completions", payload, self.key, self.timeout)
        msg = data["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        usage = data.get("usage") or {}
        self.total_tokens += int(usage.get("completion_tokens") or 0)

        obj = extract_json(content)
        if obj is None and reasoning:
            obj = extract_json(reasoning)          # 有时 JSON 落在思考段里
        if obj is None:
            obj = {"say": content[:120], "thought": "", "_parse": "failed"}
        thought = (obj.get("thought") or "").strip()
        if reasoning and not thought:
            thought = reasoning[-160:]
        want = str(obj.get("want") or "none").strip().lower()
        if want not in WANT_CHOICES:
            want = "none"
        ap = obj.get("appraisal") or {}
        def g(k):
            try:
                return max(0.0, min(1.0, float(ap.get(k, 0.5))))
            except Exception:
                return None
        return {
            "ok": True,
            "thought": thought,
            "say": (obj.get("say") or "").strip(),
            "novelty": g("novelty"), "threat": g("threat"), "control": g("control"),
            "want": want,
            "parse": obj.get("_parse", "ok"),
            "reasoning_len": len(reasoning),
            "empty_content": (not content),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
        }

    def stats(self) -> dict:
        return {"think_mode": self.think_mode, "busy": self.busy, "calls": self.n_calls,
                "done": self.n_done, "stale": self.n_stale, "rejected": self.n_rejected,
                "tokens": self.total_tokens, "last_ms": self.t_last_ms,
                "err": self.last_error[-120:]}
