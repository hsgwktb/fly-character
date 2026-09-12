#!/usr/bin/env python3
"""tune_gate_on_trace.py — 用**真实** vocal 轨迹标定发声门控参数

为什么需要它:
  speak_gate_ablation.py 用的是手编合成轨迹(量级 0.3~2.0), 结果发现线上真实
  vocal 的 p50≈2.3 —— 量级差了约 5 倍, 拿合成轨迹标出来的 θ0=0.95 只有真实
  中位数的 40%, 系统长期处在深度饱和区, 阈值几乎不起作用(实测 CV 只有 0.2)。

  所以标定必须用真实轨迹: 让角色先跑一段时间, 采样 /api/state 的 vocal 与
  dn(下行神经元项) 落到 gate_trace.jsonl, 再用它跑网格。

输入: /content/fly/gate_trace.jsonl  (每行一个采样, 含 wall / vocal / dn)
用法: 在 Colab 里跑本文件; 它只读轨迹, 不改动运行中的服务。
"""
import json
import math
import sys

import numpy as np

TRACE = "/content/fly/gate_trace.jsonl"
DT = 0.1          # 与角色主循环一致


def load(path=TRACE):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if '"vocal"' in l]
    wall = np.array([r["wall"] for r in rows], float)
    voc = np.array([r["vocal"] for r in rows], float)
    dn = np.array([r["dn"] for r in rows], float)
    tt = np.arange(wall[0], wall[-1], DT)
    return tt, np.interp(tt, wall, voc), np.interp(tt, wall, dn), voc


def gate(V, D, theta0, k, tau, jit, beta=1.0, seed=3, gen=1.5):
    """只跑门控标量动力学。阈值只做两件事: 开口抬升 ∝√超出量, 其余时间向 θ0 回落。"""
    rng = np.random.default_rng(seed)
    u = 0.0
    th = theta0
    refr = 0.0
    sp = []
    for i in range(len(V)):
        u += DT / 2 * (-u + V[i] + beta * D[i]) + 0.28 * math.sqrt(DT) * rng.standard_normal()
        if u < 0:
            u = 0.0
        th += DT * (-(th - theta0) / tau)
        if th < 0.2:
            th = 0.2
        if refr > 0:
            refr = max(0.0, refr - DT)
        if refr <= 0 and u > th:
            sp.append(i * DT)
            refr = max(1.5, 2.0 * gen)
            ex = max(0.0, u - theta0)
            th += max(0.0, k * math.sqrt(ex) * (1.0 + jit * (2 * rng.random() - 1)))
            u = 0.0
    return sp


def stats(sp):
    if len(sp) < 3:
        return None
    iv = np.diff(sp)
    return {"n": len(sp), "mean": float(iv.mean()), "cv": float(iv.std() / iv.mean())}


def main():
    tt, V0, D0, voc = load()
    print("真实轨迹: %.0f s | vocal p10=%.2f p50=%.2f p90=%.2f | dn 均值=%.3f"
          % (tt[-1], np.percentile(voc, 10), np.percentile(voc, 50),
             np.percentile(voc, 90), D0.mean()))
    # 应激工况由真实轨迹线性放大得到(单点外推, 这里只为验证"需求强就更密")
    V1 = V0 * 1.5
    print("参考(θ0=0.95 k=1.5 τ=15 j=0.8): 平静 %s" % stats(gate(V0, D0, 0.95, 1.5, 15.0, 0.8)))

    out = []
    for theta0 in (1.6, 1.9, 2.1, 2.3, 2.5):
        for k in (0.6, 1.0, 1.5, 2.2):
            for tau in (10.0, 20.0, 35.0):
                for jit in (0.4, 0.8):
                    a = stats(gate(V0, D0, theta0, k, tau, jit))
                    b = stats(gate(V1, D0, theta0, k, tau, jit))
                    if a and b:
                        out.append((theta0, k, tau, jit, a, b))

    good = [r for r in out if 12.0 <= r[4]["mean"] <= 45.0
            and r[5]["mean"] < r[4]["mean"] * 0.75
            and r[4]["cv"] >= 0.45 and r[5]["cv"] >= 0.3]
    good.sort(key=lambda r: -min(r[4]["cv"], r[5]["cv"]))
    for theta0, k, tau, jit, a, b in good[:8]:
        print("  θ0=%-5.2f k=%-5.2f τ=%-5.1f j=%-4.1f | 平静 %5.1fs n=%-3d CV=%.2f | 应激 %5.1fs n=%-3d CV=%.2f"
              % (theta0, k, tau, jit, a["mean"], a["n"], a["cv"], b["mean"], b["n"], b["cv"]))
    print("满足约束: %d / %d" % (len(good), len(out)))
    if good:
        t0, k, tau, jit, a, b = good[0]
        print("推荐: theta0=%.2f k_adapt=%.2f tau_adapt=%.1f adapt_jit=%.1f" % (t0, k, tau, jit))
        s = stats(gate(V0, D0, t0, k, tau, jit, beta=0.0))
        print("  同轨迹关掉脑项(beta_dn=0): 平静 %s (n 从 %d 变 %d)" % (s, a["n"], s["n"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
