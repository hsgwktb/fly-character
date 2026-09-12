#!/usr/bin/env python3
"""speak_gate_ablation.py — 发声门控机制 A/B：速率设定值(伺服) vs 适应动力学(涌现)

为什么单独拿出来仿真：
  线上 webui_character.py 的 Character 需要加载 166,700 神经元的连接组才能跑,
  改一次参数要几十秒。而"发声门控"本身是纯标量动力学, 与连接组无关,
  可以在这里 1 秒内跑完 3600 秒的演化, 做参数标定和机制对照。

对照的三种机制:
  A  现状 = 积分控制器追目标速率(静息 sr=20s) + 硬编码 25s 不应期
  B  只拆不应期: 仍追目标速率, 但不应期按实际生成延迟自适应
  C  改成适应动力学: 阈值不再追任何速率, 开口后抬升 k_adapt 再按 tau_adapt 回落;
     不应期同样自适应; 并接入下行神经元活动的合成项 beta_dn

判据:
  - 间隔的变异系数 CV = sd/mean。伺服把速率钉死 -> CV 很小(节拍器);
    适应动力学 -> 间隔呈宽分布, CV 明显更大。
  - 速率与内驱力的相关: 适应机制下应当正相关(需求强就说得勤)。
  - 长程不锁死: 不能退化成全程沉默或不停说话。

注意本文件用的是**合成** vocal 轨迹(量级 0.3~2.0), 只适合做机制对照。
线上默认值最终是在**真实 vocal 轨迹**上标定的(见 tune_gate_on_trace.py):
真实 vocal 的 p50≈2.3, 因此 θ0 取 2.50 —— 不要拿本文件的量级去推定线上参数。
"""
import math
import numpy as np

DT = 0.1                # 与角色主循环一致(10 Hz)
RNG = np.random.default_rng(20260912)


def vocal_trace(level, secs, tau_slow=40.0, seed=0):
    """合成一条发声驱动 trace：慢漂移 + 快噪声, 均值落在 level 附近。

    对应线上 vocal 的量级：平静约 0.3-0.6, 中等内驱力约 0.8-1.2, 强内驱力约 1.5-2.2。
    """
    r = np.random.default_rng(seed)
    n = int(secs / DT)
    slow = np.zeros(n); fast = np.zeros(n)
    a = math.exp(-DT / tau_slow); b = math.exp(-DT / 4.0)
    for i in range(1, n):
        slow[i] = a * slow[i - 1] + (1 - a) * r.standard_normal() * 0.45
        fast[i] = b * fast[i - 1] + (1 - b) * r.standard_normal() * 0.18
    v = level + slow + fast
    return np.clip(v, 0.0, None), n


def dn_trace(secs, seed=1):
    """合成"下行神经元归一化总放电"：与无聊/活动同向, 平静时很低。

    对应线上 dn_norm = mean(min(1, ro[k]/ref)) over DNp01/DNp20/DNp09。
    """
    r = np.random.default_rng(seed)
    n = int(secs / DT)
    x = np.zeros(n)
    a = math.exp(-DT / 25.0)
    for i in range(1, n):
        x[i] = a * x[i - 1] + (1 - a) * r.standard_normal()
    return np.clip(0.18 + 0.30 * x, 0.0, 1.0)


def run_gate(v, dn, mode, *, sr=20.0, theta0=0.75, k_adapt=0.5, tau_adapt=12.0,
             beta_dn=1.0, kp=2.0, stress=0.0, soc=0.0, gen_secs=1.5, jitter=0.0, seed=7):
    """跑一遍门控, 返回发声时刻列表。

    jitter: 适应抬升量的相对抖动(0=确定, 0.8=抬升量在 ±80% 之间随机)。
    真实的适应是带噪过程, 不是控制器 —— 这一点直接决定间隔分布的宽度。
    """
    rng = np.random.default_rng(seed)
    n = len(v)
    u = 0.0
    theta = theta0
    refr = 0.0
    speaks = []
    times = []
    for i in range(n):
        t = i * DT
        times.append(t)
        u += DT / 2.0 * (-u + v[i] + (beta_dn * dn[i] if mode == "C" else 0.0))
        u += 0.28 * math.sqrt(DT) * float(rng.standard_normal())
        u = max(0.0, u)

        if mode in ("A", "B"):
            # 伺服: 把最近 30s 的实际速率拉向目标速率
            recent = [s for s in speaks if t - s <= 30.0]
            rate = len(recent) / 30.0
            target = min((1.0 + 2.5 * stress + 1.5 * soc) / sr, 0.30)
            theta += DT * (kp * (rate - target) - (theta - 1.0) / 200.0)
            theta = float(np.clip(theta, 0.25, 6.0))
        else:
            # 适应: 只向基础阈值回落, 没有任何目标速率
            theta += DT * (-(theta - theta0) / max(tau_adapt, 1e-3))
            theta = float(np.clip(theta, 0.20, 8.0))

        if refr > 0:
            refr = max(0.0, refr - DT)

        if refr <= 0 and u > theta:
            speaks.append(t)
            refr = 25.0 if mode == "A" else max(1.5, 2.0 * gen_secs)
            if mode in ("A", "B"):
                theta += 0.15
            else:
                # 适应抬升 ∝ sqrt(超出量): 驱动越强适应越强(亚线性), 所以强驱动下
                # 间隔不会塌到不应期上 -> 既不饱和成节拍器, 又保持"需求强就说得勤"
                excess = max(0.0, u - theta0)
                jump = k_adapt * math.sqrt(excess)
                theta += max(0.0, jump * (1.0 + jitter * (2.0 * rng.random() - 1.0)))
            u = 0.0
    return speaks


def stats(speaks, secs):
    if len(speaks) < 2:
        return dict(n=len(speaks), mean=float("nan"), cv=float("nan"),
                    med=float("nan"), p90=float("nan"), gap=secs)
    iv = np.diff(speaks)
    return dict(n=len(speaks), mean=float(iv.mean()),
                cv=float(iv.std() / max(iv.mean(), 1e-9)),
                med=float(np.median(iv)), p90=float(np.percentile(iv, 90)),
                gap=float(secs - speaks[-1]))


def show(tag, s, secs):
    print("  %-26s 次数=%3d | 平均间隔=%6.1fs | CV=%.2f | 中位=%6.1fs | p90=%6.1fs | 末段静默=%5.1fs"
          % (tag, s["n"], s["mean"], s["cv"], s["med"], s["p90"], s["gap"]))


def main():
    SECS = 3600
    print("=" * 118)
    print("发声门控 A/B：三种机制在 3600s 演化下的间隔分布（CV 越小越像节拍器）")
    print("=" * 118)

    for label, level in (("平静 (vocal≈0.45)", 0.45), ("中等 (vocal≈0.95)", 0.95),
                         ("强内驱力 (vocal≈1.75)", 1.75)):
        v, n = vocal_trace(level, SECS, seed=hash(label) % 1000)
        dn = dn_trace(SECS)
        print("\n%s" % label)
        show("A 伺服+25s不应期(现状)", stats(run_gate(v, dn, "A"), SECS), SECS)
        show("B 伺服+自适应不应期", stats(run_gate(v, dn, "B"), SECS), SECS)
        show("C 适应动力学+脑项", stats(run_gate(v, dn, "C"), SECS), SECS)

    # ---- 参数网格：给 C 找一组"平静不太吵、应激明显变密、CV 宽"的默认值 ----
    print("\n" + "=" * 118)
    print("C 的参数标定网格（目标：平静间隔 15-35s，应激明显变密，CV>0.6）")
    print("=" * 118)
    v_c, _ = vocal_trace(0.45, 1800, seed=11)
    v_s, _ = vocal_trace(1.75, 1800, seed=12)
    dn_c, dn_s = dn_trace(1800, seed=21), dn_trace(1800, seed=22)
    print("  %-7s %-7s %-6s %-6s | %-24s | %-24s" % ("theta0", "k_adapt", "tau", "jitter", "平静", "强内驱力"))
    rows = []
    for theta0 in (0.75, 0.95, 1.05, 1.20):
        for k_adapt in (1.5, 2.0, 2.5, 3.0, 4.0):
            for tau in (6.0, 10.0, 15.0, 25.0):
                for jit in (0.4, 0.8):
                    pc = stats(run_gate(v_c, dn_c, "C", theta0=theta0, k_adapt=k_adapt,
                                        tau_adapt=tau, jitter=jit), 1800)
                    ps = stats(run_gate(v_s, dn_s, "C", theta0=theta0, k_adapt=k_adapt,
                                        tau_adapt=tau, jitter=jit), 1800)
                    if pc["n"] < 3 or ps["n"] < 3:
                        continue
                    rows.append((theta0, k_adapt, tau, jit, pc, ps))

    # 两个工况都要宽分布; 强驱动下允许更规律, 但不得塌成节拍器(CV>=0.35)
    good = [r for r in rows if 12.0 <= r[4]["mean"] <= 45.0
            and r[5]["mean"] < r[4]["mean"] * 0.7
            and r[4]["cv"] >= 0.5 and r[5]["cv"] >= 0.35]
    good.sort(key=lambda r: -min(r[4]["cv"], r[5]["cv"]))
    for theta0, k_adapt, tau, jit, pc, ps in good[:8]:
        print("  %-7.2f %-7.2f %-6.1f %-6.1f | %5.1fs n=%-3d CV=%.2f | %5.1fs n=%-3d CV=%.2f  <="
              % (theta0, k_adapt, tau, jit, pc["mean"], pc["n"], pc["cv"],
                 ps["mean"], ps["n"], ps["cv"]))
    best = good[0][:4] if good else None
    print("\n  推荐默认 (theta0, k_adapt, tau_adapt, jitter):", best if best
          else "(没有满足全部约束的, 需放宽)")
    print("  满足约束的组合数: %d / %d" % (len(good), len(rows)))

    # ---- 长程锁死检查 ----
    if best:
        theta0, k_adapt, tau, jit = best
        v, _ = vocal_trace(0.55, 7200, seed=99)
        dn = dn_trace(7200, seed=98)
        s = stats(run_gate(v, dn, "C", theta0=theta0, k_adapt=k_adapt,
                           tau_adapt=tau, jitter=jit), 7200)
        print("\n  长程 7200s (vocal≈0.55):", "次数=%d 平均间隔=%.1fs CV=%.2f 末段静默=%.1fs -> %s"
              % (s["n"], s["mean"], s["cv"], s["gap"],
                 "未锁死" if 3 <= s["n"] <= 800 else "锁死!"))
        # 脑项是否真的起作用: 关掉 beta_dn 对比
        s_off = stats(run_gate(v, dn, "C", theta0=theta0, k_adapt=k_adapt,
                               tau_adapt=tau, jitter=jit, beta_dn=0.0), 7200)
        print("  同一条 trace 关掉脑项(beta_dn=0): 次数=%d 平均间隔=%.1fs -> 脑项使发声%s"
              % (s_off["n"], s_off["mean"],
                 "变密" if s["n"] > s_off["n"] else ("变稀" if s["n"] < s_off["n"] else "无影响")))


if __name__ == "__main__":
    main()
