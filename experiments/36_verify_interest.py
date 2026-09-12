import subprocess, time, json, statistics

def sh(c):
    return subprocess.run(c, shell=True, executable="/bin/bash",
                          capture_output=True, text=True).stdout.strip()

def state():
    try:
        return json.loads(sh("curl -s -m 8 'http://127.0.0.1:8000/api/state'"))
    except Exception:
        return None

print("=== 拉取仓库更新并重启 ===")
print("courtship_index 命中:", sh("cd /content/fly-code && git pull -q 2>&1|tail -1; grep -c courtship_index webui_character.py"))
print("interest 命中:", sh("grep -c 'self.interest' /content/fly-code/webui_character.py"))
sh("pkill -f webui_character.py; sleep 1")
sh("nohup python -u /content/fly-code/webui_character.py > /content/fly/webui.log 2>&1 &")
time.sleep(18)
s = state()
print("新字段 interest =", s.get("interest"), "| params 含 mi:", "mi" in s["params"])
print("6 条曲线全在:", all(k in s["d"] for k in ["hunger","thirst","boredom","lonely","sexual","threat"]))

print("\n=== 采样 120s: 兴趣 vs 无聊 是否只是补集? ===")
print("  %6s %9s %9s %9s" % ("t", "兴趣", "无聊", "1-无聊"))
I, B = [], []
gap = (0.0, 0.0)
for i in range(30):
    time.sleep(4)
    s = state()
    if s is None: continue
    it, bo = s["interest"], s["d"]["boredom"]
    I.append(it); B.append(bo)
    g = abs(it - (1 - bo))
    if g > gap[0]: gap = (g, s["t"])
    if i % 3 == 0:
        print("  %6.0f %9.3f %9.3f %9.3f" % (s["t"], it, bo, 1 - bo))

if len(I) > 3:
    n = len(I)
    mi, mb = statistics.mean(I), statistics.mean(B)
    cov = sum((I[k]-mi)*(B[k]-mb) for k in range(n)) / n
    si = statistics.pstdev(I) or 1e-9
    sb = statistics.pstdev(B) or 1e-9
    r = cov / (si*sb)
    print("\n  兴趣均值=%.3f  无聊均值=%.3f" % (mi, mb))
    print("  相关系数 r(兴趣, 无聊) = %+.3f   (若为补集应≈ -1.000)" % r)
    print("  平均 |兴趣-(1-无聊)| = %.3f" % (sum(abs(I[k]-(1-B[k])) for k in range(n))/n))
    print("  最大偏离 %.3f (t=%.0fs)  <- 两者同时偏离补集关系的时刻" % gap)
    print("\n  结论:", "兴趣不是无聊的补集(相关明显高于 -1, 且存在偏离时刻)" if r > -0.98 else "两者近似补集, 需重新设计")
