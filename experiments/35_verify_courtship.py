import subprocess, time, json

def sh(c):
    return subprocess.run(c, shell=True, executable="/bin/bash",
                          capture_output=True, text=True).stdout.strip()

def state():
    try:
        return json.loads(sh("curl -s -m 8 'http://127.0.0.1:8000/api/state'"))
    except Exception:
        return None

print("=== 拉取仓库更新 ===")
print(sh("cd /content/fly-code && git pull -q 2>&1 | tail -2; grep -c 'courtship_index' webui_character.py"))

print("\n=== 重启服务 ===")
sh("pkill -f webui_character.py; sleep 1")
sh("nohup python -u /content/fly-code/webui_character.py > /content/fly/webui.log 2>&1 &")
time.sleep(20)
print(sh("tail -2 /content/fly/webui.log"))

s = state()
print("\n=== 新字段检查 ===")
print("ci = %s | social = %s | sexual = %s" % (s.get("ci"), s.get("social"), s["d"]["sexual"]))
print("params 含 mc/soc:", "mc" in s["params"], "soc" in s["params"])
print("初始 CI=%.3f (无同类, 应为 0)" % s["ci"])

print("\n=== 把同类放进视野 + 提高性欲倍率 ===")
sh("curl -s -m 8 -X POST -H 'Content-Type: application/json' "
   "-d '{\"soc\":1.0,\"mc\":3.0}' http://127.0.0.1:8000/api/control >/dev/null")
print("已设 soc=1.0, mc=3.0；等性欲累积…")
best = 0.0
for i in range(12):
    time.sleep(12)
    s = state()
    if s is None:
        print("  (state 读取失败)"); continue
    print("  t=%5.0fs  性欲=%.3f  社交=%.2f  CI=%.3f  court事件=%d"
          % (s["t"], s["d"]["sexual"], s["social"], s["ci"],
             sum(1 for e in s["events"] if "court" in e["text"])))
    best = max(best, s["ci"])
    if s["ci"] > 0.02 and i >= 3:
        break

print("\n=== 结论 ===")
print("求偶指数峰值 CI = %.3f  ->  %s" % (best, "指标工作正常" if best > 0 else "未观察到期偶(需更长/更高倍率)"))
