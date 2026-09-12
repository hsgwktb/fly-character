import os, subprocess
CODE = "/content/fly-code"

def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()

print("--- 验证前：模拟第 3 格会遇到的状态 ---")
print("git status:", sh("cd %s && git status --porcelain | head -5" % CODE) or "(干净)")

print()
print("--- 执行笔记本第 3 格的命令 ---")
sh("cd %s && git fetch --depth 1 origin main" % CODE)
r = subprocess.run("cd %s && git reset --hard FETCH_HEAD" % CODE,
                   shell=True, capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip()[-200:])

print()
print("--- 验证后 ---")
print("git status:", sh("cd %s && git status --porcelain | head -5" % CODE) or "(干净)")
for f in ("webui_character.py", "fly_llm.py", "ui.html", "experiments/build_graph.py"):
    p = os.path.join(CODE, f)
    print("  %-34s %s B" % (f, os.path.getsize(p) if os.path.exists(p) else "缺失"))
print("ui.html 含新界面标记 drv_bars:", sh("grep -c drv_bars %s/ui.html" % CODE))
print("webui 里 UI_B64 残留数:", sh("grep -c UI_B64 %s/webui_character.py || true" % CODE))
print("服务仍在线 GET /:", sh("curl -s -m 8 -o /dev/null -w '%{http_code} %{size_download}B' http://127.0.0.1:8000/"))
