import subprocess, time

def sh(c):
    return subprocess.run(c, shell=True, executable="/bin/bash",
                          capture_output=True, text=True).stdout.strip()

print("=== 停掉 base64 版本 ===")
sh("pkill -f webui_character.py; sleep 1")
print(sh("ps -eo pid,cmd | grep [w]ebui_character.py") or "(已停)")

print("\n=== 从 GitHub 拉代码 ===")
print(sh("rm -rf /content/fly-code && git clone --depth 1 "
         "https://github.com/hsgwktb/fly-character.git /content/fly-code 2>&1 | tail -2"))
print(sh("ls /content/fly-code"))

print("\n=== 起服务(从仓库目录) ===")
sh("nohup python -u /content/fly-code/webui_character.py > /content/fly/webui.log 2>&1 &")
time.sleep(22)
print(sh("tail -4 /content/fly/webui.log"))
print("--- 本地自检 ---")
print(sh("curl -s -m 10 'http://127.0.0.1:8000/api/state?since=0' | "
         "python -c \"import sys,json;d=json.load(sys.stdin);"
         "print('tick=%d t=%.1f fps=%.1f acts=%d grounded=%d' % (d['tick'],d['t'],d['fps'],d['n_act'],d['n_grounded']))\""))
print("--- 页面(应由仓库 ui.html 提供) ---")
print(sh("curl -s -m 10 -o /dev/null -w '%{http_code} %{size_download}B' http://127.0.0.1:8000/"))
print("--- 隧道 ---")
print(sh("grep -o 'https://[a-z0-9-]*trycloudflare.com' /content/fly/tunnel.log | head -1"))
