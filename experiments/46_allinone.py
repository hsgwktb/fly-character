import subprocess, time, os
def sh(c): return subprocess.run(c, shell=True, executable="/bin/bash", capture_output=True, text=True).stdout.strip()
B = "https://raw.githubusercontent.com/hsgwktb/fly-character/main"
print("fetch:", sh("cd /content/fly-code && for f in webui_character.py fly_llm.py ui.html; do curl -sL -o $f %s/$f; done; wc -c webui_character.py fly_llm.py ui.html" % B))
print("diag端点:", sh("grep -c '/api/diag' /content/fly-code/webui_character.py"), "| keyfix:", sh("grep -c '_read_key' /content/fly-code/fly_llm.py"))
sh("pkill -f webui_character.py; sleep 1")
subprocess.Popen("cd /content/fly-code && nohup python -u webui_character.py > /content/fly/webui.log 2>&1 &", shell=True, executable="/bin/bash")
print("角色服务已重启")
if os.path.exists("/content/cloudflared"):
    sh("pkill -f cloudflared; sleep 1")
    subprocess.Popen("nohup /content/cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate > /content/fly/tunnel.log 2>&1 &", shell=True, executable="/bin/bash")
    time.sleep(14)
    print("tunnel:", sh("grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /content/fly/tunnel.log | head -1") or "(未出)")
