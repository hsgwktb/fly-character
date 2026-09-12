import subprocess, time, os

def sh(cmd):
    return subprocess.run(cmd, shell=True, executable="/bin/bash",
                          capture_output=True, text=True).stdout.strip()

print("=== webui.log ===")
print(sh("tail -6 /content/fly/webui.log") or "(empty)")
print("=== 进程 ===", sh("ps -eo pid,etime,cmd | grep [w]ebui_character.py") or "(未运行)")
print("=== GET /api/state ===")
print(sh("curl -s -m 10 http://127.0.0.1:8000/api/state | head -c 700"))
print("=== GET / ===", sh("curl -s -m 10 -o /dev/null -w '%{http_code} %{size_download}B' http://127.0.0.1:8000/"))

print("\n=== cloudflared ===")
sh("CF=/content/cloudflared; [ -s $CF ] || wget -q -O $CF https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64; chmod +x $CF")
print(sh("ls -la /content/cloudflared"))
sh("pkill -f cloudflared; sleep 1")
sh("nohup /content/cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate > /content/tunnel.log 2>&1 &")
url = ""
for i in range(30):
    time.sleep(3)
    u = sh("grep -o 'https://[a-z0-9-]*\\.trycloudflare\\.com' /content/tunnel.log | head -1")
    if u:
        url = u.split()[0]; break
print("TUNNEL_URL:", url or "(none)")
if not url:
    print(sh("tail -20 /content/tunnel.log"))
