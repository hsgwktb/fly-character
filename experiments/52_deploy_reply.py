import base64, json, os, subprocess, time, urllib.request

CODE = "/content/fly-code"
API = "https://api.github.com/repos/hsgwktb/fly-character/contents/"

def fetch(path):
    req = urllib.request.Request(API + path, headers={
        "Accept": "application/vnd.github+json", "User-Agent": "fly"})
    d = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
    raw = base64.b64decode(d["content"])
    open(os.path.join(CODE, path), "wb").write(raw)
    return len(raw)

for f in ("webui_character.py", "fly_llm.py", "ui.html"):
    print("拉到 %-22s %d 字节" % (f, fetch(f)))

def sh(c):
    return subprocess.run(c, shell=True, executable="/bin/bash",
                          capture_output=True, text=True).stdout.strip()

print("reply_mode 标记:", sh("grep -c reply_mode %s/webui_character.py" % CODE))
print("强制回复分支  :", sh("grep -c '强制回复' %s/webui_character.py" % CODE))
print("UI 里的 reply 滑块:", sh("grep -c \"'reply'\" %s/ui.html" % CODE))

sh("pkill -f webui_character.py; sleep 1")
subprocess.Popen("cd %s && setsid nohup python -u webui_character.py "
                 "> /content/fly/webui.log 2>&1 < /dev/null &" % CODE,
                 shell=True, executable="/bin/bash")
time.sleep(16)
print()
print("日志:", sh("tail -3 /content/fly/webui.log"))
print("GET / =", sh("curl -s -m 8 -o /dev/null -w '%{http_code} %{size_download}B' http://127.0.0.1:8000/"))
print("state 里的新字段:", sh("curl -s -m 8 http://127.0.0.1:8000/api/state | "
                            "python -c \"import sys,json;d=json.load(sys.stdin);"
                            "print('reply_mode=',d.get('reply_mode'),'unanswered=',d.get('unanswered'),"
                            "'params.reply=',d['params'].get('reply'))\""))
