import subprocess, time
def sh(c): return subprocess.run(c, shell=True, executable="/bin/bash", capture_output=True, text=True).stdout.strip()
B = "https://raw.githubusercontent.com/hsgwktb/fly-character/main/ui.html"
for i in range(6):
    sh("curl -sL -H 'Cache-Control: no-cache' -o /content/fly-code/ui.html '%s?cb=%d'" % (B, int(time.time())))
    size = sh("stat -c %%s /content/fly-code/ui.html")
    ok = sh("grep -c 'display:block;background:#0d1014' /content/fly-code/ui.html")
    print("第%d次: %s 字节, 修复标记=%s" % (i+1, size, ok))
    if ok.strip() == "1":
        break
    time.sleep(4)
print("本地 GET /:", sh("curl -s -m 8 -o /dev/null -w '%{http_code} %{size_download}B' http://127.0.0.1:8000/"))
