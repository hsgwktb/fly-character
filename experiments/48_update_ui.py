import subprocess
def sh(c): return subprocess.run(c, shell=True, executable="/bin/bash", capture_output=True, text=True).stdout.strip()
B = "https://raw.githubusercontent.com/hsgwktb/fly-character/main"
print("拉取 ui.html:", sh("curl -sL -o /content/fly-code/ui.html %s/ui.html && stat -c %%s /content/fly-code/ui.html" % B))
print("本地 GET /:", sh("curl -s -m 8 -o /dev/null -w '%{http_code} %{size_download}B' http://127.0.0.1:8000/"))
print("新元素检查:", sh("grep -c 'drv_bars' /content/fly-code/ui.html"), sh("grep -c 'SPEC' /content/fly-code/ui.html"))
print("(无需重启: 后端每次请求都从磁盘读 ui.html)")
