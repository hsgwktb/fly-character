import subprocess, json, base64, urllib.request, time

def sh(c):
    return subprocess.run(c, shell=True, executable="/bin/bash",
                          capture_output=True, text=True).stdout.strip()

MARK = "display:block;background:#0d1014"
TGT = "/content/fly-code/ui.html"

# 1) 先试 git pull（git 协议不走 raw 的 CDN 缓存）
sh("cd /content/fly-code && timeout 40 git pull -q 2>&1 | tail -1")
ok = sh("grep -c '%s' %s" % (MARK, TGT))
print("git pull 后 标记数 =", ok)
if ok.strip() != "1":
    # 2) 退回 GitHub API contents（含 base64 内容，不走 raw CDN）
    for i in range(3):
        try:
            req = urllib.request.Request(
                "https://api.github.com/repos/hsgwktb/fly-character/contents/ui.html",
                headers={"Accept": "application/vnd.github+json", "User-Agent": "fly"})
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read().decode())
            raw = base64.b64decode(d["content"])
            open(TGT, "wb").write(raw)
            print("API 拉取 %d 字节, sha=%s" % (len(raw), d["sha"][:12]))
            break
        except Exception as e:
            print("API 失败:", repr(e)[:120]); time.sleep(4)

print("最终标记数 =", sh("grep -c '%s' %s" % (MARK, TGT)))
print("文件大小 =", sh("wc -c < " + TGT))
print("本地 GET / =", sh("curl -s -m 8 -o /dev/null -w '%{http_code} %{size_download}B' http://127.0.0.1:8000/"))
