import subprocess
def sh(c): return subprocess.run(c, shell=True, executable="/bin/bash", capture_output=True, text=True).stdout.strip()
B = "https://raw.githubusercontent.com/hsgwktb/fly-character/main"
print("fetch:", sh("cd /content/fly-code && for f in webui_character.py fly_llm.py ui.html; do curl -sL -o $f %s/$f; done; wc -c webui_character.py fly_llm.py ui.html" % B))
print("diag端点:", sh("grep -c '/api/diag' /content/fly-code/webui_character.py"))
print("probe端点:", sh("grep -c '/api/probe' /content/fly-code/webui_character.py"))
print("key自动读取:", sh("grep -c '_read_key' /content/fly-code/fly_llm.py"))
sh("pkill -f webui_character.py; sleep 1")
subprocess.Popen("cd /content/fly-code && nohup python -u webui_character.py > /content/fly/webui.log 2>&1 &",
                 shell=True, executable="/bin/bash")
print("RESTARTED")
