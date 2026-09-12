import subprocess
def sh(c): return subprocess.run(c, shell=True, executable="/bin/bash", capture_output=True, text=True).stdout.strip()
print("pull:", sh("cd /content/fly-code && git pull -q 2>&1 | tail -1") or "(ok)")
print("diag 端点:", sh("grep -c '/api/diag' /content/fly-code/webui_character.py"))
print("probe 端点:", sh("grep -c '/api/probe' /content/fly-code/webui_character.py"))
print("key 自动读取:", sh("grep -c '_read_key' /content/fly-code/fly_llm.py"))
sh("pkill -f webui_character.py; sleep 1")
subprocess.Popen("cd /content/fly-code && nohup python -u webui_character.py > /content/fly/webui.log 2>&1 &",
                 shell=True, executable="/bin/bash")
print("RESTARTED")
