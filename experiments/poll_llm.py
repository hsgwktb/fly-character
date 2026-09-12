import os, subprocess
print(open("/content/llm_status.txt", errors="replace").read() if os.path.exists("/content/llm_status.txt") else "(no status)")
print("proc:", subprocess.run("ps -eo pid,etime,cmd | grep -E '[d]eploy_llm|[l]lama-server' | head -3", shell=True, capture_output=True, text=True).stdout.strip() or "(none)")
