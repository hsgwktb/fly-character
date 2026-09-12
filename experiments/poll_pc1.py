import os, subprocess, json
print(open("/content/fly/pc1_status.txt", errors="replace").read() if os.path.exists("/content/fly/pc1_status.txt") else "(no status)")
print("proc:", subprocess.run("ps -eo pid,etime,cmd | grep [p]c1_drivers.py", shell=True, capture_output=True, text=True).stdout.strip() or "(finished)")
