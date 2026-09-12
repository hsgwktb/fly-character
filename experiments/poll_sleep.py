import os, subprocess
print(open("/content/fly/sleep_status.txt", errors="replace").read() if os.path.exists("/content/fly/sleep_status.txt") else "(no status)")
print("proc:", subprocess.run("ps -eo pid,cmd | grep [s]leep_test.py", shell=True, capture_output=True, text=True).stdout.strip() or "(finished)")
