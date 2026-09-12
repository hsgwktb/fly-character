import os, subprocess
p = "/content/fly/char_status.txt"
print(open(p, errors="replace").read() if os.path.exists(p) else "(no status)")
print("proc:", subprocess.run("ps -eo pid,etime,cmd | grep [c]har_loop.py", shell=True, capture_output=True, text=True).stdout.strip() or "(finished)")
