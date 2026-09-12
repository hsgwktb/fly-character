import os, subprocess
print(open("/content/fly/driver_status.txt", errors="replace").read() if os.path.exists("/content/fly/driver_status.txt") else "(no status)")
print("proc:", subprocess.run("ps -eo pid,etime,cmd | grep [f]ind_drivers.py", shell=True, capture_output=True, text=True).stdout.strip() or "(finished)")
