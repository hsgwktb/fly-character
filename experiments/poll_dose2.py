import os, subprocess
print(open("/content/fly/dose_status.txt", errors="replace").read() if os.path.exists("/content/fly/dose_status.txt") else "(no status)")
print("proc:", subprocess.run("ps -eo pid,etime,cmd | grep '[l]if_dose'", shell=True, capture_output=True, text=True).stdout.strip() or "(finished)")
