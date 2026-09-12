import os, subprocess, json
print(open("/content/fly/taste_status.txt", errors="replace").read() if os.path.exists("/content/fly/taste_status.txt") else "(no status)")
print("proc:", subprocess.run("ps -eo pid,etime,cmd | grep '[l]if_taste'", shell=True, capture_output=True, text=True).stdout.strip() or "(finished)")
