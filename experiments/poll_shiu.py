import os, subprocess
f = "/content/fly/shiu_status.txt"
print(open(f, errors="replace").read() if os.path.exists(f) else "(no status)")
print("proc:", subprocess.run("ps -eo pid,etime,cmd | grep '[l]if_shiu'", shell=True,
                             capture_output=True, text=True).stdout.strip() or "(finished)")
