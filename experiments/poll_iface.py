import os, subprocess
print(open("/content/fly/iface_status.txt", errors="replace").read() if os.path.exists("/content/fly/iface_status.txt") else "(no status)")
print("proc:", subprocess.run("ps -eo pid,etime,cmd | grep '[l]if_interface'", shell=True, capture_output=True, text=True).stdout.strip() or "(finished)")
