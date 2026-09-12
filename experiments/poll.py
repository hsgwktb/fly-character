import os
for f in ["/content/fly/reflex_status.txt", "/content/fly/reflex.log"]:
    if os.path.exists(f):
        print("=== %s ===" % f); print(open(f, errors="replace").read()[-1500:])
import subprocess
print("proc:", subprocess.run("ps -eo pid,etime,cmd | grep '[l]if_reflex'", shell=True,
                             capture_output=True, text=True).stdout.strip() or "(finished)")
