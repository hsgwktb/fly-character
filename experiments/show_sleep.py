import subprocess, json, os
print(subprocess.run("cat /content/fly/sleep.log | grep -v Warning | grep -v '^  '", shell=True, capture_output=True, text=True).stdout[-1800:])
