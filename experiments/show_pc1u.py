import subprocess
print(subprocess.run("cat /content/fly/pc1u.log", shell=True, capture_output=True, text=True).stdout[-900:])
