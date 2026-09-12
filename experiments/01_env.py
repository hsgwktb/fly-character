import os, subprocess, shutil
print("CPU cores:", os.cpu_count())
mi = dict(l.split(":") for l in open("/proc/meminfo").read().splitlines())
print("MemTotal:", mi["MemTotal"].strip(), "MemAvailable:", mi["MemAvailable"].strip())
print("disk /content:", shutil.disk_usage("/content"))
print("net:", subprocess.run(["bash","-lc","curl -s -o /dev/null -w '%{http_code}' https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather"],capture_output=True,text=True).stdout)
os.makedirs("/content/fly/data", exist_ok=True)
print("ready:", os.listdir("/content/fly"))
