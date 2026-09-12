import os, subprocess, glob
print("=== /content/fly ===")
for p in sorted(glob.glob("/content/fly/*")):
    if os.path.isdir(p):
        print("  dir ", p, os.listdir(p)[:8])
    else:
        print("  file", p, os.path.getsize(p))
print("=== 真进程 (排除 shell 自匹配) ===")
print(subprocess.run("ps -eo pid,etime,cmd | grep 'python' | grep -v grep | grep -E 'build_graph|ipykernel'",
                     shell=True, capture_output=True, text=True).stdout.strip() or "(无)")
for f in ["/content/fly/status.txt", "/content/fly/build.log"]:
    if os.path.exists(f):
        print("=== %s ===" % f)
        print(open(f, errors="replace").read()[-1200:])
