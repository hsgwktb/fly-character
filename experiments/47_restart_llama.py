import subprocess, time, os
def sh(c): return subprocess.run(c, shell=True, executable="/bin/bash", capture_output=True, text=True).stdout.strip()
print("模型:", sh("ls -la /content/model/*.gguf 2>/dev/null | tail -1") or "(无!)")
print("二进制:", sh("ls -la /content/llama.cpp/build/bin/llama-server 2>/dev/null | tail -1") or "(无!)")
key = open("/content/api_key.txt").read().strip()
m = sh("ls /content/model/*.gguf | head -1")
sh("pkill -f llama-server; sleep 2")
# setsid: 另开进程组, 这样内核被中断时不会连带杀掉它(上次就是这么死的)
cmd = ("setsid nohup /content/llama.cpp/build/bin/llama-server -m %s -ngl 99 -c 16384 -fa on --jinja "
       "--host 127.0.0.1 --port 8081 -np 2 --api-key %s --alias qwen3.8-27b-uncensored "
       "> /content/llama-server.log 2>&1 < /dev/null &") % (m, key)
subprocess.Popen(cmd, shell=True, executable="/bin/bash")
for i in range(20):
    time.sleep(4)
    code = sh("curl -s -o /dev/null -w '%%{http_code}' -m 4 http://127.0.0.1:8081/health")
    if code in ("200", "401"):
        print("就绪 %ds, health=%s" % ((i+1)*4, code)); break
else:
    print("未就绪, 日志尾部:"); print(sh("tail -8 /content/llama-server.log"))
print("gpu:", sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"))
