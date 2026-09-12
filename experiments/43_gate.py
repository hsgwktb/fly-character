import subprocess, os, json
def sh(c): return subprocess.run(c, shell=True, executable="/bin/bash", capture_output=True, text=True).stdout.strip()
print("=== 部署状态 ==="); print(open("/content/llm_status.txt", errors="replace").read() if os.path.exists("/content/llm_status.txt") else "(no status)")
print("=== 闸门结果 ===")
p = "/content/llm_gate.json"
print(open(p).read() if os.path.exists(p) else "(还没产出 → 部署可能仍在跑)")
print("=== 编译/进程 ===")
print(sh("ps -eo pid,etime,cmd | grep -E '[d]eploy_llm|[l]lama-server' | head -3"))
print("=== 显存 ==="); print(sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"))
print("=== gate 日志尾部 ===")
print(sh("grep -E '^A |^B |^C |DONE|FAILED' /content/fly/llm.log | tail -6"))
