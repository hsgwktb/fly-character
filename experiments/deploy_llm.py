"""在 Colab 上部署 Qwen3.8-27B-Uncensored 并启动 llama-server（后台任务版）。
与笔记本的差异：-c 16384 -np 2（同样显存，多一个并发槽）；末尾自动做"关思考"闸门验证。
"""
import subprocess, sys, time, os, json, base64

STATUS = "/content/llm_status.txt"
def status(m):
    open(STATUS, "w", encoding="utf-8").write("%s | %s" % (time.strftime("%H:%M:%S"), m))
    print(m, flush=True)

def sh(cmd, quiet=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if not quiet and r.stdout.strip():
        print(r.stdout.strip()[-600:], flush=True)
    return r

t0 = time.time()
try:
    status("apt + pip")
    sh("apt-get -qq update")
    sh("apt-get -qq install -y build-essential cmake ninja-build git libcurl4-openssl-dev")
    sh(f"{sys.executable} -m pip install -q fastapi uvicorn httpx requests huggingface_hub")
    status("deps done (%.0fs)" % (time.time()-t0))

    BIN = "/content/llama.cpp/build/bin/llama-server"
    if os.path.exists(BIN):
        status("llama-server 已存在，跳过编译")
    else:
        if not os.path.isdir("/content/llama.cpp"):
            status("clone llama.cpp")
            sh("git clone --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp")
        env = "export PATH=/usr/local/cuda/bin:$PATH && "
        status("cmake 配置")
        r = sh("cd /content/llama.cpp && " + env + "cmake -B build -G Ninja "
               "-DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON "
               "-DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF "
               "-DCMAKE_CUDA_ARCHITECTURES=native > /content/cmake.log 2>&1")
        if r.returncode:
            status("FAILED cmake"); raise SystemExit(1)
        status("编译中(10-20min)…")
        r = sh("cd /content/llama.cpp && " + env + "cmake --build build -j$(nproc) "
               "--target llama-server > /content/build.log 2>&1")
        if r.returncode or not os.path.exists(BIN):
            status("FAILED build"); print(sh("tail -25 /content/build.log", quiet=False)); raise SystemExit(1)
        status("编译完成 (%.0fs)" % (time.time()-t0))

    REPO = "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF"
    FILE = "Huihui-Qwen3.8-27B-abliterated-UD-IQ4_XS.gguf"
    from huggingface_hub import hf_hub_download
    status("下载模型(3-10min)…")
    MODEL = hf_hub_download(repo_id=REPO, filename=FILE, local_dir="/content/model")
    status("模型就绪 %.2f GB" % (os.path.getsize(MODEL)/1e9))

    sh("pkill -f llama-server"); time.sleep(2)
    KF = "/content/api_key.txt"
    KEY = open(KF).read().strip() if os.path.exists(KF) and open(KF).read().strip() else \
          ("sk-" + base64.urlsafe_b64encode(os.urandom(24)).decode().rstrip("="))
    open(KF, "w").write(KEY)
    CTX, NP = 16384, 2
    cmd = [BIN, "-m", MODEL, "-ngl", "99", "-c", str(CTX), "-fa", "on", "--jinja",
           "--host", "127.0.0.1", "--port", "8081", "-np", str(NP),
           "--api-key", KEY, "--alias", "qwen3.8-27b-uncensored"]
    status("启动 llama-server (ctx=%d np=%d)…" % (CTX, NP))
    srv = subprocess.Popen(cmd, stdout=open("/content/llama-server.log", "w"), stderr=subprocess.STDOUT)

    import requests
    ok = False
    for i in range(240):
        time.sleep(4)
        if srv.poll() is not None:
            print(open("/content/llama-server.log", errors="replace").read()[-2000:], flush=True)
            status("FAILED llama-server 退出"); raise SystemExit(1)
        try:
            if requests.get("http://127.0.0.1:8081/health", timeout=3).status_code in (200, 401):
                ok = True; break
        except Exception:
            pass
    if not ok:
        status("FAILED 就绪超时"); raise SystemExit(1)
    status("llama-server 就绪 %ds" % (i*4))
    print(sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader", quiet=False), flush=True)

    # ---------- 闸门验证：思考能不能关掉 ----------
    def chat(payload, timeout=300):
        return requests.post("http://127.0.0.1:8081/v1/chat/completions",
                             headers={"Authorization": "Bearer " + KEY},
                             json=payload, timeout=timeout).json()

    base = {"model": "qwen3.8-27b-uncensored", "max_tokens": 200, "temperature": 0.7}
    msgs = [{"role": "system", "content": "只输出一句很短的中文。"},
            {"role": "user", "content": "你现在很无聊，说一句。"}]
    r1 = chat(dict(base, messages=msgs))
    m1 = r1["choices"][0]["message"]
    rc1 = (m1.get("reasoning_content") or "")
    print("A 默认          : reasoning %d 字 | content %d 字" % (len(rc1), len(m1.get("content") or "")), flush=True)

    msgs2 = [{"role": "system", "content": "只输出一句很短的中文。 /no_think"},
             {"role": "user", "content": "你现在很无聊，说一句。"}]
    r2 = chat(dict(base, messages=msgs2))
    m2 = r2["choices"][0]["message"]
    rc2 = (m2.get("reasoning_content") or "")
    print("B /no_think     : reasoning %d 字 | content %d 字" % (len(rc2), len(m2.get("content") or "")), flush=True)

    r3 = chat(dict(base, messages=msgs, chat_template_kwargs={"enable_thinking": False}))
    m3 = r3["choices"][0]["message"]
    rc3 = (m3.get("reasoning_content") or "")
    print("C enable_thinking=False : reasoning %d 字 | content %d 字" % (len(rc3), len(m3.get("content") or "")), flush=True)

    verdict = "C" if len(rc3) < 10 else ("B" if len(rc2) < 10 else ("A" if len(rc1) < 10 else "NONE"))
    res = {"gate": verdict, "A": [len(rc1), len(m1.get("content") or "")],
           "B": [len(rc2), len(m2.get("content") or "")],
           "C": [len(rc3), len(m3.get("content") or "")],
           "ctx": CTX, "np": NP, "api_key": KEY}
    json.dump(res, open("/content/llm_gate.json", "w"), indent=1)
    status("DONE 关思考方案=%s (A默认/B no_think/C kwargs) 总耗时%.0fs" % (verdict, time.time()-t0))
except Exception as e:
    import traceback
    status("FAILED\n" + traceback.format_exc())
    raise
