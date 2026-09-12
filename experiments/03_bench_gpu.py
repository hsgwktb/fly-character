import os, subprocess, shutil, time
import numpy as np, torch
mi = dict(l.split(":") for l in open("/proc/meminfo").read().splitlines())
print("cores", os.cpu_count(), "| MemTotal", mi["MemTotal"].strip())
print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip())
print("disk free GB", round(shutil.disk_usage("/content").free/1e9, 1))
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA 不可用"
dev = "cuda"
print("GPU:", torch.cuda.get_device_name(0),
      "| %.1f GB" % (torch.cuda.get_device_properties(0).total_memory/1e9))

N, NNZ = 166700, 25_582_938          # MaleCNS 真实规模
rng = np.random.default_rng(0)
rows = torch.from_numpy(rng.integers(0, N, NNZ, dtype=np.int32))
cols = torch.from_numpy(rng.integers(0, N, NNZ, dtype=np.int32))
vals = torch.from_numpy(rng.random(NNZ).astype(np.float32))
order = torch.argsort(rows, stable=True)          # 按 post 排序 -> CSR
r, c, v = rows[order], cols[order], vals[order]
crow = torch.zeros(N + 1, dtype=torch.int32)
crow[1:] = torch.bincount(r, minlength=N).cumsum(0).to(torch.int32)
del rows, cols, order
print("CSR ready: nnz=%d" % v.numel())

crow_g, idx_g, val_g = crow.to(dev), c.to(dev), v.to(dev)
r_g = r.to(dev)
def sync(): torch.cuda.synchronize()

# --- A: CSR sparse.mm ---
try:
    A = torch.sparse_csr_tensor(crow_g, idx_g, val_g, (N, N))
    x = torch.zeros(N, device=dev); x[:2000] = 1.0
    for _ in range(5): y = torch.sparse.mm(A, x.view(-1, 1))
    sync(); t0 = time.time()
    for _ in range(50): y = torch.sparse.mm(A, x.view(-1, 1))
    sync(); a = (time.time()-t0)/50
    print("A CSR sparse.mm      : %.4f s/step -> %.0f M nnz/s" % (a, NNZ/a/1e6))
except Exception as e:
    a = None; print("A CSR sparse.mm 失败:", repr(e)[:160])

# --- B: COO sparse.mm ---
try:
    B = torch.sparse_coo_tensor(torch.stack([r_g, idx_g]), val_g, (N, N)).coalesce()
    x = torch.zeros(N, device=dev); x[:2000] = 1.0
    for _ in range(5): y = torch.sparse.mm(B, x.view(-1, 1))
    sync(); t0 = time.time()
    for _ in range(50): y = torch.sparse.mm(B, x.view(-1, 1))
    sync(); b = (time.time()-t0)/50
    print("B COO sparse.mm      : %.4f s/step -> %.0f M nnz/s" % (b, NNZ/b/1e6))
except Exception as e:
    b = None; print("B COO sparse.mm 失败:", repr(e)[:160])

# --- C: 完整 LIF 步 (gather + scatter-add) ---
Vth, Vr = -45.0, -52.0
def lif_step(V):
    sp = (V > Vth).to(torch.float32)
    I = torch.zeros(N, device=dev).index_add_(0, r_g, val_g * sp[idx_g])
    V = V + 0.01 * (-(V - Vr) + I)
    hit = V >= Vth
    return torch.where(hit, torch.full_like(V, Vr), V)
V = torch.full((N,), Vr, device=dev)
for _ in range(5): V = lif_step(V)
sync(); t0 = time.time()
K = 50
for _ in range(K): V = lif_step(V)
sync(); ctime = (time.time()-t0)/K
print("C 完整 LIF 步        : %.4f s -> %.1f steps/s" % (ctime, 1/ctime))

print("\n--- 1 秒生物时间需要多少墙上时间 (L4) ---")
for dt_ms, lab in ((0.1,"5 kHz 生物保真"), (0.5,"2 kHz"), (1.0,"1 kHz"), (10.0,"100 Hz 速率型")):
    steps = 1000.0/dt_ms
    print("  dt=%-5s %-14s %6d 步 -> %7.1f s" % (dt_ms, lab, steps, steps*ctime))
print("\nCPU 对照(2核): SpMV 0.0861 s/step, 完整 LIF 0.0602 s -> 16.6 steps/s")
