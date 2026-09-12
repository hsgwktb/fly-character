import numpy as np, scipy.sparse as sp, time
N, NNZ = 166700, 25_582_938          # MaleCNS 的真实规模
rng = np.random.default_rng(0)
rows = rng.integers(0, N, NNZ, dtype=np.int32)
cols = rng.integers(0, N, NNZ, dtype=np.int32)
vals = rng.random(NNZ, dtype=np.float32)
t0 = time.time(); M = sp.csr_matrix((vals, (rows, cols)), shape=(N, N)); t1 = time.time()
print("CSR build: %.1f s, nnz=%d, mem=%.0f MB" % (t1-t0, M.nnz, (M.data.nbytes+M.indices.nbytes+M.indptr.nbytes)/1e6))

x = np.zeros(N, dtype=np.float32); x[:2000] = 1.0
for _ in range(3): y = M @ x
K = 30
t0 = time.time()
for _ in range(K): y = M @ x
per = (time.time()-t0)/K
print("SpMV: %.4f s/step  ->  %.0f M nnz/s" % (per, M.nnz/per/1e6))
print("LIF 步频上限(1 SpMV/步): %.1f steps/s" % (1/per))

# 一个完整 LIF 步(SpMV + 膜电位更新 + 阈值/重置)
V = np.full(N, -52.0, dtype=np.float32); Vth = -45.0
t0 = time.time()
for _ in range(K):
    I = M @ (V > Vth).astype(np.float32)
    V = V + 0.01*(-(V+52.0) + I)
    sp_idx = V >= Vth
    V[sp_idx] = -52.0
full = (time.time()-t0)/K
print("完整 LIF 步: %.4f s  ->  %.1f steps/s" % (full, 1/full))
for dt_ms, label in ((0.1,"5 kHz 生物保真"), (0.5,"2 kHz"), (1.0,"1 kHz"), (10.0,"100 Hz 速率型")):
    steps = 1000.0/dt_ms
    print("  1 秒生物时间 @dt=%s (%s): %d 步, 墙上 %.1f s" % (dt_ms, label, steps, steps*full))
