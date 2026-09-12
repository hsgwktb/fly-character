import numpy as np, time, torch
OUT = "/content/fly/normalized"
t0 = time.time()
indptr = np.load(OUT + "/indptr.npy"); src = np.load(OUT + "/indices.npy")
w = np.load(OUT + "/weights.npy"); ids = np.load(OUT + "/neuron_ids.npy")
N = len(ids)
deg = np.diff(indptr)
post = np.repeat(np.arange(N, dtype=np.int64), deg).astype(np.int64)
print("规范化前: 边=%d" % len(src))

o = np.lexsort((src.astype(np.int64), post))        # 先按 post, 再按 pre
post, src, w = post[o], src[o], w[o]
keep = np.empty(len(src), dtype=bool); keep[0] = True
keep[1:] = (post[1:] != post[:-1]) | (src[1:] != src[:-1])
starts = np.flatnonzero(keep)
print("唯一 (post,pre) 对 = %d | 被合并的重复边 = %d" % (len(starts), len(src) - len(starts)))
w_co = np.add.reduceat(w.astype(np.float64), starts).astype(np.float32)
post_c, src_c = post[keep], src[keep]

indptr2 = np.zeros(N + 1, dtype=np.int64); np.add.at(indptr2, post_c + 1, 1); indptr2 = np.cumsum(indptr2)
print("规范化后: 边=%d 突触=%d (%.0fs)" % (len(src_c), int(w_co.sum()), time.time() - t0))

np.save(OUT + "/indptr.npy", indptr2)
np.save(OUT + "/indices.npy", src_c)
np.save(OUT + "/weights.npy", w_co)
# 行内 col 已排序?
row_ok = True
for r in (0, 1, 1000, N - 2, N - 1):
    s, e = indptr2[r], indptr2[r + 1]
    if e - s > 1 and not (np.diff(src_c[s:e]) > 0).all(): row_ok = False
print("抽样行内 col 严格递增:", row_ok)

# CPU 上做一次不变量校验 + 与 scipy 对拍
A = torch.sparse_csr_tensor(torch.from_numpy(indptr2.astype(np.int32)),
                            torch.from_numpy(src_c.astype(np.int32)),
                            torch.from_numpy(w_co), (N, N), check_invariants=True)
print("PyTorch CSR 不变量校验: 通过")
import scipy.sparse as sp
M = sp.csr_matrix((w_co, src_c, indptr2), shape=(N, N))
x = np.zeros(N, dtype=np.float32); x[:500] = 1.0
y_t = (A @ torch.from_numpy(x).view(-1, 1)).view(-1).numpy()
y_s = M @ x
print("与 scipy 对拍 最大绝对误差: %.3e" % np.abs(y_t - y_s).max())
