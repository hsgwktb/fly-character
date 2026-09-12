import numpy as np, time
OUT = "/content/fly/normalized"
indptr = np.load(OUT + "/indptr.npy"); indices = np.load(OUT + "/indices.npy")
w = np.load(OUT + "/weights.npy"); ids = np.load(OUT + "/neuron_ids.npy")
N = len(ids)
print("N=%d | col min/max = %d/%d  (越界? %s)" % (N, indices.min(), indices.max(), indices.max() >= N))
print("crow: 首=%d 末=%d 单调=%s 与边数一致=%s" %
      (indptr[0], indptr[-1], (np.diff(indptr) >= 0).all(), indptr[-1] == len(indices)))
deg = np.diff(indptr)

# 行内 col 是否已排序
row_start = np.zeros(len(indices), dtype=bool)
row_start[indptr[:-1][indptr[:-1] < len(indices)]] = True
row_start[0] = True
viol = int((~row_start[1:]) & (indices[1:] < indices[:-1])).sum() if len(indices) > 1 else 0
print("行内 col 未排序的位置数:", viol)

# 是否存在重复 (post, pre) 对
post = np.repeat(np.arange(N, dtype=np.int64), deg)
key = post * np.int64(N) + indices.astype(np.int64)
t0 = time.time(); uniq, cnt = np.unique(key, return_counts=True)
print("边数=%d 唯一对=%d 重复对=%d (%.0fs)" % (len(key), len(uniq), len(key) - len(uniq), time.time() - t0))
dup = cnt > 1
print("涉及重复的对数=%d，其中重复次数 max=%d" % (int(dup.sum()), int(cnt.max())))
if dup.any():
    idx = np.flatnonzero(dup)[:3]
    for i in idx:
        r = int(uniq[i] // N); c = int(uniq[i] % N)
        print("   例: post=%d pre=%d 重复 %d 次" % (r, c, cnt[i]))
