import os, time
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.feather as feather, pyarrow.ipc as ipc

D = "/content/fly/data"; OUT = "/content/fly/normalized"; os.makedirs(OUT, exist_ok=True)
t0 = time.time()
ann = feather.read_table(os.path.join(D, "annotations.feather")).to_pandas()
nt  = feather.read_table(os.path.join(D, "neurotransmitters.feather")).to_pandas()

retain = ann.superclass.notna() & ann.superclass.astype(str).ne("") & ~ann.status.eq("Glia")
nodes = ann.loc[retain].reset_index(drop=True)
ids = nodes.bodyId.to_numpy(dtype=np.int64)
assert len(np.unique(ids)) == len(ids)
N = len(ids); print("N =", N)

# 用文档/DOOMFLY 确认过的列: body -> consensus_nt
nt_col = "consensus_nt" if "consensus_nt" in nt.columns else "predicted_nt"
print("nt col =", nt_col)
ntmap = dict(zip(nt["body"].to_numpy(), nt[nt_col].fillna("").astype(str).to_numpy()))
raw = np.array([ntmap.get(i, "") for i in ids], dtype=object)
low = np.char.lower(raw.astype(str))
vc = pd.Series(low).value_counts()
print("神经递质分布(top8):", vc.head(8).to_dict())

sign = np.ones(N, dtype=np.float32)                       # 默认兴奋(蝇类多为胆碱能)
sign[np.isin(low, ["gaba", "glutamate", "glycine"])] = -1.0
sign[np.isin(low, ["dopamine", "serotonin", "octopamine", "tyramine", "histamine"])] = 0.0
print("符号: +1 %d, -1 %d, 0 %d" % ((sign > 0).sum(), (sign < 0).sum(), (sign == 0).sum()))

reader = ipc.open_file(pa.memory_map(os.path.join(D, "edges.feather"), "r"))
order_ids = np.argsort(ids, kind="stable"); ids_sorted = ids[order_ids]
def map_ids(x):
    pos = np.clip(np.searchsorted(ids_sorted, x), 0, N - 1)
    return order_ids[pos].astype(np.int32), ids_sorted[pos] == x

src_l, dst_l, w_l = [], [], []
for b in range(reader.num_record_batches):
    bt = reader.get_batch(b)
    pre  = bt.column(bt.schema.get_field_index("body_pre")).to_numpy(zero_copy_only=False)
    post = bt.column(bt.schema.get_field_index("body_post")).to_numpy(zero_copy_only=False)
    ww   = bt.column(bt.schema.get_field_index("weight")).to_numpy(zero_copy_only=False)
    i, oki = map_ids(pre); j, okj = map_ids(post)
    m = oki & okj
    src_l.append(i[m]); dst_l.append(j[m]); w_l.append(ww[m].astype(np.float32))
src = np.concatenate(src_l); dst = np.concatenate(dst_l); w = np.concatenate(w_l)
del src_l, dst_l, w_l
print("retained edges = %d | synapses = %d | %.0fs" % (len(src), int(w.sum()), time.time()-t0))

o2 = np.argsort(dst, kind="stable")
src, dst, w = src[o2], dst[o2], w[o2]
indptr = np.zeros(N + 1, dtype=np.int64); np.add.at(indptr, dst + 1, 1); indptr = np.cumsum(indptr)

np.save(os.path.join(OUT, "indptr.npy"), indptr)
np.save(os.path.join(OUT, "indices.npy"), src)
np.save(os.path.join(OUT, "weights.npy"), w)
np.save(os.path.join(OUT, "sign.npy"), sign)
np.save(os.path.join(OUT, "neuron_ids.npy"), ids)
pd.DataFrame({
    "node_index": np.arange(N, dtype=np.int32), "bodyId": ids,
    "superclass": nodes.superclass.astype(str).to_numpy(),
    "type": nodes.type.astype(str).to_numpy(),
    "side": nodes.somaSide.astype(str).to_numpy(),
    "neurotransmitter": raw, "sign": sign,
}).to_feather(os.path.join(OUT, "neurons.feather"))

print("saved:", sorted(os.listdir(OUT)))
print("self-loops %d | mean out-degree %.1f | 总耗时 %.0fs"
      % (int((src == dst).sum()), len(src)/N, time.time()-t0))
