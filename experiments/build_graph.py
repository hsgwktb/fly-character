import os, time, traceback
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.feather as feather, pyarrow.ipc as ipc

D = "/content/fly/data"; OUT = "/content/fly/normalized"
STATUS = "/content/fly/status.txt"
os.makedirs(OUT, exist_ok=True)

def status(msg):
    with open(STATUS, "w", encoding="utf-8") as f:
        f.write("%s | %s" % (time.strftime("%H:%M:%S"), msg))
    print(msg, flush=True)

try:
    t0 = time.time()
    status("load annotations")
    ann = feather.read_table(os.path.join(D, "annotations.feather")).to_pandas()
    nt  = feather.read_table(os.path.join(D, "neurotransmitters.feather")).to_pandas()
    retain = ann.superclass.notna() & ann.superclass.astype(str).ne("") & ~ann.status.eq("Glia")
    nodes = ann.loc[retain].reset_index(drop=True)
    ids = nodes.bodyId.to_numpy(dtype=np.int64)
    N = len(ids); status("retained N=%d" % N)

    nt_col = "consensus_nt" if "consensus_nt" in nt.columns else "predicted_nt"
    ntmap = dict(zip(nt["body"].to_numpy(), nt[nt_col].fillna("").astype(str).to_numpy()))
    raw = np.array([ntmap.get(i, "") for i in ids], dtype=object)
    low = np.char.lower(raw.astype(str))
    vc = pd.Series(low).value_counts()
    sign = np.ones(N, dtype=np.float32)
    sign[np.isin(low, ["gaba", "glutamate", "glycine"])] = -1.0
    sign[np.isin(low, ["dopamine", "serotonin", "octopamine", "tyramine", "histamine"])] = 0.0
    status("nt top=%s | sign +%d -%d 0=%d" % (dict(vc.head(4)),
           (sign > 0).sum(), (sign < 0).sum(), (sign == 0).sum()))

    reader = ipc.open_file(pa.memory_map(os.path.join(D, "edges.feather"), "r"))
    nb = reader.num_record_batches
    order_ids = np.argsort(ids, kind="stable"); ids_sorted = ids[order_ids]

    def map_ids(x):
        pos = np.clip(np.searchsorted(ids_sorted, x), 0, N - 1)
        return order_ids[pos].astype(np.int32), ids_sorted[pos] == x

    src_l, dst_l, w_l, n_edges = [], [], [], 0
    for b in range(nb):
        bt = reader.get_batch(b)
        pre  = bt.column(bt.schema.get_field_index("body_pre")).to_numpy(zero_copy_only=False)
        post = bt.column(bt.schema.get_field_index("body_post")).to_numpy(zero_copy_only=False)
        ww   = bt.column(bt.schema.get_field_index("weight")).to_numpy(zero_copy_only=False)
        i, oki = map_ids(pre); j, okj = map_ids(post)
        m = oki & okj
        src_l.append(i[m]); dst_l.append(j[m]); w_l.append(ww[m].astype(np.float32))
        n_edges += int(m.sum())
        status("batch %d/%d retained_edges=%d" % (b + 1, nb, n_edges))

    src = np.concatenate(src_l); dst = np.concatenate(dst_l); w = np.concatenate(w_l)
    status("edges=%d synapses=%d" % (len(src), int(w.sum())))

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
    status("DONE N=%d edges=%d synapses=%d selfloops=%d %.0fs"
           % (N, len(src), int(w.sum()), int((src == dst).sum()), time.time() - t0))
except Exception:
    status("FAILED\n" + traceback.format_exc())
    raise
