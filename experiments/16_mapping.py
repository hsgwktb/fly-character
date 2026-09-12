import numpy as np, pandas as pd
ann = pd.read_feather("/content/fly/normalized/neurons.feather")
raw = __import__("pyarrow.feather", fromlist=["feather"]).read_table("/content/fly/data/annotations.feather").to_pandas()
retain = raw.superclass.notna() & raw.superclass.astype(str).ne("") & ~raw.status.eq("Glia")
raw = raw.loc[retain].reset_index(drop=True)

print("=== 哪些列里有 'GF' / 'DNp01' ===")
for c in ["type", "flywireType", "hemibrainType", "mancType", "synonyms", "subclass", "class", "supertype", "group"]:
    if c in raw.columns:
        s = raw[c].astype(str)
        print("  %-14s exact GF=%d | contains DNp01=%d | exact DNp01=%d" %
              (c, int(s.eq("GF").sum()), int(s.str.contains("DNp01", regex=False).sum()), int(s.eq("DNp01").sum())))

print("\n=== DNp01 那 2 个节点在各列的值 ===")
m = raw["type"].astype(str).eq("DNp01")
print(raw.loc[m, ["bodyId", "type", "flywireType", "hemibrainType", "mancType", "synonyms"]].to_string()[:800])

print("\n=== 苦味通路诊断 ===")
OUT = "/content/fly/normalized"
indptr = np.load(OUT + "/indptr.npy"); pre = np.load(OUT + "/indices.npy")
post = np.repeat(np.arange(len(ann), dtype=np.int64), np.diff(indptr))
ts = ann.type.astype(str).str.strip()
bitter = np.flatnonzero(ts.str.match(r"^LB1[a-d]$").to_numpy())
sugar  = np.flatnonzero(ts.str.match(r"^LB3[a-d]$").to_numpy())
print("bitter n=%d sugar n=%d" % (len(bitter), len(sugar)))

bl = np.zeros(len(ann), dtype=bool); bl[bitter] = True
sl = np.zeros(len(ann), dtype=bool); sl[sugar] = True
for name, lab in (("bitter LB1", bl), ("sugar LB3", sl)):
    m = lab[pre]
    tgt = post[m]
    print("\n%s: 一级下游 %d 条边 / %d 个唯一目标" % (name, int(m.sum()), len(np.unique(tgt))))
    vc = ts.iloc[np.unique(tgt)].value_counts().head(8)
    print("   目标类型 top8:", dict(vc))

# MN9 的突触前集合
mn9 = np.flatnonzero(ts.eq("MN9").to_numpy())
pre_mn9 = np.concatenate([pre[indptr[j]:indptr[j+1]] for j in mn9]) if len(mn9) else np.array([], dtype=int)
print("\nMN9 突触前神经元数 = %d" % len(np.unique(pre_mn9)))
for name, lab in (("bitter", bl), ("sugar", sl)):
    m = lab[pre]; tgt = np.unique(post[m])
    hit = np.intersect1d(tgt, pre_mn9)
    print("   %s 的 2 跳内能到 MN9 的中间神经元数 = %d" % (name, len(hit)))
    if len(hit):
        print("      类型:", dict(pd.Series(ts.iloc[hit]).value_counts().head(5)))
