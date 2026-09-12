import numpy as np, pandas as pd, pyarrow.feather as feather

D = "/content/fly/data"
raw = feather.read_table(D + "/annotations.feather").to_pandas()
retain = raw.superclass.notna() & raw.superclass.astype(str).ne("") & ~raw.status.eq("Glia")
raw = raw.loc[retain].reset_index(drop=True)
print("保留节点:", len(raw))

ts = raw.type.fillna("").astype(str).str.strip()
syn = raw.synonyms.fillna("").astype(str)
sub = raw.subclass.fillna("").astype(str)
cls = raw["class"].fillna("").astype(str)

print("\n=== 疑似求偶相关命名在 type / synonyms / subclass / class 里 ===")
KW = ["P1", "pC1", "pC2", "aSP", "pMP", "vPR", "Mip", "fru", "dsx",
      "courtship", "song", "wing", "aDT", "pIP", "TP", "IP", "oviposition", "vpo", "ovi"]
for k in KW:
    nt = int(ts.str.fullmatch(k).sum())
    pt = int(ts.str.match(r"^" + k + r"[0-9_ab]*$").sum())
    ct = int(ts.str.contains(k, case=False, regex=False).sum())
    ns = int(syn.str.contains(k, case=False, regex=False).sum())
    print("  %-12s 精确=%-4d 前缀=%-5d type含=%-5d synonyms含=%d" % (k, nt, pt, ct, ns))

print("\n=== type 里以这些前缀开头的具体类型 (前 12) ===")
for k in ["P1", "pC1", "aSP", "pMP", "vPR", "Mip"]:
    vc = ts[ts.str.match(r"^" + k + r"")].value_counts().head(12)
    if len(vc):
        print("  %-6s: %s" % (k, ", ".join("%s(%d)" % (t, n) for t, n in vc.items())))

print("\n=== superclass 为 descending/vnc_motor 且名字含 wing 的 ===")
m = (raw.superclass.astype(str).str.contains("descending|motor")) & ts.str.contains("wing", case=False, regex=False)
print(raw.loc[m, ["bodyId", "type", "superclass", "subclass"]].head(15).to_string()[:900])

print("\n=== DNp13 / DNp11 (文献里的交配相关下行神经元) ===")
for k in ["DNp13", "DNp11", "DNp02", "DNp04"]:
    mm = ts.eq(k)
    if mm.any():
        print(" ", k, "n=", int(mm.sum()), "| superclass:", raw.loc[mm, "superclass"].unique()[:3],
              "| synonyms:", raw.loc[mm, "synonyms"].iloc[0][:80])
