import numpy as np, pandas as pd, pyarrow.feather as feather

D = "/content/fly/data"
raw = feather.read_table(D + "/annotations.feather").to_pandas()
retain = raw.superclass.notna() & raw.superclass.astype(str).ne("") & ~raw.status.eq("Glia")
raw = raw.loc[retain].reset_index(drop=True)
ts = raw.type.fillna("").astype(str).str.strip()
syn = raw.synonyms.fillna("").astype(str)
sub = raw.subclass.fillna("").astype(str)
cls = raw["class"].fillna("").astype(str)
allt = ts + " | " + syn
print("保留节点:", len(raw))

GROUPS = {
    "促眠 dFSB/扇形体": ["dFSB", "FSB", "fSB", "dFB", "FB2", "FB4", "FB5", "FB6", "FB7", "FB8",
                        "helicon", "Helicon", "ExR", "23E10", "R23E10", "fan-shaped", "fan shaped"],
    "时钟/节律":        ["PDF", "LNv", "s-LN", "l-LN", "LNd", "LPN", "DN1", "DN2", "DN3",
                        "clock", "period", "timeless", "pigment-dispersing", "cryptochrome"],
    "促醒 章鱼胺":      ["octopamine", "OA-VUM", "VUM", "Tdc2", "Tbh", "octopaminergic"],
    "促醒 多巴胺":      ["dopamine", "dopaminergic", "PAM", "PPL1", "PPL2", "PAL", "TH-"],
    "5-HT":             ["serotonin", "5-HT", "5HT", "TRH", "Tph"],
    "睡眠相关肽":       ["Dh44", "AstA", "AstC", "allatostatin", "sNPF", "NPF", "diuretic",
                        "ITP", "ion transport"],
    "GABA 能":          ["GABA", "Gad1", "GAD"],
}
for name, kws in GROUPS.items():
    hits = {}
    for kw in kws:
        m = allt.str.contains(kw, case=False, regex=False)
        if m.sum():
            hits[kw] = int(m.sum())
    print("\n=== %s ===" % name)
    if not hits:
        print("   (无命中)")
        continue
    print("   ", hits)
    # 展示 type 里前缀命中的具体类型
    for kw in list(hits)[:3]:
        m2 = ts.str.contains(kw, case=False, regex=False)
        vc = ts[m2].value_counts().head(8)
        if len(vc):
            print("    type 含 %-14s: %s" % (kw, ", ".join("%s(%d)" % (t, n) for t, n in vc.items())))

print("\n=== 有 synonyms 佐证的例子(促眠/时钟) ===")
for kw in ["dFSB", "PDF", "23E10", "AstA", "Dh44"]:
    m = syn.str.contains(kw, case=False, regex=False)
    if m.any():
        ex = raw.loc[m, ["type", "superclass", "synonyms"]].head(3)
        print("  [%s] n=%d" % (kw, int(m.sum())))
        print(ex.to_string()[:700])
