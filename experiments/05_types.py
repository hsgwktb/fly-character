import pyarrow.feather as feather, pandas as pd, numpy as np, os
D = "/content/fly/data"
ann = feather.read_table(os.path.join(D, "annotations.feather")).to_pandas()
print("shape:", ann.shape, "| columns:", list(ann.columns))
print("\nsuperclass:")
print(ann.superclass.value_counts(dropna=False).to_string())
print("\nstatus:", ann.status.value_counts(dropna=False).to_dict())

retain = ann.superclass.notna() & ann.superclass.astype(str).ne("") & ~ann.status.eq("Glia")
print("\nretained nodes:", int(retain.sum()))
t = ann.loc[retain, "type"].astype(str)
named = ~t.isin(["nan", "", "None"])
print("有 type 名称的节点:", int(named.sum()), "/", len(t))

print("\n=== 按名字前缀分布 ===")
pref = t[named].str.extract(r"^([A-Za-z]+)")[0].value_counts()
print(pref.head(15).to_string())

print("\n=== 关键行为细胞类型 ===")
for k in ["DNp01","DNp20","DNp09","DNp13","DNp02","DNg02","DNp11","MN9","LC4","LPLC2",
          "PPL1","PAM","KC","MBON","T4","T5","Mi1","L1"]:
    m = t.str.contains(k, case=False, regex=False, na=False)
    print("  %-8s -> %4d" % (k, int(m.sum())))

print("\n=== 下行神经元(DN*) 类型数 ===")
dns = t[t.str.startswith("DN") ]
print("DN* 节点数:", len(dns), "| 类型数:", dns.nunique())
print(dns.value_counts().head(20).to_string())
