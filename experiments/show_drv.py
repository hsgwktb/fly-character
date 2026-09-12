import json
d = json.load(open("/content/fly/driver_result.json"))
for k, r in d.items():
    if k.startswith("_"): continue
    print("=== %s  (目标 %d 个细胞; 1跳 %d; 2跳 %d)" % (k, r.get("n_target",0), r.get("n_hop1",0), r.get("n_hop2",0)))
    print("  1跳突触质量 top8:", ", ".join("%s(%.0f)" % (t,v) for t,v in r.get("top1",[])))
    print("  2跳突触质量 top10:", ", ".join("%s(%.0f)" % (t,v) for t,v in r.get("top2",[])[:6]))
print()
print("=== LIF 实测: 注入候选上游 -> 各输出放电率(Hz) ===")
for tgt, cands in d.get("_stim_tests", {}).items():
    if not cands: print("  %s: 未找到可识别候选" % tgt); continue
    for t, ro in cands.items():
        print("  注入 %-12s 于 %-12s -> %s" % (tgt, t, ro))
