import json
d = json.load(open("/content/fly/taste_result.json"))
print("%-6s %-13s %9s %9s %9s" % ("gain","条件","群体率","MN9","总脉冲"))
for gain, r in d.items():
    for cond, lab in (("none","静默"),("sugar","糖"),("bitter","苦"),("sugar_bitter","糖+苦")):
        x = r[cond]
        print("%-6s %-13s %9.2f %9.1f %9d" % (gain, lab, x["pop"], x["MN9"], x["n"]))
    s, sb = r["sugar"]["MN9"], r["sugar_bitter"]["MN9"]
    if s > 0:
        print("        -> 加苦味后 MN9 变化: %+.0f%% (%s)" % ((sb-s)/s*100, "抑制" if sb < s else "增强"))
    print()
