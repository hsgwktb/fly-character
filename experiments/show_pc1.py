import json
d = json.load(open("/content/fly/pc1_result.json"))
print("pC1 一级上游 top12 (突触质量):")
for t, v in d["top1"][:12]:
    print("   %-18s %8.0f" % (t, v))
print("\n注入实测 -> 目标读出 (Hz):")
print("   %-18s %-6s %9s %9s %8s" % ("注入群体","n","pC1","DNp13","群体率"))
for t, r in d["tests"].items():
    mark = "  <== 能驱动 pC1" if r["pC1"] > 20 else ""
    print("   %-18s %-6d %9.1f %9.1f %8.2f%s" % (t, r["n"], r["pC1"], r["DNp13"], r["pop"], mark))
