import json
d = json.load(open("/content/fly/shiu_result.json"))
print("populations:", d["populations"])
print()
print("%-6s %-10s %9s %9s %9s %9s %9s" % ("gain","cond","popRate","GF","DNp01","MN9","nSpk"))
for gain, r in d["runs"].items():
    for cond in ("silence","loom","sugar","bitter","random"):
        x = r[cond]
        print("%-6s %-10s %9.2f %9.1f %9.1f %9.1f %9d" %
              (gain, cond, x["pop_rate"], x["GF"], x["DNp01"], x["MN9"], x["n_spikes"]))
    print()
