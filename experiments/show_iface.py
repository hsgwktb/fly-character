import json
d = json.load(open("/content/fly/iface_result.json"))
cols = [("pop","群体率"),("feed_MN9","MN9进食"),("escape_DNp01","DNp01逃逸"),
        ("turn_DNp20","DNp20转向"),("walk_DNp09","DNp09行走")]
print("%-13s" % "通道" + "".join("%12s" % c[1] for c in cols))
for k, r in d.items():
    print("%-13s" % k + "".join("%12.1f" % r[c[0]] for c in cols))
print()
b = d["sugar"]["feed_MN9"]
print("糖 -> 进食读出: %.1f Hz" % b)
print("糖+苦 -> 进食读出: %.1f Hz  (变化 %+.0f%%)" % (d["sugar+bitter"]["feed_MN9"],
      (d["sugar+bitter"]["feed_MN9"]-b)/b*100 if b else float("nan")))
print("糖(强度0.5) -> 进食读出: %.1f Hz  (相对满强度 %+.0f%%)" % (d["sugar_half"]["feed_MN9"],
      (d["sugar_half"]["feed_MN9"]-b)/b*100 if b else float("nan")))
