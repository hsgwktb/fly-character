import json
d = json.load(open("/content/fly/dose_result.json"))
print("%8s %10s %10s %10s" % ("糖强度","MN9进食","群体率","DNp01"))
prev = None
for k in sorted(d, key=float):
    r = d[k]; v = float(k)
    mark = ""
    if prev is not None and prev[1] > 0:
        pass
    print("%8.2f %10.1f %10.2f %10.1f" % (v, r["MN9"], r["pop"], r["DNp01"]))
print()
ks = sorted(d, key=float)
mx = max(d[k]["MN9"] for k in ks)
print("MN9 饱和值 = %.1f Hz" % mx)
for k in ks:
    r = d[k]["MN9"]
    if mx > 0 and r >= 0.9*mx:
        print("达到 90%% 饱和的糖强度 = %.2f" % float(k)); break
for k in ks:
    if float(k) > 0 and d[k]["MN9"] > 0:
        print("最低可驱动强度 = %.2f (MN9=%.1f Hz)" % (float(k), d[k]["MN9"])); break
