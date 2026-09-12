import json
d = json.load(open("/content/fly/char_result5.json"))
tot = sum(d["actions"].values()); g = sum(d["grounded_hits"].values()); f = d["fallback_hits"]
print("角色 600s / 墙上 %.0fs | 动作总数 %d" % (d["wall_s"], tot))
print("总计: ", d["actions"])
print("连接组驱动 %d / 回退 %d -> %.0f%%" % (g, f, 100*g/tot))
print("读出均值(Hz): ", d["readout_mean"])
print()
print("分段(每段 150s)行为分布 —— 若某行为在某段占比 >60%% 即为锁死:")
for i, q in enumerate(d["quarters"]):
    s = sum(q.values())
    top = sorted(q.items(), key=lambda kv: -kv[1])[:3]
    print("  第%d段 n=%-3d %s  top=%s" % (i+1, s, dict(q), ", ".join("%s %.0f%%" % (k, 100*v/max(s,1)) for k,v in top)))
