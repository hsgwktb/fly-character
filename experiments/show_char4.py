import json
d = json.load(open("/content/fly/char_result4.json"))
tot = sum(d["actions"].values()); g = sum(d["grounded_hits"].values()); f = d["fallback_hits"]
print("角色 150s / 墙上 %.0fs" % d["wall_s"])
print("行为: ", d["actions"])
print("真细胞读出均值(Hz): ", d["readout_mean"])
print("连接组驱动 %d / 回退 %d (共 %d) -> %.0f%%" % (g, f, tot, 100*g/tot))
print("接地命中: ", d["grounded_hits"])
print("世界范围: ", d["world_range"])
