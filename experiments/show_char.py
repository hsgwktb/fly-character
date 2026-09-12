import json
d = json.load(open("/content/fly/char_result.json"))
tot_act = sum(d["actions"].values())
g = sum(d["grounded_hits"].values()); f = d["fallback_hits"]
print("角色时间 150s / 墙上 %.0fs" % d["wall_s"])
print("世界范围: 温度 %s  湿度 %s" % (d["world_range"]["temp"], d["world_range"]["humidity"]))
print()
print("行为触发: ", d["actions"])
print()
print("真细胞读出均值 (Hz): ", d["readout_mean"])
print()
print("动作归因: 连接组驱动 %d / 回退 %d  (共 %d, 连接组占 %.0f%%)" % (g, f, tot_act, 100*g/tot_act))
print("  各接地行为的连接组命中: ", d["grounded_hits"])
