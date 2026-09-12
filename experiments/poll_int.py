import os, subprocess, json
print(open("/content/fly/interest_status.txt", errors="replace").read() if os.path.exists("/content/fly/interest_status.txt") else "(no status)")
p = "/content/fly/interest_result.json"
if os.path.exists(p):
    d = json.load(open(p))
    print("\n兴趣: 均值%.3f 范围[%.2f, %.2f] 标准差%.3f" % (d["interest"]["mean"], d["interest"]["min"], d["interest"]["max"], d["interest"]["sd"]))
    print("无聊: 均值%.3f 范围[%.2f, %.2f] 标准差%.3f" % (d["boredom"]["mean"], d["boredom"]["min"], d["boredom"]["max"], d["boredom"]["sd"]))
    print("相关系数 r = %+.3f   (补集应为 -1.000)" % d["corr"])
    print("平均 |兴趣-(1-无聊)| = %.3f   最大偏离 %.3f" % (d["mean_abs_gap_from_complement"], d["max_gap"]))
    print("\n时间序列(每 8 个点抽 1 个):")
    for row in d["series"][::4]:
        print("   t=%6.0f  兴趣=%.3f  无聊=%.3f  1-无聊=%.3f" % (row["t"], row["interest"], row["boredom"], 1-row["boredom"]))
print("proc:", subprocess.run("ps -eo pid,cmd | grep [v]erify_interest", shell=True, capture_output=True, text=True).stdout.strip() or "(finished)")
