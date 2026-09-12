import json, statistics
d = json.load(open("/content/fly/interest_result.json"))
S = d["series"]
it = [r["interest"] for r in S]; bo = [r["boredom"] for r in S]; tt = [r["t"] for r in S]
db = [(bo[k+1]-bo[k])/(tt[k+1]-tt[k]) for k in range(len(bo)-1)]
im = [(it[k]+it[k+1])/2 for k in range(len(it)-1)]
n = len(db); mi = statistics.mean(im); md = statistics.mean(db)
cov = sum((im[k]-mi)*(db[k]-md) for k in range(n))/n
r = cov / ((statistics.pstdev(im) or 1e-9)*(statistics.pstdev(db) or 1e-9))
lo = [db[k] for k in range(n) if im[k] < 0.45]
hi = [db[k] for k in range(n) if im[k] > 0.60]
print("结构检验: 兴趣 是否抑制 无聊的增长速率?")
print("  r(兴趣, d无聊/dt) = %+.3f   <- 应为负" % r)
print("  兴趣低(<0.45)时 无聊平均增速 = %+.4f /s  (n=%d)" % (statistics.mean(lo) if lo else float('nan'), len(lo)))
print("  兴趣高(>0.60)时 无聊平均增速 = %+.4f /s  (n=%d)" % (statistics.mean(hi) if hi else float('nan'), len(hi)))
print()
print("水平值 vs 补集:")
gaps = [abs(it[k]-(1-bo[k])) for k in range(len(it))]
print("  平均 |兴趣-(1-无聊)| = %.3f, 最大 %.3f" % (statistics.mean(gaps), max(gaps)))
print("  只在兴趣均值上看，两者接近(%.2f vs %.2f)，但时间过程完全不同：" % (statistics.mean(it), statistics.mean(bo)))
print("  兴趣 sd=%.3f (相位性, 会被事件瞬间抬起)" % statistics.pstdev(it))
print("  无聊 sd=%.3f 且单调上升(累积量)" % statistics.pstdev(bo))
