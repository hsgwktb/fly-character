import subprocess, time, json, statistics, os

def sh(c):
    return subprocess.run(c, shell=True, executable="/bin/bash",
                          capture_output=True, text=True).stdout.strip()

STATUS = "/content/fly/interest_status.txt"
def status(m):
    open(STATUS, "w", encoding="utf-8").write("%s | %s" % (time.strftime("%H:%M:%S"), m))
    print(m, flush=True)

def state():
    try:
        return json.loads(sh("curl -s -m 8 'http://127.0.0.1:8000/api/state'"))
    except Exception:
        return None

status("开始采样 120s")
I, B, N, rows = [], [], [], []
for i in range(30):
    s = state()
    if s:
        I.append(s["interest"]); B.append(s["d"]["boredom"]); N.append(s.get("social", 0))
        rows.append({"t": s["t"], "interest": s["interest"], "boredom": s["d"]["boredom"]})
    time.sleep(4)

n = len(I)
mi, mb = statistics.mean(I), statistics.mean(B)
cov = sum((I[k]-mi)*(B[k]-mb) for k in range(n)) / n
si = statistics.pstdev(I) or 1e-9
sb = statistics.pstdev(B) or 1e-9
r = cov / (si*sb)
gaps = [abs(I[k] - (1 - B[k])) for k in range(n)]
res = {
    "n": n,
    "interest": {"mean": round(mi, 3), "min": round(min(I), 3), "max": round(max(I), 3),
                 "sd": round(si, 3)},
    "boredom": {"mean": round(mb, 3), "min": round(min(B), 3), "max": round(max(B), 3),
                "sd": round(sb, 3)},
    "corr": round(r, 3),
    "mean_abs_gap_from_complement": round(sum(gaps)/n, 3),
    "max_gap": round(max(gaps), 3),
    "series": rows,
}
json.dump(res, open("/content/fly/interest_result.json", "w"), indent=1)
status("DONE r=%.3f interest_range=[%.2f,%.2f] boredom_range=[%.2f,%.2f] gap_mean=%.3f"
       % (r, min(I), max(I), min(B), max(B), sum(gaps)/n))
