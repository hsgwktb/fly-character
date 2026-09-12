import numpy as np, pandas as pd, torch, time, traceback, json

OUT = "/content/fly/normalized"; dev = "cuda"; STATUS = "/content/fly/sleep_status.txt"
def status(m):
    open(STATUS, "w", encoding="utf-8").write("%s | %s" % (time.strftime("%H:%M:%S"), m))
    print(m, flush=True)

try:
    meta = pd.read_feather(OUT + "/neurons.feather")
    indptr = np.load(OUT + "/indptr.npy").astype(np.int64)
    indices = np.load(OUT + "/indices.npy").astype(np.int64)
    w = np.load(OUT + "/weights.npy"); sign = np.load(OUT + "/sign.npy")
    N = len(meta)
    ts = meta.type.astype(str).str.strip(); s_ts = pd.Series(ts)
    def idx(pat): return np.flatnonzero(s_ts.str.match(pat).to_numpy())

    STIM = {
        "FB(sleep)":  idx(r"^FB[2-8]"),                       # 扇形体 = 促眠区
        "ExR":        idx(r"^ExR"),
        "LNv(clock)": idx(r"^(s-LNv|l-LNv|5thsLNv.*)$"),       # PDF 阳性主节律器 = 促醒
        "DN1(clock)": idx(r"^DN1"),
        "OA-VUM":     idx(r"^OA-VUM"),                         # 章鱼胺 = 促醒
        "PAM(DA)":    idx(r"^PAM"),
        "5HT":        idx(r"^5-HT"),
    }
    READ = {"DNp09(走)": idx(r"^DNp09$"), "DNp20(转)": idx(r"^DNp20$"),
            "MN9(食)": idx(r"^MN9$"), "DNp01(逃)": idx(r"^DNp01$")}
    status("stim " + ", ".join("%s=%d" % (k, len(v)) for k, v in STIM.items()))

    crow = torch.from_numpy(indptr.astype(np.int32)).to(dev)
    col = torch.from_numpy(indices.astype(np.int32)).to(dev)
    sg = torch.from_numpy(sign).to(dev)
    Wraw = torch.from_numpy(w).to(dev) * sg[col.long()]
    V0 = VR = -52.0; VTH = -45.0; T_MBR = 20.0; TAU = 5.0; TRFC = 2.2; TDLY = 1.8
    WS = 0.275; RP = 150.0; FP = 250.0; DT = 0.1
    D = int(round(TDLY/DT)); DG = float(np.exp(-DT/TAU))
    RT = {k: torch.from_numpy(v.astype(np.int64)).to(dev) for k, v in READ.items() if len(v)}

    def run(stim_np, gain, T_ms=300, seed=0):
        A = torch.sparse_csr_tensor(crow, col, Wraw*(WS*gain), (N, N))
        g = torch.Generator(device=dev).manual_seed(seed)
        V = torch.full((N,), V0, device=dev); G = torch.zeros(N, device=dev)
        rf = torch.zeros(N, device=dev); spk = torch.zeros(N, device=dev)
        buf = [torch.zeros(N, device=dev) for _ in range(D)]
        si = torch.from_numpy(np.asarray(stim_np).astype(np.int64)).to(dev)
        p = RP*DT/1000.0
        out = {k: 0 for k in RT}; tot = 0
        for _ in range(int(T_ms/DT)):
            rf = torch.clamp(rf - DT, min=0.0); act = rf <= 0
            I = torch.sparse.mm(A, buf.pop(0).view(-1, 1)).view(-1); buf.append(spk)
            G = torch.where(act, G*DG + I, G)
            V = torch.where(act, V + (DT/T_MBR)*(V0 - V + G), V)
            if len(si):
                h = torch.rand(len(si), device=dev, generator=g) < p
                if bool(h.any()):
                    V[si[h]] += WS*FP
            s = act & (V > VTH)
            V = torch.where(s, torch.full_like(V, VR), V)
            G = torch.where(s, torch.zeros_like(G), G)
            rf = torch.where(s, torch.full_like(rf, TRFC), rf)
            spk = s.float(); tot += int(s.sum())
            for k in RT: out[k] += int(s[RT[k]].sum())
        sec = T_ms/1000.0
        r = {k: out[k]/max(RT[k].numel(), 1)/sec for k in RT}
        r["pop"] = tot/N/sec
        return r

    res = {}
    for gain in (0.65, 1.30):
        base = run(np.array([], dtype=int), gain)
        res["gain%.2f_base" % gain] = base
        status("gain=%.2f 基线 -> %s" % (gain, {k: round(v,1) for k,v in base.items()}))
        for name, ii in STIM.items():
            if len(ii) == 0: continue
            r = run(ii, gain)
            res["gain%.2f_%s" % (gain, name)] = r
            d9 = r.get("DNp09(走)", 0) - base.get("DNp09(走)", 0)
            status("gain=%.2f 注入 %-11s(n=%-4d) -> 走=%.1f(Δ%+.1f) 转=%.1f 食=%.1f 逃=%.1f pop=%.2f"
                   % (gain, name, len(ii), r.get("DNp09(走)",0), d9, r.get("DNp20(转)",0),
                      r.get("MN9(食)",0), r.get("DNp01(逃)",0), r["pop"]))
    json.dump({k: {kk: round(vv,2) for kk,vv in v.items()} for k,v in res.items()},
              open("/content/fly/sleep_result.json","w"), indent=1)
    status("DONE")
except Exception:
    status("FAILED\n" + traceback.format_exc()); raise
