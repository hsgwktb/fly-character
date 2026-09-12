import numpy as np, pandas as pd, torch, time, traceback

OUT = "/content/fly/normalized"; dev = "cuda"
STATUS = "/content/fly/pc1_status.txt"
def status(m):
    open(STATUS, "w", encoding="utf-8").write("%s | %s" % (time.strftime("%H:%M:%S"), m))
    print(m, flush=True)

try:
    meta = pd.read_feather(OUT + "/neurons.feather")
    indptr = np.load(OUT + "/indptr.npy").astype(np.int64)
    indices = np.load(OUT + "/indices.npy").astype(np.int64)
    w = np.load(OUT + "/weights.npy"); sign = np.load(OUT + "/sign.npy")
    N = len(meta)
    ts = meta.type.astype(str).str.strip()
    s_ts = pd.Series(ts)
    def idx(pat):
        return np.flatnonzero(s_ts.str.match(pat).to_numpy())

    PC1 = idx(r"^pC1"); DNP13 = idx(r"^DNp13$"); ASPS = idx(r"^aSP"); VPR = idx(r"^vPR")
    status("pC1=%d DNp13=%d aSP=%d vPR=%d" % (len(PC1), len(DNP13), len(ASPS), len(VPR)))

    # ---- 1) pC1 一级上游按类型聚合 ----
    wsum = {}
    for j in PC1:
        cols = indices[indptr[j]:indptr[j+1]]
        for c, ww in zip(cols, w[indptr[j]:indptr[j+1]]):
            wsum[ts[c]] = wsum.get(ts[c], 0.0) + float(ww)
    top1 = sorted(wsum.items(), key=lambda kv: -kv[1])
    status("pC1 上游 top12: " + ", ".join("%s(%.0f)" % (t, v) for t, v in top1[:12]))

    # ---- 2) LIF 实测注入 ----
    crow = torch.from_numpy(indptr.astype(np.int32)).to(dev)
    col = torch.from_numpy(indices.astype(np.int32)).to(dev)
    sg = torch.from_numpy(sign).to(dev)
    Wraw = torch.from_numpy(w).to(dev) * sg[col.long()]
    V0 = VR = -52.0; VTH = -45.0; T_MBR = 20.0; TAU = 5.0; TRFC = 2.2; TDLY = 1.8
    WS = 0.275; RP = 150.0; FP = 250.0; GAIN = 0.65; DT = 0.1
    D = int(round(TDLY/DT)); DG = float(np.exp(-DT/TAU))
    A = torch.sparse_csr_tensor(crow, col, Wraw*(WS*GAIN), (N, N))
    TGT = {"pC1": torch.from_numpy(PC1.astype(np.int64)).to(dev),
           "DNp13": torch.from_numpy(DNP13.astype(np.int64)).to(dev)}

    def run(stim_np, T_ms=200, seed=0):
        g = torch.Generator(device=dev).manual_seed(seed)
        V = torch.full((N,), V0, device=dev); G = torch.zeros(N, device=dev)
        rf = torch.zeros(N, device=dev); spk = torch.zeros(N, device=dev)
        buf = [torch.zeros(N, device=dev) for _ in range(D)]
        si = torch.from_numpy(np.asarray(stim_np).astype(np.int64)).to(dev)
        p = RP*DT/1000.0
        out = {k: 0 for k in TGT}; tot = 0
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
            for k in TGT:
                out[k] += int(s[TGT[k]].sum())
        sec = T_ms/1000.0
        r = {k: out[k]/max(TGT[k].numel(), 1)/sec for k in TGT}
        r["pop"] = tot/N/sec
        return r

    b = run(np.array([], dtype=int))
    status("基线 -> pC1=%.1f DNp13=%.1f pop=%.2f" % (b["pC1"], b["DNp13"], b["pop"]))

    results = {}
    cands = [t for t, _ in top1 if t and t != "None" and not t.startswith("pC1")][:8]
    for t in cands + ["aSP", "vPR"]:
        ii = idx("^" + t + "$") if t in ("aSP", "vPR") else np.flatnonzero(s_ts.eq(t).to_numpy())
        if len(ii) == 0:
            continue
        r = run(ii)
        results[t] = {"n": int(len(ii)), "pC1": round(r["pC1"], 1),
                      "DNp13": round(r["DNp13"], 1), "pop": round(r["pop"], 2)}
        status("注入 %-16s(n=%-4d) -> pC1=%7.1f DNp13=%7.1f pop=%6.2f"
               % (t, len(ii), r["pC1"], r["DNp13"], r["pop"]))
    import json
    json.dump({"top1": top1[:20], "tests": results},
              open("/content/fly/pc1_result.json", "w"), indent=1)
    status("DONE")
except Exception:
    status("FAILED\n" + traceback.format_exc())
    raise
