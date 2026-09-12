import os, time, json, traceback
import numpy as np, pandas as pd, torch
OUT = "/content/fly/normalized"; STATUS = "/content/fly/dose_status.txt"; dev = "cuda"
def status(m):
    open(STATUS, "w", encoding="utf-8").write("%s | %s" % (time.strftime("%H:%M:%S"), m)); print(m, flush=True)
try:
    status("load")
    meta = pd.read_feather(OUT + "/neurons.feather")
    indptr = np.load(OUT + "/indptr.npy").astype(np.int32); indices = np.load(OUT + "/indices.npy").astype(np.int32)
    w = np.load(OUT + "/weights.npy"); sign = np.load(OUT + "/sign.npy"); N = len(meta)
    crow = torch.from_numpy(indptr).to(dev); col = torch.from_numpy(indices).to(dev)
    sg = torch.from_numpy(sign).to(dev); Wraw = torch.from_numpy(w).to(dev) * sg[col.long()]
    ts = meta.type.astype(str).str.strip()
    def sel(fn): return torch.from_numpy(np.flatnonzero(fn(ts).to_numpy()).astype(np.int64)).to(dev)
    SUGAR = sel(lambda s: s.str.match(r"^LB3[a-d]$")); MN9 = sel(lambda s: s.eq("MN9"))
    DNP01 = sel(lambda s: s.eq("DNp01"))

    v0 = -52.0; v_rst = -52.0; v_th = -45.0
    t_mbr = 20.0; tau_syn = 5.0; t_rfc = 2.2; t_dly = 1.8
    w_syn = 0.275; r_poi = 150.0; f_poi = 250.0
    dt = 0.1; D = int(round(t_dly/dt)); decay_g = float(np.exp(-dt/tau_syn))

    def run(gain, intensity, T_ms=300, seed=0):
        A = torch.sparse_csr_tensor(crow, col, Wraw * (w_syn * gain), (N, N))
        g = torch.Generator(device=dev).manual_seed(seed)
        v = torch.full((N,), v0, device=dev); gs = torch.zeros(N, device=dev)
        refr = torch.zeros(N, device=dev); spk = torch.zeros(N, device=dev)
        buf = [torch.zeros(N, device=dev) for _ in range(D)]
        p = (r_poi * intensity) * dt / 1000.0
        tot = 0; mn9 = 0; dnp = 0
        for i in range(int(T_ms/dt)):
            refr = torch.clamp(refr - dt, min=0.0); act = refr <= 0
            I = torch.sparse.mm(A, buf.pop(0).view(-1,1)).view(-1); buf.append(spk)
            gs = torch.where(act, gs*decay_g + I, gs)
            v = torch.where(act, v + (dt/t_mbr)*(v0 - v + gs), v)
            if p > 0:
                h = torch.rand(len(SUGAR), device=dev, generator=g) < p
                if bool(h.any()): v[SUGAR[h]] += w_syn * f_poi
            s = act & (v > v_th)
            v = torch.where(s, torch.full_like(v, v_rst), v); gs = torch.where(s, torch.zeros_like(gs), gs)
            refr = torch.where(s, torch.full_like(refr, t_rfc), refr)
            spk = s.float(); tot += int(s.sum()); mn9 += int(s[MN9].sum()); dnp += int(s[DNP01].sum())
        sec = T_ms/1000.0
        return {"pop": tot/N/sec, "MN9": mn9/len(MN9)/sec, "DNp01": dnp/len(DNP01)/sec}

    GAIN = 0.65
    curve = {}
    for inten in (0.0, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00):
        r = run(GAIN, inten); curve["%.2f" % inten] = r
        status("sugar强度=%.2f -> MN9=%7.1f Hz | 群体=%5.2f | 逃逸=%5.1f" % (inten, r["MN9"], r["pop"], r["DNp01"]))
    json.dump(curve, open("/content/fly/dose_result.json", "w"), indent=1)
    status("DONE")
except Exception:
    status("FAILED\n" + traceback.format_exc()); raise
