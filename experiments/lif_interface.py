import os, time, json, traceback
import numpy as np, pandas as pd, torch
OUT = "/content/fly/normalized"; STATUS = "/content/fly/iface_status.txt"; dev = "cuda"
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

    SENSE = {                                  # 感觉输入 -> 识别细胞
        "sugar": sel(lambda s: s.str.match(r"^LB3[a-d]$")),
        "bitter": sel(lambda s: s.str.match(r"^LB1[a-d]$")),
        "loom": torch.cat([sel(lambda s: s.str.startswith("LC4")), sel(lambda s: s.str.startswith("LPLC2"))]),
    }
    MOTOR = {                                  # 行为读出 <- 识别细胞
        "feed_MN9": sel(lambda s: s.eq("MN9")),
        "escape_DNp01": sel(lambda s: s.eq("DNp01")),
        "turn_DNp20": sel(lambda s: s.eq("DNp20")),
        "walk_DNp09": sel(lambda s: s.eq("DNp09")),
    }
    status("sense " + ", ".join("%s=%d" % (k, len(v)) for k, v in SENSE.items()))
    status("motor " + ", ".join("%s=%d" % (k, len(v)) for k, v in MOTOR.items()))

    v0 = -52.0; v_rst = -52.0; v_th = -45.0
    t_mbr = 20.0; tau_syn = 5.0; t_rfc = 2.2; t_dly = 1.8
    w_syn = 0.275; r_poi = 150.0; f_poi = 250.0
    dt = 0.1; D = int(round(t_dly/dt)); decay_g = float(np.exp(-dt/tau_syn))

    def run(gain, stims, T_ms=300, seed=0):
        """stims: {通道名: 强度 0..1}; 强度按比例缩放该通道的激活率"""
        A = torch.sparse_csr_tensor(crow, col, Wraw * (w_syn * gain), (N, N))
        g = torch.Generator(device=dev).manual_seed(seed)
        v = torch.full((N,), v0, device=dev); gs = torch.zeros(N, device=dev)
        refr = torch.zeros(N, device=dev); spk = torch.zeros(N, device=dev)
        buf = [torch.zeros(N, device=dev) for _ in range(D)]
        pops = {k: t for k, t in SENSE.items()}
        pops.update(MOTOR)
        cnt = {k: 0 for k in pops}; tot = 0
        probs = {k: (r_poi * inten) * dt / 1000.0 for k, inten in stims.items()}
        for i in range(int(T_ms/dt)):
            refr = torch.clamp(refr - dt, min=0.0); act = refr <= 0
            I = torch.sparse.mm(A, buf.pop(0).view(-1,1)).view(-1); buf.append(spk)
            gs = torch.where(act, gs*decay_g + I, gs)
            v = torch.where(act, v + (dt/t_mbr)*(v0 - v + gs), v)
            for k, p in probs.items():
                if p <= 0: continue
                idx = SENSE[k]
                h = torch.rand(len(idx), device=dev, generator=g) < p
                if bool(h.any()): v[idx[h]] += w_syn * f_poi
            s = act & (v > v_th)
            v = torch.where(s, torch.full_like(v, v_rst), v); gs = torch.where(s, torch.zeros_like(gs), gs)
            refr = torch.where(s, torch.full_like(refr, t_rfc), refr)
            spk = s.float(); tot += int(s.sum())
            for k, idx in pops.items(): cnt[k] += int(s[idx].sum())
        sec = T_ms/1000.0
        r = {"pop": tot/N/sec}
        for k, idx in pops.items(): r[k] = cnt[k]/max(len(idx),1)/sec
        return r

    GAIN = 0.65; res = {}
    conds = {
        "none": {}, "sugar": {"sugar": 1.0}, "bitter": {"bitter": 1.0}, "loom": {"loom": 1.0},
        "sugar+bitter": {"sugar": 1.0, "bitter": 1.0}, "sugar_half": {"sugar": 0.5},
    }
    for name, st in conds.items():
        res[name] = run(GAIN, st); r = res[name]
        status("%-13s pop=%6.2f | MN9=%8.1f DNp01=%8.1f DNp20=%7.1f DNp09=%7.1f"
               % (name, r["pop"], r["feed_MN9"], r["escape_DNp01"], r["turn_DNp20"], r["walk_DNp09"]))
    json.dump(res, open("/content/fly/iface_result.json", "w"), indent=1)
    status("DONE")
except Exception:
    status("FAILED\n" + traceback.format_exc()); raise
