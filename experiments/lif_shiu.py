import os, time, json, traceback
import numpy as np, pandas as pd, torch
OUT = "/content/fly/normalized"; STATUS = "/content/fly/shiu_status.txt"; dev = "cuda"
def status(m):
    open(STATUS, "w", encoding="utf-8").write("%s | %s" % (time.strftime("%H:%M:%S"), m))
    print(m, flush=True)

try:
    status("load")
    meta = pd.read_feather(OUT + "/neurons.feather")
    indptr = np.load(OUT + "/indptr.npy").astype(np.int32)
    indices = np.load(OUT + "/indices.npy").astype(np.int32)
    w = np.load(OUT + "/weights.npy"); sign = np.load(OUT + "/sign.npy")
    N = len(meta)
    crow = torch.from_numpy(indptr).to(dev); col = torch.from_numpy(indices).to(dev)
    sg = torch.from_numpy(sign).to(dev); wv = torch.from_numpy(w).to(dev)
    Wraw = wv * sg[col.long()]

    ts = meta.type.astype(str).str.strip()
    def sel(fn):
        return torch.from_numpy(np.flatnonzero(fn(ts).to_numpy()).astype(np.int64)).to(dev)
    POP = {
        "GF":      sel(lambda s: s.eq("GF")),
        "DNp01":   sel(lambda s: s.eq("DNp01")),
        "LC4":     sel(lambda s: s.str.startswith("LC4")),
        "LPLC2":   sel(lambda s: s.str.startswith("LPLC2")),
        "MN9":     sel(lambda s: s.eq("MN9")),
        "sugar":   sel(lambda s: s.str.match(r"^LB3[a-d]$")),
        "bitter":  sel(lambda s: s.str.match(r"^LB1[a-d]$")),
    }
    status("populations " + ", ".join("%s=%d" % (k, len(v)) for k, v in POP.items()))

    # ---- Shiu 2024 常数 (flybench configs/shiu2024.yaml) ----
    v0 = -52.0; v_rst = -52.0; v_th = -45.0
    t_mbr = 20.0; tau_syn = 5.0; t_rfc = 2.2; t_dly = 1.8
    w_syn = 0.275; r_poi = 150.0; f_poi = 250.0
    dt = 0.1; D = int(round(t_dly / dt)); decay_g = float(np.exp(-dt / tau_syn))
    p_act = r_poi * dt / 1000.0
    status("Shiu constants loaded: D=%d steps delay, decay_g=%.4f, p_act=%.4f" % (D, decay_g, p_act))

    def build(gain):
        return torch.sparse_csr_tensor(crow, col, Wraw * (w_syn * gain), (N, N))

    def run(A, T_ms, stim=None, seed=0):
        g = torch.Generator(device=dev).manual_seed(seed)
        v = torch.full((N,), v0, device=dev); gs = torch.zeros(N, device=dev)
        refr = torch.zeros(N, device=dev); spk = torch.zeros(N, device=dev)
        buf = [torch.zeros(N, device=dev) for _ in range(D)]
        steps = int(T_ms / dt); tot = 0; pop_spk = {k: 0 for k in POP}
        for i in range(steps):
            refr = torch.clamp(refr - dt, min=0.0)
            act = refr <= 0
            I = torch.sparse.mm(A, buf.pop(0).view(-1, 1)).view(-1)
            buf.append(spk)
            gs = torch.where(act, gs * decay_g + I, gs)
            v = torch.where(act, v + (dt / t_mbr) * (v0 - v + gs), v)
            if stim is not None and len(stim):
                h = torch.rand(len(stim), device=dev, generator=g) < p_act
                if bool(h.any()):
                    v[stim[h]] += w_syn * f_poi
            s = act & (v > v_th)
            v = torch.where(s, torch.full_like(v, v_rst), v)
            gs = torch.where(s, torch.zeros_like(gs), gs)
            refr = torch.where(s, torch.full_like(refr, t_rfc), refr)
            spk = s.float(); tot += int(s.sum())
            for k, idx in POP.items():
                pop_spk[k] += int(s[idx].sum())
        sec = T_ms / 1000.0
        out = {"pop_rate": tot / N / sec, "n_spikes": tot}
        for k, idx in POP.items():
            out[k] = pop_spk[k] / max(len(idx), 1) / sec
        return out

    R = {"populations": {k: len(v) for k, v in POP.items()}, "runs": {}}
    T = 300
    for gain in (0.45, 0.55, 0.65, 0.80):
        A = build(gain)
        r = {}
        r["silence"] = run(A, T, stim=None, seed=1)
        status("gain=%.2f silence pop=%.3f GF=%.1f" % (gain, r["silence"]["pop_rate"], r["silence"]["GF"]))
        r["loom"] = run(A, T, stim=torch.cat([POP["LC4"], POP["LPLC2"]]), seed=1)
        status("gain=%.2f loom pop=%.2f GF=%.1f DNp01=%.1f" % (gain, r["loom"]["pop_rate"], r["loom"]["GF"], r["loom"]["DNp01"]))
        r["sugar"] = run(A, T, stim=POP["sugar"], seed=1)
        status("gain=%.2f sugar MN9=%.1f" % (gain, r["sugar"]["MN9"]))
        r["bitter"] = run(A, T, stim=POP["bitter"], seed=1)
        status("gain=%.2f bitter MN9=%.1f" % (gain, r["bitter"]["MN9"]))
        rng = np.random.default_rng(0)
        rnd = torch.from_numpy(rng.choice(N, len(POP["sugar"]) + len(POP["bitter"]), replace=False).astype(np.int64)).to(dev)
        r["random"] = run(A, T, stim=rnd, seed=1)
        status("gain=%.2f random pop=%.2f MN9=%.1f GF=%.1f" % (gain, r["random"]["pop_rate"], r["random"]["MN9"], r["random"]["GF"]))
        R["runs"][str(gain)] = r
        del A
        torch.cuda.empty_cache()
        json.dump(R, open("/content/fly/shiu_result.json", "w"), indent=1)
    status("DONE")
except Exception:
    status("FAILED\n" + traceback.format_exc()); raise
