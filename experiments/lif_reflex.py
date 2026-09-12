import os, time, json, traceback, sys
import numpy as np, pandas as pd, torch
OUT = "/content/fly/normalized"; STATUS = "/content/fly/reflex_status.txt"
dev = "cuda"
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
    crow = torch.from_numpy(indptr).to(dev)
    col  = torch.from_numpy(indices).to(dev)
    sg   = torch.from_numpy(sign).to(dev)
    wv   = torch.from_numpy(w).to(dev)
    Wraw = wv * sg[col.long()]                      # 带符号的突触强度(突触数)

    t = meta.type.astype(str)
    def idx_of(pat):
        return torch.from_numpy(np.flatnonzero(t.str.contains(pat, regex=False).to_numpy())
                                .astype(np.int64)).to(dev)
    LC = torch.cat([idx_of("LC4"), idx_of("LPLC2")])
    DNP01 = idx_of("DNp01")
    status("LC4+LPLC2=%d DNp01=%d" % (len(LC), len(DNP01)))

    Vr, Vth, tau, dt = -52.0, -45.0, 20.0, 0.1
    def make(T_ms, w_syn, gain, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        A = torch.sparse_csr_tensor(crow, col, Wraw * (w_syn * gain), (N, N), check_invariants=False)
        return A, g

    def run(A, g, T_ms, stim=None, stim_rate=0.0):
        V = torch.full((N,), Vr, device=dev); spk = torch.zeros(N, device=dev)
        steps = int(T_ms / dt); tot = 0; dn = 0; first = None
        for i in range(steps):
            I = torch.sparse.mm(A, spk.view(-1, 1)).view(-1)
            V = V + (dt / tau) * (-(V - Vr) + I)
            s = V >= Vth
            V = torch.where(s, torch.full_like(V, Vr), V)
            if stim is not None and stim_rate > 0:
                h = torch.zeros(N, device=dev, dtype=torch.bool)
                h[stim] = torch.rand(len(stim), device=dev, generator=g) < stim_rate * dt / 1000.0
                s = s | h
            spk = s.float()
            c = int(spk.sum()); tot += c; dn += int(spk[DNP01].sum())
            if first is None and c > 0: first = i * dt
        sec = T_ms / 1000.0
        return tot / N / sec, tot, dn / len(DNP01) / sec, first

    status("silence sweep")
    rows = []
    for gain in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        A, g = make(300, 0.275, gain, 0)
        rate, tot, dnr, _ = run(A, g, 300)
        rows.append((gain, rate, tot)); status("gain=%.1f base_rate=%.3f n=%d" % (gain, rate, tot))
        del A
        torch.cuda.empty_cache()

    ok = [r for r in rows if r[1] < 0.5]
    gain = max(ok, key=lambda r: r[0])[0] if ok else 1.0
    status("chosen gain=%.1f (静默判据 <0.5Hz)" % gain)

    res = {"gain": gain, "silence": rows}
    T = 300
    A, g = make(T, 0.275, gain, 1)
    b = run(A, g, T); status("baseline rate=%.3f DNp01=%.3f" % (b[0], b[2]))
    s = run(A, g, T, stim=LC, stim_rate=100.0)
    status("LC4/LPLC2 stim 100Hz -> rate=%.3f DNp01=%.3f (首活动 %.1fms)" % (s[0], s[2], s[3] or -1))
    rng = np.random.default_rng(0)
    ctrl = torch.from_numpy(rng.choice(N, len(LC), replace=False).astype(np.int64)).to(dev)
    c = run(A, g, T, stim=ctrl, stim_rate=100.0)
    status("对照: 随机 %d 个神经元同样刺激 -> rate=%.3f DNp01=%.3f" % (len(LC), c[0], c[2]))

    res.update({"baseline": b[:3], "loom": s[:3], "control": c[:3]})
    json.dump(res, open("/content/fly/reflex_result.json", "w"), indent=1)
    status("DONE baseline_DNp01=%.3f loom_DNp01=%.3f control_DNp01=%.3f"
           % (b[2], s[2], c[2]))
except Exception:
    status("FAILED\n" + traceback.format_exc()); raise
