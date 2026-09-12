import numpy as np, pandas as pd, torch, time
OUT = "/content/fly/normalized"; dev = "cuda"
meta = pd.read_feather(OUT + "/neurons.feather")
indptr = np.load(OUT + "/indptr.npy"); indices = np.load(OUT + "/indices.npy")
w = np.load(OUT + "/weights.npy"); sign = np.load(OUT + "/sign.npy")
N = len(meta)
crow = torch.from_numpy(indptr.astype(np.int64)).to(dev)
col  = torch.from_numpy(indices.astype(np.int32)).to(dev)
sg   = torch.from_numpy(sign).to(dev)
wv   = torch.from_numpy(w).to(dev)
W = wv * sg[col.long()]
dropped = float((sg[col.long()] == 0).float().mul(wv).sum() / wv.sum())
print("被 sign=0(调质) 丢弃的突触质量占比: %.1f%%" % (100*dropped))

t = meta.type.astype(str)
def idx_of(pat):
    return torch.from_numpy(np.flatnonzero(t.str.contains(pat, regex=False).to_numpy()).astype(np.int64)).to(dev)
IDX = {k: idx_of(k) for k in ["DNp01", "LC4", "LPLC2", "DNp20", "MN9"]}
print({k: len(v) for k, v in IDX.items()})

Vr, Vth, tau, dt = -52.0, -45.0, 20.0, 0.1

def build_A(w_syn, gain):
    return torch.sparse_csr_tensor(crow, col, W * (w_syn * gain), (N, N))

def run(A, T_ms=300, bg_rate=0.0, stim_idx=None, stim_rate=0.0, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    V = torch.full((N,), Vr, device=dev)
    spk = torch.zeros(N, device=dev)
    steps = int(T_ms / dt); n_spk = 0; dn_spk = 0
    dnp = IDX["DNp01"]
    for i in range(steps):
        I = torch.sparse.mm(A, spk.view(-1, 1)).view(-1)
        V = V + (dt/tau) * (-(V - Vr) + I)
        s = V >= Vth
        V = torch.where(s, torch.full_like(V, Vr), V)
        if bg_rate > 0:
            s = s | (torch.rand(N, device=dev, generator=g) < bg_rate*dt/1000.0)
        if stim_idx is not None and stim_rate > 0:
            s = s | (torch.rand(N, device=dev, generator=g) < stim_rate*dt/1000.0) & torch.zeros(N, device=dev, dtype=torch.bool)
        spk = s.float()
        n_spk += int(spk.sum()); dn_spk += int(spk[dnp].sum())
    sec = T_ms/1000.0
    return n_spk/N/sec, n_spk, dn_spk/len(dnp)/sec

print("\n=== 静默测试 (无输入, 300 ms) — flybench 判据: 群体率 < 0.5 Hz ===")
for gain in (0.5, 1.0, 2.0, 4.0, 8.0):
    A = build_A(0.275, gain)
    rate, tot, dnr = run(A, T_ms=300)
    print("  w_syn=0.275 gain=%-4.1f -> 群体率 %8.3f Hz  (总数 %d)" % (gain, rate, tot))
    del A
