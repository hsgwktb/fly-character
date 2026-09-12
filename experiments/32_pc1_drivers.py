import numpy as np, pandas as pd, torch, json, time

OUT = "/content/fly/normalized"; dev = "cuda"
meta = pd.read_feather(OUT + "/neurons.feather")
indptr = np.load(OUT + "/indptr.npy").astype(np.int64)
indices = np.load(OUT + "/indices.npy").astype(np.int64)
w = np.load(OUT + "/weights.npy"); sign = np.load(OUT + "/sign.npy")
N = len(meta)
ts = meta.type.astype(str).str.strip()

def idx(pat):
    return np.flatnonzero(pd.Series(ts).str.match(pat).to_numpy())

PC1 = idx(r"^pC1")
DNP13 = idx(r"^DNp13$")
ASPS = idx(r"^aSP")
VPR = idx(r"^vPR")
print("pC1=%d  DNp13=%d  aSP=%d  vPR=%d" % (len(PC1), len(DNP13), len(ASPS), len(VPR)))

# ---------- 1) pC1 的上游按类型聚合 ----------
def presyn(js):
    return np.concatenate([indices[indptr[j]:indptr[j+1]] for j in js]) if len(js) else np.array([], dtype=np.int64)

h1 = np.unique(presyn(PC1))
wsum = {}
for j in PC1:
    cols = indices[indptr[j]:indptr[j+1]]
    for c, ww in zip(cols, w[indptr[j]:indptr[j+1]]):
        wsum[ts[c]] = wsum.get(ts[c], 0.0) + float(ww)
top1 = sorted(wsum.items(), key=lambda kv: -kv[1])
print("\npC1 一级上游类型 top15 (按突触质量):")
for t, v in top1[:15]:
    print("   %-16s %8.0f" % (t, v))
print("   (一级上游总细胞数 %d)" % len(h1))

# ---------- 2) LIF 实测: 注入候选能否驱动 pC1 / DNp13 ----------
crow = torch.from_numpy(indptr.astype(np.int32)).to(dev)
col = torch.from_numpy(indices.astype(np.int32)).to(dev)
sg = torch.from_numpy(sign).to(dev)
Wraw = torch.from_numpy(w).to(dev) * sg[col.long()]
V0=VR=-52.0; VTH=-45.0; T_MBR=20.0; TAU=5.0; TRFC=2.2; TDLY=1.8
WS=0.275; RP=150.0; FP=250.0; GAIN=0.65; DT=0.1
D=int(round(TDLY/DT)); DG=float(np.exp(-DT/TAU))
A = torch.sparse_csr_tensor(crow, col, Wraw*(WS*GAIN), (N,N))
TGT = {"pC1": torch.from_numpy(PC1.astype(np.int64)).to(dev),
       "DNp13": torch.from_numpy(DNP13.astype(np.int64)).to(dev)}

def run(stim_np, T_ms=200, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    V = torch.full((N,), V0, device=dev); G = torch.zeros(N, device=dev)
    rf = torch.zeros(N, device=dev); spk = torch.zeros(N, device=dev)
    buf = [torch.zeros(N, device=dev) for _ in range(D)]
    si = torch.from_numpy(stim_np.astype(np.int64)).to(dev)
    p = RP*DT/1000.0
    out = {k: 0 for k in TGT}; tot = 0
    for _ in range(int(T_ms/DT)):
        rf = torch.clamp(rf - DT, min=0.0); act = rf <= 0
        I = torch.sparse.mm(A, buf.pop(0).view(-1,1)).view(-1); buf.append(spk)
        G = torch.where(act, G*DG + I, G)
        V = torch.where(act, V + (DT/T_MBR)*(V0 - V + G), V)
        if len(si):
            h = torch.rand(len(si), device=dev, generator=g) < p
            if bool(h.any()): V[si[h]] += WS*FP
        s = act & (V > VTH)
        V = torch.where(s, torch.full_like(V, VR), V); G = torch.where(s, torch.zeros_like(G), G)
        rf = torch.where(s, torch.full_like(rf, TRFC), rf)
        spk = s.float(); tot += int(s.sum())
        for k in TGT: out[k] += int(s[TGT[k]].sum())
    sec = T_ms/1000.0
    r = {k: out[k]/max(TGT[k].numel(),1)/sec for k in TGT}
    r["pop"] = tot/N/sec
    return r

print("\n=== 注入实测 (200 ms) ===")
print("  基线(无输入)      ->", {k: round(v,1) for k,v in run(np.array([],dtype=int)).items()})
# 候选: 一级上游里质量最大的几类(排除 pC1 自身)
cands = [t for t, _ in top1 if t and t != "None" and not t.startswith("pC1")][:6]
for t in cands:
    ii = idx("^" + t.replace(" ", "") + "$")
    if len(ii) == 0: ii = np.flatnonzero((ts == t).to_numpy())
    if len(ii) == 0: continue
    r = run(ii)
    print("  注入 %-16s(n=%-4d) -> pC1=%7.1f DNp13=%7.1f pop=%6.2f" % (t, len(ii), r["pC1"], r["DNp13"], r["pop"]))
# 也试 aSP / vPR 两个求偶相关群体
for name, ii in (("aSP", ASPS), ("vPR", VPR)):
    if len(ii):
        r = run(ii)
        print("  注入 %-16s(n=%-4d) -> pC1=%7.1f DNp13=%7.1f pop=%6.2f" % (name, len(ii), r["pC1"], r["DNp13"], r["pop"]))
