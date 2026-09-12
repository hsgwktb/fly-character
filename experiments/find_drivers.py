import os, time, json, traceback
import numpy as np, pandas as pd, torch

OUT = "/content/fly/normalized"
STATUS = "/content/fly/driver_status.txt"
dev = "cuda"

def status(m):
    open(STATUS, "w", encoding="utf-8").write("%s | %s" % (time.strftime("%H:%M:%S"), m))
    print(m, flush=True)

try:
    status("load")
    meta = pd.read_feather(OUT + "/neurons.feather")
    indptr = np.load(OUT + "/indptr.npy").astype(np.int64)
    indices = np.load(OUT + "/indices.npy").astype(np.int64)
    w = np.load(OUT + "/weights.npy"); sign = np.load(OUT + "/sign.npy")
    N = len(meta)
    ts = meta.type.astype(str).str.strip().to_numpy()
    sc = meta.superclass.astype(str).str.strip().to_numpy()

    def pools(pat):
        m = pd.Series(ts).str.match(pat).to_numpy()
        return np.flatnonzero(m)

    TARGETS = {"MN9(eat)": r"^MN9$", "DNp01(flee)": r"^DNp01$",
               "DNp20(approach)": r"^DNp20$", "DNp09(explore)": r"^DNp09$"}

    # ---- 突触前回溯: CSR 行=post, 列=pre ----
    def pre_of(js):
        return np.concatenate([indices[indptr[j]:indptr[j + 1]] for j in js]) if len(js) else np.array([], dtype=np.int64)

    report = {}
    for label, pat in TARGETS.items():
        tgt = pools(pat)
        if len(tgt) == 0:
            report[label] = {"error": "no target cell"}; continue
        hop1 = np.unique(pre_of(tgt))
        # 1 跳: 按类型聚合突触质量
        post1 = np.repeat(tgt, np.diff(indptr)[tgt])
        s1 = pd.Series(ts[hop1])
        # 累加每条边权重到其 pre 的类型
        wsum = {}
        for j in tgt:
            cols = indices[indptr[j]:indptr[j + 1]]
            for c, ww in zip(cols, w[indptr[j]:indptr[j + 1]]):
                wsum[ts[c]] = wsum.get(ts[c], 0.0) + float(ww)
        top1 = sorted(wsum.items(), key=lambda kv: -kv[1])[:8]
        hop2 = np.unique(pre_of(hop1))
        wsum2 = {}
        for j in hop1:
            cols = indices[indptr[j]:indptr[j + 1]]
            for c, ww in zip(cols, w[indptr[j]:indptr[j + 1]]):
                wsum2[ts[c]] = wsum2.get(ts[c], 0.0) + float(ww)
        top2 = sorted(wsum2.items(), key=lambda kv: -kv[1])[:10]
        report[label] = {"n_target": int(len(tgt)), "n_hop1": int(len(hop1)), "n_hop2": int(len(hop2)),
                         "top1": [(t, round(v, 1)) for t, v in top1],
                         "top2": [(t, round(v, 1)) for t, v in top2]}
        status("%s: hop1=%d hop2=%d | top1=%s" % (label, len(hop1), len(hop2),
               ", ".join("%s(%.0f)" % (t, v) for t, v in top1[:5])))

    # ---- 用 LIF 实测候选上游是否真能驱动 ----
    crow = torch.from_numpy(indptr.astype(np.int32)).to(dev)
    col = torch.from_numpy(indices.astype(np.int32)).to(dev)
    sg = torch.from_numpy(sign).to(dev)
    Wraw = torch.from_numpy(w).to(dev) * sg[col.long()]
    v0 = -52.0; v_rst = -52.0; v_th = -45.0
    t_mbr = 20.0; tau_syn = 5.0; t_rfc = 2.2; t_dly = 1.8
    w_syn = 0.275; r_poi = 150.0; f_poi = 250.0; GAIN = 0.65
    DT = 0.1; D = int(round(t_dly / DT)); decay_g = float(np.exp(-DT / tau_syn))
    A = torch.sparse_csr_tensor(crow, col, Wraw * (w_syn * GAIN), (N, N))

    def run(stim_idx, intensity=1.0, T_ms=200, seed=0):
        g = torch.Generator(device=dev).manual_seed(seed)
        V = torch.full((N,), v0, device=dev); G = torch.zeros(N, device=dev)
        rf = torch.zeros(N, device=dev); spk = torch.zeros(N, device=dev)
        buf = [torch.zeros(N, device=dev) for _ in range(D)]
        p = (r_poi * intensity) * DT / 1000.0
        out = {k: 0 for k in TARGETS}
        for _ in range(int(T_ms / DT)):
            rf = torch.clamp(rf - DT, min=0.0); act = rf <= 0
            I = torch.sparse.mm(A, buf.pop(0).view(-1, 1)).view(-1); buf.append(spk)
            G = torch.where(act, G * decay_g + I, G)
            V = torch.where(act, V + (DT / t_mbr) * (v0 - V + G), V)
            if p > 0 and len(stim_idx):
                h = torch.rand(len(stim_idx), device=dev, generator=g) < p
                if bool(h.any()):
                    V[stim_idx[h]] += w_syn * f_poi
            s = act & (V > v_th)
            V = torch.where(s, torch.full_like(V, v_rst), V)
            G = torch.where(s, torch.zeros_like(G), G)
            rf = torch.where(s, torch.full_like(rf, t_rfc), rf)
            spk = s.float()
            for label, pat in TARGETS.items():
                idx = TGT_IDX[label]
                out[label] += int(s[idx].sum())
        sec = T_ms / 1000.0
        return {k: out[k] / max(TGT_IDX[k].numel(), 1) / sec for k in TARGETS}

    TGT_IDX = {}
    for label, pat in TARGETS.items():
        idx = pools(pat)
        TGT_IDX[label] = torch.from_numpy(idx.astype(np.int64)).to(dev)

    # 候选: 每个目标的 1 跳里突触质量最大的"可识别"类型(排除同名下游/自身)
    CAND = {"MN9(eat)": "^LB3[a-d]$", "DNp01(flee)": "^LPLC2$",
            "DNp20(approach)": None, "DNp09(explore)": None}
    SENSORY = ("LB", "ORN", "LC", "LPLC", "T4", "T5", "Mi", "Tm", "L1", "L2", "L3", "R")
    tests = {}
    for label, r in report.items():
        if "top1" not in r:
            continue
        cands = []
        for t, v in r["top1"]:
            if t in ("None", "") or not isinstance(t, str):
                continue
            if any(t.startswith(p) for p in SENSORY) or sc[np.flatnonzero(ts == t)[0]] in (
                    "cb_sensory", "ol_sensory", "vnc_sensory", "visual_projection", "ascending_neuron"):
                cands.append(t)
            if len(cands) >= 3:
                break
        tests[label] = {}
        for t in cands:
            idx = np.flatnonzero(ts == t)
            if len(idx) == 0:
                continue
            si = torch.from_numpy(idx.astype(np.int64)).to(dev)
            ro = run(si, 1.0)
            tests[label][t] = {k: round(v, 1) for k, v in ro.items()}
            status("注入 %-14s(%s, n=%d) -> %s" % (label, t, len(idx), tests[label][t]))
    report["_stim_tests"] = tests
    json.dump(report, open("/content/fly/driver_result.json", "w"), indent=1, ensure_ascii=False)
    status("DONE")
except Exception:
    status("FAILED\n" + traceback.format_exc())
    raise
