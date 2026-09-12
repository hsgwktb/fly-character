import numpy as np, pandas as pd, torch, time, traceback, json
OUT="/content/fly/normalized"; dev="cuda"; STATUS="/content/fly/pc1u_status.txt"
def status(m):
    open(STATUS,"w",encoding="utf-8").write("%s | %s"%(time.strftime("%H:%M:%S"),m)); print(m,flush=True)
try:
    meta=pd.read_feather(OUT+"/neurons.feather")
    indptr=np.load(OUT+"/indptr.npy").astype(np.int64); indices=np.load(OUT+"/indices.npy").astype(np.int64)
    w=np.load(OUT+"/weights.npy"); sign=np.load(OUT+"/sign.npy"); N=len(meta)
    ts=meta.type.astype(str).str.strip(); s_ts=pd.Series(ts)
    PC1=np.flatnonzero(s_ts.str.match(r"^pC1").to_numpy())
    DNP13=np.flatnonzero(s_ts.eq("DNp13").to_numpy())
    # pC1 的完整一级上游
    up=np.unique(np.concatenate([indices[indptr[j]:indptr[j+1]] for j in PC1]))
    status("pC1=%d 一级上游=%d" % (len(PC1), len(up)))
    crow=torch.from_numpy(indptr.astype(np.int32)).to(dev); col=torch.from_numpy(indices.astype(np.int32)).to(dev)
    sg=torch.from_numpy(sign).to(dev); Wraw=torch.from_numpy(w).to(dev)*sg[col.long()]
    V0=VR=-52.0; VTH=-45.0; T_MBR=20.0; TAU=5.0; TRFC=2.2; TDLY=1.8
    WS=0.275; RP=150.0; FP=250.0; DT=0.1; D=int(round(TDLY/DT)); DG=float(np.exp(-DT/TAU))
    TGT={"pC1":torch.from_numpy(PC1.astype(np.int64)).to(dev),
         "DNp13":torch.from_numpy(DNP13.astype(np.int64)).to(dev)}
    si=torch.from_numpy(up.astype(np.int64)).to(dev); p=RP*DT/1000.0
    for GAIN in (0.65, 1.3, 2.0):
        A=torch.sparse_csr_tensor(crow,col,Wraw*(WS*GAIN),(N,N))
        g=torch.Generator(device=dev).manual_seed(0)
        V=torch.full((N,),V0,device=dev); G=torch.zeros(N,device=dev)
        rf=torch.zeros(N,device=dev); spk=torch.zeros(N,device=dev)
        buf=[torch.zeros(N,device=dev) for _ in range(D)]
        out={k:0 for k in TGT}; tot=0
        for _ in range(2000):
            rf=torch.clamp(rf-DT,min=0.0); act=rf<=0
            I=torch.sparse.mm(A,buf.pop(0).view(-1,1)).view(-1); buf.append(spk)
            G=torch.where(act,G*DG+I,G); V=torch.where(act,V+(DT/T_MBR)*(V0-V+G),V)
            h=torch.rand(len(si),device=dev,generator=g)<p
            if bool(h.any()): V[si[h]]+=WS*FP
            s=act&(V>VTH)
            V=torch.where(s,torch.full_like(V,VR),V); G=torch.where(s,torch.zeros_like(G),G)
            rf=torch.where(s,torch.full_like(rf,TRFC),rf); spk=s.float(); tot+=int(s.sum())
            for k in TGT: out[k]+=int(s[TGT[k]].sum())
        status("gain=%.2f 注入全部一级上游 -> pC1=%.1f DNp13=%.1f pop=%.2f"
               % (GAIN, out["pC1"]/len(PC1)/0.2, out["DNp13"]/len(DNP13)/0.2, tot/N/0.2))
    status("DONE")
except Exception:
    status("FAILED\n"+traceback.format_exc()); raise
