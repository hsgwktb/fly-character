import os, time, json, math, traceback
import numpy as np, pandas as pd, torch
OUT = "/content/fly/normalized"; STATUS = "/content/fly/char_status.txt"; dev = "cuda"
def status(m):
    open(STATUS, "w", encoding="utf-8").write("%s | %s" % (time.strftime("%H:%M:%S"), m)); print(m, flush=True)
try:
    status("load graph")
    meta = pd.read_feather(OUT + "/neurons.feather")
    indptr = np.load(OUT + "/indptr.npy").astype(np.int32); indices = np.load(OUT + "/indices.npy").astype(np.int32)
    w = np.load(OUT + "/weights.npy"); sign = np.load(OUT + "/sign.npy"); N = len(meta)
    crow = torch.from_numpy(indptr).to(dev); col = torch.from_numpy(indices).to(dev)
    sg = torch.from_numpy(sign).to(dev); Wraw = torch.from_numpy(w).to(dev) * sg[col.long()]
    ts = meta.type.astype(str).str.strip()
    def sel(fn): return torch.from_numpy(np.flatnonzero(fn(ts).to_numpy()).astype(np.int64)).to(dev)
    SENSE = {"sugar": sel(lambda s: s.str.match(r"^LB3[a-d]$")),
             "bitter": sel(lambda s: s.str.match(r"^LB1[a-d]$")),
             "loom": torch.cat([sel(lambda s: s.str.startswith("LC4")), sel(lambda s: s.str.startswith("LPLC2"))])}
    # 真细胞读出: MN9->eat, DNp01->flee, DNp20->approach(转向), DNp09->explore(行走)
    GROUND = {"eat": ("MN9", 50.0), "flee": ("DNp01", 300.0),
              "approach": ("DNp20", 40.0), "explore": ("DNp09", 40.0)}
    MOTOR = {b: sel(lambda s, n=n: s.eq(n)) for b, (n, _) in GROUND.items()}
    status("grounded behaviors: %s | fallback: %s" % (list(GROUND), ["drink","cool","warm","seek_humid","groom","court","rest"]))

    v0 = -52.0; v_rst = -52.0; v_th = -45.0
    t_mbr = 20.0; tau_syn = 5.0; t_rfc = 2.2; t_dly = 1.8
    w_syn = 0.275; r_poi = 150.0; f_poi = 250.0; GAIN = 0.65
    DT = 1.0                                   # 脑步长 1 ms (角色 100 ms/tick, 每 tick 15 步)
    D = int(round(t_dly/DT)); decay_g = float(np.exp(-DT/tau_syn))
    A = torch.sparse_csr_tensor(crow, col, Wraw * (w_syn * GAIN), (N, N))
    st = {"v": torch.full((N,), v0, device=dev), "g": torch.zeros(N, device=dev),
          "refr": torch.zeros(N, device=dev), "spk": torch.zeros(N, device=dev),
          "buf": [torch.zeros(N, device=dev) for _ in range(D)], "rng": torch.Generator(device=dev).manual_seed(0)}

    def brain_step(sensory):
        """sensory: {通道: 强度0..1}; 返回各读出细胞的放电率(Hz)"""
        p = {k: (r_poi*inten)*DT/1000.0 for k, inten in sensory.items()}
        cnt = {b: 0 for b in MOTOR}; steps = 15
        for _ in range(steps):
            st["refr"] = torch.clamp(st["refr"] - DT, min=0.0); act = st["refr"] <= 0
            I = torch.sparse.mm(A, st["buf"].pop(0).view(-1,1)).view(-1); st["buf"].append(st["spk"])
            st["g"] = torch.where(act, st["g"]*decay_g + I, st["g"])
            st["v"] = torch.where(act, st["v"] + (DT/t_mbr)*(v0 - st["v"] + st["g"]), st["v"])
            for k, pp in p.items():
                if pp <= 0: continue
                idx = SENSE[k]; h = torch.rand(len(idx), device=dev, generator=st["rng"]) < pp
                if bool(h.any()): st["v"][idx[h]] += w_syn*f_poi
            s = act & (st["v"] > v_th)
            st["v"] = torch.where(s, torch.full_like(st["v"], v_rst), st["v"])
            st["g"] = torch.where(s, torch.zeros_like(st["g"]), st["g"])
            st["refr"] = torch.where(s, torch.full_like(st["refr"], t_rfc), st["refr"])
            st["spk"] = s.float()
            for b in MOTOR: cnt[b] += int(s[MOTOR[b]].sum())
        sec = steps*DT/1000.0
        return {b: cnt[b]/max(len(MOTOR[b]),1)/sec for b in MOTOR}

    # ---------- 角色层 (与 flytwin 同构) ----------
    BASE = ("hunger","thirst","heat","cold","dry","humid","threat")
    BEH = {"eat":({"hunger":2.2},0.0,1.5,0.30,1.00,3.0,2.0), "drink":({"thirst":2.2},0.0,1.2,0.30,1.00,2.0,2.0),
           "cool":({"heat":2.4},0.05,1.0,0.28,1.00,3.0,1.0), "warm":({"cold":2.4},0.05,1.0,0.28,1.00,3.0,1.0),
           "seek_humid":({"dry":2.0},0.0,1.5,0.28,1.05,3.0,2.0), "groom":({"humid":1.2},0.25,2.0,0.34,1.05,3.0,3.0),
           "approach":({"lonely":2.2},0.10,2.0,0.30,1.00,3.0,2.0), "court":({"sexual":2.2,"lonely":0.6},0.0,2.0,0.30,1.00,4.0,6.0),
           "explore":({},0.70,2.5,0.38,1.00,4.0,1.0), "rest":({"fatigue":1.8},0.10,3.0,0.30,1.05,6.0,2.0),
           "flee":({"threat":4.5},0.0,0.5,0.18,0.70,0.6,1.0)}
    PRIO = {"eat":1.0,"drink":1.0,"cool":1.0,"warm":1.0,"seek_humid":1.0,"groom":0.85,"approach":0.95,"court":0.95,"explore":0.55,"rest":1.0,"flee":1.2}
    BGAIN = {"eat":1.6,"flee":1.6,"approach":0.8,"explore":0.8}   # 真细胞读出 -> 冲动 的增益

    rngp = np.random.default_rng(0)
    body = {"energy":0.70,"hydration":0.70,"temp":25.0,"fatigue":0.10,"boredom":0.05,
            "last_social":0.0,"last_mating":0.0,"activity":0.0}
    world = {"food":1.0,"water":1.0,"flies_near":0,"threat":0.0}
    dt = 0.1; t = 0.0; day = 0.0
    def drives():
        d = {"hunger":min(1,max(0,1-body["energy"])), "thirst":min(1,max(0,1-body["hydration"])),
             "heat":min(1,max(0,(body["temp"]-27)/8)), "cold":min(1,max(0,(22-body["temp"])/8)),
             "dry":min(1,max(0,(0.40-world["humidity"] if "humidity" in world else 0)/0.40)),
             "humid":0.0, "lonely":min(1,max(0,(t-body["last_social"]-25)/120)),
             "sexual":min(1,max(0,(t-body["last_mating"]-55)/180)),
             "fatigue":min(1,max(0,body["fatigue"])), "threat":min(1,max(0,world["threat"])),
             "boredom":min(1,max(0,body["boredom"]))}
        return d
    world["humidity"] = 0.55
    EAT, DRINK, COOL = 0, 0, 0
    res = {"n_ticks":0, "actions":{}, "grounded_hits":{b:0 for b in GROUND}, "fallback_hits":0,
           "readout_sum":{b:0.0 for b in GROUND}, "n_readout":0, "n_speak":0, "utter":[]}
    TICKS = 1500                                # 1500 * 100ms = 150 秒角色时间
    t0 = time.time()
    for tick in range(TICKS):
        day += dt/120.0
        base_t = 25.0 + 6.0*math.sin(2*math.pi*day); base_h = 0.55 + 0.20*math.sin(2*math.pi*day+1.2)
        world["temp"] = world.get("temp",25.0) + dt*(-(world.get("temp",25.0)-base_t)/30.0) + 0.10*math.sqrt(dt)*float(rngp.standard_normal())
        world["humidity"] += dt*(-(world["humidity"]-base_h)/40.0) + 0.015*math.sqrt(dt)*float(rngp.standard_normal())
        world["humidity"] = float(np.clip(world["humidity"],0.05,0.98))
        world["food"] = min(1.0, world["food"]+dt*0.0015); world["water"] = min(1.0, world["water"]+dt*0.0020)
        world["threat"] = max(0.0, world["threat"]-dt*0.05)
        if rngp.random() < dt*0.030: world["threat"] = min(1.0, world["threat"]+0.5+0.4*float(rngp.random()))
        if rngp.random() < dt*0.05: world["flies_near"] = int(np.clip(world["flies_near"]+int(rngp.integers(-1,2)),0,4))
        body["energy"] = max(0.0, body["energy"]-dt*(0.0050+0.0030*body["activity"]))
        body["hydration"] = max(0.0, body["hydration"]-dt*0.0045)
        body["fatigue"] = min(1.0, body["fatigue"]+dt*0.0060*(0.4+body["activity"]))
        body["temp"] += dt/20.0*(world["temp"]-body["temp"]) + dt*0.15*body["activity"]
        body["boredom"] = min(1.0, body["boredom"]+dt*0.014)
        body["activity"] *= math.exp(-dt/3.0)

        d = drives()
        # 驱动 -> 识别感觉通道 (用剂量曲线定出的可用带 0.10~0.50)
        sugar = float(np.clip(world["food"],0,1)) * (0.10 + 0.40*float(np.clip(d["hunger"],0,1)))
        loom  = 0.10 + 0.40*float(np.clip(d["threat"],0,1)) if d["threat"] > 0.02 else 0.0
        ro = brain_step({"sugar": sugar, "loom": loom})
        for b in ro: res["readout_sum"][b] += ro[b]; 
        res["n_readout"] += 1

        # 冲动: 真细胞读出替换掉原先的随机 brain_bias
        best, best_s = None, 0.0
        for name,(wts,base,tau,sigma,thr,dur,cool) in BEH.items():
            drv = base + sum(w*d[k] for k,w in wts.items())
            if name in GROUND:
                term = BGAIN[name]*(ro[name]/GROUND[name][1])      # 归一化后的真细胞读出
                src = "connectome"
            else:
                term
