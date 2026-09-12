import os, time, json, math, traceback
import numpy as np, pandas as pd, torch

OUT = "/content/fly/normalized"
STATUS = "/content/fly/char_status.txt"
dev = "cuda"

def status(m):
    open(STATUS, "w", encoding="utf-8").write("%s | %s" % (time.strftime("%H:%M:%S"), m))
    print(m, flush=True)

try:
    status("load graph")
    meta = pd.read_feather(OUT + "/neurons.feather")
    indptr = np.load(OUT + "/indptr.npy").astype(np.int32)
    indices = np.load(OUT + "/indices.npy").astype(np.int32)
    w = np.load(OUT + "/weights.npy"); sign = np.load(OUT + "/sign.npy")
    N = len(meta)
    crow = torch.from_numpy(indptr).to(dev); col = torch.from_numpy(indices).to(dev)
    sg = torch.from_numpy(sign).to(dev)
    Wraw = torch.from_numpy(w).to(dev) * sg[col.long()]

    ts = meta.type.astype(str).str.strip()
    def sel(fn):
        return torch.from_numpy(np.flatnonzero(fn(ts).to_numpy()).astype(np.int64)).to(dev)

    SENSE = {
        "sugar": sel(lambda s: s.str.match(r"^LB3[a-d]$")),
        "loom": torch.cat([sel(lambda s: s.str.startswith("LC4")),
                           sel(lambda s: s.str.startswith("LPLC2"))]),
        # 由 find_drivers.py 回溯并经 LIF 实测验证的上游驱动:
        #   LC9 -> DNp09(行走)  105 Hz ;  VS -> DNp20(转向)  112 Hz
        "walk_vis": sel(lambda s: s.str.match(r"^LC9$")),
        "turn_vis": sel(lambda s: s.str.match(r"^VS$")),
    }
    status("sense pools: " + ", ".join("%s=%d" % (k, v.numel()) for k, v in SENSE.items()))
    # 分母 = 该通道"实测可达峰值"(Hz), 使读出项归一到 [0,1], 与驱动项同尺度。
    # 用错分母会让读出项压过驱动项 -> 行为锁死(实测过: approach 占 60%)。
    GROUND = {"eat": ("MN9", 55.0), "flee": ("DNp01", 280.0),
              "approach": ("DNp20", 120.0), "explore": ("DNp09", 120.0)}
    MOTOR = {}
    for b, (nm, _) in GROUND.items():
        MOTOR[b] = sel(lambda s, n=nm: s.eq(n))
    status("grounded=%s fallback=%s" % (list(GROUND), ["drink","cool","warm","seek_humid","groom","court","rest"]))

    v0 = -52.0; v_rst = -52.0; v_th = -45.0
    t_mbr = 20.0; tau_syn = 5.0; t_rfc = 2.2; t_dly = 1.8
    w_syn = 0.275; r_poi = 150.0; f_poi = 250.0; GAIN = 0.65
    DT = 1.0
    D = int(round(t_dly / DT)); decay_g = float(np.exp(-DT / tau_syn))
    BRAIN_STEPS = 15
    A = torch.sparse_csr_tensor(crow, col, Wraw * (w_syn * GAIN), (N, N))
    st = {"v": torch.full((N,), v0, device=dev), "g": torch.zeros(N, device=dev),
          "refr": torch.zeros(N, device=dev), "spk": torch.zeros(N, device=dev),
          "buf": [torch.zeros(N, device=dev) for _ in range(D)],
          "rng": torch.Generator(device=dev).manual_seed(0)}

    def brain_step(sensory):
        p = {k: (r_poi * inten) * DT / 1000.0 for k, inten in sensory.items()}
        cnt = {b: 0 for b in MOTOR}
        for _ in range(BRAIN_STEPS):
            st["refr"] = torch.clamp(st["refr"] - DT, min=0.0)
            act = st["refr"] <= 0
            I = torch.sparse.mm(A, st["buf"].pop(0).view(-1, 1)).view(-1)
            st["buf"].append(st["spk"])
            st["g"] = torch.where(act, st["g"] * decay_g + I, st["g"])
            st["v"] = torch.where(act, st["v"] + (DT / t_mbr) * (v0 - st["v"] + st["g"]), st["v"])
            for k, pp in p.items():
                if pp <= 0:
                    continue
                idx = SENSE[k]
                h = torch.rand(len(idx), device=dev, generator=st["rng"]) < pp
                if bool(h.any()):
                    st["v"][idx[h]] += w_syn * f_poi
            s = act & (st["v"] > v_th)
            st["v"] = torch.where(s, torch.full_like(st["v"], v_rst), st["v"])
            st["g"] = torch.where(s, torch.zeros_like(st["g"]), st["g"])
            st["refr"] = torch.where(s, torch.full_like(st["refr"], t_rfc), st["refr"])
            st["spk"] = s.float()
            for b in MOTOR:
                cnt[b] += int(s[MOTOR[b]].sum())
        sec = BRAIN_STEPS * DT / 1000.0
        return {b: cnt[b] / max(len(MOTOR[b]), 1) / sec for b in MOTOR}

    BASE = ("hunger", "thirst", "heat", "cold", "dry", "humid", "threat")
    BEH = {
        "eat": ({"hunger": 2.2}, 0.0, 1.5, 0.30, 1.00, 3.0, 2.0),
        "drink": ({"thirst": 2.2}, 0.0, 1.2, 0.30, 1.00, 2.0, 2.0),
        "cool": ({"heat": 2.4}, 0.05, 1.0, 0.28, 1.00, 3.0, 1.0),
        "warm": ({"cold": 2.4}, 0.05, 1.0, 0.28, 1.00, 3.0, 1.0),
        "seek_humid": ({"dry": 2.0}, 0.0, 1.5, 0.28, 1.05, 3.0, 2.0),
        "groom": ({"humid": 1.2}, 0.25, 2.0, 0.34, 1.05, 3.0, 3.0),
        "approach": ({"lonely": 2.2}, 0.10, 2.0, 0.30, 1.00, 3.0, 2.0),
        "court": ({"sexual": 2.2, "lonely": 0.6}, 0.0, 2.0, 0.30, 1.00, 4.0, 6.0),
        "explore": ({}, 0.70, 2.5, 0.38, 1.00, 4.0, 1.0),
        "rest": ({"fatigue": 1.8}, 0.10, 3.0, 0.30, 1.05, 6.0, 2.0),
        "flee": ({"threat": 4.5}, 0.0, 0.5, 0.18, 0.70, 0.6, 1.0),
    }
    PRIO = {"eat": 1.0, "drink": 1.0, "cool": 1.0, "warm": 1.0, "seek_humid": 1.0,
            "groom": 0.85, "approach": 0.95, "court": 0.95, "explore": 0.55,
            "rest": 1.0, "flee": 1.2}
    BGAIN = {"eat": 1.0, "flee": 1.0, "approach": 1.0, "explore": 1.0}

    rngp = np.random.default_rng(0)
    body = {"energy": 0.70, "hydration": 0.70, "temp": 25.0, "fatigue": 0.10,
            "boredom": 0.05, "last_social": 0.0, "last_mating": 0.0, "activity": 0.0}
    world = {"food": 1.0, "water": 1.0, "flies_near": 0, "threat": 0.0,
             "humidity": 0.55, "temp": 25.0, "novelty": 1.0}
    # 习惯化: 反复触发某接地通道 -> 其读出贡献衰减, 防止"驱动永不满足 -> 行为锁死"
    hab = {b: 0.0 for b in GROUND}
    dt = 0.1; t = 0.0; day = 0.0

    def drives():
        return {
            "hunger": min(1, max(0, 1 - body["energy"])),
            "thirst": min(1, max(0, 1 - body["hydration"])),
            "heat": min(1, max(0, (body["temp"] - 27) / 8)),
            "cold": min(1, max(0, (22 - body["temp"]) / 8)),
            "dry": min(1, max(0, (0.40 - world["humidity"]) / 0.40)),
            "humid": min(1, max(0, (world["humidity"] - 0.75) / 0.25)),
            "lonely": min(1, max(0, (t - body["last_social"] - 25) / 120)),
            "sexual": min(1, max(0, (t - body["last_mating"] - 55) / 180)),
            "fatigue": min(1, max(0, body["fatigue"])),
            "threat": min(1, max(0, world["threat"])),
            "boredom": min(1, max(0, body["boredom"])),
        }

    res = {"actions": {}, "grounded_hits": {b: 0 for b in GROUND},
           "fallback_hits": 0, "readout_sum": {b: 0.0 for b in GROUND}, "n_readout": 0,
           "n_court": 0, "temp_min": 99.0, "temp_max": -99.0,
           "hum_min": 9.0, "hum_max": -9.0}
    TICKS = int(os.environ.get("FLY_TICKS", "1500"))
    NQ = 4
    res["quarters"] = [{} for _ in range(NQ)]          # 按时间段统计, 用于检测行为锁死
    t0 = time.time()
    for tick in range(TICKS):
        day += dt / 120.0
        base_t = 25.0 + 6.0 * math.sin(2 * math.pi * day)
        base_h = 0.55 + 0.20 * math.sin(2 * math.pi * day + 1.2)
        world["temp"] += dt * (-(world["temp"] - base_t) / 30.0) + 0.10 * math.sqrt(dt) * float(rngp.standard_normal())
        world["humidity"] += dt * (-(world["humidity"] - base_h) / 40.0) + 0.015 * math.sqrt(dt) * float(rngp.standard_normal())
        world["humidity"] = float(np.clip(world["humidity"], 0.05, 0.98))
        world["food"] = min(1.0, world["food"] + dt * 0.0015)
        world["water"] = min(1.0, world["water"] + dt * 0.0020)
        # 场景新颖度: 探索消耗它, 时间缓慢恢复它 —— 这自然形成"探索-恢复"循环, 不会锁死
        world["novelty"] += dt * 0.006 * (1.0 - world["novelty"])
        for k in hab:
            hab[k] *= math.exp(-dt / 20.0)
        world["threat"] = max(0.0, world["threat"] - dt * 0.05)
        if rngp.random() < dt * 0.030:
            world["threat"] = min(1.0, world["threat"] + 0.5 + 0.4 * float(rngp.random()))
        if rngp.random() < dt * 0.05:
            world["flies_near"] = int(np.clip(world["flies_near"] + int(rngp.integers(-1, 2)), 0, 4))
        body["energy"] = max(0.0, body["energy"] - dt * (0.0050 + 0.0030 * body["activity"]))
        body["hydration"] = max(0.0, body["hydration"] - dt * 0.0045)
        body["fatigue"] = min(1.0, body["fatigue"] + dt * 0.0060 * (0.4 + body["activity"]))
        body["temp"] += dt / 20.0 * (world["temp"] - body["temp"]) + dt * 0.15 * body["activity"]
        body["boredom"] = min(1.0, body["boredom"] + dt * 0.014)
        body["activity"] *= math.exp(-dt / 3.0)
        res["temp_min"] = min(res["temp_min"], world["temp"]); res["temp_max"] = max(res["temp_max"], world["temp"])
        res["hum_min"] = min(res["hum_min"], world["humidity"]); res["hum_max"] = max(res["hum_max"], world["humidity"])

        d = drives()
        sugar = float(np.clip(world["food"], 0, 1)) * (0.10 + 0.40 * float(np.clip(d["hunger"], 0, 1)))
        loom = (0.10 + 0.40 * float(np.clip(d["threat"], 0, 1))) if d["threat"] > 0.02 else 0.0
        # 视觉通道编码的是"外界有没有东西可看", 不是"我想不想看":
        #   walk_vis <- 场景新颖度(探索会消耗) x 无聊增益
        #   turn_vis <- 视野里有无同类    x 孤独增益
        # 内部驱动只做增益调制(与 hunger 调制 sugar 同一套模式), 因此不存在"永不满足"。
        social = 1.0 if world["flies_near"] > 0 else 0.0
        walk_vis = float(np.clip(world["novelty"], 0, 1)) * (0.10 + 0.40 * float(np.clip(d["boredom"], 0, 1)))
        turn_vis = social * (0.10 + 0.40 * float(np.clip(d["lonely"], 0, 1)))
        ro = brain_step({"sugar": sugar, "loom": loom,
                         "walk_vis": walk_vis, "turn_vis": turn_vis})
        for b in ro:
            res["readout_sum"][b] += ro[b]
        res["n_readout"] += 1

        best, best_s, best_src = None, 0.0, ""
        stress = max(d[k] for k in BASE)
        for name in BEH:
            wts, base, tau, sigma, thr, dur, cool = BEH[name]
            drv = base + sum(wt * d[k] for k, wt in wts.items())
            if name in GROUND:
                term = (BGAIN[name] * (ro[name] / GROUND[name][1])
                        * (1.0 - 0.7 * min(1.0, hab[name])))
                src = "connectome"
            else:
                term = 0.0
                src = "fallback"
            u = drv + term + sigma * 0.3 * float(rngp.standard_normal())
            score = u * PRIO[name] * (1.0 + 1.5 * stress)
            if score > thr and score > best_s:
                best, best_s, best_src = name, score, src
        if best:
            res["actions"][best] = res["actions"].get(best, 0) + 1
            q = res["quarters"][min(NQ - 1, tick * NQ // TICKS)]
            q[best] = q.get(best, 0) + 1
            if best_src == "connectome":
                res["grounded_hits"][best] += 1
            else:
                res["fallback_hits"] += 1
            if best == "eat":
                if world["food"] > 0.02:
                    body["energy"] = min(1.0, body["energy"] + 0.15); world["food"] -= 0.03
            elif best == "drink":
                if world["water"] > 0.02:
                    body["hydration"] = min(1.0, body["hydration"] + 0.12); world["water"] -= 0.03
            elif best == "cool":
                body["temp"] -= 0.3
            elif best == "warm":
                body["temp"] += 0.3
            elif best == "seek_humid":
                world["humidity"] = min(1.0, world["humidity"] + 0.02)
            elif best == "groom":
                body["boredom"] = max(0.0, body["boredom"] - 0.03)
            elif best == "approach":
                if world["flies_near"] > 0:
                    body["last_social"] = t
                elif rngp.random() < 0.25:
                    world["flies_near"] += 1
            elif best == "court":
                if world["flies_near"] > 0:
                    body["last_mating"] = t; res["n_court"] += 1
            elif best == "explore":
                body["boredom"] = max(0.0, body["boredom"] - 0.10)
                world["novelty"] = max(0.0, world["novelty"] - 0.35)   # 探索消耗新颖度
                if rngp.random() < 0.20:
                    world["food"] = min(1.0, world["food"] + 0.04)
                if rngp.random() < 0.15:
                    world["water"] = min(1.0, world["water"] + 0.04)
            elif best == "rest":
                body["fatigue"] = max(0.0, body["fatigue"] - 0.06)
            elif best == "flee":
                world["threat"] = max(0.0, world["threat"] - 0.5)
            body["activity"] = 0.6
            if best in hab:
                hab[best] = min(1.0, hab[best] + 0.35)
        t += dt
        if tick % 150 == 0:
            status("tick %d/%d t=%.0fs acts=%d reads MN9=%.1f DNp01=%.1f DNp20=%.1f DNp09=%.1f | %.0fs"
                   % (tick, TICKS, t, sum(res["actions"].values()),
                      ro["eat"], ro["flee"], ro["approach"], ro["explore"], time.time() - t0))

    res["n_ticks"] = TICKS
    res["wall_s"] = round(time.time() - t0, 1)
    res["readout_mean"] = {b: round(res["readout_sum"][b] / max(res["n_readout"], 1), 2) for b in GROUND}
    res["world_range"] = {"temp": [round(res.pop("temp_min"), 1), round(res.pop("temp_max"), 1)],
                          "humidity": [round(res.pop("hum_min"), 2), round(res.pop("hum_max"), 2)]}
    json.dump(res, open("/content/fly/char_result5.json", "w"), indent=1)
    status("DONE wall=%.0fs acts=%s grounded=%s fallback=%d"
           % (res["wall_s"], res["actions"], res["grounded_hits"], res["fallback_hits"]))
except Exception:
    status("FAILED\n" + traceback.format_exc())
    raise
