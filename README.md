# fly-character — 果蝇数字角色

用**真实的成年雄果蝇中枢神经系统连接组**（MaleCNS v1.0，166,700 神经元 / 25,582,938 有向连接）
当作一个虚拟角色的大脑：识别细胞构成的感觉→脑→行为通路驱动它的行动，
一个"想说话的冲动越过阈值"的门控机制决定它什么时候开口说话，
并提供浏览器里的活体观测台可以实时看曲线、调参数。

> 这不是"用果蝇脑做的语言模型"。**语言来自发声通道的模板/LLM，果蝇脑只负责行为**
> —— 这是刻意的划分：连接组能不能真的驱动行为，必须和"模型会不会说话"分开评估。

---

## 快速开始（Colab）

```bash
# 1) 取代码（公开仓库，不需要 token）
cd /content && git clone --depth 1 https://github.com/hsgwktb/fly-character.git fly-code

# 2) 取连接组数据（免登录直链，约 1.11 GB）
mkdir -p /content/fly/data && cd /content/fly/data
B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
curl -sLO $B/body-annotations-male-cns-v1.0-minconf-0.5.feather
curl -sLO $B/body-neurotransmitters-male-cns-v1.0.feather
curl -sLO $B/connectome-weights-male-cns-v1.0-minconf-0.5.feather

# 3) 建图（约 5 分钟，产出 213 MB 的 CSR）
python /content/fly-code/experiments/build_graph.py 2>/dev/null || \
  echo "build_graph 见 README 说明，或用 experiments/ 里的脚本自建"

# 4) 起观测台（独立进程，端口 8000）
nohup python -u /content/fly-code/webui_character.py > /content/fly/webui.log 2>&1 &

# 5) 暴露到公网
[ -x /content/cloudflared ] || curl -sL -o /content/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x /content/cloudflared
nohup /content/cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate \
  > /content/fly/tunnel.log 2>&1 &
sleep 20 && grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /content/fly/tunnel.log | head -1
```

环境变量：
- `FLY_DATA`：图产物目录，默认 `/content/fly/normalized`
- `FLY_UI_PORT`：WebUI 端口，默认 `8000`（**不要用 8080**，Colab 自带 node 占用）

依赖：`torch`（Colab 预装 CUDA 版）、`numpy`、`pandas`、`pyarrow`。

---

## 架构

```
世界/身体 ──> 内驱力(11 维) ─┬─ 刺激量 × (0.10 + 0.40×驱动) ─> 识别感觉细胞
                             │        sugar←食物×饥饿   loom←威胁
                             │        walk_vis←场景新颖度×无聊   turn_vis←视野内同类×孤独
                             │
                             └──> 真实 MaleCNS LIF（α突触 τ=5ms、不应期 2.2ms、突触延迟 1.8ms）
                                        │
                                        └──> 识别输出细胞 ──> 行为倾向
                                               MN9→进食  DNp01→逃逸  DNp20→转向  DNp09→行走
                                        ＋ 其余 7 个行为回退到手写权重

想说话的冲动 u_speak ← 底噪 + 唤醒 + 情感强度 + 寂寞(受无聊门控) + 性欲 + 最大亏缺 + 无聊 + 反刍
u_speak > θ(t) 时开口；θ 是对"最近发声频率"做负反馈的积分控制器，目标率随内驱力上浮
```

**关键设计：接地必须是"输入+输出"的配对。** 只把读出接到识别细胞、却没有识别细胞的上游
输入去驱动它，等于接了一个永远为 0 的通道（实测过：DNp09/DNp20 一开始读出恒为 0）。

**另一个关键设计：视觉通道编码"刺激"，不编码"驱动"。** LC9/VS 是视觉细胞，
该编码"外界有没有东西可看"。内部驱动只做增益调制——这与"饥饿放大糖通道增益"
是同一套模式，从结构上排除了"驱动永不满足导致行为锁死"。

---

## 观测台页面

| 区域 | 内容 |
|---|---|
| 顶栏 | t / tick / **fps** / 连接组驱动占比 / 发声模式；暂停·重置 |
| 参数 | 脑增益、**脑步数每 tick**（时间尺度旋钮）、习惯化强度、感觉带上下限、四个驱动倍率、静息目标发声率、阈值积分增益 κ、**自适应阈值开关** |
| 图 1 | 内驱力曲线（饥饿/口渴/热冷/无聊/孤独/威胁） |
| 图 2 | **想说话冲动 u 与阈值 θ**，越过即开口 |
| 图 3 | 识别细胞读出（MN9 / DNp01 / DNp20 / DNp09） |
| 事件流 | 行为与台词，每条带 `连接组` / `回退` 标记 |

「自适应阈值」开关是刻意的教学开关：关掉就能直接看到固定阈值的病态
（发声率失控、间隔像节拍器）。

---

## 实测数据（可复现）

图与官方公布完全吻合：**166,700 节点 / 25,582,938 有向边 / 0 重复边**，
与 SciPy 稀疏矩阵对拍**最大绝对误差 0.000e+00**。

反射验收（Shiu 2024 常数，gain 0.65，300 ms）：

| 条件 | 群体放电率 | DNp01 巨纤维 |
|---|---|---|
| 静默 | 0.0 Hz | 0.0 |
| 刺激 LC4+LPLC2（484 个 looming 细胞） | 2.03 Hz | **1053 → 285 Hz**（加不应期后） |
| 对照：随机 484 个神经元同样刺激 | 182.7 Hz | **0.0** |

味觉（同一张图）：

| 条件 | MN9 进食 |
|---|---|
| 糖（LB3a–d） | 55.0 Hz |
| 苦（LB1a–d） | 0.0 Hz |
| **糖 + 苦** | **0.0 Hz（−100%）**，而群体率反而上升 → 通路特异的抑制 |

角色闭环（1500 tick = 150 s 角色时间）：连接组驱动占比 **56%**（600 s 长程测试），
行为覆盖 8 种，四段中最高的单行为占比仅 28%（无锁死）。

---

## 诚实的局限

1. **只有 4/11 个行为是真细胞驱动的**：`eat / flee / approach / explore`。
   其余（drink / cool / warm / seek_humid / groom / court / rest）在 MaleCNS 注释里
   找不到对应细胞类型，只能回退到手写权重。**回退比例必须一直公开**，否则会高估连接组贡献。
2. **连接组贡献占比必须和行为分布一起报**。本项目踩过两次坑：一次是"脑项幅值占比"
   被分母骗到 0.62，一次是未归一尺度让一个行为锁死、把占比刷到 92.6%（假象）。
3. **脑的时间尺度是压缩的**：默认每 100 ms 角色时间只跑 15 ms 神经时间（0.15×），
   所以读出绝对值不可跨配置比较，只有相对排序可用。页面上的"脑步数每 tick"就是这个旋钮。
4. **发声是模板句**，不是真 LLM。接真模型要改发声通道，且必须异步。
5. **未验证**：苦味抑制是否确实经由 Scapula（只证明了行为极性正确，没证明中间神经元身份）。
6. 未知递质一律按兴奋处理；胺类调质置 0（占 1.5% 突触质量）。这些约定会影响具体通路强弱。

---

## 文件

| 文件 | 作用 |
|---|---|
| `webui_character.py` | 观测台后端：连接组识别接口 + 说话门控 + HTTP 服务（独立进程） |
| `ui.html` | 观测台前端（后端从同目录读它，无 base64 嵌入） |
| `experiments/lif_shiu.py` | Shiu 2024 真实 LIF 标定（静默/looming/糖/苦 × 各 gain） |
| `experiments/lif_interface.py` | 识别接口耦合矩阵 |
| `experiments/find_drivers.py` | 回溯目标输出的上游驱动，并用 LIF 实测注入验证 |
| `experiments/lif_taste.py` | 糖/苦/糖+苦 抑制性检验 |
| `experiments/lif_dose.py` | 糖通道剂量-反应曲线 |
| `experiments/char_loop.py` | 角色主循环（接进连接组后端，含长程稳定性测试） |

---

## 数据来源与引用

- Google Research + HHMI Janelia, 2026-09-03：MaleCNS v1.0 完整连接组
  （*Sexual dimorphism in the complete connectome of the Drosophila male central nervous system*, Cell）
- Shiu et al. 2024, *Nature* 634:210–219：全脑 LIF 模型与神经递质符号约定
- 常数与标定口径参照 [flybench](https://github.com/brandoncho369/flybench) 的 `shiu2024.yaml` / `malecns_minecraft.yaml`
- 模式/接口设计受 [DOOMFLY](https://github.com/nftechie/doomfly)（含负面结果与对照组）启发

## 许可

代码 MIT。上游连接组数据与模型常数保留其各自条款（CC BY 4.0 / 论文引用）。

---

## 求偶 / 性欲指标（含一次有结论的失败尝试）

页面顶栏显示 **性欲 sexual** 与 **求偶指数 CI**，并有一条专门的三线图
（性欲 / 求偶指数 / 社交显著性）。

**求偶指数的口径沿用果蝇行为学的 CI(courtship index)**：不是加权和，而是
**观察窗口（60 s）内处于求偶行为的时间占比**，因此可以直接解释、也能和文献里的 CI 对照。
它和"性欲"是两个不同的量：性欲是内部状态，CI 是**表达出来的**求偶。
所以 CI=0 不一定代表"不想"，而是"视野里没有同类"——页面上的
"人工社交显著性"滑块就是把一只同类放进视野，用来观察性欲如何转成实际求偶。

### 为什么求偶没有像进食/逃逸那样接地（实测）

MaleCNS 里确实有求偶细胞：**`pC1` 156 个**（即文献里的 P1 求偶命令神经元）、
`aSP` 46、`vPR` 17、`DNp13` 2（同义词 `Kimura 2015: pMN1; Ruta 2010: DN1`）、
`fru` 在 synonyms 里出现 190 次。于是按项目一贯做法回溯 + 实测：

| 注入范围 | pC1 (Hz) | DNp13 (Hz) | 群体率 (Hz) |
|---|---|---|---|
| 单个上游类型（4–16 个细胞，如 `mAL_m8`/`SMP702m`/`oviIN`） | **0.0** | 0.0 | ≈0 |
| 视觉细胞 `LC16`（182 个） | 3.0 | 0.0 | 0.39 |
| **全部 7,892 个一级上游（gain 0.65）** | **161.8** | 255.0 | 11.43 |
| 全部一级上游（gain 1.30） | 173.7 | 292.5 | 16.23 |

**结论：pC1 不是"不能驱动"，而是"不能选择性驱动"。** 它是高阈值多模态整合器，
只有把近 8000 个上游细胞一起点亮才驱动得起来——这与它的生物学一致
（P1 需要信息素 + 视觉 + 听觉 + 唤醒共同汇聚）。而"一起点亮 8000 个细胞"不是一个可用的
感觉通道，它同时把群体率从 ~2 Hz 推到 11 Hz、把 DNp13 打到 255 Hz。

因此**求偶指标是程序层的，不是连接组接地的**——这一点必须在页面上和文档里都说清楚，
否则会把回退当贡献。改法也明确：
1. 恢复**胺类调质**（当前把多巴胺/血清素/章鱼胺置 0，占 1.5% 突触质量），
   pC1 的激活在真实果蝇里高度依赖这些门控；
2. 接入**信息素通道**（cVA → ORN → aSP/pMP），构成真正的多模态输入；
3. `DNp13` 实测能响应到 255 Hz，一旦有了选择性输入，它是现成的读出点。
