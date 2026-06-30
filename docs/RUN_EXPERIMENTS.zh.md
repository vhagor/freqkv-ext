# RST-KV 实验指导手册（H100）

本手册给出**按步骤照抄即可运行**的命令。配套理论与判生死标准见
`docs/EXPERIMENT_PLAN.zh.md`（实验编号 E1–E6 与此处一致）。

本轮要在 H100 上跑的是**全部小算力、训练无关**的三件事，目标是在投入大规模训练
（E4）之前把核心主张 **C2** 的成败定下来：

| 步骤 | 脚本 | 验证主张 | 预计耗时 | 输出 |
|------|------|----------|----------|------|
| E1（文本版）| `scripts/analyze_spectrum.py --no-plot` | C1：频率梳位置 | 单卡 ~15 min | 峰值对齐表（文本）|
| E2 + E3 | `scripts/rate_distortion.py` | C2：混合码优于 DCT | 单卡 ~1–2 h | 率失真表 + α 曲线 + 注水分配（文本）|

> **所有脚本都已改成「文本优先」**：默认把 markdown 表格直接打到 stdout，同时落盘到
> `--out-dir`。E1 用 `--no-plot` 跳过画图。你只需把终端里的表格**整段复制**发回即可。

---

## 0. 前置：仓库与模型

### 0.1 需要的两个仓库

| 仓库 | 用途 | 获取方式 |
|------|------|----------|
| `freqkv-ext`（本仓库）| 我们的实验代码 | 推到你的 GitHub 后在 H100 上 `git clone`，或 `rsync` 本地目录 |
| `FreqKV`（官方）| 基线 monkey-patch（E1/E2 用不到，E4 才用）| `h100_setup.sh` 会自动 clone |

E1/E2/E3 **只依赖 `freqkv-ext`**，不需要 FreqKV 官方仓库，也不需要 flash-attn。

### 0.2 模型

| 模型 | RoPE base | HF 链接 | 备注 |
|------|-----------|---------|------|
| LLaMA-2-7B | 1e4 | `meta-llama/Llama-2-7b-hf` | 主力，已在上一轮验证过梳齿（pair0→bin652）|
| LLaMA-3-8B | 5e5 | `meta-llama/Meta-Llama-3-8B` | E1 跨模型用：验证梳齿随 base 收拢 |
| Mistral-7B | 1e6 | `mistralai/Mistral-7B-v0.1` | E1 跨模型用 |

你上一轮用的是本地路径 `/root/llama2-7b/`。下面命令统一用环境变量 `MODEL`，按你的
实际路径设置即可。

---

## 1. 环境（首次或换机器时）

```bash
# 把 freqkv-ext 放到 $WORKSPACE 下（rsync 或 git clone），然后：
export WORKSPACE=/workspace            # 按你的机器改
cd $WORKSPACE/freqkv-ext

# 一键装环境（uv + torch CUDA + 依赖；E1/E2 其实只要 cpu/gpu extra 即可）
bash scripts/h100_setup.sh             # 含 FreqKV clone + flash-attn，E4 才需要

# 如果只想先跑 E1/E2/E3（不碰训练），可以用更轻的安装：
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[gpu]"             # 或 .[cpu]，E1/E2 都够用
```

设置公共变量（每次新开终端都先跑）：

```bash
cd $WORKSPACE/freqkv-ext
source .venv/bin/activate
export MODEL=/root/llama2-7b/          # 改成你的 LLaMA-2-7B 路径
export PYTHONPATH=$WORKSPACE/freqkv-ext/src:$PYTHONPATH
```

---

## 2. E1（文本版）——确认频率梳位置（C1）

目的：把上一轮「图上看着对齐」升级成**可复制粘贴的数字表**：每对 pair 的预测齿位
`n_i = round(theta_i · N / 2π)` vs 实测峰位、峰锐度、DC 占比。

```bash
python scripts/analyze_spectrum.py \
    --model_name_or_path "$MODEL" \
    --seq-len 4096 --num-samples 8 \
    --layers 0 8 16 31 \
    --rope-base 10000 \
    --no-plot \
    --peak-pairs 0 1 4 8 16 32 63 \
    --out-dir results/e1_llama2
```

会打印形如：

```
## Layer 16
| pair | theta_i | predicted n_i | observed peak | |peak-pred| | sharpness(peak/median) | DC frac |
| 0 | 1 | 652 | 651 | 1 | ... | ... |
...
```

**判读（C1 成立的标志）**：`|peak-pred|` 普遍很小（个位数 bin），`sharpness` 明显
大于 1（梳齿尖锐）。把整段 `# E1 peak-alignment` 文本复制回来即可。

**跨模型（强证据，可选）**：换 base 重跑两次，验证齿位按 base 收拢——

```bash
python scripts/analyze_spectrum.py --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --seq-len 4096 --num-samples 8 --layers 0 8 16 31 --rope-base 500000 \
    --no-plot --out-dir results/e1_llama3
python scripts/analyze_spectrum.py --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --seq-len 4096 --num-samples 8 --layers 0 8 16 31 --rope-base 1000000 \
    --no-plot --out-dir results/e1_mistral
```

预测：base 越大，同一 pair 的 `predicted n_i` 越小（齿向 DC 收拢），实测峰应随之
移动。三个模型都对上 → C1 从「单点观察」升级为「随 base 缩放的定律」。

---

## 3. E2 + E3（核心）——离线率失真 + 预算分配（C2）

这是**判生死**的实验，完全不训练。一条命令同时产出：

- **E2 主表**：每个 γ × 每种码（dct / dft_rope / wavelet / rst）的 K 误差、V 误差、
  **注意力输出误差**。
- **E2 α 曲线**：RST 的 bulk/residual 预算比 α 扫描，给出每个 γ 的最优 α。
- **E3 注水分配**：每对 pair 按能量做 water-filling，对比均匀分配的保留能量。
- **GO/NO-GO 提示**：自动算「最小 γ 下 DCT 与 RST 的注意力误差差」。

```bash
python scripts/rate_distortion.py \
    --model_name_or_path "$MODEL" \
    --seq-len 2048 --num-samples 8 \
    --layers 0 8 16 31 \
    --gammas 0.5 0.25 0.125 0.0625 0.03125 \
    --alpha-sweep 0.0 0.5 0.7 0.85 1.0 \
    --residual-domain time \
    --k-domain natural \
    --dtype float16 \
    --out-dir results/e2_llama2
```

**参数说明 / 调参**：

- `--seq-len 2048`：注意力误差要算 `[N,N]` 矩阵，逐 (batch, head) 循环。N=2048 在
  单卡上很轻；想看更长程效应可调到 4096（显存/时间 ~4×）。
- `--num-samples 8`：抓取的校准样本数。8 个 2K 序列足够稳定；想更稳可加到 16。
- `--k-domain natural`（默认）：DCT/小波在 **pre-RoPE** 上压、再补 RoPE（忠于 FreqKV
  真实管线）；DFT-RoPE/RST 直接在 **post-RoPE** 上压。这是最公平的「真实部署」对比。
  用 `--k-domain post` 可做「同一信号下纯基底对比」的消融。
- `--residual-domain time|wavelet`：残差用时域 top-k 还是小波域 top-k（E6 残差基消融）。
- 数据集默认 `EleutherAI/pile` 的 `test` split 流式读取；如需换用
  `--dataset wikitext --dataset-split 'test' --text-field text` 等。

**预计耗时**：4 层 × 5 个 γ × 4 种码 + α 扫描 + 注水，N=2048 下单卡约 1–2 h（主要
花在注意力 `[N,N]` 上；小波在 CPU 上跑，是次要瓶颈）。

### 判读标准（直接对应 EXPERIMENT_PLAN 的 C2）

看每层 **attn relerr** 列，重点在小 γ（0.125 / 0.0625 / 0.03125）：

- ✅ **C2 支撑**：`rst` 的 attn relerr 在小 γ 下**明显低于** `dct`（脚本末尾的
  GO/NO-GO 提示里 `mean(attn relerr DCT - RST) > 0.01`）。→ 进入 E4 训练。
- ⚠️ **打平**：rst ≈ dct → 在 PG-19 这类平滑文本上残差没用武之地；需要换更像
  needle / 代码的校准文本再看（残差的主战场）。
- ❌ **C2 被证伪**：rst 不优于 dct → 退回单分量（纯 dft_rope 或纯 wavelet），按
  EXPERIMENT_PLAN 的决策树降级为 measurement 论文。

**α 曲线**：预期 γ 大时最优 α→1（带通够用），γ 小时最优 α 下降（残差变重要）。这条
曲线本身就是一个新发现（论文 Fig 3）。

**E3 注水**：看 `gain` 列。若普遍 > 0 → 非均匀分配有价值（论文 Fig 4 成立）；若 ≈ 0
→ 各 pair 能量差不多，E3 可砍。

### 要回传给我的文本

把 `results/e2_llama2/rate_distortion.md` **整份内容**贴回来即可（它就是终端打印的
全部表格）。`rate_distortion.json` 留在服务器备查，不用贴。

---

## 3b. NE0 + NE1（诊断）——小波为什么赢？（C2 复盘）

E2 显示小波大幅领先、DFT-RoPE≈DCT、RST 反而更差。在转向新方法前，必须先坐实
**小波的优势来自哪里**。这一步完全离线、单卡 <1h、纯文本输出。

```bash
python scripts/diagnose_outliers.py \
    --model_name_or_path "$MODEL" \
    --seq-len 2048 --num-samples 8 \
    --layers 0 8 16 31 \
    --gammas 0.5 0.25 0.125 \
    --anchor-m 1 4 16 \
    --out-dir results/ne1_llama2
```

脚本一次性回答两件事：

- **NE0（基底 vs 域）**：把 DCT/小波分别在 **pre-RoPE** 和 **post-RoPE** 上重建 K。
  - `basis@post > 0` → 小波是真正更好的**基底**（不是只占了"压 pre-RoPE 更容易"的便宜）。
  - `dom.effect` → post-RoPE 比 pre-RoPE 本质上难压多少。
- **NE1（离群归因）**：K 的逐通道峰度、能量最集中的 token 位置（是否就是 sink/t=0）、
  DCT 与小波各自的重建误差有多少落在这些离群 token 上，以及把 top-m 离群 token
  作为**无损锚点**后，DCT↔小波的差距是否被抹平。

**判读（脚本末尾 `## Read-out` 自动给）**：

| 观察 | 含义 | 下一步 |
|------|------|--------|
| `basis@post` 明显 > 0 | 小波基底本身更优 | 小波就是方法主体 |
| DCT 误差大量落在 top-m token，小波几乎不落 | "全局变换抹离群"假设成立 | 离群/sink 是关键 |
| 锚点后 DCT 追平小波 | 赢点几乎全是离群 | 走"变换 + 无损锚点"（便宜路线 NE2）|
| 锚点后小波仍赢 | 赢点也在平滑主体 | 小波为主，锚点为辅 |

把 `results/ne1_llama2/diagnose.md` 整份贴回来即可。

## 3c. NE2'（诊断）——小波赢是「基底」还是「自适应选择」？

NE0 已证"小波是更好的基底"，但 NE0 拿 DCT-低通 比 小波-topk，**混淆了基底与选择规则**
（FreqKV 的 DCT 保留前 L 个低频，小波保留幅值最大的 L 个）。这一步把两者拆开，并刻画
K 的平滑性类别。单卡 <1h，纯文本。

```bash
python scripts/diagnose_basis.py \
    --model_name_or_path "$MODEL" \
    --seq-len 2048 --num-samples 8 \
    --layers 0 8 16 31 \
    --gammas 0.5 0.25 0.125 \
    --out-dir results/ne2b_llama2
```

**判读（脚本末尾自动给）**：

| 观察 | 含义 | 方法选择 |
|------|------|----------|
| basis effect ≫ selection effect | 赢点在基底 | 小波是方法主体 |
| selection effect ≫ basis effect | 赢点在自适应选择 | 用 **adaptive-DCT(top-k)** 即可，更省、保留快变换 |
| 两者相当 | 都重要 | 自适应选择 + 小波基 |
| first-difference kurtosis ≫ value kurtosis | K 是分段光滑/有界变差 | 解释"为何小波更优"的机理（写进论文）|

把 `results/ne2b_llama2/diagnose_basis.md` 整份贴回来。

## 4. 本地连通性自检（在 H100 跑之前，可选）

在任意机器（含本地 RTX5060 / 纯 CPU）先确认脚本能跑通、再上 H100：

```bash
# 率失真全链路（合成 AR(1)+spike）
python scripts/rate_distortion.py --dry-run \
    --seq-len 256 --num-samples 2 --layers 0 4 \
    --gammas 0.5 0.25 0.125 --device cpu --dtype float32 \
    --out-dir results/rd_dryrun

# 离群诊断全链路（合成 AR(1)+sink+spike）
python scripts/diagnose_outliers.py --dry-run \
    --seq-len 256 --num-samples 2 --layers 0 4 \
    --gammas 0.5 0.25 0.125 --device cpu --dtype float32 \
    --out-dir results/ne1_dryrun

# 基底/选择诊断全链路（合成分段光滑信号）
python scripts/diagnose_basis.py --dry-run \
    --seq-len 256 --num-samples 2 --layers 0 4 \
    --gammas 0.5 0.25 0.125 --device cpu --dtype float32 \
    --out-dir results/ne2b_dryrun

# 单元测试（48 个）
uv run --extra cpu pytest -q
```

诊断 dry-run 应看到：DCT 误差大量落在 sink（t=0）token 上、小波几乎不落，且
`basis@post > 0`——说明脚本与离群归因逻辑正确。（跑完删掉 `results/*_dryrun`。）

---

## 5. E4（训练 + 下游评测）——仅当 E2 通过才跑

**前置 gate：E2 显示 RST 在小 γ 占优。** 否则不要烧 H100。

通过后，用已有脚本（`scripts/train.py` + `scripts/eval_*.py`，或 `h100_run_all.sh`）
跑四个变体 `dct / dft_rope / wavelet / rst`，评测重点放在 γ ≤ 0.125 的高压缩区、
LongBench 子任务、以及 **Needle-in-a-Haystack**（残差分量的决定性测试）。具体命令见
`docs/RUN_H100.zh.md`。把 E2 选出的最优 α 通过 `--alpha` 传给 RST 训练变体。

---

## 6. 故障排查

| 现象 | 处理 |
|------|------|
| `No module named freqkv_ext` | 先 `source .venv/bin/activate`，或设 `PYTHONPATH=.../src` |
| `pytest: No such file` | 用 `uv run --extra cpu pytest`（pytest 在 cpu/dev extra 里）|
| 注意力步骤 OOM | 调小 `--seq-len`（2048→1024）或 `--num-samples` |
| 小波很慢 | 正常，PyWavelets 在 CPU；E2 阶段只关心精度，速度在 E5 才处理 |
| 数据集下载失败 | 换 `--dataset wikitext --dataset-split test`，或离线准备文本 |
| 峰位对不上预测 | 确认 `--rope-base` 与模型一致（L2=1e4，L3=5e5，Mistral=1e6）|

---

## 附录 A：上一轮频谱图片的新命名

上一轮 `results/` 里的时间戳图片已重命名，便于查找：

| 旧文件 | 新文件 | 内容 |
|--------|--------|------|
| 20260629-214924.png | `layer00_spectrum.png` | 第 0 层 pre/post-RoPE 频谱 |
| 20260629-214917.png | `layer00_sparsity.png` | 第 0 层 DCT/DFT/小波稀疏度小提琴图 |
| 20260629-214932.png | `layer04_spectrum.png` | 第 4 层频谱 |
| 20260629-214928.png | `layer04_sparsity.png` | 第 4 层稀疏度 |
| 20260629-214939.png | `layer08_spectrum.png` | 第 8 层频谱 |
| 20260629-214935.png | `layer08_sparsity.png` | 第 8 层稀疏度 |
| 20260629-214946.png | `layer16_spectrum.png` | 第 16 层频谱 |
| 20260629-214942.png | `layer16_sparsity.png` | 第 16 层稀疏度 |
| 20260629-214953.png | `layer31_spectrum.png` | 第 31 层频谱 |
| 20260629-214950.png | `layer31_sparsity.png` | 第 31 层稀疏度 |
| 20260629-214958.png | `sparsity_summary.png` | 逐层稀疏度对比汇总 |

## 附录 B：本轮新增/改动的代码

| 文件 | 作用 |
|------|------|
| `src/freqkv_ext/rdcodecs.py` | E2/E3 的「保留系数→重建到 N」码 + 注意力误差 + 注水分配 |
| `src/freqkv_ext/capture.py` | GQA-aware 抓取 Q/K/V（E2 用）|
| `src/freqkv_ext/transforms/rst_hybrid.py` | RST 混合码（FreqKV 定长接口，E4 训练用）|
| `scripts/rate_distortion.py` | E2 + E3 主脚本（文本输出）|
| `scripts/analyze_spectrum.py` | 新增 `--no-plot` + E1 文本峰值对齐表 |
| `tests/test_rdcodecs.py`, `tests/test_rst_transform.py` | 新增单测（共 38 个全过）|
