# freqkv-ext

基于 DSP 视角扩展的 KV cache 压缩实验框架：把 FreqKV 的 DCT-II 算子替换成**DFT
（RoPE-aware bandpass）**和**小波（自适应阈值）**两条新路。目标是用同一套
FreqKV scaffolding 在公平基线下检验两个假设：

1. **DFT RoPE-aware bandpass**：由 DFT 调制定理，RoPE 在 pair-complex 上是每对
   独立的频域整体平移 $\theta_i$。一个 RoPE-matched 的带通在同等压缩预算下，
   应当比 FreqKV 的均匀低通更好地保住和 attention 相关的信息——尤其是对那些
   $\theta_i$ 较大、编码短程位置信息的早期 hidden pair。

2. **小波 + 自适应阈值**：时间-频率局部化的基函数，应当能保住 FreqKV 全局 DCT
   抹平掉的局部细节信号（needle、代码符号、公式、表头数值）。

DSP 推导和精确算法请见 [`docs/METHOD.zh.md`](docs/METHOD.zh.md)，数学原理见
[`docs/MATH.zh.md`](docs/MATH.zh.md)，创新点和实验 TODO 见
[`docs/INNOVATIONS.zh.md`](docs/INNOVATIONS.zh.md)。

## 目录结构

```
freqkv-ext/
├── pyproject.toml                  # uv 管理依赖，[cpu]/[gpu] extras 分开
├── src/freqkv_ext/
│   ├── transforms/
│   │   ├── dct_baseline.py         # 严格复刻 FreqKV 的 DCT
│   │   ├── dft_lowpass.py          # DFT 低通（sanity baseline）
│   │   ├── dft_rope_aware.py       # 每对 pair 以 theta_i 为中心的带通（新算子）
│   │   └── wavelet.py              # DWT + 硬阈值
│   ├── rope_utils.py               # theta_i、bin 偏移、pair↔complex 转换
│   ├── patch.py                    # monkey-patch FreqKV.dct_compress
│   └── spectrum.py                 # K 频谱抽取工具
├── scripts/
│   ├── analyze_spectrum.py         # 实验 1（小算力）：pre/post-RoPE 频谱
│   ├── eval_ppl.py                 # 实验 2（H100）：PG-19 / Proof-pile PPL
│   ├── eval_longbench.py           # 实验 3（H100）：LongBench
│   ├── eval_needle.py              # 实验 4（H100）：Needle-in-a-Haystack
│   ├── train.py                    # 微调入口（H100）
│   ├── h100_setup.sh               # H100 一键装环境
│   └── h100_run_all.sh             # H100 全流水线
├── tests/                          # CPU-only 单元测试（24 个全过）
└── docs/                           # 英文 + 中文文档
```

## H100 服务器一键流程（明天开跑用）

```bash
# 1. 把 freqkv-ext 整目录 rsync 到 /workspace/freqkv-ext。
# 2. 装环境（自动装 uv / clone FreqKV / 装 torch+cuda / 装 flash-attn / sanity check）：
bash /workspace/freqkv-ext/scripts/h100_setup.sh

# 3. 跑全流水线（spectrum → train × 4 → PPL × 4 × 3）：
source /workspace/freqkv-ext/.venv/bin/activate
MODEL_PATH=/path/to/Llama-2-7b-hf bash /workspace/freqkv-ext/scripts/h100_run_all.sh
```

可调参见 ``scripts/h100_run_all.sh`` 顶部的 env vars，包括按 stage 跳过、改方法
列表、改训练长度、改评测长度等。

## 期望结果

- **频谱分析图**：每张 layer png 在 post-RoPE 子图里应该看到每对 pair 的能量峰
  落在标注的红色虚线 $n_i = \mathrm{round}(\theta_i N / (2\pi))$ 附近。如果
  对齐，DFT-RoPE 这条路就有了实证根基；如果不对齐，先回来讨论失败模式再决定
  是否继续花算力。
- **PG-19 PPL**：预期排序 ``dft_rope ~ dct < dft_lowpass``。DCT 在 AR(1) 平滑
  信号上近似最优（KLT），DFT-RoPE 应当持平而不付出 pre-RoPE caching 的代价。
  小波可能略输 DCT。
- **LongBench**：DFT-RoPE 应当在 HotpotQA / 代码 / RULER NIAH 上反超 DCT，因为
  这些是 FreqKV 自己掉点的子任务。
- **Needle**：对 DFT-RoPE 的决定性测试。如果在超过原始窗口（8K+）的 needle 上
  没有显著改善，DFT-RoPE 的假设在实证上被否定，把精力转向小波路径。

## 作者备注

本仓库故意不动 FreqKV 的代码。所有 compressor 通过 ``freqkv_ext.patch.install``
在运行时替换 ``llama_attn_replace_dct_mempe.dct_compress``，然后跳进 FreqKV 自己
的 ``fine-tune.py`` / ``eval.py``。这样 baseline 严格对齐，只差一个压缩算子，
排除了由 scaffolding 差异引入的奇怪 bug。
