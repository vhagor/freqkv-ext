# 方法：DFT RoPE-aware 与小波 KV 压缩

本文给出两个新压缩算子的 DSP 推导，并与 FreqKV 的 DCT-II 低通对照。

## 1. 一段话回顾 FreqKV

对一段 cache $K \in \mathbb{R}^{N \times d}$（pre-RoPE），FreqKV 沿序列轴做
DCT-II，保留前 $L = \gamma N$ 个系数，做 IDCT，再乘以 $\sqrt{L / N}$ 复原幅度。
压缩后的 cache 仍以 **pre-RoPE** 形式写回；RoPE 在 attention 时按"cache 内部
位置" $[0, S + L + r_\mathrm{recent})$（**不是**原始序列位置）现场应用。这种
"位置重指派"是 FreqKV 不需要 position 外推就能把上下文扩展到 256K 的关键。

为什么是 DCT 而不是 DFT？对一阶平稳 AR 过程 $x[n] = \rho\, x[n-1] + e[n]$，
当 $\rho \to 1$ 时其 Karhunen-Loève 变换（对协方差矩阵对角化的正交基）渐近退化
到 DCT-II。LLaMA 的 KV 沿序列轴经验上很接近这个 regime（相邻 token 高度相关），
所以 DCT-II 的能量集中性在这里近似最优；它还是实数变换、隐含偶对称边界（无
Gibbs 泄漏）。

## 2. RoPE 在频域里就是一个频移

LLaMA 的 RoPE 把相邻 hidden 维 $(d_{2i}, d_{2i+1})$ 配成一对，在位置 $t$ 把第
$i$ 对旋转 $\theta_i t$：

$$
\theta_i \;=\; \mathrm{base}^{-2i/d}, \qquad i = 0, 1, \dots, d/2 - 1.
$$

把 pair 视为复数 $c_i(t) = k_{2i}(t) + j\, k_{2i+1}(t)$，RoPE 变成

$$
c_i^{\mathrm{RoPE}}(t) \;=\; c_i(t) \, e^{j \theta_i t}.
$$

沿序列轴做 DFT，按调制定理：

$$
\begin{aligned}
C_i^{\mathrm{RoPE}}[\omega]
  &= \sum_{t} c_i^{\mathrm{RoPE}}(t) \, e^{-j \omega t} \\
  &= \sum_{t} c_i(t) \, e^{-j (\omega - \theta_i) t} \\
  &= C_i\!\left[\omega - \theta_i\right].
\end{aligned}
$$

**RoPE 是按 pair 的频域整体平移。** 对长度 $N$ 的 DFT，$\theta_i$ 对应 bin
偏移

$$
n_i \;=\; \mathrm{round}\!\left(\frac{\theta_i N}{2\pi}\right) \bmod N.
$$

## 3. 由此推出的几条结论

### 3.1 pre-RoPE 与 post-RoPE 的可互换性

FreqKV 必须 cache **pre-RoPE** K，因为 DCT 没有相位维度承接 RoPE 写入的位置
信息。若在 DCT 下 cache post-RoPE K，那些 token 的原始位置就被烤进了 cache；
做迭代压缩时，新 token 进入会让 attention 调用 out-of-bound 的 RoPE 位置 →
performance breakdown。

DFT 保留相位，RoPE 写入的位置信息能完整通过变换。由频移定理，任何想要的"压缩
token 位置"都可以通过对保留频段乘一个线性相位实现。pre / post-RoPE cache 在
DFT 域里**等价可互换**，每对 pair 只需 $\mathcal{O}(d/2)$ 次复数乘法。

### 3.2 信息究竟住在哪个频带

FreqKV 经验图（论文 Figure 1）显示 pre-RoPE K 的能量集中在低 bin。调制定理告诉
我们：post-RoPE 后第 $i$ 对的能量被搬到 bin 附近 $n_i$。对 LLaMA-2
（$d_\mathrm{head} = 128$，$\mathrm{base} = 10000$）：

- $\theta_0 = 1$ rad/sample。$N = 2048$ 时 $n_0 \approx 326$。
- $\theta_{d/4} \approx 0.1$ rad/sample，$n_{d/4} \approx 33$。
- $\theta_{d/2 - 1} \approx 1/10000$ rad/sample，$n_{d/2 - 1} \approx 0$。

FreqKV 的统一低通保留 bins $[0, L)$。**对低 $\theta$ 的 pair**（$n_i$ 接近
0），这就是 post-RoPE 的最优带；**对高 $\theta$ 的 pair**，FreqKV 切掉了
post-RoPE 实际能量所在的频带。这就是我们对 FreqKV 在 needle、代码、数值这类
依赖短程位置细节的任务上掉点的 DSP 解释假设。

### 3.3 RoPE-matched bandpass 算法

`dft_rope_aware_compress(x, L)`：

1. 把 $x \in \mathbb{R}^{B \times H \times N \times d}$ 重排为 pair-complex
   张量 $c \in \mathbb{C}^{B \times H \times N \times (d/2)}$，沿序列轴 DFT。
2. 对每对 pair $i$，gather 以 $n_i$ 为中心的 $L$ 个 bin（循环索引：
   $(n_i - L/2 + k) \bmod N$，$k = 0, \dots, L-1$）。
3. 把每对保留下的带通乘以 $e^{-j 2\pi (n_i / N) t}$（$t = 0, \dots, L-1$），把
   能量平移回基带——这样下游 RoPE 用 cache 内部位置重新加上去时，相位关系合得
   上。
4. 长度 $L$ 的 IDFT，幅度乘 $\sqrt{L / N}$（沿用 FreqKV 的约定）。
5. 复数对 → 实数对，恢复形状。

调用方应当传 **post-RoPE** K（当 `kv_type == "key"` 时）。包里附带的
`freqkv_ext.patch._wrap_with_rope_for_key` 包装器会在调用前先把 pre-RoPE K
旋转成 post-RoPE，保持 FreqKV 原 attention 路径不动。

### 3.4 这一版**还没有**做到的事

当前实现是 DFT 域里压缩，但 attention 仍然在时域计算（IDFT 回来再做）。下一步
（见 `docs/INNOVATIONS.md` 的 Phase 3）：

- 用 Parseval：

  $$
  \langle q, k \rangle \;=\; \sum_{t} q[t]\, k^{*}[t]
  \;=\; \frac{1}{N} \sum_{\omega} Q[\omega]\, K^{*}[\omega],
  $$

  attention 留在 DFT 域。
- V 侧把 IDFT 离线吸进 $W_o$（PALU 式融合），跳过推理期 IDFT。
- K 侧需要写自定义 kernel；先做 V，K 留到后面。

## 4. 小波路径

默认 `db4`、$\log_2 N$ 层小波分解；硬阈值保留每个 $(B, H, \text{hidden})$ 通道
前 $\gamma N d_\mathrm{head}$ 个最大幅度系数；逆变换后截到长度 $L$、再按
$\sqrt{L / N}$ 校准幅度。

选小波而不是 DCT / DFT 的理由：一个突变事件（needle、字面常量、函数名）对
DCT / DFT 来说是"时域 delta $=$ 频域常数"，会激活所有高频 bin，第一个被低通
切掉；而对小波它只激活相应尺度上的少数局部系数，硬阈值正好留下来。

代价：

- 当前实现用 PyWavelets 在 CPU 上做 DWT，是端到端的瓶颈；后续会换成 GPU 上的
  分层 depthwise conv 实现（见 `docs/INNOVATIONS.md` 的 Phase 3 (j)）。
- "硬截到长度 $L$" 是为了适配 FreqKV 的等长 cache 接口，会丢一些信息。原生
  小波 cache 应当直接存稀疏系数集，这是更大的接口重构，留到下一版。

## 5. 频谱分析脚本验证的是什么

`scripts/analyze_spectrum.py` 用 forward hook 抽取 frozen LLaMA 每层 `k_proj`
的输出（即 pre-RoPE K），手动 apply RoPE 得到 post-RoPE K，分别画 DFT 功率谱。
每张图叠加预测的 $n_i = \theta_i N / (2\pi)$ 红色虚线。

如果 post-RoPE 图中各 pair 的能量峰**确实**落在红线附近，调制定理在真实模型上
被验证、RoPE-matched bandpass 命中目标——后续 H100 训练 + 评测才有意义。

如果**不**对齐——可能是 RMSNorm 重缩放、sink token DC 泄漏、或 GQA 改变 K 的
统计——需要先做诊断（见 `docs/INNOVATIONS.md` 的 Phase 0 (b')），再决定是否
继续。

这个实验在单卡 24 GB GPU 上能跑（LLaMA-2-7B fp16 $\approx$ 14 GB 权重 + 少量
激活），是后续 H100 投入的 gate。
