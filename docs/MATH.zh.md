# 数学原理

这份笔记把 `freqkv-ext` 里用到的 DSP 恒等式集中列一遍。是参考手册，不是教科
书；完整推导见 Oppenheim & Schafer（信号）或 Daubechies（小波）。

## 1. 离散傅里叶变换（DFT）

对长度 $N$ 的复序列 $x[0], \dots, x[N-1]$，DFT 与 IDFT 为

$$
X[k] \;=\; \sum_{n=0}^{N-1} x[n] \, e^{-j 2\pi k n / N}, \qquad k = 0, 1, \dots, N-1,
$$

$$
x[n] \;=\; \frac{1}{N} \sum_{k=0}^{N-1} X[k] \, e^{+j 2\pi k n / N}.
$$

矩阵形式 $X = F_N x$。NumPy / PyTorch 用非酉约定（$1/N$ 放在逆变换里）。

**Parseval 恒等式**（能量守恒）：

$$
\sum_{n=0}^{N-1} \bigl|x[n]\bigr|^2 \;=\; \frac{1}{N} \sum_{k=0}^{N-1} \bigl|X[k]\bigr|^2.
$$

这是"频域里做 attention"的形式化依据：两个序列的内积等于（差一个 $1/N$ 因子）
它们 DFT 的内积。

## 2. 调制定理（`freqkv-ext` 的核心恒等式）

若 $y[n] = x[n] \, e^{j \omega_0 n}$ 是 $x[n]$ 的频率调制版本，则

$$
\begin{aligned}
Y[k] &= \sum_{n=0}^{N-1} x[n] \, e^{j \omega_0 n} \, e^{-j 2\pi k n / N} \\
     &= \sum_{n=0}^{N-1} x[n] \, e^{-j \left(\tfrac{2\pi k}{N} - \omega_0\right) n} \\
     &= X\!\left[k - \frac{\omega_0 N}{2\pi}\right].
\end{aligned}
$$

bin 索引形式：时域乘 $e^{j \omega_0 n}$ 等价于频域**循环平移**
$\mathrm{round}\!\left(\omega_0 N / 2\pi\right)$ 个 bin。

把"RoPE 视作 DFT bin 平移"，依据就是这一条——RoPE 本质上是 pair-complex 上沿
时间乘 $e^{j \theta_i t}$。

## 3. DCT-II 与 AR(1) KLT 的关系

序列 $x[0], \dots, x[N-1]$ 的 Type-II DCT 为

$$
y[k] \;=\; \alpha_k \sum_{n=0}^{N-1} x[n] \cos\!\left(\frac{\pi k (2n+1)}{2N}\right),
$$

其中 $\alpha_0 = \sqrt{1/N}$、$\alpha_k = \sqrt{2/N}$（$k > 0$）。它是 $x$ 的
偶对称镜像扩展信号的 DFT，所以边界无 Gibbs 跳变。

对平稳 AR(1) 过程

$$
x[n] \;=\; \rho \, x[n-1] + e[n], \qquad |\rho| < 1,
$$

其 Karhunen-Loève 变换（对协方差矩阵对角化的正交基）随 $\rho \to 1$ 渐近退化到
DCT-II。这是 JPEG / FreqKV 选 DCT-II 的经典理由：对**实数、平滑、非周期**信号，
DCT-II 的能量集中性在 $L^2$ 意义下可证地优于 DFT。

LLaMA 沿序列轴的 pre-RoPE K 经验上接近 $\rho \to 1$ 区（相邻 token 高度相关），
这就是为什么 FreqKV 论文 Figure 1 显示 pre-RoPE 能量集中在低 DCT bin。

## 4. RoPE 的 pair-complex 形式

LLaMA RoPE 作用在每个 head 的 $d$ 维 K 上（$d$ 偶数）。它把相邻 hidden 维配成
复 channel $c_i = k_{2i} + j\, k_{2i+1}$，在位置 $t$ 把第 $i$ 对旋转

$$
c_i^{\mathrm{RoPE}}(t) \;=\; c_i(t) \, e^{j \theta_i t}, \qquad
\theta_i \;=\; \mathrm{base}^{-2i/d}.
$$

LLaMA-2 和 7B 级 LLaMA-3 用 $\mathrm{base} = 10000$，长上下文变体用
$\mathrm{base} = 500000$。$\theta_i$ 在 $i$ 上是几何级数：

| $i$ | $\theta_i$（base=10000, $d=128$） |
|-----|------------------------------------|
| 0   | $1.0$ rad/sample                   |
| 16  | $0.1$ rad/sample                   |
| 32  | $0.01$ rad/sample                  |
| 48  | $0.001$ rad/sample                 |
| 63  | $1.15 \times 10^{-4}$ rad/sample   |

最快的对每个 token 转约 $1$ rad，最慢的转约 $10^{-4}$ rad。跨 pair 角度跨 4 个
数量级。

## 5. `dft_rope_aware_compress` 用的核心恒等式

对 $c_i(t)$ apply RoPE，做长度 $N$ 的 DFT：

$$
\begin{aligned}
C_i^{\mathrm{RoPE}}[k]
  &= \sum_{t=0}^{N-1} c_i(t) \, e^{j \theta_i t} \, e^{-j 2\pi k t / N} \\
  &= \sum_{t=0}^{N-1} c_i(t) \, e^{-j \left(\tfrac{2\pi k}{N} - \theta_i\right) t} \\
  &= C_i\!\left[k - n_i\right], \qquad
     n_i \;=\; \mathrm{round}\!\left(\frac{\theta_i N}{2\pi}\right) \bmod N.
\end{aligned}
$$

**结论**：apply 到 $c_i(t)$ 上的 RoPE 把那一对 pair 的 DFT 平移 $n_i$ 个 bin。
$n_i$ 由 RoPE base 和 $N$ 完全决定，与模型、与输入无关。

代入 $\mathrm{base} = 10000$、$d = 128$：

| $i$ | $\theta_i$              | $n_i$（$N=2048$）| $n_i$（$N=4096$）|
|-----|--------------------------|------------------|------------------|
| 0   | $1.0$                    | 326              | 652              |
| 16  | $0.1$                    | 33               | 65               |
| 32  | $0.01$                   | 3                | 7                |
| 48  | $0.001$                  | 0                | 1                |
| 63  | $1.15 \times 10^{-4}$    | 0                | 0                |

即使 $N = 4096$，post-RoPE 能量也只在一个相当窄的包络里：$n_0 = 652$ 是
最大的，$i \gtrsim 32$ 之后基本回到 0。LLaMA-2-7B 上的实证刚好印证：pair 0
峰在 bin 652，pair 32 峰约 bin 7，pair 63 不动。

FreqKV 的统一低通保留 $[0, L)$（$L = \gamma N$）。FreqKV **漏掉** $n_0$
（即丢掉 pair 0 的 post-RoPE 能量中心）的条件是

$$
L < n_0
\;\iff\; \gamma < \frac{n_0}{N} = \frac{\theta_0}{2\pi} \approx 0.159.
$$

这是关键的实用界：$\gamma = 0.5$（FreqKV 默认）下，$[0, L)$ 远比 $n_0$ 大，
**FreqKV 并没有丢掉任何 pair 的能量中心**——相对 RoPE-matched bandpass 的
slack 只在"每个 channel 用同一个 $[0, L)$ 窗 vs 每对 pair 用各自不同的窗"
这个**形状自适应**层面（小 delta）。$\gamma \le 0.15$ 之后 FreqKV 开始丢
pair 0 的 post-RoPE 频带；$\gamma$ 再小，更多高 $\theta$ pair 跟着掉。
**这才是 DFT-RoPE bandpass 结构性地胜过 FreqKV 的区间**——和 FreqKV 论文
Table 3 在 $\gamma \to 0.01$ 处 PPL 爆炸的现象完全一致。

## 6. 为什么 FreqKV 仍然对，并且 slack 在哪

FreqKV cache 的是 **pre-RoPE** K。attention 时按"cache 内部位置"再 apply RoPE。
由调制定理，这等价于把 cache 的频谱按 pair 平移到各自的 $\theta_i$ 频带。所以
**最终** attention 看到的 post-RoPE K 语义自洽——FreqKV 不是丢错信息，而是用
"pre-RoPE 的统一低通"丢的。

slack 来自：pre-RoPE 低通丢的是 $[0, N) \setminus [0, L)$ 全部能量，里面包含
的短程位置高频细节经过 RoPE 之后落在像 bin 652 这样的位置。"post-RoPE 匹配带通"
在同样总预算下可以**按 pair 选不同的 pre-RoPE bin 子集**，更精细地保住
post-RoPE 的 attention 行为。

一句话：FreqKV 用对了 RoPE；`dft_rope` 的论点是它**没有按 RoPE 编 budget**。

## 7. 小波基础（给小波 compressor 看的）

离散小波变换（DWT）把 $x[0], \dots, x[N-1]$ 分解成

$$
x \;=\; \sum_{j, k} c_{j,k} \, \psi_{j,k}, \qquad
\psi_{j,k}(n) \;=\; 2^{j/2} \, \psi\!\left(2^j n - k\right),
$$

它是由单一母小波 $\psi$ 生成的尺度-平移族。**正交**族（Daubechies "db4"、
"db8"、"sym8"）的系数 $c_{j,k}$ 由共轭正交镜像滤波器对的 FIR 滤波级联产生；
$N$ 点 DWT 计算量 $\mathcal{O}(N)$。

和 KV 压缩相关的关键性质：

- **时频局部化**：每个 $\psi_{j,k}$ 时域有限支持，频域有限带宽（Heisenberg–
  Gabor trade-off）。信号里一个局部"针"事件只在恰当尺度上激活少数 $c_{j,k}$；
  阈值操作能廉价地留下。
- **多分辨**：尺度 $j = 0, 1, 2, \dots$ 对应越来越粗的视图。和 FreqKV 的直觉
  "近 token 细、远 token 粗"对上。
- **硬阈值**：按幅度保留前 $K$ 个系数，其余置零。重建是 $L^2$ 意义下到保留基
  的最佳投影，是该基上的最优稀疏近似。
- **边界处理**：边界 padding（"symmetric" 或 "periodization"）会引入小伪影；
  KV 里我们保留 sink token 不动，相当于一种带 anchor 的边界处理器。

当前 `wavelet_adaptive_compress` 的逻辑：

1. 沿序列轴 PyWavelets DWT。
2. 硬阈值，每个 channel 保留 $L \cdot d_\mathrm{head}$ 个系数（$L$ 是 FreqKV
   风格的目标长度）。
3. 逆 DWT。
4. 截到长度 $L$，幅度乘 $\sqrt{L / N}$ 与 FreqKV 的 IDCT 约定对齐。

"硬截到长度 $L$"是为了贴合 FreqKV 等长 cache 接口的妥协。原生小波 cache 应该
直接存稀疏系数集——这是更大的接口重构（要改 cache 数据结构），留到后续。

## 8. RST-KV：bulk + 稀疏残差分解的数学形式

本节给出本仓库提出的方法 **RST-KV（RoPE-Spectral Transform coding with sparse
residual）** 的严格形式。它把 §5 的"每对 pair 的 RoPE 平移"升级成一个完整的
率失真编码。

### 8.1 记号

固定一个 head，序列长 $N$、head 维 $d$。pre-RoPE key $k_t \in \mathbb{R}^d$，
按 §4 配对成复序列 $c_i(t) = k_{2i}(t) + j\,k_{2i+1}(t)$，$i = 0, \dots, d/2-1$。
post-RoPE 复序列与其序列轴 DFT 记作

$$
\tilde{c}_i(t) = c_i(t)\, e^{j \theta_i t}, \qquad
\widetilde{C}_i[\omega] = C_i[\omega - n_i], \qquad
n_i = \mathrm{round}\!\left(\frac{\theta_i N}{2\pi}\right) \bmod N .
$$

把 post-RoPE 的实数 key 张量记作 $\widetilde{K} \in \mathbb{R}^{N \times d}$。

### 8.2 核心分解

$$
\boxed{\;\widetilde{K} \;=\; \underbrace{B}_{\text{spectral bulk}} \;+\; \underbrace{R}_{\text{sparse residual}}, \qquad
B = \mathcal{P}_{\mathcal{B}}\,\widetilde{K}, \quad R = (\mathcal{I} - \mathcal{P}_{\mathcal{B}})\,\widetilde{K}\;}
$$

其中 $\mathcal{P}_{\mathcal{B}}$ 是 §8.3 的 RoPE 匹配带通**正交投影**。编码后重建为

$$
\widehat{K} \;=\; B \;+\; \widehat{R}, \qquad \widehat{R} = \mathcal{T}_S(R),
$$

$\mathcal{T}_S$ 为 §8.4 的稀疏硬阈值算子。这与 robust PCA 的 low-rank + sparse
分解同构，只是发生在 RoPE 频域：$B$ 是位置可预测的结构主体，$\widehat{R}$ 是局
部事件。

### 8.3 bulk：每对 pair 的 RoPE 匹配带通

第 $i$ 对以 $n_i$ 为中心、宽 $L_i$ 的 bin 集合：

$$
\mathcal{B}_i = \Big\{\,(n_i + \delta) \bmod N \;:\; \delta = -\lfloor L_i/2 \rfloor, \dots, \lceil L_i/2 \rceil - 1 \,\Big\}, \qquad |\mathcal{B}_i| = L_i .
$$

带通（频域置零带外）+ IDFT 重建：

$$
\widehat{C}_i^{\,\mathrm{bulk}}[\omega] = \widetilde{C}_i[\omega]\,\mathbf{1}[\omega \in \mathcal{B}_i], \qquad
b_i(t) = \frac{1}{N} \sum_{\omega \in \mathcal{B}_i} \widetilde{C}_i[\omega]\, e^{+j 2\pi \omega t / N} .
$$

因为带通 = 在正交 Fourier 基上选子集，$\mathcal{P}_{\mathcal{B}}$ 是正交投影：
$\mathcal{P}_{\mathcal{B}}^2 = \mathcal{P}_{\mathcal{B}} = \mathcal{P}_{\mathcal{B}}^{*}$。

- 对 $V$（无 RoPE）：令 $n_i \equiv 0$，带通退化为普通低通（代码 `is_key=False`）。
- **FreqKV 是特例**：所有对共用以 $0$ 为中心的同一窗，即
  $n_i \equiv 0,\ L_i \equiv L = \gamma N$。

### 8.4 residual：稀疏编码

残差 $R = (\mathcal{I} - \mathcal{P}_{\mathcal{B}})\,\widetilde{K}$。对第 $i$ 对的
残差序列 $r_i(t)$（时域）或其小波系数 $w_i = \mathrm{DWT}(r_i)$，取幅度最大的
$S_i$ 项：

$$
\Omega_i = \operatorname*{arg\,top\text{-}S_i}_{t} \, |r_i(t)|, \qquad
\widehat{r}_i(t) =
\begin{cases}
r_i(t), & t \in \Omega_i \\
0, & \text{otherwise.}
\end{cases}
$$

这就是 $\mathcal{T}_S$（代码 `residual_domain="time"` 用时域 top-$S$，`="wavelet"`
用小波域 top-$S$）。一个 needle / 代码符号在时域是单点尖峰，极少的 $S_i$ 即可精
确补回——这正是带通主体漏掉、而 FreqKV 永远丢掉的局部信息。

### 8.5 失真分解（正交性 ⇒ 勾股）

因 $\mathcal{P}_{\mathcal{B}}$ 正交，$B \perp R$，于是

$$
\|\widetilde{K}\|^2 = \|B\|^2 + \|R\|^2,
$$

且重建失真坍缩为**只与被丢弃的残差项有关**：

$$
\boxed{\;\big\|\widetilde{K} - \widehat{K}\big\|^2 = \big\|R - \widehat{R}\big\|^2 = \sum_i \sum_{t \notin \Omega_i} |r_i(t)|^2\;}
$$

即总失真 = 带外能量中没被稀疏残差捞回来的尾部。带通廉价吃掉集中在 $n_i$ 附近的结
构能量，残差精确吃掉少数大的离群项，两者支撑不重叠——这是"互补分量"而非"方法
杂糅"的数学依据。

### 8.6 率失真预算分配

总预算为每通道保留 $\gamma N$ 个实标量。预算对齐说明：$1$ 个复 bin $= 2$ 实数
$=$ 覆盖一对的 $2$ 个通道 $= 1$ 实数 / 通道，故与 DCT 的 $\gamma$ 可直接比较。
预算两分：

$$
\underbrace{\alpha \gamma N}_{\text{bulk: } \sum_i L_i / (d/2)} \;+\; \underbrace{(1-\alpha)\gamma N}_{\text{residual: } \sum_i S_i / (d/2)} \;=\; \gamma N .
$$

**(a) bulk 内部的 per-pair 带宽分配（注水）。** 给定 bulk 预算
$M = \alpha \gamma N \cdot \tfrac{d}{2}$ 个 bin，最大化保留能量

$$
\max_{\{L_i\}} \ \sum_i \sum_{\omega \in \mathcal{B}_i(L_i)} \big|\widetilde{C}_i[\omega]\big|^2
\quad \text{s.t.} \quad \sum_i L_i = M .
$$

其 KKT 解是一个**统一功率水位 $\lambda$**：

$$
\boxed{\;\text{保留 bin } (i,\omega) \iff \big|\widetilde{C}_i[\omega]\big|^2 \ge \lambda, \qquad
L_i(\lambda) = \#\{\omega : |\widetilde{C}_i[\omega]|^2 \ge \lambda\}\;}
$$

$\lambda$ 由 $\sum_i L_i(\lambda) = M$ 定出。能量集中（大 $\theta_i$）的对自动分到
更多带宽。FreqKV 的均匀 $L_i \equiv L$ 是 $\lambda$ 取常数的次优特例。代码
`water_fill_allocation` 在"峰关于 $n_i$ 单峰"假设下用边际增益贪心实现此解。

**(b) bulk / residual 的劈分 $\alpha$。** 在总预算 $\gamma$ 下选

$$
\alpha^\star(\gamma) = \arg\min_{\alpha \in [0,1]} \ \big\| \widetilde{K} - \widehat{K}_\alpha \big\|^2 .
$$

预期：$\gamma$ 大 $\Rightarrow \alpha^\star \to 1$（带通够用）；$\gamma$ 小
$\Rightarrow \alpha^\star$ 下降（残差变关键）。$\alpha^\star(\gamma)$ 曲线由
`rate_distortion.py` 的 $\alpha$ 扫描直接产出。

### 8.7 真正的目标：注意力输出失真 + Parseval 压缩域

最终影响模型的是注意力输出，故真正最小化的是

$$
D \;=\; \frac{\big\| \operatorname{softmax}(q K^\top / \sqrt{d})\,V - \operatorname{softmax}(q \widehat{K}^\top / \sqrt{d})\,\widehat{V} \big\|}{\big\| \operatorname{softmax}(q K^\top / \sqrt{d})\,V \big\|},
$$

$K$ 重建误差是其可计算代理。由 Parseval（§1），内积可在频域算：

$$
\langle q_t, \tilde{k}_s \rangle = \frac{1}{N} \sum_i \sum_{\omega} Q_i[\omega]\, \widetilde{C}_i^{*}[\omega] .
$$

若把 $Q, K$ 都限制在保留的 $L$ 个 bin 上，注意力打分只需在 $L$ 个 bin 上做；$V$
侧的 IDFT 可离线融进 $W_o$，解码期免在线 RoPE GEMM 与 IDFT——这是只有 DFT 路径
有、小波路径给不了的系统收益。

### 8.8 与代码的对应

| 公式 | 代码 |
|------|------|
| 调制定理 $n_i = \mathrm{round}(\theta_i N / 2\pi)$ | `rope_utils.thetas_to_bin_offsets` |
| 带通投影 $\mathcal{P}_{\mathcal{B}}$ + IDFT | `rdcodecs.dft_rope_keep_reconstruct` |
| 分解 $\widehat{K} = B + \mathcal{T}_S(R)$ | `rdcodecs.rst_keep_reconstruct` |
| 硬阈值 $\mathcal{T}_S$ | `rdcodecs._topk_keep_lastdim` |
| 注水水位 $\lambda$ | `rdcodecs.water_fill_allocation` + `pair_energy_curves` |
| 注意力失真 $D$ | `rdcodecs.causal_attention_output` |
| 定长接口（训练 E4）| `transforms/rst_hybrid.rst_compress` |

**一句话总括**：RST-KV 把 post-RoPE 的 $\widetilde{K}$ 做正交分解
$\widetilde{K} = \mathcal{P}_{\mathcal{B}}\widetilde{K} + (\mathcal{I} - \mathcal{P}_{\mathcal{B}})\widetilde{K}$，
前者用以 $n_i$ 为中心、带宽由注水水位 $\lambda$ 决定的 per-pair 带通编码，后者用
硬阈值 $\mathcal{T}_S$ 稀疏编码，劈分比 $\alpha$ 由率失真最优化决定。FreqKV 是
$n_i \equiv 0,\ L_i \equiv L,\ \alpha \equiv 1$ 的退化特例。

## 9. 仓库里的数值约定

- 所有变换内部用 **float32** 计算，末了再 cast 回输入 dtype——和 FreqKV 的
  `dct` / `idct` 保持一致。
- DFT 用非酉约定 $X[k] = \sum_{n} x[n]\, e^{-j 2\pi k n / N}$（PyTorch 默认）；
  $\sqrt{L / N}$ 幅度重缩放是 FreqKV 约定，保证"带内已涵盖所有信号能量"时
  压缩前后**总能量**（不是 per-sample 能量）守恒。
- DCT 用 orthonormal `"ortho"` 归一化（PyTorch / SciPy 默认 `"ortho"`）；
  DCT-II 接 DCT-III 在同归一化下是恒等。
- 小波默认 PyWavelets `"symmetric"` 边界；需要时通过 `mode` 切到
  `"periodization"`。

## 10. 推荐阅读

- A. V. Oppenheim, R. W. Schafer. *Discrete-Time Signal Processing* (Pearson)。
  第 8 章（DFT）、9 章（FFT）、4 章（采样）。
- I. Daubechies. *Ten Lectures on Wavelets* (SIAM)。第 1、5、6 章。
- N. Ahmed, T. Natarajan, K. R. Rao. *Discrete Cosine Transform*
  (IEEE Trans. Comput., 1974). DCT-II 原始论文。
- J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*
  (arXiv 2104.09864). RoPE 定义。
