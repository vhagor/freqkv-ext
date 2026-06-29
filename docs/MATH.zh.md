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
| 16  | $0.316$ rad/sample                 |
| 32  | $0.10$ rad/sample                  |
| 48  | $0.0316$ rad/sample                |
| 63  | $1.0 \times 10^{-4}$ rad/sample    |

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

代入 $N = 2048$、$\mathrm{base} = 10000$、$d = 128$：

| $i$ | $\theta_i$           | $n_i$ |
|-----|----------------------|-------|
| 0   | $1.0$                | 326   |
| 16  | $0.316$              | 103   |
| 32  | $0.10$               | 33    |
| 48  | $0.0316$             | 10    |
| 63  | $1 \times 10^{-4}$   | 0     |

所以 $N = 2048$ 的 DFT 中，post-RoPE 能量"住"在 326（早 pair）到 0（晚 pair）
之间的 bin 上。FreqKV 的统一低通保留 $[0, L)$：$L = 1024$ 时它覆盖了
$i \ge 8$ 的 $n_i$；**对 $i = 0, \dots, 7$ 这八个最快 pair，post-RoPE 频带中心
已经在 $[0, L)$ 外**，FreqKV 把这些 channel 的 post-RoPE 相关内容当低频丢了
（**注意正确性不受影响**：FreqKV 保留这些 channel 的 pre-RoPE 能量，attention
时再 apply RoPE 仍然能复出正确语义；这里只是 budget 分配不最优，理论上
RoPE-matched bandpass 能在同样预算下做得更细——见 §6）。

## 6. 为什么 FreqKV 仍然对，并且 slack 在哪

FreqKV cache 的是 **pre-RoPE** K。attention 时按"cache 内部位置"再 apply RoPE。
由调制定理，这等价于把 cache 的频谱按 pair 平移到各自的 $\theta_i$ 频带。所以
**最终** attention 看到的 post-RoPE K 语义自洽——FreqKV 不是丢错信息，而是用
"pre-RoPE 的统一低通"丢的。

slack 来自：pre-RoPE 低通丢的是 $[0, N) \setminus [0, L)$ 全部能量，里面包含
的短程位置高频细节经过 RoPE 之后落在像 bin 326 这样的位置。"post-RoPE 匹配带通"
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

## 8. 仓库里的数值约定

- 所有变换内部用 **float32** 计算，末了再 cast 回输入 dtype——和 FreqKV 的
  `dct` / `idct` 保持一致。
- DFT 用非酉约定 $X[k] = \sum_{n} x[n]\, e^{-j 2\pi k n / N}$（PyTorch 默认）；
  $\sqrt{L / N}$ 幅度重缩放是 FreqKV 约定，保证"带内已涵盖所有信号能量"时
  压缩前后**总能量**（不是 per-sample 能量）守恒。
- DCT 用 orthonormal `"ortho"` 归一化（PyTorch / SciPy 默认 `"ortho"`）；
  DCT-II 接 DCT-III 在同归一化下是恒等。
- 小波默认 PyWavelets `"symmetric"` 边界；需要时通过 `mode` 切到
  `"periodization"`。

## 9. 推荐阅读

- A. V. Oppenheim, R. W. Schafer. *Discrete-Time Signal Processing* (Pearson)。
  第 8 章（DFT）、9 章（FFT）、4 章（采样）。
- I. Daubechies. *Ten Lectures on Wavelets* (SIAM)。第 1、5、6 章。
- N. Ahmed, T. Natarajan, K. R. Rao. *Discrete Cosine Transform*
  (IEEE Trans. Comput., 1974). DCT-II 原始论文。
- J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*
  (arXiv 2104.09864). RoPE 定义。
