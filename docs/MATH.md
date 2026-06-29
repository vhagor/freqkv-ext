# Mathematical foundations

This note collects the DSP identities used by `freqkv-ext`. It is meant as a
compact reference, not a textbook; for full derivations consult Oppenheim &
Schafer (signals) or Daubechies (wavelets).

## 1. Discrete Fourier transform (DFT)

For a finite complex sequence $x[0], \dots, x[N-1]$, the DFT and inverse are

$$
X[k] \;=\; \sum_{n=0}^{N-1} x[n] \, e^{-j 2\pi k n / N}, \qquad k = 0, 1, \dots, N-1,
$$

$$
x[n] \;=\; \frac{1}{N} \sum_{k=0}^{N-1} X[k] \, e^{+j 2\pi k n / N}.
$$

In matrix form $X = F_N x$. The unitary convention uses
$F_N = (1/\sqrt{N})\,[e^{-j 2\pi k n / N}]$; in NumPy / PyTorch the un-unitary
convention above is used and the $1/N$ lives in the inverse.

**Parseval's identity** (energy preservation):

$$
\sum_{n=0}^{N-1} \bigl|x[n]\bigr|^2 \;=\; \frac{1}{N} \sum_{k=0}^{N-1} \bigl|X[k]\bigr|^2.
$$

This is the formal justification for "spectrum-domain attention": the inner
product of two sequences equals (up to a $1/N$ factor) the inner product of
their DFTs.

## 2. The modulation theorem (the key identity for freqkv-ext)

If $y[n] = x[n] \, e^{j \omega_0 n}$ is a frequency-modulated version of $x[n]$, then

$$
\begin{aligned}
Y[k] &= \sum_{n=0}^{N-1} x[n] \, e^{j \omega_0 n} \, e^{-j 2\pi k n / N} \\
     &= \sum_{n=0}^{N-1} x[n] \, e^{-j \left(\tfrac{2\pi k}{N} - \omega_0\right) n} \\
     &= X\!\left[k - \frac{\omega_0 N}{2\pi}\right].
\end{aligned}
$$

In bin-index form: multiplication by $e^{j \omega_0 n}$ in time corresponds to
a **circular shift** of the spectrum by $\mathrm{round}\!\left(\omega_0 N / 2\pi\right)$
bins.

This identity is the structural justification for treating RoPE — which IS a
per-pair time-domain multiplication by $e^{j \theta_i t}$ — as a per-pair
DFT-bin shift.

## 3. DCT-II and its relation to KLT for AR(1)

The Type-II DCT of $x[0], \dots, x[N-1]$ is

$$
y[k] \;=\; \alpha_k \sum_{n=0}^{N-1} x[n] \cos\!\left(\frac{\pi k (2n+1)}{2N}\right),
$$

with $\alpha_0 = \sqrt{1/N}$ and $\alpha_k = \sqrt{2/N}$ for $k > 0$. It is
the DFT of the even-symmetric mirror extension of $x$ (so there is no Gibbs
discontinuity at boundaries).

For a stationary AR(1) process

$$
x[n] \;=\; \rho\, x[n-1] + e[n], \qquad |\rho| < 1,
$$

the Karhunen-Loève transform (the orthogonal basis that diagonalizes the
covariance matrix) asymptotes to the DCT-II as $\rho \to 1$. This is the
classical reason JPEG / FreqKV pick DCT-II over DFT: for *real, smooth,
non-periodic* signals, DCT-II's energy compaction is provably better than DFT
in the $L^2$ sense.

LLaMA's pre-RoPE K states along the sequence axis are empirically close to
the $\rho \to 1$ regime (adjacent tokens are highly correlated), which is why
FreqKV's empirical Figure 1 shows pre-RoPE energy concentrated in low DCT
bins.

## 4. RoPE in pair-complex form

LLaMA's RoPE acts on each head's $d$-dim K (with $d$ even). Conceptually it
pairs adjacent hidden dims into a complex channel
$c_i = k_{2i} + j\, k_{2i+1}$ and rotates pair $i$ at position $t$ by

$$
c_i^{\mathrm{RoPE}}(t) \;=\; c_i(t) \, e^{j \theta_i t}, \qquad
\theta_i \;=\; \mathrm{base}^{-2i/d},
$$

with $\mathrm{base} = 10000$ for LLaMA-2 and 7B-class LLaMA-3 (long-context
variants use $\mathrm{base} = 500000$). The angles form a geometric series
in $i$:

| $i$ | $\theta_i$ (base=10000, $d=128$) |
|-----|-----------------------------------|
| 0   | $1.0$ rad/sample                  |
| 16  | $0.316$ rad/sample                |
| 32  | $0.10$ rad/sample                 |
| 48  | $0.0316$ rad/sample               |
| 63  | $1.0 \times 10^{-4}$ rad/sample   |

The "fastest" pair rotates by about $1$ rad per token, the "slowest" by
about $10^{-4}$ rad per token. Across pairs the angles cover four orders of
magnitude.

## 5. The central identity used by `dft_rope_aware_compress`

Apply RoPE to $c_i(t)$ and take the length-$N$ DFT:

$$
\begin{aligned}
C_i^{\mathrm{RoPE}}[k]
  &= \sum_{t=0}^{N-1} c_i(t) \, e^{j \theta_i t} \, e^{-j 2\pi k t / N} \\
  &= \sum_{t=0}^{N-1} c_i(t) \, e^{-j \left(\tfrac{2\pi k}{N} - \theta_i\right) t} \\
  &= C_i\!\left[k - n_i\right], \qquad
     n_i \;=\; \mathrm{round}\!\left(\frac{\theta_i N}{2\pi}\right) \bmod N.
\end{aligned}
$$

**Take-away**: RoPE applied to $c_i(t)$ shifts that pair's DFT by $n_i$ bins.
$n_i$ is hardwired by RoPE base and $N$; it does not depend on the model or
the input.

Plugging in $N = 2048$, $\mathrm{base} = 10000$, $d = 128$:

| $i$ | $\theta_i$            | $n_i$ |
|-----|------------------------|-------|
| 0   | $1.0$                 | 326   |
| 16  | $0.316$               | 103   |
| 32  | $0.10$                | 33    |
| 48  | $0.0316$              | 10    |
| 63  | $1 \times 10^{-4}$    | 0     |

So in a 2048-bin DFT, post-RoPE energy "lives" in bins ranging from 326
(early pairs) down to 0 (late pairs). FreqKV's uniform low-pass keeps bins
$[0, L)$. For $L = 1024$ it covers $n_i$ for $i \ge 8$; **for the eight
fastest pairs ($i = 0, \dots, 7$), the post-RoPE band center is outside
$[0, L)$**, and FreqKV is throwing away the post-RoPE-relevant content of
those channels (it *does* preserve the pre-RoPE energy of those channels,
which after RoPE applied at attention time still reconstructs the right
answer, so this is not a correctness violation; it is, however, a place
where a smarter, RoPE-matched band could in principle do better — see §6).

## 6. Why FreqKV is still correct, and where the slack is

FreqKV caches **pre-RoPE** K. At attention time it applies RoPE to the cached
pre-RoPE K with positions allocated *within the cache*. By the modulation
theorem this is equivalent to taking the cached spectrum and
frequency-shifting each pair to its $\theta_i$ band. So the **final**
post-RoPE K, viewed in attention, is consistent with the original semantics
— FreqKV is not throwing away information through misalignment; it is
throwing away information through **uniform low-pass on the pre-RoPE
spectrum**.

The slack exists because the "pre-RoPE low-pass" loses *every* energy
component outside $[0, L)$ (in pre-RoPE), and that energy includes the
high-frequency short-range positional detail that, **after** RoPE, ends up at
bins like 326. A "post-RoPE matched bandpass" can keep different (smaller)
sets of pre-RoPE bins per pair, chosen to preserve the post-RoPE attention
behavior more faithfully under the same total budget.

In short: FreqKV correctly uses RoPE; `dft_rope_aware` argues it doesn't
**budget for RoPE**.

## 7. Wavelet basics (for the wavelet compressor)

The discrete wavelet transform (DWT) decomposes $x[0], \dots, x[N-1]$ into

$$
x \;=\; \sum_{j,k} c_{j,k} \, \psi_{j,k}, \qquad
\psi_{j,k}(n) \;=\; 2^{j/2} \, \psi\!\left(2^j n - k\right),
$$

a scale-translation family generated by a single mother wavelet $\psi$. For
an *orthogonal* family (Daubechies "db4", "db8", "sym8"), the $c_{j,k}$ are
produced by a cascade of FIR filter banks with conjugate quadrature mirror
filters; an $N$-point DWT costs $\mathcal{O}(N)$ operations.

Key properties relevant to KV compression:

- **Time-frequency localization**: each $\psi_{j,k}$ has finite support in
  time AND finite bandwidth in frequency (Heisenberg–Gabor tradeoff). A
  localized "needle" in the signal puts energy into a small number of
  $c_{j,k}$ at the appropriate scale; thresholding keeps it cheaply.
- **Multiresolution**: scales $j = 0, 1, 2, \dots$ correspond to coarser
  and coarser views. Aligned with FreqKV's intuition that "near tokens are
  fine, far tokens are coarse".
- **Hard thresholding**: keep the top $K$ coefficients by magnitude, set the
  rest to zero. The reconstructed signal is the $L^2$ projection onto the
  kept basis, which is the optimal sparse approximation in that basis.
- **Boundary handling**: signal padding ("symmetric" or "periodization") at
  the edges introduces small artifacts; for KV with sink tokens we keep the
  sink unmolested as a wavelet boundary handler.

The current `wavelet_adaptive_compress` implements:

1. DWT($x$ along seq axis) using PyWavelets.
2. Hard-threshold to keep $L \cdot d_\mathrm{head}$ coefficients per channel
   (where $L$ is the FreqKV-style target length).
3. Inverse DWT.
4. Truncate to length $L$ and amplitude-rescale by $\sqrt{L / N}$ for
   amplitude parity with FreqKV's IDCT convention.

The "truncate to length $L$" is a forcing compromise to fit FreqKV's
fixed-length cache interface. A principled wavelet cache would store the
sparse coefficient set directly; that is a larger refactor (different cache
data structure) reserved for later.

## 8. Numerical conventions in this repo

- All transforms execute in **float32** internally and cast back to the
  input dtype at the end, matching FreqKV's `dct` / `idct` implementations.
- DFT uses the un-unitary convention
  $X[k] = \sum_{n} x[n]\, e^{-j 2\pi k n / N}$ (PyTorch default); the
  $\sqrt{L/N}$ amplitude rescale is FreqKV's convention, preserving **total
  energy** (NOT per-sample energy) under uniform low-pass / band selection
  of bins that contain all the signal energy.
- DCT uses the orthonormal "ortho" normalization (PyTorch / SciPy default
  for `"ortho"`); DCT-II followed by DCT-III with the same normalization is
  the identity.
- Wavelet uses PyWavelets "symmetric" boundary by default; "periodization"
  is available via `mode` kwarg if needed.

## 9. Suggested reading

- A. V. Oppenheim, R. W. Schafer. *Discrete-Time Signal Processing*
  (Pearson). Ch. 8 (DFT), 9 (FFT), 4 (sampling).
- I. Daubechies. *Ten Lectures on Wavelets* (SIAM). Ch. 1, 5, 6.
- N. Ahmed, T. Natarajan, K. R. Rao. *Discrete Cosine Transform* (IEEE
  Trans. Comput., 1974). The original DCT-II paper.
- J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position
  Embedding* (arXiv 2104.09864). The RoPE definition.
