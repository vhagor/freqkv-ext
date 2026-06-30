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
| 16  | $0.1$ rad/sample                  |
| 32  | $0.01$ rad/sample                 |
| 48  | $0.001$ rad/sample                |
| 63  | $1.15 \times 10^{-4}$ rad/sample  |

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

Plugging in $\mathrm{base} = 10000$, $d = 128$:

| $i$ | $\theta_i$              | $n_i$ ($N=2048$) | $n_i$ ($N=4096$) |
|-----|--------------------------|------------------|------------------|
| 0   | $1.0$                    | 326              | 652              |
| 16  | $0.1$                    | 33               | 65               |
| 32  | $0.01$                   | 3                | 7                |
| 48  | $0.001$                  | 0                | 1                |
| 63  | $1.15 \times 10^{-4}$    | 0                | 0                |

So even in a 4096-bin DFT, post-RoPE energy stays in a fairly narrow envelope:
$n_0 = 652$ at the high end, falling to 0 quickly for $i \gtrsim 32$. The
empirical confirmation on LLaMA-2-7B is exactly this: pair 0 peaks at
bin 652, pair 32 peaks near bin 7, pair 63 stays at bin 0.

FreqKV's uniform low-pass keeps bins $[0, L)$ with $L = \gamma N$. The
condition under which FreqKV's band **misses** $n_0$ (and thus loses pair 0's
post-RoPE energy center) is

$$
L < n_0
\;\iff\; \gamma < \frac{n_0}{N} = \frac{\theta_0}{2\pi} \approx 0.159.
$$

This is the key practical bound: at $\gamma = 0.5$ (FreqKV default), the band
$[0, L)$ comfortably contains every $n_i$ and **FreqKV is not throwing away
any pair's center** — the slack relative to a RoPE-matched bandpass is only
in *per-channel band shape* (FreqKV uses a single $[0, L)$ window for every
pair; bandpass uses different windows). At $\gamma \le 0.15$, FreqKV begins
to lose pair 0's post-RoPE band entirely; below that, more high-$\theta$
pairs follow. **This is the regime where DFT-RoPE bandpass is structurally
expected to outperform FreqKV** — and it lines up with FreqKV's own Table 3
showing PPL explosion at $\gamma \to 0.01$.

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
bins like 652. A "post-RoPE matched bandpass" can keep different (smaller)
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

## 8. RST-KV: math of the bulk + sparse-residual decomposition

This section gives the rigorous form of the method proposed in this repo,
**RST-KV (RoPE-Spectral Transform coding with sparse residual)**. It upgrades
the "per-pair RoPE shift" of §5 into a complete rate-distortion code.

### 8.1 Notation

Fix a head; sequence length $N$, head dim $d$. The pre-RoPE key
$k_t \in \mathbb{R}^d$ is paired (§4) into complex sequences
$c_i(t) = k_{2i}(t) + j\,k_{2i+1}(t)$, $i = 0, \dots, d/2-1$. The post-RoPE
complex sequence and its sequence-axis DFT are

$$
\tilde{c}_i(t) = c_i(t)\, e^{j \theta_i t}, \qquad
\widetilde{C}_i[\omega] = C_i[\omega - n_i], \qquad
n_i = \mathrm{round}\!\left(\frac{\theta_i N}{2\pi}\right) \bmod N .
$$

Write the post-RoPE real key tensor as $\widetilde{K} \in \mathbb{R}^{N \times d}$.

### 8.2 The core decomposition

$$
\boxed{\;\widetilde{K} \;=\; \underbrace{B}_{\text{spectral bulk}} \;+\; \underbrace{R}_{\text{sparse residual}}, \qquad
B = \mathcal{P}_{\mathcal{B}}\,\widetilde{K}, \quad R = (\mathcal{I} - \mathcal{P}_{\mathcal{B}})\,\widetilde{K}\;}
$$

where $\mathcal{P}_{\mathcal{B}}$ is the RoPE-matched bandpass **orthogonal
projection** of §8.3. The encoded reconstruction is

$$
\widehat{K} \;=\; B \;+\; \widehat{R}, \qquad \widehat{R} = \mathcal{T}_S(R),
$$

with $\mathcal{T}_S$ the sparse hard-threshold operator of §8.4. This is
isomorphic to robust PCA's low-rank + sparse decomposition, but in the RoPE
frequency domain: $B$ is the position-predictable structured bulk,
$\widehat{R}$ captures localized events.

### 8.3 Bulk: per-pair RoPE-matched bandpass

For pair $i$, the band of $L_i$ bins centered at $n_i$:

$$
\mathcal{B}_i = \Big\{\,(n_i + \delta) \bmod N \;:\; \delta = -\lfloor L_i/2 \rfloor, \dots, \lceil L_i/2 \rceil - 1 \,\Big\}, \qquad |\mathcal{B}_i| = L_i .
$$

Bandpass (zero out-of-band) + IDFT reconstruction:

$$
\widehat{C}_i^{\,\mathrm{bulk}}[\omega] = \widetilde{C}_i[\omega]\,\mathbf{1}[\omega \in \mathcal{B}_i], \qquad
b_i(t) = \frac{1}{N} \sum_{\omega \in \mathcal{B}_i} \widetilde{C}_i[\omega]\, e^{+j 2\pi \omega t / N} .
$$

Since bandpass = selecting a subset of an orthonormal Fourier basis,
$\mathcal{P}_{\mathcal{B}}$ is an orthogonal projection:
$\mathcal{P}_{\mathcal{B}}^2 = \mathcal{P}_{\mathcal{B}} = \mathcal{P}_{\mathcal{B}}^{*}$.

- For $V$ (no RoPE): set $n_i \equiv 0$, bandpass degenerates to a plain
  low-pass (code path `is_key=False`).
- **FreqKV is the special case**: every pair shares one window centered at
  $0$, i.e. $n_i \equiv 0,\ L_i \equiv L = \gamma N$.

### 8.4 Residual: sparse coding

Residual $R = (\mathcal{I} - \mathcal{P}_{\mathcal{B}})\,\widetilde{K}$. For
pair $i$'s residual series $r_i(t)$ (time domain) or its wavelet coefficients
$w_i = \mathrm{DWT}(r_i)$, keep the $S_i$ largest-magnitude entries:

$$
\Omega_i = \operatorname*{arg\,top\text{-}S_i}_{t} \, |r_i(t)|, \qquad
\widehat{r}_i(t) =
\begin{cases}
r_i(t), & t \in \Omega_i \\
0, & \text{otherwise.}
\end{cases}
$$

This is $\mathcal{T}_S$ (`residual_domain="time"` uses time-domain top-$S$,
`="wavelet"` uses wavelet-domain top-$S$). A needle / code symbol is a single
time-domain spike, so a tiny $S_i$ restores it exactly — precisely the local
information the smooth bulk misses and FreqKV discards forever.

### 8.5 Distortion decomposition (orthogonality ⇒ Pythagoras)

Since $\mathcal{P}_{\mathcal{B}}$ is orthogonal, $B \perp R$, hence

$$
\|\widetilde{K}\|^2 = \|B\|^2 + \|R\|^2,
$$

and the reconstruction distortion collapses to **only the discarded residual
entries**:

$$
\boxed{\;\big\|\widetilde{K} - \widehat{K}\big\|^2 = \big\|R - \widehat{R}\big\|^2 = \sum_i \sum_{t \notin \Omega_i} |r_i(t)|^2\;}
$$

Total distortion = the tail of out-of-band energy not recovered by the sparse
residual. The bandpass cheaply absorbs the structured energy concentrated
near $n_i$; the residual exactly absorbs the few large outliers; their
supports do not overlap — the mathematical reason this is "complementary
components" rather than "a mashup of methods".

### 8.6 Rate-distortion budget allocation

Budget = $\gamma N$ retained real scalars per channel. Budget parity: $1$
complex bin $= 2$ reals $=$ covers the $2$ channels of a pair $= 1$ real /
channel, so it is directly comparable to DCT's $\gamma$. Two-way split:

$$
\underbrace{\alpha \gamma N}_{\text{bulk: } \sum_i L_i / (d/2)} \;+\; \underbrace{(1-\alpha)\gamma N}_{\text{residual: } \sum_i S_i / (d/2)} \;=\; \gamma N .
$$

**(a) Per-pair bandwidth inside the bulk (water-filling).** Given a bulk
budget of $M = \alpha \gamma N \cdot \tfrac{d}{2}$ bins, maximize retained
energy

$$
\max_{\{L_i\}} \ \sum_i \sum_{\omega \in \mathcal{B}_i(L_i)} \big|\widetilde{C}_i[\omega]\big|^2
\quad \text{s.t.} \quad \sum_i L_i = M .
$$

Its KKT solution is a **single power level $\lambda$ (water level)**:

$$
\boxed{\;\text{keep bin } (i,\omega) \iff \big|\widetilde{C}_i[\omega]\big|^2 \ge \lambda, \qquad
L_i(\lambda) = \#\{\omega : |\widetilde{C}_i[\omega]|^2 \ge \lambda\}\;}
$$

with $\lambda$ chosen so $\sum_i L_i(\lambda) = M$. Energy-concentrated pairs
(large $\theta_i$) automatically get more bandwidth. FreqKV's uniform
$L_i \equiv L$ is the suboptimal special case of constant $\lambda$. Code
`water_fill_allocation` implements this via greedy marginal gain under the
"unimodal around $n_i$" assumption.

**(b) The bulk / residual split $\alpha$.** Under total budget $\gamma$,

$$
\alpha^\star(\gamma) = \arg\min_{\alpha \in [0,1]} \ \big\| \widetilde{K} - \widehat{K}_\alpha \big\|^2 .
$$

Expected trend: large $\gamma \Rightarrow \alpha^\star \to 1$ (bandpass
suffices); small $\gamma \Rightarrow \alpha^\star$ drops (residual matters).
The $\alpha^\star(\gamma)$ curve is produced directly by the $\alpha$ sweep in
`rate_distortion.py`.

### 8.7 The real objective: attention-output distortion + Parseval

What ultimately matters is the attention output, so the true objective is

$$
D \;=\; \frac{\big\| \operatorname{softmax}(q K^\top / \sqrt{d})\,V - \operatorname{softmax}(q \widehat{K}^\top / \sqrt{d})\,\widehat{V} \big\|}{\big\| \operatorname{softmax}(q K^\top / \sqrt{d})\,V \big\|},
$$

with $K$ reconstruction error as its tractable surrogate. By Parseval (§1),
the inner product can be computed in frequency:

$$
\langle q_t, \tilde{k}_s \rangle = \frac{1}{N} \sum_i \sum_{\omega} Q_i[\omega]\, \widetilde{C}_i^{*}[\omega] .
$$

Restricting $Q, K$ to the retained $L$ bins makes the attention score an
$L$-bin computation; the $V$-side IDFT folds offline into $W_o$, removing the
online RoPE GEMM and IDFT at decode — a systems gain unique to the DFT path
(the wavelet path cannot provide it).

### 8.8 Mapping to code

| Formula | Code |
|---------|------|
| Modulation theorem $n_i = \mathrm{round}(\theta_i N / 2\pi)$ | `rope_utils.thetas_to_bin_offsets` |
| Bandpass projection $\mathcal{P}_{\mathcal{B}}$ + IDFT | `rdcodecs.dft_rope_keep_reconstruct` |
| Decomposition $\widehat{K} = B + \mathcal{T}_S(R)$ | `rdcodecs.rst_keep_reconstruct` |
| Hard threshold $\mathcal{T}_S$ | `rdcodecs._topk_keep_lastdim` |
| Water level $\lambda$ | `rdcodecs.water_fill_allocation` + `pair_energy_curves` |
| Attention distortion $D$ | `rdcodecs.causal_attention_output` |
| Fixed-length interface (training E4) | `transforms/rst_hybrid.rst_compress` |

**In one sentence**: RST-KV orthogonally decomposes the post-RoPE
$\widetilde{K} = \mathcal{P}_{\mathcal{B}}\widetilde{K} + (\mathcal{I} - \mathcal{P}_{\mathcal{B}})\widetilde{K}$,
encodes the former with a per-pair bandpass centered at $n_i$ whose width is
set by the water level $\lambda$, encodes the latter with a hard threshold
$\mathcal{T}_S$, and sets the split $\alpha$ by rate-distortion optimization.
FreqKV is the degenerate case $n_i \equiv 0,\ L_i \equiv L,\ \alpha \equiv 1$.

## 9. Numerical conventions in this repo

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

## 10. Suggested reading

- A. V. Oppenheim, R. W. Schafer. *Discrete-Time Signal Processing*
  (Pearson). Ch. 8 (DFT), 9 (FFT), 4 (sampling).
- I. Daubechies. *Ten Lectures on Wavelets* (SIAM). Ch. 1, 5, 6.
- N. Ahmed, T. Natarajan, K. R. Rao. *Discrete Cosine Transform* (IEEE
  Trans. Comput., 1974). The original DCT-II paper.
- J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position
  Embedding* (arXiv 2104.09864). The RoPE definition.
