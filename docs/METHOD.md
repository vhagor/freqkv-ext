# Method: DFT-RoPE-aware and wavelet KV compressors

This document derives the DSP backing for the two new compressors and contrasts
them with FreqKV's DCT-II low-pass.

## 1. FreqKV in one paragraph

For a cache segment $K \in \mathbb{R}^{N \times d}$ (pre-RoPE), FreqKV applies
DCT-II along the sequence axis, retains the lowest $L = \gamma N$
coefficients, runs IDCT, and rescales by $\sqrt{L / N}$. The compressed cache
is written back as pre-RoPE K; RoPE is applied on the fly at attention time
using positions in $[0, S + L + r_\mathrm{recent})$ *within* the cache (not the
original sequence). This "position re-anchoring" is what lets FreqKV extend
the context window without out-of-bound position embeddings.

Why DCT and not DFT? For an AR(1) source with correlation $\rho \to 1$, the
Karhunen-Loève transform asymptotes to DCT-II. LLaMA's KV states along the
sequence axis are empirically close to that regime (FreqKV Figure 1). DCT-II
is real, has implicit even-symmetric boundary (no Gibbs leakage), and
concentrates energy well.

## 2. RoPE in the frequency domain

LLaMA's RoPE pairs adjacent hidden dimensions $(d_{2i}, d_{2i+1})$ and
rotates pair $i$ by angle $\theta_i t$ at sequence position $t$, where

$$
\theta_i \;=\; \mathrm{base}^{-2i/d}, \qquad i = 0, 1, \dots, d/2 - 1.
$$

In the pair-complex view $c_i(t) = k_{2i}(t) + j\, k_{2i+1}(t)$, RoPE becomes

$$
c_i^{\mathrm{RoPE}}(t) \;=\; c_i(t) \, e^{j \theta_i t}.
$$

Taking the sequence-axis DFT and applying the modulation theorem:

$$
\begin{aligned}
C_i^{\mathrm{RoPE}}[\omega]
  &= \sum_{t} c_i^{\mathrm{RoPE}}(t) \, e^{-j \omega t} \\
  &= \sum_{t} c_i(t) \, e^{-j (\omega - \theta_i) t} \\
  &= C_i\!\left[\omega - \theta_i\right].
\end{aligned}
$$

**RoPE is a per-pair frequency shift of the sequence-axis spectrum.** For a
length-$N$ DFT, $\theta_i$ corresponds to bin offset

$$
n_i \;=\; \mathrm{round}\!\left(\frac{\theta_i N}{2\pi}\right) \bmod N.
$$

## 3. Consequences for compression

### 3.1 Pre-RoPE vs post-RoPE caching

FreqKV caches **pre-RoPE** K because the DCT basis has no phase dimension to
carry the position information that RoPE would imprint. Caching post-RoPE
under DCT bakes in original positions, which become invalid after iterative
compression (we'd be using out-of-bound RoPE positions at attention time).

DFT preserves phase, so the position info encoded by RoPE survives the
transform. By the shift theorem, *any* desired position assignment for the
compressed tokens can be realized by a single linear phase multiplier on the
retained spectrum. Pre- and post-RoPE caches are inter-convertible at
$\mathcal{O}(d/2)$ complex multiplies per retained bin.

### 3.2 Where the relevant information lives

Pre-RoPE K's spectrum (FreqKV's empirical Figure 1) concentrates at low bins
across all pairs. Post-RoPE, the modulation theorem moves pair $i$'s mass to
bins around $n_i$. For LLaMA-2 ($d_\mathrm{head} = 128$,
$\mathrm{base} = 10000$):

- $\theta_0 = 1$ rad/sample. With $N = 2048$, $n_0 \approx 326$.
- $\theta_{d/4} \approx 0.1$ rad/sample, so $n_{d/4} \approx 33$.
- $\theta_{d/2 - 1} \approx 1/10000$ rad/sample, so $n_{d/2 - 1} \approx 0$.

A uniform low-pass that keeps bins $[0, L)$ is RoPE-aware only for the
high-index pairs (small $\theta$). For the early-index pairs (large
$\theta$), it discards the band where the post-RoPE energy actually lives.
This is our candidate explanation for FreqKV's empirical weakness on tasks
that depend on short-range positional detail (Needle-in-a-Haystack at the
original window boundary; numerical / code retrieval).

### 3.3 RoPE-matched bandpass

Algorithm `dft_rope_aware_compress(x, L)`:

1. Reshape $x \in \mathbb{R}^{B \times H \times N \times d}$ into a
   pair-complex tensor $c \in \mathbb{C}^{B \times H \times N \times (d/2)}$
   and DFT along the sequence axis.
2. For each pair $i$, gather $L$ bins centered at $n_i$ (circular indexing:
   $(n_i - L/2 + k) \bmod N$ for $k = 0, \dots, L-1$).
3. Demodulate each pair's retained band by $e^{-j 2\pi (n_i / N) t}$ for
   $t = 0, \dots, L-1$. This shifts the band back to baseband so that
   downstream RoPE re-application (with cache-internal positions) composes
   correctly.
4. Inverse DFT of length $L$ and rescale by $\sqrt{L / N}$.
5. Reshape back to real pairs.

The caller is expected to pass **post-RoPE** K when `kv_type == "key"`. The
package ships a wrapper (`freqkv_ext.patch._wrap_with_rope_for_key`) that
rotates pre-RoPE K with RoPE before feeding it to the compressor, so
FreqKV's unmodified attention path continues to work.

### 3.4 What this does NOT yet give us

This first implementation does compression in DFT domain but still does
attention in time domain (after inverse transform). A future version can
keep attention in DFT domain via Parseval:

$$
\langle q, k \rangle \;=\; \sum_{t} q[t]\, k^{*}[t]
\;=\; \frac{1}{N} \sum_{\omega} Q[\omega] \, K^{*}[\omega].
$$

This unlocks two extra wins:

- **Skip IDFT** on the key path. Attention is computed on the retained $L$
  bins.
- **Fuse the IDFT** of value into $W_o$ (PALU-style), removing it from the
  inference path entirely.

These require a custom attention kernel and are deferred. The current
implementation is the minimal step that lets us *measure* whether the
RoPE-matched band actually helps PPL / Needle / LongBench.

## 4. Wavelet compressor

We use orthogonal / biorthogonal DWT (default `db4`, $\log_2 N$ levels) along
the sequence axis, hard-threshold to keep the $\gamma N d_\mathrm{head}$
largest coefficients per (batch, head, hidden) channel, then inverse-transform
and truncate to length $L$.

Reason for wavelets vs DCT / DFT for this problem: a localized event (a
needle token, a numeric literal, a function name) deposits energy across all
DCT / DFT high-frequency bins (delta-in-time $=$ constant-in-frequency).
Wavelet coefficients are time-frequency localized, so the same event lives in
a small number of coefficients at the appropriate scale; thresholding keeps
it cheaply.

Caveats:

- Current implementation does the DWT on CPU via PyWavelets. For end-to-end
  speed on H100, this is a bottleneck and should be replaced by a GPU-native
  DWT (e.g. as a stacked depthwise conv). Documented in `docs/RUN_H100.md`.
- Truncating to length $L$ after wavelet thresholding is a forcible
  conformance with FreqKV's interface and loses information. The principled
  form keeps the sparse coefficient set as the cache; this is a larger
  refactor reserved for a follow-up.

## 5. What the spectrum analysis script validates

`scripts/analyze_spectrum.py` extracts pre- and post-RoPE K from a frozen
LLaMA via forward hooks on `k_proj`, then plots:

- the per-pair DFT power spectrum,
- the predicted $n_i$ overlaid as a red dotted line per selected pair.

If the post-RoPE plots show clean peaks at the predicted $n_i$, the modulation
theorem is empirically active in the actual model and the RoPE-matched
bandpass is correctly targeted. If the spectrum is dominated by $n = 0$
across all pairs even post-RoPE (e.g. due to extreme dynamic range or
RMSNorm side-effects), the bandpass hypothesis weakens and we should
preregister this in the writeup before running compute-heavy evals.

This experiment is small enough to run on a single 24 GB GPU for LLaMA-2-7B
(seq_len $=$ 4K, 8 samples, fp16 $\approx$ 14 GB weights + small activations)
and is designed to gate the rest of the H100 spend.
