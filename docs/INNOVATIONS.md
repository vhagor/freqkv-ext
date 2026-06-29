# Innovations over FreqKV, and experiment TODO list

This document is a working record of what we claim is genuinely new in
``freqkv-ext`` relative to the FreqKV paper (ICLR 2026), what the current
implementation has actually shipped, and what experiments are still required
to validate or refute the claims.

## A. Why FreqKV is the right reference point

FreqKV opens a useful frame: KV cache as a 1D signal along the sequence axis,
compressible by classical transform coding. Within that frame, FreqKV picks
the simplest tool (DCT-II) and the simplest selection rule (uniform low-pass)
and shows that, after light fine-tuning, this is enough to extend LLaMA-2-7B
from 4K to 256K context with negligible PPL drift. Its weak spots, in our
reading of the paper:

1. **RoPE is treated as something to be avoided** (cache pre-RoPE, apply RoPE
   on the fly each attention step). The compression decision is made before
   the model's actual position-aware representation exists.
2. **Uniform low-pass is RoPE-blind**: the band ``[0, L)`` in pre-RoPE is a
   poor proxy for the band that survives RoPE for high-θ pairs.
3. **Position re-anchoring is a "trick"**: compressed tokens are assigned
   fake positions 0..L-1 within the cache; this works empirically but has no
   information-theoretic justification.
4. **Local features get smeared**: DCT is a global basis; localized
   features (needles, code symbols) excite *all* high-frequency coefficients
   and are the first thing dropped under any low-pass.

## B. What ``freqkv-ext`` brings to the table

Five distinct improvements over FreqKV, separated by status:

| # | Improvement | Theory | Code | Empirical |
|---|---|---|---|---|
| 1 | Per-pair θ_i-matched bandpass (vs FreqKV's uniform low-pass) | ✓ | ✓ | □ |
| 2 | Post-RoPE caching is safe in DFT domain (pre/post inter-convertible by a known phase) | ✓ | ✓ via wrapper | □ |
| 3 | Compressed-token positions become a design variable via linear-phase multipliers | ✓ | partial | □ |
| 4 | Online-RoPE GEMM removed (cache stores post-RoPE) | ✓ | ✗ (needs attention-path edit) | □ |
| 5 | Compressed-domain attention via Parseval (skip IDFT) | ✓ | ✗ (needs custom kernel) | □ |
| 6 | Wavelet operator: time-frequency localized basis preserves local details that DCT smears | ✓ | ✓ | □ |

"Empirical" means measured on a real LLaMA-2-7B (or larger) with PPL / Needle
/ LongBench numbers. Currently zero of the six claims are empirically
substantiated; only the **mathematical identity** that improvement #1 rests on
(the modulation theorem, RoPE-as-frequency-shift) has been numerically verified
on a synthetic input (see ``tests/test_transforms.py::test_theta_bin_offsets_match_modulation``).

### Improvement 1 in detail (the headline claim)

Apply LLaMA RoPE to a pair-complex K sequence. The modulation theorem gives

$$
C_i^{\mathrm{post\text{-}RoPE}}[k] \;=\; C_i^{\mathrm{pre\text{-}RoPE}}[k - n_i],
\qquad n_i \;=\; \mathrm{round}\!\left(\frac{\theta_i N}{2\pi}\right).
$$

So the **post-RoPE relevant band** of pair $i$ is centered at bin $n_i$,
NOT at bin 0. For LLaMA-2 with $N = 2048$, $n_i$ ranges from 326
($i = 0$, fast pair) down to 0 ($i = 63$, slow pair).

FreqKV's uniform low-pass keeps bins $[0, L)$ of pre-RoPE; after on-the-fly
RoPE this band becomes $[n_i, n_i + L) \bmod N$. For high-$\theta$ pairs
this is roughly the same as the post-RoPE band (just rotated). So FreqKV's
choice IS correct as far as preserving final post-RoPE energy, when $L$ is
reasonably large. **The slack appears at high compression** (small $L$) and
**for short-range positional information** that lives in narrow post-RoPE
bands near non-zero $n_i$.

A RoPE-matched bandpass selects bins $[n_i - L/2,\, n_i + L/2]$ (per pair) of
the **post-RoPE** spectrum directly. Same budget $L$; smarter allocation.

### Improvement 6 in detail (the wavelet path)

A localized event at position $t_0$ looks like a delta in time and a
**constant** in DCT / DFT (i.e. excites all high-frequency coefficients
uniformly). The wavelet basis represents the same event as a small handful
of large coefficients at the appropriate scale, around $t_0$ only.
Threshold the small coefficients away and you preserve the local event for
free.

In the KV compression context this directly attacks the FreqKV weak spot on
needle / code / numerics. The cost is that wavelets are typically worse than
DCT for *smooth* signals (which is most of natural text), so PG-19 PPL may
move the wrong way unless we use the wavelet's multiresolution structure
carefully.

## C. Experiment TODO list

Group by phase. Each line ends with the artifact that "closes" it.

### Phase 0 — DSP claim validation (RTX 5060 with INT4 or H100)

- [ ] **(a) Pre-RoPE K spectrum on LLaMA-2-7B**. Run
      ``scripts/analyze_spectrum.py`` with ``--model_name_or_path
      meta-llama/Llama-2-7b-hf`` (or local mirror), seq_len=4096, 16 samples,
      layers {0, 4, 8, 16, 31}. Verify pre-RoPE energy is concentrated near
      bin 0 (reproducing FreqKV's Figure 1).
      *Artifact*: ``out/spectrum/layer*.png`` showing low-freq peaks pre-RoPE.

- [ ] **(b) Post-RoPE K spectrum on LLaMA-2-7B**. Same script, same outputs;
      check that the per-pair peaks line up with the red dotted lines
      (predicted $n_i$). Inspect specifically pair 0 (high $\theta$, peak should
      be ~bin 326) and pair 32 (mid θ, peak ~bin 33).
      *Artifact*: same PNGs; pass/fail per pair logged.

- [ ] **(b') Failure-mode characterization**. If (b) fails partially (e.g.
      peaks present but broad, or only for low-θ pairs), characterize the
      cause:
        - RMSNorm scaling? Subtract per-token mean before DFT and re-check.
        - Attention sinks? Drop the first ``sink_size`` tokens and re-check.
        - Layer-specific? Earlier layers might be sharper than later.
      *Artifact*: short writeup of which pairs / layers show clean shifts.

- [ ] **(a') Wavelet-vs-DCT sparsity on LLaMA-2-7B**. Same script run as
      (a) / (b) also emits ``layer*_sparsity.png``, ``sparsity_summary.png``,
      and ``sparsity.json`` reporting the fraction of coefficients needed to
      capture 95% of $L^2$ energy in DCT, DFT (rfft), and wavelet (default
      ``db4``) bases. **GO/NO-GO rule** (printed at end of run): if median
      $\text{wavelet}_{p50}$ across layers is materially lower than DCT
      $p_{50}$ (e.g. by $\geq 0.05$), the wavelet path has a structural
      foothold on this model and should be pursued. If DCT is at parity or
      better, deprioritize wavelet.
      *Artifact*: ``sparsity_summary.png`` (per-layer median lines) and the
      gate decision in stdout.

### Phase 1 — Training-free plug-in eval (H100, ~1 day)

- [ ] **(c1) PPL sanity at γ=0.5 on FreqKV's released SFT ckpt**. Patch the
      compressor as ``dct`` and reproduce FreqKV's PG-19 PPL number. This is
      the anchor.
- [ ] **(c2) Same setup, ``dft_lowpass``**. Should match (c1) within noise.
      If it doesn't, there's a bug in the DFT scaffolding.
- [ ] **(c3) Same setup, ``dft_rope``**. Expected: roughly matches; the
      training-free regime is unfair to the bandpass operator because the
      model wasn't trained for the new spectrum allocation.
- [ ] **(c4) Same setup, ``wavelet``**. Expected: worse than DCT (training-
      free regime + smooth signal disadvantage).
*Artifacts*: a 4-row PPL table at multiple seq lengths.

### Phase 2 — Full training + eval (H100, ~3-5 days)

- [ ] **(d) Train all four variants** at 8K with the FreqKV LongLoRA recipe.
      Output: ``ckpts/{dct,dft_lowpass,dft_rope,wavelet}_8192``.
- [ ] **(e1) PPL on PG-19 test** at 8K, 16K, 32K. Reproduce FreqKV Table 2
      for ``dct``; report deltas for the other three.
- [ ] **(e2) PPL on Proof-pile test** at 8K, 16K, 32K. Special interest:
      wavelet should help here (math symbols are localized).
- [ ] **(f) LongBench full evaluation**. Compare per-task scores; expected
      ``dft_rope`` wins on HotpotQA / 2WikiMQA / RULER NIAH; ``wavelet`` wins
      on LCC / code tasks.
- [ ] **(g) Needle-in-a-Haystack** at 1K..16K, depths {0, 0.25, 0.5, 0.75,
      1.0}. **This is the decisive test for improvement #1.** Expected:
      ``dft_rope`` recovers needles in the 8K-16K band where FreqKV's heatmap
      starts to fail.

### Phase 3 — Systems-level wins (the harder follow-ups)

- [ ] **(h) Compressed-domain V attention**. Fuse ``D^T`` (inverse DFT) into
      ``W_o``; cache stores compressed V latent; attention output is
      ``softmax · cached_V · M_v`` where ``M_v`` is precomputed offline.
      Skip IDFT entirely on V side.
      *Risk*: numerical mismatch from fusion; need careful unit tests.
- [ ] **(i) Online RoPE elimination**. With (h) and post-RoPE K caching,
      the attention path no longer needs to apply RoPE at decode time.
      Measure decode latency / TTFT delta on long contexts.
- [ ] **(j) Wavelet GPU kernel**. The current wavelet path runs on CPU via
      PyWavelets; replace with a stacked depthwise conv on GPU for speed.

### Phase 4 — Writing

- [ ] **(k) Negative-result audit**. For every failed expectation in Phases
      0-3, log the cause. The paper writeup must report failures honestly.
- [ ] **(l) Ablations**: γ sweep, sink size, recent window size, layer-wise
      γ allocation.
- [ ] **(m) Limitations section** anchored on the failure-mode taxonomy
      from (k) and (l).

## D. Decision rules

We do not commit to writing a paper until at least one of:

- (g) ``dft_rope`` shows ≥10% absolute improvement over ``dct`` on Needle
  accuracy at >=8K with matching γ.
- (f) ``dft_rope`` shows ≥1 point average improvement on LongBench QA subset
  with same fine-tune budget.
- (e2) ``wavelet`` shows ≥0.1 PPL improvement over ``dct`` on Proof-pile,
  AND ≥5% absolute improvement on a math/code subtask.

If none of these triggers fire after Phase 2, the project ends with the
spectrum analysis as a standalone DSP note ("how RoPE looks in the DFT
domain"), without a full ICLR-style paper.

## E. What's already on disk

- ``src/freqkv_ext/transforms/dft_rope_aware.py`` — implementation of
  improvement #1 and the demodulation part of #3.
- ``src/freqkv_ext/transforms/wavelet.py`` — implementation of #6 (CPU; #J is the GPU follow-up).
- ``src/freqkv_ext/patch.py::_wrap_with_rope_for_key`` — bridges FreqKV's
  pre-RoPE caching to our post-RoPE compressor (improvement #2's code).
- ``tests/test_transforms.py::test_theta_bin_offsets_match_modulation`` —
  numerical verification of the modulation theorem; this is the *only* claim
  that's currently validated, and it's only on synthetic data.
- ``scripts/h100_setup.sh`` + ``scripts/h100_run_all.sh`` — Phase 0-2
  automation.

## F. Open questions we cannot answer from the desk

1. Does the post-RoPE spectrum of real LLaMA-2-7B actually peak at the
   predicted $n_i$? (Phase 0.)
2. If yes, how sharp are the peaks? Sharper -> higher upper bound on
   improvement #1's gain.
3. Are different layers in the same model "spectrally homogeneous", or do
   early/late layers need different L per pair? (If the latter, the budget
   allocator becomes a research subproblem in itself.)
4. Does γ=0.5 hide the differences? At γ=0.1 or 0.01, FreqKV's PPL goes up
   fast (paper Table 3 at 1% retention); this is the regime where smarter
   band selection should help most.

These four questions structure the Phase 0-2 experiment plan, and the answers
will determine whether the project pivots to a different track or pushes
through to publication.
