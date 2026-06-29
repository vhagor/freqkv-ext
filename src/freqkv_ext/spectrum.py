"""Utilities for extracting KV state spectra from a frozen LLaMA-family model.

Used by ``scripts/analyze_spectrum.py`` to validate the central DSP claim of
this project:

    1. Pre-RoPE K has energy concentrated near zero frequency along the sequence
       axis (the FreqKV observation).

    2. Post-RoPE K has the pair-``i`` energy shifted to bin ``n_i ~ theta_i * N / (2*pi)``,
       i.e. RoPE realizes a per-pair frequency shift (modulation theorem).

The script registers PyTorch forward hooks on each ``LlamaAttention`` layer's
``k_proj`` / ``q_proj`` / ``rotary_emb`` to capture pre-RoPE and post-RoPE
key tensors without modifying the model code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class LayerSpectrum:
    layer_idx: int
    pre_rope_power: np.ndarray  # [d_pair, N] real energy of DFT along seq
    post_rope_power: Optional[np.ndarray] = None  # same shape; only if available


@dataclass
class SpectrumResult:
    seq_len: int
    head_dim: int
    layers: list[LayerSpectrum] = field(default_factory=list)


def power_spectrum_pair_complex(x: torch.Tensor) -> np.ndarray:
    """Compute average energy spectrum along sequence for pair-complex view.

    Args:
        x: ``[bsz, num_heads, seq_len, head_dim]``, real.

    Returns:
        ``[head_dim//2, seq_len]`` numpy float32 array of mean-over-(batch, heads)
        squared magnitude of the per-pair DFT along the sequence axis.
    """
    bsz, num_heads, N, head_dim = x.shape
    pair = x.reshape(bsz, num_heads, N, head_dim // 2, 2)
    c = torch.view_as_complex(pair.to(torch.float32).contiguous())  # [B, H, N, d_pair]
    c = c.permute(0, 1, 3, 2)  # [B, H, d_pair, N]
    C = torch.fft.fft(c, dim=-1)
    p = (C.real**2 + C.imag**2).mean(dim=(0, 1))  # [d_pair, N]
    return p.cpu().numpy()


def apply_llama_rope_to_key(
    k: torch.Tensor,
    rope_base: float = 10000.0,
    positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply LLaMA RoPE to a key tensor of shape ``[B, H, N, head_dim]``.

    Uses the per-pair complex formulation; the result is real-valued K rotated by
    ``exp(j theta_i * t)``.
    """
    bsz, num_heads, N, head_dim = k.shape
    if positions is None:
        positions = torch.arange(N, device=k.device, dtype=torch.float32)
    i = torch.arange(head_dim // 2, dtype=torch.float32, device=k.device)
    thetas = rope_base ** (-2.0 * i / head_dim)
    pair = k.reshape(bsz, num_heads, N, head_dim // 2, 2)
    c = torch.view_as_complex(pair.to(torch.float32).contiguous())
    phase = torch.exp(1j * positions.unsqueeze(1) * thetas.unsqueeze(0))  # [N, d_pair]
    c_rot = c * phase.unsqueeze(0).unsqueeze(0)
    out_pair = torch.view_as_real(c_rot)
    return out_pair.reshape(bsz, num_heads, N, head_dim).to(k.dtype)
