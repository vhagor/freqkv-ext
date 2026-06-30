"""Capture pre-RoPE Q/K and V from a frozen LLaMA-family model via hooks.

Used by ``scripts/rate_distortion.py`` (experiment E2). Generalizes the
key-only capture in :mod:`freqkv_ext.spectrum` to also grab Q and V so we can
measure the *attention-output* error of each codec, not just K reconstruction.

GQA-aware: q_proj produces ``num_attention_heads`` heads, while k_proj/v_proj
produce ``num_key_value_heads`` heads. Tensors are returned in
``[samples, heads, seq_len, head_dim]`` layout (CPU float32).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LayerQKV:
    layer_idx: int
    q_pre: torch.Tensor  # [S, Hq, N, D]   pre-RoPE
    k_pre: torch.Tensor  # [S, Hkv, N, D]  pre-RoPE
    v: torch.Tensor      # [S, Hkv, N, D]


def capture_qkv(
    model,
    tokenizer,
    prompts: list[str],
    seq_len: int,
    layers: list[int],
    device: str,
    dtype: torch.dtype,
) -> tuple[list[LayerQKV], int, int, int]:
    """Run ``prompts`` through ``model`` and capture q/k/v projections.

    Returns ``(layer_qkv_list, num_q_heads, num_kv_heads, head_dim)``.
    """
    config = model.config
    num_q_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_q_heads)
    head_dim = config.hidden_size // num_q_heads

    layer_modules = list(model.model.layers)
    want = set(layers)
    cap_q: dict[int, torch.Tensor] = {}
    cap_k: dict[int, torch.Tensor] = {}
    cap_v: dict[int, torch.Tensor] = {}
    hooks = []

    def _mk_hook(store, li, n_heads):
        def _hook(_module, _inp, out):
            B, N, _ = out.shape
            store[li] = (
                out.detach().to(torch.float32).cpu()
                .reshape(B, N, n_heads, head_dim).transpose(1, 2)
            )
        return _hook

    for li, layer in enumerate(layer_modules):
        if li not in want:
            continue
        attn = layer.self_attn
        hooks.append(attn.q_proj.register_forward_hook(_mk_hook(cap_q, li, num_q_heads)))
        hooks.append(attn.k_proj.register_forward_hook(_mk_hook(cap_k, li, num_kv_heads)))
        hooks.append(attn.v_proj.register_forward_hook(_mk_hook(cap_v, li, num_kv_heads)))

    acc_q = {li: [] for li in want}
    acc_k = {li: [] for li in want}
    acc_v = {li: [] for li in want}
    try:
        for prompt in prompts:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=seq_len, padding="max_length",
            ).to(device)
            with torch.no_grad():
                model(**inputs)
            for li in want:
                acc_q[li].append(cap_q[li])
                acc_k[li].append(cap_k[li])
                acc_v[li].append(cap_v[li])
    finally:
        for h in hooks:
            h.remove()

    out = []
    for li in sorted(want):
        if not acc_q[li]:
            continue
        out.append(LayerQKV(
            layer_idx=li,
            q_pre=torch.cat(acc_q[li], dim=0),
            k_pre=torch.cat(acc_k[li], dim=0),
            v=torch.cat(acc_v[li], dim=0),
        ))
    return out, num_q_heads, num_kv_heads, head_dim
