"""Tests for the RST compressor under FreqKV's fixed-length interface."""

from __future__ import annotations

import torch

from freqkv_ext.transforms import METHODS, get_compressor


def test_rst_registered():
    assert "rst" in METHODS
    assert get_compressor("rst") is METHODS["rst"]


def test_rst_output_length_and_dtype():
    rst = get_compressor("rst")
    x = torch.randn(2, 4, 64, 16, dtype=torch.float16)
    out = rst(x, compress_len=16, kv_type="key")
    assert out.shape == (2, 4, 16, 16)
    assert out.dtype == torch.float16


def test_rst_passthrough_when_no_compression():
    rst = get_compressor("rst")
    x = torch.randn(1, 2, 32, 8)
    assert torch.equal(rst(x, compress_len=32), x)
    assert rst(x, compress_len=0).shape[2] == 0


def test_rst_value_path_runs():
    rst = get_compressor("rst")
    x = torch.randn(1, 2, 48, 8)
    out = rst(x, compress_len=12, kv_type="value")
    assert out.shape == (1, 2, 12, 8)
