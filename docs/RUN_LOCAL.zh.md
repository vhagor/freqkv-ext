# 本地开发

本地路径**不跑重活**：代码改动、单元测试、dry-run 画图。所有重的推理 / 训练都
放到 H100 上跑。

## 前置

- ``uv``（已经装在 ``~/.local/bin/uv``）。
- Python 3.11（uv 缺则自动装）。
- 大约 250 MB 空间放 CPU-only torch wheel。

## 启动一次性

```bash
cd /home/vhagor/workbench/freqkv-ext

# 创建 CPU-only venv（已存在的话跳过）。
uv venv --python 3.11 .venv-cpu
uv pip install --python .venv-cpu/bin/python \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    numpy scipy pywavelets pytest matplotlib einops 'torch>=2.1'
```

## 跑单元测试

```bash
PYTHONPATH="$PWD/src" ./.venv-cpu/bin/python -m pytest tests -v
```

期望：**24 passed**。最关键的是
``test_theta_bin_offsets_match_modulation``——它在数值上验证了"对基带信号 apply
RoPE，DFT 峰落在 $\theta_i N / (2\pi)$"这个调制定理恒等式。任何代码改动后
这个测试如果挂了，说明 RoPE-频移的等式被破坏，DFT-RoPE-aware compressor 就不再
正确。

## Dry-run 频谱分析（无需 LLM）

验证画图 pipeline 不依赖任何 LLM。用 AR(1) 合成信号代替 K：

```bash
PYTHONPATH="$PWD/src" ./.venv-cpu/bin/python scripts/analyze_spectrum.py \
    --dry-run --seq-len 256 --num-samples 2 --layers 0 4 8 \
    --device cpu --dtype float32 --out-dir ./out/spectrum_dryrun
```

输出在 ``out/spectrum_dryrun/layerNN.png``。pre-RoPE 子图能量应当集中在低 bin
（AR(1) 的特征），post-RoPE 子图应该看到每对 pair 在红色虚线处有清晰峰。

## 可选：在 RTX 5060 上跑小模型

8 GB 显存装不下 LLaMA-2-7B（fp16 ≈ 14 GB），但小 LLaMA-架构模型能装，可以用
真实激活做频谱（只 forward 几个 prompt，不训练）：

```bash
PYTHONPATH="$PWD/src" python scripts/analyze_spectrum.py \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --seq-len 1024 --num-samples 4 \
    --layers 0 4 8 16 \
    --dtype float16 --device cuda \
    --out-dir ./out/spectrum_tinyllama
```

注意：
- TinyLlama 用 base=10000 RoPE（和 LLaMA-2 一致），红线预测仍然适用。
- TinyLlama 的 RMSNorm 缩放和 LLaMA-2 不一样，绝对峰高在跨模型时不可直接比，
  但 bin 位置的预测仍然有效。

## 工作流建议

- 改 ``src/freqkv_ext/transforms/*.py``。每次有意义的改动后跑一次 ``pytest tests``。
- ``patch.py`` 在本地测不了（需要 FreqKV 在 PYTHONPATH，并且需要有可工作的
  attention 路径）；改动后留到 H100 review。
- 加新的压缩算子：写一个带标准签名的函数
  ``(x, compress_len, seq_dim=2, kv_type='key', **kwargs) -> Tensor``，注册到
  ``src/freqkv_ext/transforms/__init__.py::METHODS``，再补单元测试。
