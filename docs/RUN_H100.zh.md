# H100 实验手册

本文是 ``freqkv-ext`` 在多卡 H100 服务器上的执行剧本。假设：

- 8 × H100-80GB 或同档。
- Linux + CUDA 12 toolkit + 最新 NVIDIA 驱动。
- 能访问 ``meta-llama/Llama-2-7b-hf``（或本地权重）。
- 能访问 PyPI 和（可选）HuggingFace Hub。

下面所有命令假设工作目录是 ``/workspace``，``FreqKV`` 和 ``freqkv-ext`` 并排
放在那里。

## 0. 目录布局

```
/workspace/
├── FreqKV/                         # 从 LUMIA-Group/FreqKV clone
├── freqkv-ext/                     # 本仓库
├── models/                         # 本地模型快照（可选）
├── data/                           # FreqKV 提供的 PG-19 / Proof-pile bin
├── LongBench/                      # 从 THUDM/LongBench clone（可选）
└── LLMTest_NeedleInAHaystack/      # 从 gkamradt/... clone（可选）
```

## 1. 一键装环境

把 ``freqkv-ext`` 整目录 rsync 到 ``/workspace/freqkv-ext``，然后：

```bash
bash /workspace/freqkv-ext/scripts/h100_setup.sh
```

这个脚本会幂等地完成：

1. 装 uv（如果没有）。
2. clone FreqKV 到 ``/workspace/FreqKV``（如果没有）。
3. 在 ``/workspace/freqkv-ext`` 下建 ``.venv``，装 CUDA torch、本包的 ``[gpu]``
   extras、FreqKV 的 requirements.txt、以及 flash-attn（10-30 分钟）。
4. sanity-check：能 import torch+cuda、transformers 4.43.0、freqkv_ext、
   FreqKV 的 monkey-patch 模块、以及 flash_attn。

完成后激活 venv：

```bash
source /workspace/freqkv-ext/.venv/bin/activate
```

如果 flash-attn 默认版本装失败，可以指定一个已知能 build 的版本：

```bash
FLASH_ATTN_PIN="flash-attn==2.6.3" bash /workspace/freqkv-ext/scripts/h100_setup.sh
```

## 2. 一键跑流水线

激活 venv 之后：

```bash
MODEL_PATH=/path/to/Llama-2-7b-hf bash /workspace/freqkv-ext/scripts/h100_run_all.sh
```

按 stage 跳过：

```bash
# 只跑频谱分析（gate）：
STAGE_TRAIN=0 STAGE_PPL=0 bash scripts/h100_run_all.sh

# 训练 + PPL，不开 LongBench/Needle：
STAGE_LONGBENCH=0 STAGE_NEEDLE=0 bash scripts/h100_run_all.sh

# 只跑某几个方法：
METHODS="dct dft_rope" bash scripts/h100_run_all.sh
```

env 全部入口（顶部列在 ``h100_run_all.sh`` 的注释里）：

| 变量 | 默认 | 含义 |
|---|---|---|
| ``STAGE_SPECTRUM`` | 1 | 是否跑 stage 0 频谱分析 |
| ``STAGE_TRAIN`` | 1 | 是否跑 stage 1 训练 |
| ``STAGE_PPL`` | 1 | 是否跑 stage 2 PPL |
| ``STAGE_LONGBENCH`` | 0 | 需要 ``LONGBENCH_ROOT`` |
| ``STAGE_NEEDLE`` | 0 | 需要 ``NEEDLE_ROOT`` |
| ``METHODS`` | ``dct dft_lowpass dft_rope wavelet`` | 方法列表 |
| ``MODEL_PATH`` | ``meta-llama/Llama-2-7b-hf`` | 基础模型 |
| ``OUTPUT_DIR`` | ``$WORKSPACE/freqkv-ext/out`` | 输出根目录 |
| ``CKPT_DIR`` | ``$WORKSPACE/ckpts`` | 训练 ckpt 根目录 |
| ``NUM_GPUS`` | 8 | accelerate 进程数 |
| ``SEQ_LEN`` | 8192 | 训练上下文长度 |
| ``EVAL_SEQS`` | ``8192 16384 32768`` | 评测上下文长度 |
| ``ROPE_BASE`` | 10000.0 | LLaMA-2/3 默认 |
| ``HEAD_DIM`` | 128 | LLaMA-2-7B 默认 |
| ``KEY_OFFSET`` | 0 | ``FREQKVEXT_KEY_OFFSET`` |

## 3. 关键判断点：频谱分析的 go / no-go

``h100_run_all.sh`` 的 stage 0 跑完后请打开 ``$OUTPUT_DIR/spectrum/layerNN.png``。
post-RoPE 子图里有一条条**红色虚线**——它们是预测的
$n_i = \mathrm{round}(\theta_i N / (2\pi))$。

**判断规则**：
- 红线和实际峰位置**对齐**（至少在低 ``i`` 和高 ``i`` 两端各看一对清楚的对齐）
  → DFT-RoPE 路径成立，继续 stage 1+。
- 红线和峰**完全不对齐**或峰太宽以至于看不出来 → 停下，先做 ``docs/INNOVATIONS.zh.md``
  的 (b') 失败模式诊断（去 RMSNorm 缩放重画、去 sink token 重画、按层分别检查），
  再决定是否花 H100 时间训练。

## 4. 单步分阶执行（不用 run_all）

### 4.1 频谱分析（约 5 分钟 / 1 GPU）

```bash
python scripts/analyze_spectrum.py \
    --model_name_or_path /workspace/models/Llama-2-7b-hf \
    --seq-len 4096 --num-samples 16 \
    --layers 0 4 8 16 31 \
    --out-dir /workspace/freqkv-ext/out/spectrum
```

### 4.2 训练单方法

```bash
accelerate launch --num_processes 8 scripts/train.py \
    --ext-method dft_rope \
    --variant lm \
    --model_name_or_path /workspace/models/Llama-2-7b-hf \
    --bf16 True \
    --output_dir /workspace/ckpts/dft_rope_8k \
    --model_max_length 8192 \
    --use_flash_attn True \
    --low_rank_training True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 --warmup_steps 20 --logging_steps 1 \
    --save_strategy steps --save_steps 200 \
    --deepspeed /workspace/FreqKV/ds_configs/stage2.json
```

训练完合并 LoRA：

```bash
cd /workspace/ckpts/dft_rope_8k && \
    python /workspace/FreqKV/zero_to_fp32.py . pytorch_model.bin && \
    bash /workspace/FreqKV/merge.sh
```

### 4.3 PPL 单方法 / 单长度

```bash
PYTHONPATH=src python scripts/eval_ppl.py \
    --ext-method dft_rope \
    --base_model /workspace/ckpts/dft_rope_8k/merged \
    --seq_len 32768 --context_size 8192 \
    --data_path /workspace/FreqKV/data/pg19/test.bin \
    --output_dir /workspace/out/ppl/dft_rope_32768
```

### 4.4 LongBench / Needle

需要先 clone 上游仓库：

```bash
git clone https://github.com/THUDM/LongBench /workspace/LongBench
git clone https://github.com/gkamradt/LLMTest_NeedleInAHaystack /workspace/LLMTest_NeedleInAHaystack

export LONGBENCH_ROOT=/workspace/LongBench
export NEEDLE_ROOT=/workspace/LLMTest_NeedleInAHaystack
```

然后：

```bash
PYTHONPATH=src python scripts/eval_longbench.py \
    --ext-method dft_rope \
    --model-path /workspace/ckpts/dft_rope_8k/merged \
    --longbench-root "$LONGBENCH_ROOT" \
    --task all --out /workspace/out/longbench/dft_rope.jsonl

PYTHONPATH=src python scripts/eval_needle.py \
    --ext-method dft_rope \
    --model-path /workspace/ckpts/dft_rope_8k/merged \
    --needle-root "$NEEDLE_ROOT" \
    --context-lengths 1000 2000 4000 8000 12000 16000 \
    --depths 0.0 0.25 0.5 0.75 1.0 \
    --out /workspace/out/needle/dft_rope.jsonl
```

## 5. 结果汇总表格（建议格式）

照 FreqKV 论文 Table 2/3/4 的风格：

| 压缩算子 | PG-19 PPL @ 32K | LongBench avg | NIAH @ 16K |
|---|---|---|---|
| DCT (FreqKV baseline) | (基线) | (基线) | (基线) |
| DFT 低通 | 应当接近 DCT | 应当接近 DCT | 应当接近 DCT |
| DFT RoPE-aware | 持平或更好 | QA / 代码子集更好 | **预期显著更好** |
| 小波 | 可能略差 | 接近 | 平均略差，代码 / needle 更好 |

任何偏离"应当持平"的数据都是论文素材：要么有 bug，要么是真信息论现象。两种
情况都能写，但叙事完全不同。

## 6. 常见踩坑

- **DFT-RoPE wrapper 的位置偏移**：``patch.py`` 默认按 ``positions =
  arange(N)`` 旋转 pre-RoPE K。FreqKV 迭代压缩里那一段 K 在原始序列的位置其实是
  ``[sink_size, sink_size + fft_span)``。如果想严格对齐，运行前 ``export
  FREQKVEXT_KEY_OFFSET=<sink_size>``。第一次粗跑可以用默认值（常数偏移只是个
  全局相位，不影响 bandpass 中心位置）。

- **transformers 版本**：FreqKV 的 monkey-patch 锁着 transformers 4.43.0 的
  ``LlamaAttention.forward`` 签名。新版本 transformers 重构了 cache 和
  position embedding，不要升级，除非同时改 ``llama_attn_replace_dct_mempe.py``。

- **flash-attn 构建**：需要 CUDA 12。如果遇到莫名编译报错，``pip install
  flash-attn==2.6.3 --no-build-isolation`` 是已知能 build 的 pin。

- **deepspeed + low_rank_training**：如果训练时遇到 deepspeed 在 LoRA 参数上
  形状不匹配，把 ``"zero_optimization": {"stage": 2}`` 写明，避免 ZeRO-3，
  除非顺便把 ``"prefetch_bucket_size": auto`` 也加上。
