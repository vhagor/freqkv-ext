# DFT 相对 FreqKV 的改进与创新点 / 实验 TODO

这份文档是 ``freqkv-ext`` 相对 FreqKV 论文（ICLR 2026）的工作记录：我们声明哪些
是真正新的、当前代码实现到了哪里、还需要哪些实验来验证或否定这些声明。

## A. 为什么 FreqKV 是合适的参照对象

FreqKV 提出了一个有效的框架：把 KV cache 视为沿序列轴的 1D 信号，用经典变换编码
压缩。在这个框架内 FreqKV 选了最简单的工具（DCT-II）和最简单的选择规则（均匀
低通），证明只要再轻微微调，就能把 LLaMA-2-7B 从 4K 推到 256K 上下文且 PPL 几乎
不掉。在我们读论文时看到的弱点：

1. **把 RoPE 当作要绕开的事**（cache pre-RoPE，attention 时现场加 RoPE）。
   压缩决策发生在模型真正"看到位置"之前。
2. **均匀低通对 RoPE 是盲的**：pre-RoPE 的 ``[0, L)`` 频带对高 θ 的 pair 不是
   post-RoPE 的最优代理。
3. **位置重指派是个 trick**：被压缩的 token 在 cache 内被赋"假位置" ``0..L-1``。
   经验上 work，但没有信息论意义上的依据。
4. **局部特征被抹平**：DCT 是全局基，"针"事件激活**所有**高频系数，是任何低通
   第一个丢掉的东西。

## B. ``freqkv-ext`` 带来的改变

相对 FreqKV 的五个改进，按状态分类：

| # | 改进 | 理论 | 代码 | 实证 |
|---|---|---|---|---|
| 1 | 按 pair 的 θ_i 匹配带通（替代 FreqKV 的均匀低通） | ✓ | ✓ | □ |
| 2 | 在 DFT 域 cache post-RoPE 是安全的（pre/post 可由已知相位互换） | ✓ | ✓（wrapper） | □ |
| 3 | 压缩 token 的"位置"成为可调设计变量（线性相位乘法） | ✓ | 部分 | □ |
| 4 | 推理期 RoPE GEMM 可消去（cache 直存 post-RoPE） | ✓ | ✗（需改 attention 路径） | □ |
| 5 | 压缩域 attention（Parseval，跳过 IDFT） | ✓ | ✗（需自定义 kernel） | □ |
| 6 | 小波算子：时-频局部基保住 DCT 抹掉的局部细节 | ✓ | ✓ | □ |

"实证"列表示：在真实 LLaMA-2-7B（或更大）上用 PPL / Needle / LongBench 测过。
**当前 6 条声明里没有一条有实证支撑**；仅有改进 #1 所依赖的**数学恒等式**
（调制定理，即 RoPE 在 DFT 域是频移）在合成输入上做过数值验证（见
``tests/test_transforms.py::test_theta_bin_offsets_match_modulation``）。

### 改进 1 详解（头条声明）

对 pair-complex K 序列 apply LLaMA RoPE。由调制定理：

$$
C_i^{\mathrm{post\text{-}RoPE}}[k] \;=\; C_i^{\mathrm{pre\text{-}RoPE}}[k - n_i],
\qquad n_i \;=\; \mathrm{round}\!\left(\frac{\theta_i N}{2\pi}\right).
$$

所以 pair $i$ 的 **post-RoPE 相关频带**中心在 bin $n_i$，**不是**在 bin 0。
LLaMA-2 $N = 4096$ 时 $n_i$ 从 652（$i = 0$，高频 pair）到 0（$i \gtrsim
48$，慢 pair）。

FreqKV 的统一低通保留 pre-RoPE 的 $[0, L)$；推理期 apply RoPE 后这段变成
$[n_i, n_i + L) \bmod N$。对高 $\theta$ 的 pair，这和 post-RoPE 的目标频带
大体重合（只是旋转过去）。所以 $L$ 充分大时 FreqKV 的最终 post-RoPE 能量
IS 保住的。**slack 在高压缩率下**（$L$ 小）和**对短程位置信息**（住在非零
$n_i$ 附近窄带里的内容）出现。

RoPE-matched bandpass 直接在 **post-RoPE** 频谱上按 pair 选
$[n_i - L/2,\, n_i + L/2]$ 个 bin。同样的预算 $L$，更聪明的分配。

### 改进 6 详解（小波路径）

位置 $t_0$ 处的一个局部事件，对 DCT / DFT 看是"时域 delta $=$ 频域常数"——
所有高频系数被一致激活。同一个事件对小波基只在合适尺度上产生少数大幅度系数，
仅在 $t_0$ 周围；硬阈值正好留下来。

在 KV 压缩语境里这直接打击 FreqKV 在 needle / 代码 / 数值任务上的弱点。代价是
小波在**平滑信号**上一般弱于 DCT（自然语言大部分是平滑的），所以 PG-19 PPL
可能往错方向走，除非小心地用小波的多分辨结构。

## C. 实验 TODO

按阶段组织，每条结尾给出"closed"判据。

### Phase 0 — DSP 声明验证（RTX 5060 + INT4 或 H100）

- [ ] **(a) LLaMA-2-7B 的 pre-RoPE K 频谱**。``scripts/analyze_spectrum.py``
      跑 ``--model_name_or_path meta-llama/Llama-2-7b-hf``（或本地镜像），
      seq_len=4096，16 个 samples，layers {0, 4, 8, 16, 31}。验证 pre-RoPE
      能量集中在 bin 0 附近（复现 FreqKV Figure 1）。
      *Artifact*：``out/spectrum/layer*.png`` 的 pre-RoPE 子图低频有峰。

- [ ] **(b) LLaMA-2-7B 的 post-RoPE K 频谱**。同脚本同输出；检查 per-pair 峰
      和红色虚线（预测 $n_i$）对齐。重点检查 pair 0（高 $\theta$，$N=4096$ 时
      峰应在 bin $\approx 652$）和 pair 32（中 $\theta = 0.01$，峰在 bin
      $\approx 7$）。
      *Artifact*：同样的 PNG；按 pair 记录 pass / fail。

- [ ] **(b') 失败模式刻画**。如果 (b) 部分失败（如峰在但宽，或只对低 $\theta$
      pair 对齐），定位原因：
        - RMSNorm 缩放？DFT 前先减 per-token 均值再画。
        - Attention sink？丢掉前 ``sink_size`` token 再画。
        - 层依赖？早层和晚层可能不一样。
      *Artifact*：一份简短文字记录哪些 pair / 哪些层有干净频移。

- [ ] **(a') LLaMA-2-7B 上的 wavelet-vs-DCT 稀疏度 gate**。同一次
      ``analyze_spectrum.py`` 还会输出 ``layer*_sparsity.png``、
      ``sparsity_summary.png`` 和 ``sparsity.json``，报告在 DCT、DFT（rfft）和
      wavelet（默认 ``db4``）三种基下，每条沿序列轴的 per-channel 实信号要保住
      95% $L^2$ 能量分别需要保留多少比例的系数。**GO/NO-GO 规则**（脚本末尾会
      打印）：如果跨层 wavelet 的 $p_{50}$ 比 DCT 的 $p_{50}$ 显著更小
      （$\geq 0.05$），说明小波在这模型上有**结构性**优势，值得推；如果 DCT
      持平或反胜，把小波路线降优先级。
      *Artifact*：``sparsity_summary.png`` 的每层中位曲线和 stdout 的 gate 判定。

### Phase 1 — Training-free plug-in eval（H100，约 1 天）

- [ ] **(c1) PPL sanity，在 FreqKV 公开 SFT ckpt 上 γ=0.5**。把 compressor 设
      为 ``dct``，复现 FreqKV 的 PG-19 PPL 数值。这是锚点。
- [ ] **(c2) 同上，换 ``dft_lowpass``**。应该在 noise 内贴近 (c1)。如果不贴近，
      DFT scaffolding 里有 bug。
- [ ] **(c3) 同上，换 ``dft_rope``**。期望：差不多。training-free 对带通算子
      不公平，因为模型没为新频带分配训练过。
- [ ] **(c4) 同上，换 ``wavelet``**。期望：略差于 DCT（training-free + 平滑信号
      劣势）。
*Artifact*：一张 4 行 × 多 seq 长度的 PPL 表。

### Phase 2 — 完整训练 + 评测（H100，约 3-5 天）

- [ ] **(d) 训四个变体**。FreqKV 的 LongLoRA recipe，8K 训练。输出
      ``ckpts/{dct, dft_lowpass, dft_rope, wavelet}_8192``。
- [ ] **(e1) PG-19 test 上 PPL**，8K / 16K / 32K。复现 FreqKV Table 2 的
      ``dct`` 行；报告其他三种的 delta。
- [ ] **(e2) Proof-pile test 上 PPL**，8K / 16K / 32K。重点关注：小波应当在
      这里有帮助（数学符号是局部的）。
- [ ] **(f) LongBench 全部任务**。按子任务比较；期望 ``dft_rope`` 在
      HotpotQA / 2WikiMQA / RULER NIAH 上反超；``wavelet`` 在 LCC / 代码任务
      上反超。
- [ ] **(g) Needle-in-a-Haystack** 1K..16K，深度 {0, 0.25, 0.5, 0.75, 1.0}。
      **这是对改进 #1 的决定性测试**。期望：``dft_rope`` 在 8K-16K 这个 FreqKV
      热力图开始翻车的区间里把 needle 找回来。

### Phase 3 — 系统层改造（更难的后续）

- [ ] **(h) 压缩域 V attention**。把 ``D^T``（逆 DFT）融进 ``W_o``；cache 存
      压缩 V 隐空间；attention 输出 ``softmax · cached_V · M_v``，``M_v``
      离线预算。完全跳过 V 的 IDFT。
      *风险*：融合带来的数值不一致；要细的单元测试。
- [ ] **(i) 推理期 RoPE 消去**。配合 (h) 和 post-RoPE K cache，attention 路径
      不再需要解码时 apply RoPE。测长 context 上的 decode latency / TTFT
      变化。
- [ ] **(j) 小波 GPU kernel**。当前小波在 CPU 上跑（PyWavelets），换成 GPU 上
      的分层 depthwise conv。

### Phase 4 — 写作

- [ ] **(k) 负面结果审计**。Phase 0-3 里每一处期望落空都记录原因。论文里要
      诚实报告失败。
- [ ] **(l) Ablation**：γ 扫、sink 大小、recent window 大小、按层不同 γ。
- [ ] **(m) Limitations**：基于 (k)(l) 的失败模式分类。

## D. 决策规则

至少满足以下之一才进入论文写作：

- (g) ``dft_rope`` 在 >=8K 的 Needle accuracy 上，在同样 γ 下比 ``dct`` 至少
  绝对 +10%。
- (f) ``dft_rope`` 在 LongBench QA 子集均分上，同样 fine-tune budget 下比
  ``dct`` 至少 +1 分。
- (e2) ``wavelet`` 在 Proof-pile 上比 ``dct`` PPL 至少低 0.1，**且**在
  某个数学 / 代码子任务上至少绝对 +5%。

如果 Phase 2 跑完三条都不触发，项目以频谱分析作为独立 DSP 笔记（"RoPE 在 DFT
域长什么样"）收尾，不强行做完整 ICLR 级论文。

## E. 当前磁盘上已经有的东西

- ``src/freqkv_ext/transforms/dft_rope_aware.py`` — 改进 #1 的实现，以及 #3
  的 demodulation 部分。
- ``src/freqkv_ext/transforms/wavelet.py`` — 改进 #6（CPU 上；GPU 跟在 (j)）。
- ``src/freqkv_ext/patch.py::_wrap_with_rope_for_key`` — 衔接 FreqKV 的
  pre-RoPE cache 与我们的 post-RoPE compressor（改进 #2 的代码）。
- ``tests/test_transforms.py::test_theta_bin_offsets_match_modulation`` —
  调制定理的数值验证；这是**当前唯一**被证过的声明，且只在合成数据上。
- ``scripts/h100_setup.sh`` + ``scripts/h100_run_all.sh`` — Phase 0-2 的自动
  化。

## F. 桌面上回答不了的开放问题

1. 真实 LLaMA-2-7B 的 post-RoPE 频谱是不是真的在预测 $n_i$ 处有峰？
   （Phase 0。）
2. 如果是，峰多锐？越锐 → 改进 #1 收益的上界越高。
3. 同一个模型里不同 layer 是不是"频谱同质"，还是早 / 晚层需要不同的 per-pair
   ``L``？（如果是后者，budget 分配本身就成了一个研究子问题。）
4. γ=0.5 是不是把差异掩盖掉了？γ=0.1 或 0.01（FreqKV 论文 Table 3 的 1% 保留
   regime）才是 FreqKV PPL 抬起来最快的区间，也是更聪明的带选择应当帮助最大
   的地方。

这四个问题决定了 Phase 0-2 的实验结构，它们的答案决定项目下一步是另辟蹊径还是
推到发表。
