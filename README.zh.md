# exactkv

**大模型解码用的无损 KV 缓存压缩：缓存更小，attention 更快，输出一个比特都不变。**

![bit-exact](https://img.shields.io/badge/output-bit--exact-brightgreen) ![KV](https://img.shields.io/badge/KV%20cache-%E2%88%9220%25-blue) ![attention](https://img.shields.io/badge/decode%20attention-1.2%C3%97%20vs%20FlashInfer-orange) ![license](https://img.shields.io/badge/license-Apache--2.0-lightgrey)

常见的 KV 缓存瘦身方法（FP8、INT4、丢弃 token）都会改变模型的输出，exactkv 不会。它让 bf16 的 KV 缓存在显存里一直保持压缩，在 decode attention 做张量核乘法之前，于寄存器里把每个 key、value 原样还原。没有任何近似，不用校准，不用重新训练，也不写解压后的副本。

- **逐位精确。** 开不开压缩，贪心解码的 token 完全相同、logits 逐位相同。已在 Qwen2.5-7B、Qwen3-8B、Llama-3.1-8B 上端到端验证，并覆盖全部 65,536 种 bf16 取值。
- **比 FlashInfer 更快。** RTX 4090 上，decode attention 内核比 FlashInfer 的 bf16 内核快 1.19–1.24 倍，实测的 12 种配置全部更快（3 个模型，4K–128K 上下文，batch 1–32）。
- **KV 显存少 20%，能多放 25% 的 token。** Qwen2.5-7B 在一张 24 GB 卡上，能缓存的 token 从约 12.6 万增加到约 15.8 万。
- **解码吞吐最高 1.23 倍**（同一张卡，因为能同时服务更多序列）。
- **小巧好改。** 约 600 行 Python + Triton：既可以只用缓存加一个 attention 调用接进自己的解码循环，也可以直接用现成的批量解码引擎跑 Hugging Face 的 Qwen、Llama 模型。

| Qwen2.5-7B-Instruct，RTX 4090 24 GB，提示 4096 token + 生成 256 token | bf16 KV + FlashInfer | **exactkv** |
|---|---|---|
| 每个序列的 KV | 238 MiB | **190 MiB** |
| 能同时放下的序列数 | 28 | **36** |
| 解码吞吐 | 1064 token/秒 | **1308 token/秒（1.23 倍）** |
| 贪心解码 logits 与不压缩相比 | | **逐位相同** |

## 为什么能更快

解码受显存带宽限制：每一步都要把整个 KV 缓存读一遍，张量核大部分时间在等数据。exactkv 少读约 20% 的字节，多出来的解码计算放在本来空闲的运算单元上做，所以虽然每个数都要解码，attention 反而更快。

## 原理

bf16 每个数 = 1 位符号 + 8 位指数 + 7 位尾数。符号和尾数近乎随机，原样保留。指数很浪费：同一个通道里连续 16 个 token 的数量级差不多。

例子：某一页（16 个 token）某个通道的指数在 106 到 121 之间。exactkv 只存一次基准 121 − 15 = 106，每个数只存"指数 − 基准"（4 位），旁边是它原来的符号 + 尾数（1 字节）。每个数 8 + 4 位，加上基准摊下来的 8/16 位，约 12.6 位，原来是 16 位。还原时把基准加回去，得到原来的 16 位，一位不差。

真实模型上靠两个细节：
- RoPE 之后约 0.04% 的 key 恰好等于 0（两个舍入后的项正好抵消）。含 0 的组用码 0 表示"零"，由基准字节的最高位标记。没有这一招，约 5% 的行会放不下。
- 如果某一行（一个 token、一个头）里有数比组内最大值小 2^15 倍以上，这一行按原始 bf16 存，真实模型上占 0.05–0.6%。预留空间不够时会自动扩容，任何数据都不会丢。

完整格式见 [`exactkv/codec.py`](exactkv/codec.py)，内核见 [`exactkv/kernels.py`](exactkv/kernels.py)。

## 结果（RTX 4090 24 GB，Triton 3.7.1，PyTorch 2.13，FlashInfer 0.6.18）

**decode attention 内核。** KV 取自各模型中间层的真实数据（4096 token 的 prefill），平铺到下表规模。FlashInfer 读不压缩的 bf16（`BatchDecodeWithPagedKVCacheWrapper`，张量核路径），exactkv 读压缩数据。每个 exactkv 输出都和不压缩参照比过，12 组全部逐位相同。

| batch × 上下文 | Qwen2.5-7B（28 个 q 头 / 4 个 kv 头） | Qwen3-8B（32 / 8） | Llama-3.1-8B（32 / 8） |
|---|---|---|---|
| 1 × 32K | 106.4 → **86.8 µs（1.23x）** | 201.3 → **163.5 µs（1.23x）** | 200.7 → **163.2 µs（1.23x）** |
| 1 × 128K | 342.8 → **286.3 µs（1.20x）** | 617.4 → **508.4 µs（1.21x）** | 616.7 → **507.9 µs（1.21x）** |
| 8 × 8K | 200.4 → **162.3 µs（1.23x）** | 342.9 → **287.5 µs（1.19x）** | 342.6 → **287.1 µs（1.19x）** |
| 32 × 4K | 343.9 → **286.6 µs（1.20x）** | 613.6 → **506.8 µs（1.21x）** | 613.7 → **506.4 µs（1.21x）** |

含原始行预留空间：每个数 12.7 位（−20%）。

**单卡端到端解码。** 同一个解码程序、同一份权重，只有 KV 缓存和 decode attention 不同。提示 4096 token，贪心生成 256 token，KV 预算 = 载入权重后的空闲显存 − 1.6 GiB。

| 模型 | 每个序列的 KV | 能放下的序列数 | 解码吞吐 |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 238 → **190 MiB** | 28 → **36** | 1064 → **1308 token/秒（1.23 倍）** |
| Qwen3-8B | 612 → **488 MiB** | 9 → **11** | 313 → **365 token/秒（1.17 倍）** |
| Llama-3.1-8B-Instruct | 544 → **434 MiB** | 11 → **14** | 427 → **525 token/秒（1.23 倍）** |

吞吐的提升来自同样显存放下更多序列。batch 相同时，每步速度与 bf16 相当（Qwen2.5、Llama-3.1 快 1.04 倍；Qwen3-8B 为 0.95 倍，这个最小的 Python 循环在该规模下受 CPU 发射内核的速度限制，CUDA Graph 已在计划中）。

## 正确性

`python tests/test_exactkv.py` 在 GPU 上检查：
- 全部 65,536 种 bf16 位模式（0、次正规数、±inf、NaN）和仿真数据，经 PyTorch 参照实现和 GPU 内核编码再解码，逐位不变；
- GPU 编码器写出的字节和参照实现完全一样；
- 读压缩缓存的 decode attention 与不压缩参照逐位相同（4 种形状、GQA 比 1–8、最后一页不满、两种 warp 数）；
- 原始行池多次溢出的缓存仍然精确；
- 设置 `EXACTKV_MODEL=模型路径` 时：完整模型 3 条提示 × 40 步贪心解码，压缩与不压缩的 token 相同、logits 逐位相同。已在 Qwen2.5-7B-Instruct、Qwen3-8B、Llama-3.1-8B-Instruct 上验证。

"逐位相同"指：开关压缩不改变输出的任何一位。exactkv 的不压缩参照内核按同样的顺序做同样的运算。不同的 attention 内核（exactkv 的参照、FlashInfer 等）彼此之间本来就差最后几位。

## 安装

```bash
pip install git+https://github.com/zyfyz666/exactkv
```

需要 NVIDIA GPU、PyTorch、Triton。引擎需要 Transformers；FlashInfer 只在跑基准时需要。

## 用法

```python
import torch, exactkv
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16, device_map="cuda")
eng = exactkv.Engine(model, B=8, T_max=4096 + 256, backend="exactkv")   # 或 backend="bf16"
tokens, logits = eng.generate(prompt_ids, n_new=256)                    # prompt_ids: [8, 4096]
```

接进自己的解码循环见英文版 [Usage](README.md#usage)。

## 工程亮点

- **从构造上保证逐位相同。** 不压缩参照内核按同样的顺序做同样的运算，连 Triton 选的张量核操作数布局都一致，测试逐位比较两者。
- **在用的地方解码。** 数值在矩阵乘之前于寄存器里还原，从不写回显存。实测：先把（3 位格式的）压缩缓存解压到 bf16 缓冲区再跑 FlashInfer，比完全不压缩还慢 1.8–2.6 倍。融合才把"少读字节"变成"更快"。
- **选最快的格式，不是最密的。** 更密的 3 位码存得更少（每个数 11.7 位），但在 attention 里解码太费指令，只和 FlashInfer 打平。4 位布局每个数约十条整数运算就能还原，是我们找到的最快的格式。
- **不用校准。** 基准指数在写入每一页时按"页 × 头 × 通道"现场计算，适应任何模型和输入。
- **自愈。** 原始行超出预留时，池自动扩容并重新编码该页。整个模型每 16 步只需一次主机同步来检查。

## 适用场景

- 同样的输出，每张卡服务更多用户或更长上下文；
- 24 GB 消费级显卡上跑长上下文；
- 强化学习 rollout、评测、回归测试等输出不能变的场合。

## 目前支持与下一步

目前：NVIDIA GPU（Triton，已在 RTX 4090 测试），bf16 KV，head dim 128，全注意力；Qwen2 / Qwen2.5、Qwen3、Llama-3 系列；同一 batch 长度相同、连续缓存。

下一步：引擎加 CUDA Graph；分页 KV、接入 vLLM 和 SGLang；滑动窗口与其他 head 大小；更多 GPU；解码更便宜后换更密的格式。

## 复现

```bash
python tests/test_exactkv.py                                    # 端到端测试需设 EXACTKV_MODEL
python bench/attention.py --model 模型路径 --out results/attention.json
python bench/e2e.py --model 模型路径 --out results/e2e.json
```

## 许可

Apache-2.0。
