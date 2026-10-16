# exactkv

**Lossless KV-cache compression for LLM decoding. Smaller cache, faster attention, and not a single bit of output changes.**

![bit-exact](https://img.shields.io/badge/output-bit--exact-brightgreen) ![KV](https://img.shields.io/badge/KV%20cache-%E2%88%9220%25-blue) ![attention](https://img.shields.io/badge/decode%20attention-1.2%C3%97%20vs%20FlashInfer-orange) ![license](https://img.shields.io/badge/license-Apache--2.0-lightgrey)

The usual ways to shrink the KV cache (FP8, INT4, dropping tokens) change what the model says. exactkv doesn't. It keeps the bf16 KV cache compressed in GPU memory and rebuilds every key and value exactly, in registers, right before the tensor-core matmul of decode attention. There is no approximation, no calibration, no retraining, and no decompressed copy.

- **Bit-exact.** Greedy decoding gives identical tokens and bit-identical logits with and without compression, verified end to end on Qwen2.5-7B, Qwen3-8B and Llama-3.1-8B, and on all 65,536 possible bf16 values.
- **Faster than FlashInfer.** The decode-attention kernel is 1.19–1.24x faster than FlashInfer's bf16 kernel on an RTX 4090, in all 12 configurations we measured (3 models, 4K–128K context, batch 1–32).
- **20% less KV memory, 25% more tokens.** Qwen2.5-7B on one 24 GB card holds about 158K cached tokens instead of 126K.
- **Up to 1.23x decode throughput** on the same GPU, from fitting more sequences.
- **Small and hackable.** About 600 lines of Python and Triton: a cache plus one attention call for your own decoding loop, or a ready-made batched engine for Hugging Face Qwen and Llama models.

| Qwen2.5-7B-Instruct, RTX 4090 24 GB, 4096-token prompts + 256 generated | bf16 KV + FlashInfer | **exactkv** |
|---|---|---|
| KV per sequence | 238 MiB | **190 MiB** |
| sequences that fit | 28 | **36** |
| decode throughput | 1064 tok/s | **1308 tok/s (1.23x)** |
| greedy logits vs uncompressed | | **bit-identical** |

## Why it works

Decoding is limited by memory bandwidth: every step reads the whole KV cache, while the tensor cores mostly wait. exactkv reads about 20% fewer bytes and does the extra decoding work on the otherwise idle arithmetic units, so attention gets faster even though every value has to be decoded.

## How it works

A bf16 number is 1 sign bit, 8 exponent bits and 7 mantissa bits. Sign and mantissa look random and are kept as they are. The exponent is wasteful: within one channel, 16 consecutive tokens have similar magnitudes.

Example: one channel of one 16-token page holds exponents between 106 and 121. exactkv stores the base 121 − 15 = 106 once, and for each value only `exponent − base` in 4 bits, next to its original sign + mantissa byte. That is 8 + 4 bits per value plus 8/16 for the base: about 12.6 bits instead of 16. Decoding adds the base back, which gives the original 16 bits exactly.

Two details make it work on real models:
- After RoPE, about 0.04% of keys are exactly 0 (two rounded terms cancel). Groups containing a 0 use code 0 for "zero", flagged by the top bit of the base byte. Without this trick, about 5% of rows would not fit.
- A row (one token of one head) with a value more than 2^15 below its group's maximum is stored as raw bf16: 0.05–0.6% of rows on real models. The raw pool grows by itself when a workload needs more, so nothing is ever dropped.

Format: [`exactkv/codec.py`](exactkv/codec.py). Kernels: [`exactkv/kernels.py`](exactkv/kernels.py).

## Results (RTX 4090 24 GB, Triton 3.7.1, PyTorch 2.13, FlashInfer 0.6.18)

**Decode attention kernel.** Real keys/values from each model's middle layer (a 4096-token prefill), tiled to the sizes below. FlashInfer runs on the uncompressed bf16 cache (`BatchDecodeWithPagedKVCacheWrapper`, tensor-core path); exactkv on the compressed one. Every exactkv output was checked against the uncompressed reference: bit-identical in all 12 cases. [`results/attention_*.json`](results)

| batch × context | Qwen2.5-7B (28 q / 4 kv heads) | Qwen3-8B (32 / 8) | Llama-3.1-8B (32 / 8) |
|---|---|---|---|
| 1 × 32K | 106.4 → **86.8 µs (1.23x)** | 201.3 → **163.5 µs (1.23x)** | 200.7 → **163.2 µs (1.23x)** |
| 1 × 128K | 342.8 → **286.3 µs (1.20x)** | 617.4 → **508.4 µs (1.21x)** | 616.7 → **507.9 µs (1.21x)** |
| 8 × 8K | 200.4 → **162.3 µs (1.23x)** | 342.9 → **287.5 µs (1.19x)** | 342.6 → **287.1 µs (1.19x)** |
| 32 × 4K | 343.9 → **286.6 µs (1.20x)** | 613.6 → **506.8 µs (1.21x)** | 613.7 → **506.4 µs (1.21x)** |

Cache size including the raw-row reserve: 12.7 bits per value (−20%).

**End-to-end decoding on one GPU.** Same engine and weights; only the KV cache and decode attention differ. 4096-token prompts, 256 greedy tokens, KV budget = free memory after loading the weights minus 1.6 GiB. [`results/e2e_*.json`](results)

| model | KV per sequence | sequences that fit | decode throughput |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 238 → **190 MiB** | 28 → **36** | 1064 → **1308 tok/s (1.23x)** |
| Qwen3-8B | 612 → **488 MiB** | 9 → **11** | 313 → **365 tok/s (1.17x)** |
| Llama-3.1-8B-Instruct | 544 → **434 MiB** | 11 → **14** | 427 → **525 tok/s (1.23x)** |

The throughput gain comes from fitting more sequences into the same memory. At equal batch, a decode step runs at about the speed of bf16 (1.04x on Qwen2.5 and Llama-3.1, 0.95x on Qwen3-8B, where this minimal Python loop is limited by launching kernels from the CPU; CUDA graphs are on the roadmap).

## Correctness

`python tests/test_exactkv.py` checks, on the GPU:
- all 65,536 bf16 bit patterns (zeros, subnormals, ±inf, NaNs) and realistic data survive encode → decode bit for bit, both in the PyTorch reference and in the GPU kernels;
- the GPU encoder writes exactly the same bytes as the reference;
- decode attention on the compressed cache equals the uncompressed reference bit for bit (4 shapes, GQA ratios 1–8, partial last pages, two warp counts);
- a cache whose raw pool overflows many times stays exact;
- with `EXACTKV_MODEL=/path/to/model`: 3 prompts × 40 greedy steps through the full model give identical tokens and bit-identical logits with and without compression. Verified on Qwen2.5-7B-Instruct, Qwen3-8B and Llama-3.1-8B-Instruct.

"Bit-identical" means: switching compression on or off changes no bit of the output. exactkv's uncompressed reference kernel does the same arithmetic in the same order. Different attention kernels (exactkv's reference, FlashInfer, others) differ from one another in the last bits, as usual.

## Install

```bash
pip install git+https://github.com/zyfyz666/exactkv
```

Needs an NVIDIA GPU, PyTorch and Triton. Transformers for the engine; FlashInfer only for the benchmarks.

## Usage

```python
import torch, exactkv
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16, device_map="cuda")
eng = exactkv.Engine(model, B=8, T_max=4096 + 256, backend="exactkv")   # or backend="bf16"
tokens, logits = eng.generate(prompt_ids, n_new=256)                    # prompt_ids: [8, 4096]
```

Or use the cache and the kernel in your own decoding loop:

```python
k_cache = exactkv.CompressedKV(B, T_max, n_kv_heads, 128)
v_cache = exactkv.CompressedKV(B, T_max, n_kv_heads, 128)
for b in range(B):                                   # after prefill: keys/values as [1, T, H, D] bf16
    k_cache.prefill_slot(b, k[b:b + 1]); v_cache.prefill_slot(b, v[b:b + 1])
# each decode step, for each layer:
k_cache.append(k_new); v_cache.append(v_new)         # [B, H, D] bf16, after RoPE for keys
out = exactkv.decode_attention(q, k_cache, v_cache)  # q: [B, Hq, D] bf16
# once per step, after all layers (needed only when the step crossed a 16-token page boundary):
exactkv.check_overflow(all_caches)
```

## Engineering highlights

- **Bit-identical by construction.** The uncompressed reference kernel performs the same operations in the same order, down to the tensor-core operand layout Triton chooses, and the tests compare the two bit for bit.
- **Decoded where it is used.** Values are rebuilt in registers just before the matmul and never written back. In our tests, decompressing a (3-bit) compressed cache into a bf16 buffer first and then running FlashInfer was 1.8–2.6x *slower* than not compressing at all. Fusing is what turns fewer bytes into speed.
- **The fast format, not the densest.** Denser 3-bit codes store less (11.7 bits per value) but cost so many instructions to decode inside attention that they only matched FlashInfer. The 4-bit layout decodes in about ten integer operations per value and was the fastest we found.
- **No calibration.** The base exponent is computed per 16-token page, per head and per channel when the page is written, so it adapts to any model and any input.
- **Self-healing.** If a workload produces more raw rows than reserved, the pool grows and the page is encoded again. One host sync per 16 decode steps for the whole model checks this.

## Good for

- serving more users or longer contexts per GPU with exactly the same outputs;
- long-context inference on 24 GB consumer cards;
- RL rollouts, evaluations and regression tests, where outputs must not move.

## Supported today and next

Today: NVIDIA GPUs through Triton (tested on RTX 4090), bf16 KV, head dim 128, full attention; Qwen2 / Qwen2.5, Qwen3 and Llama-3 families; equal-length batches in a contiguous cache.

Next: CUDA graphs in the engine; paged KV and integration with vLLM and SGLang; sliding-window attention and other head sizes; more GPUs; denser formats as decoding gets cheaper.

## Reproduce

```bash
python tests/test_exactkv.py                                    # EXACTKV_MODEL=... for the end-to-end test
python bench/attention.py --model /path/to/model --out results/attention.json
python bench/e2e.py --model /path/to/model --out results/e2e.json
```

## License

Apache-2.0.
