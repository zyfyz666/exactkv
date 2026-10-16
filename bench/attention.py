"""Decode-attention kernel benchmark: exactkv (compressed KV) vs FlashInfer (bf16 KV).

With --model, keys/values come from a real prefill of that model (one middle layer, tiled to
the benchmark sizes); otherwise from synthetic data with per-channel scales.
  python bench/attention.py --model /path/to/Qwen2.5-7B-Instruct --out results/attention.json
"""
import argparse
import json
import os
import sys

import torch
from triton.testing import do_bench

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import exactkv  # noqa: E402
from exactkv import kernels  # noqa: E402
from baselines import FlashInferBF16  # noqa: E402


def real_kv(path, T=4096):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    from e2e import prompts
    ids = prompts(tok, 1, T)
    with torch.no_grad():
        pkv = model(ids, use_cache=True).past_key_values
    i = model.config.num_hidden_layers // 2
    lay = pkv.layers[i] if hasattr(pkv, "layers") else None
    k, v = (lay.keys, lay.values) if lay is not None else (pkv.key_cache[i], pkv.value_cache[i])
    Hq = model.config.num_attention_heads
    del model, pkv
    torch.cuda.empty_cache()
    T0 = k.shape[2] // 16 * 16
    return k[0, :, :T0].transpose(0, 1).contiguous(), v[0, :, :T0].transpose(0, 1).contiguous(), Hq


def synthetic_kv(Hkv=8, T=4096):
    g = torch.Generator(device="cuda").manual_seed(0)
    s = torch.exp(torch.randn(Hkv, 128, device="cuda", generator=g) * 0.7)
    return ((torch.randn(T, Hkv, 128, device="cuda", generator=g) * s).to(torch.bfloat16),
            torch.randn(T, Hkv, 128, device="cuda", generator=g).to(torch.bfloat16), 32)


def tile(x, B, T):
    return x.repeat(-(-B * T // x.shape[0]), 1, 1)[: B * T].reshape(B, T, *x.shape[1:]).contiguous()


def compress(x):
    B, T, H, D = x.shape
    c = exactkv.CompressedKV(B, T, H, D)
    for b in range(B):
        c.prefill_slot(b, x[b:b + 1])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--out")
    a = ap.parse_args()
    K0, V0, Hq = real_kv(a.model) if a.model else synthetic_kv()
    Hkv = K0.shape[1]
    fi = FlashInferBF16()
    rows = []
    for B, T in [(1, 32768), (1, 131072), (8, 8192), (32, 4096)]:
        k, v = tile(K0, B, T), tile(V0, B, T)
        q = (torch.randn(B, Hq, 128, device="cuda") * k.float().std()).to(torch.bfloat16)
        fi.plan(B, T, T, Hq, Hkv, 128)
        t_fi = do_bench(lambda: fi.w.run(q, (fi.paged(k), fi.paged(v))))
        kc, vc = compress(k), compress(v)
        t_ex, best = min((do_bench(lambda: kernels.attention_compressed(q, kc, vc, T, num_warps=nw)), nw)
                         for nw in (4, 8))
        same = torch.equal(kernels.attention_compressed(q, kc, vc, T, num_warps=best).view(torch.int16),
                           kernels.attention_reference(q, k, v, T, num_warps=best).view(torch.int16))
        t_ref = do_bench(lambda: kernels.attention_reference(q, k, v, T, num_warps=best))
        bits = (kc.nbytes() + vc.nbytes()) * 8 / (2 * k.numel())
        r = dict(B=B, T=T, Hq=Hq, Hkv=Hkv, kv_bf16_mib=round(4 * k.numel() / 2**20), bits_per_value=round(bits, 2),
                 flashinfer_us=round(t_fi * 1e3, 1), exactkv_us=round(t_ex * 1e3, 1), reference_us=round(t_ref * 1e3, 1),
                 speedup_vs_flashinfer=round(t_fi / t_ex, 3), bit_identical_to_reference=same)
        print(json.dumps(r), flush=True)
        rows.append(r)
        del k, v, kc, vc
        torch.cuda.empty_cache()
    if a.out:
        json.dump(dict(model=a.model or "synthetic", gpu=torch.cuda.get_device_name(0), rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
