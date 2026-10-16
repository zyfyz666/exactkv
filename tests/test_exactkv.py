"""Tests. Needs a CUDA GPU with Triton. Run: python tests/test_exactkv.py (or pytest).
The end-to-end test runs only if EXACTKV_MODEL points at a local Hugging Face model."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import exactkv  # noqa: E402
from exactkv import codec, kernels  # noqa: E402

DEV = "cuda"


def _same(a, b):
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def _gpu_encode(x):
    """Encode [B,T,H,D] with the GPU kernel into fresh buffers (T a multiple of 16)."""
    B, T, H, D = x.shape
    c = exactkv.CompressedKV(B, T, H, D, reserve=1.0)                  # room for every row to be raw
    st = (x.stride(0), 16 * x.stride(1), x.stride(1), x.stride(2))
    kernels.encode_pages(x, st, c._buf(), 0, 0, B, T // 16)
    assert int(c.cnt.item()) <= c.cap
    return c


def _inputs():
    g = torch.Generator(device=DEV).manual_seed(0)
    yield "all 65536 bf16 bit patterns", torch.randperm(65536, device=DEV, generator=g).to(torch.int16) \
        .view(torch.bfloat16).view(4, 128, 1, 128)
    scale = torch.exp(torch.randn(4, 128, device=DEV, generator=g))
    scale[0, 5] = 300.0
    x = (torch.randn(2, 512, 4, 128, device=DEV, generator=g) * scale).to(torch.bfloat16)
    x[0, 7, 1, 3] = 0.0
    x[1, 9, 2, 5] = -0.0
    x[1, 100, 2, :] = 1e-30
    x[0, 200, 3, 64] = float("inf")
    x[1, 300, 0, 1] = float("nan")
    yield "gaussian + outlier channel + zeros/inf/nan", x
    yield "constant", torch.full((1, 64, 2, 128), 1.5, device=DEV, dtype=torch.bfloat16)
    yield "all zeros", torch.zeros(1, 64, 2, 128, device=DEV, dtype=torch.bfloat16)


def test_codec_roundtrip():
    for name, x in _inputs():
        c = codec.encode_reference(x)
        assert _same(codec.decode_reference(c), x), name


def test_gpu_encoder_matches_reference_and_decodes_exactly():
    for name, x in _inputs():
        ref = codec.encode_reference(x)
        c = _gpu_encode(x)
        for k in ("sm", "code", "base", "rowmask"):
            assert torch.equal(getattr(c, k), ref[k]), (name, k)
        assert _same(kernels.decode_to_bf16(c.buffers(), x.shape[1]), x), name


def test_attention_bit_identical():
    g = torch.Generator(device=DEV).manual_seed(1)
    for B, T, Hq, Hkv in [(1, 4096, 28, 4), (3, 1000, 32, 8), (2, 333, 32, 32), (4, 17, 16, 2)]:
        TS = -(-T // 16) * 16
        scale = torch.exp(torch.randn(Hkv, 128, device=DEV, generator=g) * 0.7)
        k = (torch.randn(B, TS, Hkv, 128, device=DEV, generator=g) * scale).to(torch.bfloat16)
        v = torch.randn(B, TS, Hkv, 128, device=DEV, generator=g).to(torch.bfloat16)
        k[:, ::97, :, 3] = 0.0
        q = (torch.randn(B, Hq, 128, device=DEV, generator=g) * 0.3).to(torch.bfloat16)
        kc, vc = _gpu_encode(k), _gpu_encode(v)
        for nw in (4, 8):
            o_c = kernels.attention_compressed(q, kc.buffers(), vc.buffers(), T, num_warps=nw)
            o_r = kernels.attention_reference(q, k, v, T, num_warps=nw)
            assert _same(o_c, o_r), (B, T, Hq, Hkv, nw)
        G = Hq // Hkv
        kf, vf = k[:, :T].double().repeat_interleave(G, 2), v[:, :T].double().repeat_interleave(G, 2)
        o64 = torch.einsum("bht,bthd->bhd", (torch.einsum("bhd,bthd->bht", q.double(), kf) / 128 ** 0.5).softmax(-1), vf)
        assert (o_r.double() - o64).abs().max() < 2e-2


def test_cache_append_and_pool_growth():
    """Many rows that do not fit, a tiny initial pool: the pool must grow and stay exact."""
    g = torch.Generator(device=DEV).manual_seed(2)
    B, H, D, T0, TS = 2, 4, 128, 1000, 1200

    def make(n):
        x = torch.randn(B, n, H, D, device=DEV, generator=g).to(torch.bfloat16)
        x[torch.rand(B, n, H, device=DEV, generator=g) < 0.1, 7] = 1e-30   # forces raw rows
        return x
    c, r = exactkv.CompressedKV(B, TS, H, D, reserve=0.001), exactkv.BF16KV(B, TS, H, D)
    x0 = make(T0)
    for b in range(B):
        c.prefill_slot(b, x0[b:b + 1].contiguous())
        r.prefill_slot(b, x0[b:b + 1].contiguous())
    xs = make(TS - T0)
    q = torch.randn(B, 32, D, device=DEV, generator=g).to(torch.bfloat16)
    for t in range(TS - T0):
        c.append(xs[:, t].contiguous())
        r.append(xs[:, t].contiguous())
        exactkv.check_overflow([c])
        if t % 37 == 0 or t == TS - T0 - 1:
            assert _same(kernels.decode_to_bf16(c.buffers(), c.T), r.buf[:, :c.T]), t
            assert _same(exactkv.decode_attention(q, c, c), exactkv.decode_attention(q, r, r)), t
    assert c.grows > 0


def test_end_to_end_model():
    path = os.environ.get("EXACTKV_MODEL")
    if not path:
        print("  (skipped: set EXACTKV_MODEL to a local Qwen2/Qwen3/Llama-3 checkpoint)")
        return
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=DEV).eval()
    torch.manual_seed(0)
    ids = torch.randint(1000, 20000, (3, 500), device=DEV)
    out = {}
    for be in ("bf16", "exactkv"):
        eng = exactkv.Engine(model, 3, 500 + 40, backend=be)
        out[be] = eng.generate(ids, 40)
        del eng
    assert torch.equal(out["bf16"][0], out["exactkv"][0])
    assert _same(out["bf16"][1], out["exactkv"][1])


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
