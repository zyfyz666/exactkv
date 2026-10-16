"""The storage format, as a plain PyTorch reference (the GPU encoder in kernels.py
produces the same bytes). Lossless: decode(encode(x)) == x bit for bit, for every
bf16 bit pattern including zeros, subnormals, infinities and NaNs.

A bf16 value is sign (1 bit) | exponent (8) | mantissa (7). Sign and mantissa are
close to random and are kept as they are. The exponent is not: within one channel,
the 16 tokens of a page have similar magnitudes. Per page (16 tokens) x KV head x
channel the format stores one base byte, and per element a 4-bit offset from it:

  sm      uint8 [B, T, H, D]      sign and mantissa, unchanged
  code    uint8 [B, T, H, D/2]    4-bit exponent offsets; one byte holds channels j and j + D/2
  base    uint8 [B, T/16, H, D]   < 128: exponent = base + code (a 16-step window)
                                  >= 128: group with zeros: code 0 means exponent 0,
                                  otherwise exponent = base - 128 + code (15 steps)
  rowmask int32 [B, T/16, H]      bit p set: token p of the page is stored raw
  rowoff  int32 [B, T/16, H]      index of the page's first raw row in `raw`
  raw     bf16  [N_raw, D]        rows (one token, one head) that contain a value more
                                  than 2^15 below its group's maximum

12 bits per value plus 0.5 (base) plus about 0.06 (raw rows and offsets) on real
models: about 12.6 bits instead of 16.

Why the zero group: after RoPE, about 0.04% of keys are exactly 0 (two rounded
terms cancel). Without a code for 0 they would push about 5% of rows to raw storage.
"""
import torch

P = 16


def encode_reference(x):
    """x: [B, T, H, D] bf16, T a multiple of 16. Returns a dict of tensors (raw rows in page order)."""
    B, T, H, D = x.shape
    assert T % P == 0 and x.dtype == torch.bfloat16
    NP = T // P
    u = x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e = (u >> 7) & 0xFF
    sm = (((u >> 8) & 0x80) | (u & 0x7F)).to(torch.uint8)
    ev = e.view(B, NP, P, H, D)
    b = (ev.amax(dim=2) - 15).clamp(min=0)
    zm = (ev == 0).any(dim=2) & (b > 0)
    bf, zf = b[:, :, None], zm[:, :, None]
    code = torch.where(zf & (ev == 0), torch.zeros_like(ev), ev - bf)
    lo = torch.where(zf, torch.ones_like(bf), torch.zeros_like(bf))
    esc = ((code < lo) & ~(zf & (ev == 0))) | (bf >= 128)
    code = code.clamp(0, 15).view(B, T, H, D)
    packed = (code[..., : D // 2] | (code[..., D // 2:] << 4)).to(torch.uint8)
    base = torch.where(zm, b + 128, b).clamp(max=255).to(torch.uint8)
    er = esc.any(dim=-1).permute(0, 1, 3, 2)                       # [B, NP, H, P]
    rowmask = (er.to(torch.int32) << torch.arange(P, device=x.device, dtype=torch.int32)).sum(-1, dtype=torch.int32)
    cnt = er.sum(-1, dtype=torch.int32).flatten()
    rowoff = (torch.cumsum(cnt, 0, dtype=torch.int32) - cnt).view(B, NP, H)
    raw = x.contiguous().view(B, NP, P, H, D).permute(0, 1, 3, 2, 4)[er].contiguous()
    return dict(shape=(B, T, H, D), sm=sm, code=packed, base=base, rowmask=rowmask.contiguous(),
                rowoff=rowoff.contiguous(), raw=raw)


def decode_reference(c):
    """Inverse of encode_reference (expects raw rows in page order, as encode_reference writes them)."""
    B, T, H, D = c["shape"]
    NP = T // P
    code = c["code"].to(torch.int32)
    code = torch.cat([code & 15, code >> 4], dim=-1)
    bb = c["base"].to(torch.int32).repeat_interleave(P, dim=1)
    e = (bb & 127) + code
    e = torch.where((bb >= 128) & (code == 0), torch.zeros_like(e), e)
    sm = c["sm"].to(torch.int32)
    bits = ((sm & 0x80) << 8) | ((e & 0xFF) << 7) | (sm & 0x7F)
    x = bits.to(torch.int16).view(torch.bfloat16).clone()
    xv = x.view(B, NP, P, H, D).permute(0, 1, 3, 2, 4)
    er = ((c["rowmask"][..., None] >> torch.arange(P, device=x.device)) & 1).bool()
    xv[er] = c["raw"]
    return x


def bits_per_value(c):
    n = c["sm"].numel()
    return sum(c[k].numel() * c[k].element_size() for k in ("sm", "code", "base", "rowmask", "rowoff", "raw")) * 8 / n
