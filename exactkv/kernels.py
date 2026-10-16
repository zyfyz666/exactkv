"""Triton kernels: decode attention over the compressed KV cache, the matching
uncompressed reference, and the page encoder.

Decode attention splits the tokens across programs (split-K). One program handles one
(sequence, KV head) pair and a run of tokens; the G query heads that share the
KV head are packed into one tile, so each KV element is read once for all G
heads. Partial results (running max, running sum, unnormalized output) of the
splits are merged by `_reduce_kernel`.

Bit-identity. The compressed kernel rebuilds every K/V element exactly, then
runs the same arithmetic as the reference kernel. Two details make the outputs
equal bit for bit, not just within rounding:
  * the head dimension is processed as two halves (channels j and j + D/2 share
    one byte of exponent codes), and the reference splits it the same way;
  * Triton picks the tensor-core operand layout (kWidth) from the narrowest
    integer type in a tile's def chain. The compressed tile is assembled from
    uint8 loads, so the reference also assembles bf16 from bytes. Otherwise the
    two kernels would sum products in a different order.
Selections are done with integer masks, not `tl.where` on boolean tensors:
Triton can propagate a boolean into the MMA operand layout and hit a compiler
assertion (Triton 3.7).
"""
import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
P = 16  # tokens per page


# ----------------------------------------------------------------------------- shared pieces

@triton.jit
def _reduce_kernel(Op, Mp, Lp, Out, NSPLIT, D: tl.constexpr, BLOCK_S: tl.constexpr):
    row = tl.program_id(0)  # b * Hq + h
    s = tl.arange(0, BLOCK_S)
    sm = s < NSPLIT
    m = tl.load(Mp + row * NSPLIT + s, mask=sm, other=-float("inf"))
    l = tl.load(Lp + row * NSPLIT + s, mask=sm, other=0.0)
    w = tl.exp2(m - tl.max(m, 0))
    d = tl.arange(0, D)
    o = tl.load(Op + (row * NSPLIT + s)[:, None] * D + d[None, :], mask=sm[:, None], other=0.0)
    tl.store(Out + row * D + d, (tl.sum(o * w[:, None], 0) / tl.sum(l * w, 0)).to(Out.dtype.element_ty))


@triton.jit
def _attn_block(q_lo, q_hi, k_lo, k_hi, v_lo, v_hi, nmask, m_i, l_i, acc_lo, acc_hi, qk_scale):
    """One block of online softmax. Scores use base 2 (qk_scale includes log2 e)."""
    s = tl.dot(q_lo, tl.trans(k_lo))
    s = tl.dot(q_hi, tl.trans(k_hi), s) * qk_scale
    s = tl.where(nmask[None, :], s, -float("inf"))
    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    pb = p.to(v_lo.dtype)
    acc_lo = tl.dot(pb, v_lo, acc_lo * alpha[:, None])
    acc_hi = tl.dot(pb, v_hi, acc_hi * alpha[:, None])
    return m_new, l_i, acc_lo, acc_hi


@triton.jit
def _load_q(Q, b, hk, Hq, G, D: tl.constexpr, BLOCK_H: tl.constexpr):
    h = tl.arange(0, BLOCK_H)
    dh = tl.arange(0, D // 2)
    ptr = Q + (b * Hq + hk * G + h)[:, None] * D + dh[None, :]
    hm = (h < G)[:, None]
    return tl.load(ptr, mask=hm, other=0.0), tl.load(ptr + D // 2, mask=hm, other=0.0)


@triton.jit
def _store_partial(Op, Mp, Lp, b, hk, split, NSPLIT, Hq, G, m_i, l_i, acc_lo, acc_hi,
                   D: tl.constexpr, BLOCK_H: tl.constexpr):
    h = tl.arange(0, BLOCK_H)
    hm = h < G
    row = b * Hq + hk * G + h
    tl.store(Mp + row * NSPLIT + split, m_i, mask=hm)
    tl.store(Lp + row * NSPLIT + split, l_i, mask=hm)
    dh = tl.arange(0, D // 2)
    base = (row * NSPLIT + split)[:, None] * D + dh[None, :]
    tl.store(Op + base, acc_lo, mask=hm[:, None])
    tl.store(Op + base + D // 2, acc_hi, mask=hm[:, None])


# ----------------------------------------------------------------------------- decoding one tile

@triton.jit
def _popc16(x):
    x = x - ((x >> 1) & 0x5555)
    x = (x & 0x3333) + ((x >> 2) & 0x3333)
    x = (x + (x >> 4)) & 0x0F0F
    return (x + (x >> 8)) & 0x1F


@triton.jit
def _dec_half(sm, bb, c, keep, r):
    """Rebuild bf16 bit patterns. bb < 128: exponent = bb + c. bb >= 128 (a group
    containing zeros): c == 0 means exponent 0, else exponent = bb - 128 + c.
    keep: all ones for normal rows, 0 for raw rows; r: the raw value (0 elsewhere)."""
    nz = ((c + 15) >> 4) | (1 - (bb >> 7))           # 0 only for code 0 in a zero group
    e = ((bb & 127) + c) * nz
    bits = (((sm & 0x80) << 8) | (e << 7) | (sm & 0x7F)) & keep
    return (bits | r).to(tl.int16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _load_ctile(SM, CODE, BASE, RMASK, ROFF, RAW, b, hk, offs_n, nmask, TS, H,
                D: tl.constexpr, P: tl.constexpr):
    """Rebuild a [BLOCK_N, D] bf16 tile (as two halves), bit-exact. TS: buffer length in tokens."""
    dh = tl.arange(0, D // 2)
    row = (b * TS + offs_n) * H + hk
    page = (b * (TS // P) + offs_n // P) * H + hk
    m2 = nmask[:, None]
    c = tl.load(CODE + row[:, None] * (D // 2) + dh[None, :], mask=m2, other=0).to(tl.int32)
    sm_lo = tl.load(SM + row[:, None] * D + dh[None, :], mask=m2, other=0).to(tl.int32)
    sm_hi = tl.load(SM + row[:, None] * D + (dh + D // 2)[None, :], mask=m2, other=0).to(tl.int32)
    b_lo = tl.load(BASE + page[:, None] * D + dh[None, :], mask=m2, other=0).to(tl.int32)
    b_hi = tl.load(BASE + page[:, None] * D + (dh + D // 2)[None, :], mask=m2, other=0).to(tl.int32)
    # rows stored raw: their slot in the raw pool = page offset + popcount of earlier raw rows in the page
    rm = tl.load(RMASK + page, mask=nmask, other=0)
    pos = offs_n % P
    raw_row = (rm >> pos) & 1
    ridx = tl.load(ROFF + page, mask=nmask, other=0) + _popc16(rm & ((1 << pos) - 1))
    rmask2 = ((raw_row != 0) & nmask)[:, None]
    keep = (raw_row - 1)[:, None]
    r_lo = tl.load(RAW + ridx[:, None] * D + dh[None, :], mask=rmask2, other=0).to(tl.int32) & 0xFFFF
    r_hi = tl.load(RAW + ridx[:, None] * D + (dh + D // 2)[None, :], mask=rmask2, other=0).to(tl.int32) & 0xFFFF
    return _dec_half(sm_lo, b_lo, c & 15, keep, r_lo), _dec_half(sm_hi, b_hi, c >> 4, keep, r_hi)


@triton.jit
def _ld_bytes(ptr, mask):
    """Load uncompressed bf16 as two bytes (little endian). Same values; it only makes
    Triton choose the same operand layout as the compressed path (see module docstring)."""
    lo = tl.load(ptr, mask=mask, other=0).to(tl.int32)
    hi = tl.load(ptr + 1, mask=mask, other=0).to(tl.int32)
    return ((hi << 8) | lo).to(tl.int16).to(tl.bfloat16, bitcast=True)


# ----------------------------------------------------------------------------- attention kernels

@triton.jit
def _split_c_kernel(Q, SMk, CDk, BSk, RMk, ROk, RWk, SMv, CDv, BSv, RMv, ROv, RWv,
                    Op, Mp, Lp, T, TS, Hq, G, NSPLIT, SPLIT_LEN, qk_scale,
                    D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, P: tl.constexpr):
    bh = tl.program_id(0)
    split = tl.program_id(1)
    Hkv = Hq // G
    b = bh // Hkv
    hk = bh % Hkv
    q_lo, q_hi = _load_q(Q, b, hk, Hq, G, D, BLOCK_H)
    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc_lo = tl.zeros([BLOCK_H, D // 2], tl.float32)
    acc_hi = tl.zeros([BLOCK_H, D // 2], tl.float32)
    start = split * SPLIT_LEN
    end = tl.minimum(start + SPLIT_LEN, T)
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nmask = offs_n < end
        k_lo, k_hi = _load_ctile(SMk, CDk, BSk, RMk, ROk, RWk, b, hk, offs_n, nmask, TS, Hkv, D, P)
        v_lo, v_hi = _load_ctile(SMv, CDv, BSv, RMv, ROv, RWv, b, hk, offs_n, nmask, TS, Hkv, D, P)
        m_i, l_i, acc_lo, acc_hi = _attn_block(q_lo, q_hi, k_lo, k_hi, v_lo, v_hi, nmask,
                                               m_i, l_i, acc_lo, acc_hi, qk_scale)
    _store_partial(Op, Mp, Lp, b, hk, split, NSPLIT, Hq, G, m_i, l_i, acc_lo, acc_hi, D, BLOCK_H)


@triton.jit
def _split_ref_kernel(Q, K, V, Op, Mp, Lp, T, TS, Hq, G, NSPLIT, SPLIT_LEN, qk_scale,
                      D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr):
    """Uncompressed reference: K, V are [B, TS, Hkv, D] bf16 passed as uint8 views."""
    bh = tl.program_id(0)
    split = tl.program_id(1)
    Hkv = Hq // G
    b = bh // Hkv
    hk = bh % Hkv
    q_lo, q_hi = _load_q(Q, b, hk, Hq, G, D, BLOCK_H)
    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc_lo = tl.zeros([BLOCK_H, D // 2], tl.float32)
    acc_hi = tl.zeros([BLOCK_H, D // 2], tl.float32)
    dh = tl.arange(0, D // 2)
    start = split * SPLIT_LEN
    end = tl.minimum(start + SPLIT_LEN, T)
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nmask = offs_n < end
        off = (((b * TS + offs_n) * Hkv + hk)[:, None] * D + dh[None, :]) * 2
        m2 = nmask[:, None]
        k_lo, k_hi = _ld_bytes(K + off, m2), _ld_bytes(K + off + D, m2)
        v_lo, v_hi = _ld_bytes(V + off, m2), _ld_bytes(V + off + D, m2)
        m_i, l_i, acc_lo, acc_hi = _attn_block(q_lo, q_hi, k_lo, k_hi, v_lo, v_hi, nmask,
                                               m_i, l_i, acc_lo, acc_hi, qk_scale)
    _store_partial(Op, Mp, Lp, b, hk, split, NSPLIT, Hq, G, m_i, l_i, acc_lo, acc_hi, D, BLOCK_H)


@triton.jit
def _dec_only_kernel(SM, CD, BS, RM, RO, RW, Out, T, TS, H, D: tl.constexpr, P: tl.constexpr,
                     BLOCK_N: tl.constexpr):
    """Decode the compressed cache back to bf16 [B, T, H, D] (for tests)."""
    bh = tl.program_id(0)
    b = bh // H
    hk = bh % H
    dh = tl.arange(0, D // 2)
    for n0 in range(0, T, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nmask = offs_n < T
        x_lo, x_hi = _load_ctile(SM, CD, BS, RM, RO, RW, b, hk, offs_n, nmask, TS, H, D, P)
        ptr = Out + ((b * T + offs_n) * H + hk)[:, None] * D + dh[None, :]
        tl.store(ptr, x_lo, mask=nmask[:, None])
        tl.store(ptr + D // 2, x_hi, mask=nmask[:, None])


# ----------------------------------------------------------------------------- page encoder

@triton.jit
def _enc_half(u):
    """u: [P, D/2] bf16 bit patterns (int32). Returns the base byte per channel, 4-bit codes,
    and which elements do not fit (their whole row is then stored raw)."""
    e = (u >> 7) & 0xFF
    b = tl.maximum(tl.max(e, 0) - 15, 0)
    is0 = e == 0
    zm = (tl.max(is0.to(tl.int32), 0) > 0) & (b > 0)    # group has zeros the plain window can't hold
    zf = zm[None, :]
    code = tl.where(zf & is0, 0, e - b[None, :])
    lo = tl.where(zf, 1, 0)
    esc = ((code < lo) & ~(zf & is0)) | (b[None, :] >= 128)
    code = tl.minimum(tl.maximum(code, 0), 15)
    base = tl.minimum(tl.where(zm, b + 128, b), 255)
    return base, code, esc


@triton.jit
def _encode_pages_kernel(SRC, s_b, s_pg, s_p, s_h, SM, CODE, BASE, RMASK, ROFF, RAW, CNT, RAW_CAP,
                         b0, pg0, NPG, H, TS, D: tl.constexpr, P: tl.constexpr):
    """One program encodes one page (P tokens) of one KV head. Raw rows get slots in the raw
    pool through an atomic counter; slots at or past RAW_CAP are not written (the caller
    detects that from the counter and re-encodes after growing the pool)."""
    pid = tl.program_id(0)
    hk = pid % H
    pg = (pid // H) % NPG
    bi = pid // (H * NPG)
    b = b0 + bi
    gpg = pg0 + pg
    p = tl.arange(0, P)
    dh = tl.arange(0, D // 2)
    src = SRC + bi * s_b + pg * s_pg + p[:, None] * s_p + hk * s_h
    x_lo = tl.load(src + dh[None, :])
    x_hi = tl.load(src + (dh + D // 2)[None, :])
    u_lo = x_lo.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    u_hi = x_hi.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    bl, cl, el = _enc_half(u_lo)
    bh, ch, eh = _enc_half(u_hi)
    row = (b * TS + gpg * P + p) * H + hk
    tl.store(SM + row[:, None] * D + dh[None, :], (((u_lo >> 8) & 0x80) | (u_lo & 0x7F)).to(tl.uint8))
    tl.store(SM + row[:, None] * D + (dh + D // 2)[None, :], (((u_hi >> 8) & 0x80) | (u_hi & 0x7F)).to(tl.uint8))
    tl.store(CODE + row[:, None] * (D // 2) + dh[None, :], (cl | (ch << 4)).to(tl.uint8))
    page = (b * (TS // P) + gpg) * H + hk
    tl.store(BASE + page * D + dh, bl.to(tl.uint8))
    tl.store(BASE + page * D + D // 2 + dh, bh.to(tl.uint8))
    er = ((tl.max(el.to(tl.int32), 1) + tl.max(eh.to(tl.int32), 1)) > 0).to(tl.int32)
    off = tl.atomic_add(CNT, tl.sum(er, 0))
    tl.store(RMASK + page, tl.sum(er << p, 0))
    tl.store(ROFF + page, off)
    ridx = off + tl.cumsum(er, 0) - er
    wm = ((er != 0) & (ridx < RAW_CAP))[:, None]
    tl.store(RAW + ridx[:, None] * D + dh[None, :], x_lo, mask=wm)
    tl.store(RAW + ridx[:, None] * D + (dh + D // 2)[None, :], x_hi, mask=wm)


# ----------------------------------------------------------------------------- Python wrappers

def choose_split(B, Hkv, T, BLOCK_N, target=512):
    """Number of token splits and split length (a multiple of BLOCK_N). Plain integer math (hot path)."""
    ns = max(1, min(-(-T // BLOCK_N), target // max(1, B * Hkv)))
    split_len = -(-(-(-T // ns)) // BLOCK_N) * BLOCK_N
    return -(-T // split_len), split_len


_WS = {}  # split partials, reused across calls on the same stream (kernels on one stream run in order)


def _prepare(q, T, Hkv, BLOCK_N):
    B, Hq, D = q.shape
    ns, sl = choose_split(B, Hkv, T, BLOCK_N)
    key = (q.device.index, torch.cuda.current_stream().cuda_stream, B * Hq, ns, D)
    ws = _WS.get(key)
    if ws is None:
        f32 = dict(device=q.device, dtype=torch.float32)
        ws = _WS[key] = (torch.empty(B * Hq, ns, D, **f32), torch.empty(B * Hq, ns, **f32), torch.empty(B * Hq, ns, **f32))
    G = Hq // Hkv
    return B, Hq, D, ns, sl, ws, max(16, 1 << (G - 1).bit_length())


def _reduce(q, ns, ws):
    out = torch.empty_like(q)
    B, Hq, D = q.shape
    _reduce_kernel[(B * Hq,)](*ws, out, ns, D=D, BLOCK_S=1 << (ns - 1).bit_length())
    return out


def _cache_args(c):
    """(six tensors, TS, Hkv) from a CompressedKV (cached there) or from its .buffers() dict."""
    if isinstance(c, dict):
        return ((c["sm"], c["code"], c["base"], c["rowmask"], c["rowoff"], c["raw"].view(torch.int16)),
                c["shape"][1], c["shape"][2])
    return c.kernel_args(), c.TS, c.H


def attention_compressed(q, kc, vc, T, BLOCK_N=64, num_warps=4, num_stages=1):
    """q [B, Hq, D] bf16 contiguous; kc, vc: CompressedKV objects (or their .buffers() dicts); T: valid tokens.
    Bit-identical to `attention_reference` called with the same BLOCK_N / num_warps / num_stages."""
    ka, TS, Hkv = _cache_args(kc)
    va, _, _ = _cache_args(vc)
    B, Hq, D, ns, sl, ws, BH = _prepare(q, T, Hkv, BLOCK_N)
    _split_c_kernel[(B * Hkv, ns)](q, *ka, *va, *ws, T, TS, Hq, Hq // Hkv, ns, sl, D ** -0.5 * LOG2E,
                                   D=D, BLOCK_H=BH, BLOCK_N=BLOCK_N, P=P, num_warps=num_warps, num_stages=num_stages)
    return _reduce(q, ns, ws)


def attention_reference(q, k, v, T, BLOCK_N=64, num_warps=4, num_stages=1):
    """Same computation on uncompressed bf16 K, V of shape [B, TS, Hkv, D] (contiguous)."""
    TS, Hkv = k.shape[1], k.shape[2]
    B, Hq, D, ns, sl, ws, BH = _prepare(q, T, Hkv, BLOCK_N)
    _split_ref_kernel[(B * Hkv, ns)](q, k.view(torch.uint8), v.view(torch.uint8), *ws, T, TS, Hq, Hq // Hkv,
                                     ns, sl, D ** -0.5 * LOG2E, D=D, BLOCK_H=BH, BLOCK_N=BLOCK_N,
                                     num_warps=num_warps, num_stages=num_stages)
    return _reduce(q, ns, ws)


def decode_to_bf16(c, T):
    """Compressed buffers -> bf16 [B, T, H, D] through the same tile decoder attention uses."""
    B, TS, H, D = c["shape"]
    out = torch.empty(B, T, H, D, dtype=torch.bfloat16, device=c["sm"].device)
    _dec_only_kernel[(B * H,)](c["sm"], c["code"], c["base"], c["rowmask"], c["rowoff"], c["raw"].view(torch.int16),
                               out, T, TS, H, D=D, P=P, BLOCK_N=64)
    return out


def encode_pages(src, strides, buf, b0, pg0, nb, npg):
    """Encode nb sequences x npg pages. src points at sequence b0's first page to encode;
    strides = (seq, page, token, head) in elements. buf: dict with sm, code, base, rowmask,
    rowoff, raw, cnt, cap, TS, H, D."""
    _encode_pages_kernel[(nb * npg * buf["H"],)](
        src, *strides, buf["sm"], buf["code"], buf["base"], buf["rowmask"], buf["rowoff"], buf["raw"],
        buf["cnt"], buf["cap"], b0, pg0, npg, buf["H"], buf["TS"], D=buf["D"], P=P, num_warps=4)
