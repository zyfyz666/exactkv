"""KV caches with one interface: `prefill_slot(b, x)`, `append(x)`, `T`, `nbytes()`.

CompressedKV keeps one layer's keys (or values) in the format of codec.py. Full pages
are encoded on the GPU as they fill. The newest, partly filled page is held as bf16 in
a fixed area at the end of the raw pool, with all its rows marked raw, so attention
reads it like any other page. The raw pool layout is [rows that did not fit | tail area].

Rows that do not fit are rare (0.05-0.6% of rows on Qwen2.5/Qwen3/Llama-3.1), but there
is no hard bound, so the pool grows when needed: after each page the counter is checked
(`check_overflow`, one host sync per 16 decode steps for the whole model) and an
overflowing page is encoded again into a larger pool. Data is never silently dropped.
"""
import torch
from .kernels import encode_pages, P


class CompressedKV:
    def __init__(self, B, TS, H, D, device="cuda", reserve=0.01):
        """B sequences of up to TS tokens (multiple of 16), H KV heads, head dim D.
        reserve: initial raw-row capacity as a fraction of all rows (grows on demand)."""
        assert TS % P == 0 and D == 128, "D = 128 only (Qwen2/2.5/3, Llama-3 family)"
        NP = TS // P
        self.B, self.TS, self.H, self.D, self.dev = B, TS, H, D, device
        self.sm = torch.zeros(B, TS, H, D, dtype=torch.uint8, device=device)
        self.code = torch.zeros(B, TS, H, D // 2, dtype=torch.uint8, device=device)
        self.base = torch.zeros(B, NP, H, D, dtype=torch.uint8, device=device)
        self.rowmask = torch.zeros(B, NP, H, dtype=torch.int32, device=device)
        self.rowoff = torch.zeros(B, NP, H, dtype=torch.int32, device=device)
        self.cnt = torch.zeros(1, dtype=torch.int32, device=device)
        self._before = torch.zeros(1, dtype=torch.int32, device=device)   # counter before the last page
        self.cap = max(64, int(B * TS * H * reserve))
        self.raw = torch.zeros(self.cap + B * H * P, D, dtype=torch.bfloat16, device=device)
        self.T = 0
        self.grows = 0
        self._pending = None                                              # page waiting for check_overflow
        self._refresh()

    # -- internals
    def _refresh(self):
        """Views and kernel arguments that depend on the raw pool; rebuilt only when it grows (hot path)."""
        self._tailv = self.raw[self.cap:].view(self.B, self.H, P, self.D)
        self._toff = (self.cap + torch.arange(self.B * self.H, device=self.dev, dtype=torch.int32) * P).view(self.B, self.H)
        self._kargs = (self.sm, self.code, self.base, self.rowmask, self.rowoff, self.raw.view(torch.int16))

    def kernel_args(self):
        return self._kargs

    def _tail_off(self):
        return self._toff

    def _tail(self):
        return self._tailv

    def _open_tail(self, pg):
        self.rowmask[:, pg] = (1 << P) - 1
        self.rowoff[:, pg] = self._toff

    def _buf(self):
        return dict(sm=self.sm, code=self.code, base=self.base, rowmask=self.rowmask, rowoff=self.rowoff,
                    raw=self.raw, cnt=self.cnt, cap=self.cap, TS=self.TS, H=self.H, D=self.D)

    def _grow(self, need):
        new_cap = max(2 * self.cap, need + self.cap // 4)
        raw = torch.zeros(new_cap + self.B * self.H * P, self.D, dtype=torch.bfloat16, device=self.dev)
        raw[: self.cap] = self.raw[: self.cap]
        raw[new_cap:] = self.raw[self.cap:]
        self.raw, self.cap = raw, new_cap
        self.grows += 1
        self._refresh()
        if self.T < self.TS:
            self.rowoff[:, self.T // P] = self._tail_off()                # the open tail page moved

    def _encode(self, src, strides, b0, pg0, nb, npg):
        self._before.copy_(self.cnt)
        encode_pages(src, strides, self._buf(), b0, pg0, nb, npg)

    def _redo(self, src, strides, b0, pg0, nb, npg):
        need = int(self.cnt.item())
        if need <= self.cap:
            return
        self._grow(need)
        self.cnt.copy_(self._before)
        encode_pages(src, strides, self._buf(), b0, pg0, nb, npg)
        assert int(self.cnt.item()) <= self.cap

    # -- interface
    def prefill_slot(self, b, x):
        """x: [1, T, H, D] bf16 for sequence slot b. All slots must have the same T."""
        T = x.shape[1]
        Tf = T // P * P
        if Tf:
            st = (x.stride(0), P * x.stride(1), x.stride(1), x.stride(2))
            self._encode(x, st, b, 0, 1, Tf // P)
            self._redo(x, st, b, 0, 1, Tf // P)                            # prefill may sync
        if Tf < self.TS and b == 0:
            self._open_tail(Tf // P)
        self._tail()[b, :, : T - Tf] = x[0, Tf:].transpose(0, 1)
        self.T = T

    def append(self, x):
        """x: [B, H, D] bf16, one new token per sequence."""
        assert self._pending is None, "call check_overflow() after a page boundary before appending again"
        self._tailv[:, :, self.T % P] = x
        self.T += 1
        if self.T % P == 0:
            pg = self.T // P - 1
            t = self._tailv
            st = (t.stride(0), 0, t.stride(2), t.stride(1))
            self._encode(t, st, 0, pg, self.B, 1)
            self._pending = (st, pg)
            if self.T < self.TS:
                self._open_tail(pg + 1)

    def finish_page(self, overflowed):
        """Called by check_overflow with this cache's result."""
        if self._pending is not None and overflowed:
            st, pg = self._pending
            self._redo(self._tail(), st, 0, pg, self.B, 1)
        self._pending = None

    def buffers(self):
        return dict(shape=(self.B, self.TS, self.H, self.D), sm=self.sm, code=self.code, base=self.base,
                    rowmask=self.rowmask, rowoff=self.rowoff, raw=self.raw)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.sm, self.code, self.base, self.rowmask, self.rowoff, self.raw))


class BF16KV:
    """Plain bf16 cache [B, TS, H, D] with the same interface (the reference)."""

    def __init__(self, B, TS, H, D, device="cuda"):
        self.buf = torch.zeros(B, TS, H, D, dtype=torch.bfloat16, device=device)
        self.T = 0
        self.cnt = None

    def prefill_slot(self, b, x):
        self.buf[b, : x.shape[1]] = x[0]
        self.T = x.shape[1]

    def append(self, x):
        self.buf[:, self.T] = x
        self.T += 1

    def nbytes(self):
        return self.buf.numel() * 2


def check_overflow(caches):
    """After a page boundary: one host sync for all compressed caches, then re-encode any
    page that overflowed its raw pool. Returns the number of caches that had to grow."""
    cs = [c for c in caches if isinstance(c, CompressedKV) and c._pending is not None]
    if not cs:
        return 0
    over = (torch.cat([c.cnt for c in cs]).cpu() > torch.tensor([c.cap for c in cs], dtype=torch.int32)).tolist()
    for c, o in zip(cs, over):
        c.finish_page(o)
    return sum(over)
