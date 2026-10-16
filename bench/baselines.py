"""FlashInfer bf16 decode attention as a drop-in for exactkv.Engine (the speed baseline)."""
import torch
import exactkv

P = 16


class FlashInferBF16:
    """Holds the shared FlashInfer wrapper; the cache itself is a plain exactkv.BF16KV."""

    def __init__(self, workspace_mb=256):
        import flashinfer
        self.w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(workspace_mb << 20, dtype=torch.uint8, device="cuda"), "NHD", use_tensor_cores=True)

    def plan(self, B, TS, T, Hq, Hkv, D):
        npg = -(-T // P)
        idx = (torch.arange(B)[:, None] * (TS // P) + torch.arange(npg)[None, :]).flatten().to(torch.int32)
        self.w.plan(torch.arange(B + 1, dtype=torch.int32) * npg, idx,
                    torch.full((B,), T - (npg - 1) * P, dtype=torch.int32), Hq, Hkv, D, P,
                    q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)

    @staticmethod
    def paged(buf):
        B, TS, H, D = buf.shape
        return buf.view(B * TS // P, P, H, D)

    def attn(self, q, kc, vc):
        return self.w.run(q, (self.paged(kc.buf), self.paged(vc.buf)))

    def engine(self, model, B, T_max):
        """An exactkv.Engine whose decode attention is FlashInfer over a bf16 cache."""
        def before(eng, T):
            self.plan(eng.B, eng.TS, T, eng.Hq, eng.Hkv, eng.D)
        return exactkv.Engine(model, B, T_max, backend="bf16", attn=self.attn, before_step=before)
