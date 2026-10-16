"""A minimal batched greedy decoding loop for Hugging Face Qwen2 / Qwen2.5 / Qwen3 / Llama-3
models. It reuses the model's own weights and modules and only replaces the KV cache and
decode attention, so two engines that differ only in the cache produce identical results
wherever the cache is lossless.

Prefill runs through the model's own forward pass (one sequence at a time, to save memory)
and the resulting keys/values are written into the cache. All sequences in a batch have the
same prompt length.
"""
import importlib
import torch
from .cache import CompressedKV, BF16KV, check_overflow
from .attention import decode_attention

BACKENDS = {"exactkv": CompressedKV, "bf16": BF16KV}


class Engine:
    def __init__(self, model, B, T_max, backend="exactkv", attn=None, before_step=None):
        """backend: 'exactkv' / 'bf16', or a cache class with the same interface.
        attn(q, k_cache, v_cache) and before_step(engine, T) let benchmarks plug in other kernels."""
        cfg = model.config
        self.m = model
        self.L, self.Hq, self.Hkv = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.Hq
        assert cfg.model_type in ("qwen2", "qwen3", "llama"), f"supported: Qwen2/2.5, Qwen3, Llama-3 (got {cfg.model_type})"
        assert not getattr(cfg, "use_sliding_window", False), "sliding-window attention is not supported"
        assert self.D == 128, "head dim 128 only"
        self.B, self.TS = B, -(-T_max // 16) * 16
        cls = BACKENDS.get(backend, backend)
        self.kv = [(cls(B, self.TS, self.Hkv, self.D), cls(B, self.TS, self.Hkv, self.D)) for _ in range(self.L)]
        self.attn = attn or decode_attention
        self.before_step = before_step
        self.rope = importlib.import_module(type(model.model.layers[0].self_attn).__module__).apply_rotary_pos_emb
        self.qk_norm = hasattr(model.model.layers[0].self_attn, "k_norm")   # Qwen3
        self.grows = 0

    @torch.no_grad()
    def prefill(self, ids):
        """ids: [B, T] token ids. Returns the greedy next token of each sequence, [B]."""
        nxt = []
        for b in range(ids.shape[0]):
            out = self.m.model(ids[b:b + 1], use_cache=True)
            pkv = out.past_key_values
            for i in range(self.L):
                lay = pkv.layers[i] if hasattr(pkv, "layers") else None
                k, v = (lay.keys, lay.values) if lay is not None else (pkv.key_cache[i], pkv.value_cache[i])
                self.kv[i][0].prefill_slot(b, k.transpose(1, 2).contiguous())
                self.kv[i][1].prefill_slot(b, v.transpose(1, 2).contiguous())
            nxt.append(self.m.lm_head(self.m.model.norm(out.last_hidden_state[:, -1])).argmax(-1))
            del out, pkv
        return torch.cat(nxt)

    @torch.no_grad()
    def step(self, tok):
        """tok: [B] current tokens. Returns logits [B, vocab] (bf16) for the next token."""
        m, B, Hq, Hkv, D = self.m, self.B, self.Hq, self.Hkv, self.D
        T = self.kv[0][0].T
        if self.before_step:
            self.before_step(self, T + 1)
        h = m.model.embed_tokens(tok.view(B, 1))
        cos, sin = m.model.rotary_emb(h, torch.full((1, 1), T, device=h.device))
        for i, layer in enumerate(m.model.layers):
            att = layer.self_attn
            x = layer.input_layernorm(h)
            q = att.q_proj(x).view(B, 1, Hq, D)
            k = att.k_proj(x).view(B, 1, Hkv, D)
            v = att.v_proj(x).view(B, 1, Hkv, D)
            if self.qk_norm:
                q, k = att.q_norm(q), att.k_norm(k)
            q, k = self.rope(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
            kc, vc = self.kv[i]
            kc.append(k[:, :, 0])
            vc.append(v[:, 0])
            o = self.attn(q[:, :, 0].contiguous(), kc, vc)
            h = h + att.o_proj(o.reshape(B, 1, Hq * D))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        if (T + 1) % 16 == 0:
            self.grows += check_overflow([c for kv in self.kv for c in kv])
        return m.lm_head(m.model.norm(h))[:, 0]

    @torch.no_grad()
    def generate(self, ids, n_new):
        """Greedy decoding: returns (tokens [B, n_new], logits [B, n_new - 1, vocab])."""
        t = self.prefill(ids)
        toks, logits = [t], []
        for _ in range(n_new - 1):
            lg = self.step(t)
            logits.append(lg)
            t = lg.argmax(-1)
            toks.append(t)
        return torch.stack(toks, 1), torch.stack(logits, 1)

    def kv_bytes(self):
        return sum(c.nbytes() for kv in self.kv for c in kv)
