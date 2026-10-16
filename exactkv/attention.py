"""Decode attention over either cache type. For one query token per sequence."""
from .cache import CompressedKV, BF16KV
from .kernels import attention_compressed, attention_reference


def decode_attention(q, k_cache, v_cache, **kw):
    """q: [B, Hq, D] bf16 contiguous (after RoPE). k_cache / v_cache: both CompressedKV or both BF16KV,
    already holding the current token. Returns [B, Hq, D] bf16.

    The two cache types give bit-identical results for the same contents and kernel parameters."""
    if isinstance(k_cache, CompressedKV):
        return attention_compressed(q, k_cache, v_cache, k_cache.T, **kw)
    if isinstance(k_cache, BF16KV):
        return attention_reference(q, k_cache.buf, v_cache.buf, k_cache.T, **kw)
    raise TypeError(type(k_cache))
