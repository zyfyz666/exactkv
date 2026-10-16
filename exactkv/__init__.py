"""exactkv: a lossless (bit-exact) compressed KV cache for LLM decoding, with decode
attention that reads the compressed cache directly."""
from .cache import CompressedKV, BF16KV, check_overflow
from .attention import decode_attention
from .codec import encode_reference, decode_reference, bits_per_value
from .kernels import attention_compressed, attention_reference, decode_to_bf16, encode_pages
from .engine import Engine

__version__ = "0.1.0"
__all__ = ["CompressedKV", "BF16KV", "check_overflow", "decode_attention", "Engine",
           "encode_reference", "decode_reference", "bits_per_value",
           "attention_compressed", "attention_reference", "decode_to_bf16", "encode_pages"]
