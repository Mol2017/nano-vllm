import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    # [n_tokens, n_kv_heads, head_dim]
    key: torch.Tensor, 
    value: torch.Tensor,
    # [n_blocks, block_size, n_kv_heads, head_dim]
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor,
    # [n_tokens]
    slot_mapping: torch.Tensor
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # Launch 1D kernel with N threads, each thread handles one token.
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(
        self,
        # [n_q_tokens, n_heads, head_dim]
        q: torch.Tensor, 
        # [n_kv_tokens, n_kv_heads, head_dim]
        k: torch.Tensor, 
        v: torch.Tensor
    ):
        context = get_context()
        # [n_blocks, block_size, n_kv_heads, head_dim]
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            
            # [n_q_tokens, n_heads, head_dim]
            o = flash_attn_varlen_func(
                # [n_q_tokens, n_heads, head_dim]
                q, 
                # [n_kv_tokens, n_kv_heads, head_dim]
                k, 
                v,      
                max_seqlen_q=context.max_seqlen_q, 
                # [batch_size + 1]          
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k, 
                # [batch_size + 1]          
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale, 
                block_table=context.block_tables,
                causal=True
            )
        else:    # decode
            # [n_tokens, 1, n_kv_heads, head_dim]
            o = flash_attn_with_kvcache(
                # [n_tokens, 1, n_heads, head_dim]
                q.unsqueeze(1),
                # [n_blocks, block_size, n_kv_heads, head_dim]
                k_cache, 
                v_cache,
                # [batch_size]
                cache_seqlens=context.context_lens,
                # [batch_size, n_blocks]
                block_table=context.block_tables, 
                softmax_scale=self.scale,
                causal=True
            )
        return o
