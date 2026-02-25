import torch
from flash_attn import flash_attn_func
import torch.nn.functional as F

# Setup tensors (must be fp16 or bf16)
q = torch.randn(2, 512, 8, 128, dtype=torch.bfloat16, device="cuda")
k = torch.randn(2, 512, 8, 128, dtype=torch.bfloat16, device="cuda")
v = torch.randn(2, 512, 8, 128, dtype=torch.bfloat16, device="cuda")

# Flash Attention
output_flash = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)

# Regular Attention Computation
def regular_attention(q, k, v, causal=True):
    # q, k, v: [batch, seqlen, num_heads, head_dim]
    batch, seqlen, num_heads, head_dim = q.shape
    
    # Reshape to [batch, num_heads, seqlen, head_dim]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Compute attention scores: Q @ K^T
    # [batch, num_heads, seqlen, head_dim] @ [batch, num_heads, head_dim, seqlen]
    # -> [batch, num_heads, seqlen, seqlen]
    scale = head_dim ** -0.5
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Apply causal mask
    if causal:
        causal_mask = torch.triu(torch.ones(seqlen, seqlen, device=q.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
    
    # Apply softmax
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    # Apply attention to values: attn_weights @ V
    # [batch, num_heads, seqlen, seqlen] @ [batch, num_heads, seqlen, head_dim]
    # -> [batch, num_heads, seqlen, head_dim]
    output = torch.matmul(attn_weights, v)
    
    # Reshape back to [batch, seqlen, num_heads, head_dim]
    output = output.transpose(1, 2)
    
    return output

output_regular = regular_attention(q, k, v, causal=True)

# Compare results
print("Flash Attention output shape:", output_flash.shape)
print("Regular Attention output shape:", output_regular.shape)

# Check if outputs are close (allowing for numerical differences)
max_diff = torch.max(torch.abs(output_flash - output_regular)).item()
mean_diff = torch.mean(torch.abs(output_flash - output_regular)).item()
relative_diff = mean_diff / torch.mean(torch.abs(output_regular)).item()

print(f"\nMax absolute difference: {max_diff:.6f}")
print(f"Mean absolute difference: {mean_diff:.6f}")
print(f"Relative difference: {relative_diff:.6f}")
