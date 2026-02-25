import torch
import subprocess

# GPU info
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Torch info
print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")

# CUDA compiler info
result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
print(result.stdout.strip().split("\n")[-1])

# Test torch on GPU
try:
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    print("✅ Torch GPU compute: OK")
except Exception as e:
    print(f"❌ Torch GPU compute: {e}")

# Test flash attention
try:
    from flash_attn import flash_attn_func
    q = torch.randn(1, 10, 2, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 10, 2, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 10, 2, 64, device="cuda", dtype=torch.float16)
    flash_attn_func(q, k, v)
    torch.cuda.synchronize()
    import flash_attn
    print(f"✅ Flash Attention ({flash_attn.__version__}): OK")
except ImportError:
    print("❌ Flash Attention: not installed")
except Exception as e:
    print(f"❌ Flash Attention: {e}")