import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        # [n_tokens, hidden_dim]
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Normalize: 
        # ret = x / sqrt(mean(x²) + eps) * self.weight
        orig_dtype = x.dtype
        x = x.float()
        # [n_tokens, 1]
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # [n_tokens, hidden_dim]
        x.mul_(torch.rsqrt(var + self.eps))
        # [n_tokens, hidden_dim]
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        # [n_tokens, hidden_dim]
        x: torch.Tensor,
        # [n_tokens, hidden_dim]
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Normalize: 
        # sum = x + residual
        # ret = sum / sqrt(mean(sum²) + eps) * self.weight
        orig_dtype = x.dtype
        # [n_tokens, hidden_dim]
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        # [n_tokens, 1]
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # [n_tokens, hidden_dim]
        x.mul_(torch.rsqrt(var + self.eps))
        # [n_tokens, hidden_dim]
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        # [n_tokens, hidden_dim]
        x: torch.Tensor,
        # [n_tokens, hidden_dim]
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
