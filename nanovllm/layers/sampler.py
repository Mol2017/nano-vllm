import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        """
        Sample token ids from the logits with temperature scaling.
        
        Args:
            logits: [batch_size, vocab_size]
            temperatures: [batch_size]
                
        Returns:
            sample_tokens: [batch_size]
        """
        # [batch_size, vocab_size]]
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        # [batch_size, vocab_size]
        probs = torch.softmax(logits, dim=-1)

        # Introduce Gumbel noise for sampling
        # [batch_size]
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
