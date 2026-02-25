import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # trainable weight
        # [vocab_size/tp_size, hidden_dim]
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(
            self, 
            # [n_tokens]
            x: torch.Tensor
        ):
        if self.tp_size > 1:
            # [n_tokens]
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # Shift token ids to local partition indices.
            # [n_tokens]
            x = mask * (x - self.vocab_start_idx)

        # [n_tokens, hidden_dim]
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # [n_tokens, hidden_dim]
            y = mask.unsqueeze(1) * y
            # Sum across all GPUs.
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(
            self,
            # [n_tokens, hidden_dim]
            x: torch.Tensor
        ):
        # If it's prefill, only compute logits for the last tokens.
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        
        # [n_tokens, vocab_size/tp_size]
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            # [n_tokens, vocab_size]
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
