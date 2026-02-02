import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `forward` faster
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None, position_ids=None):
        # x: [batch, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
            
        if position_ids is not None:
            # Custom position indexing (Permutation Equivariance)
            # position_ids: [batch, seq_len]
            # cached: [1, 1, max_len, dim] -> [max_len, dim]
            cos = self.cos_cached.squeeze(0).squeeze(0)
            sin = self.sin_cached.squeeze(0).squeeze(0)
            
            # Gather: [batch, seq_len] -> [batch, seq_len, dim]
            # Use F.embedding for efficiency
            # Ensure position_ids are clamped to max_len
            position_ids = position_ids.clamp(max=self.max_seq_len_cached - 1)
            
            cos_out = F.embedding(position_ids, cos).unsqueeze(1) # [batch, 1, seq, dim]
            sin_out = F.embedding(position_ids, sin).unsqueeze(1)
            
            return cos_out.to(dtype=x.dtype), sin_out.to(dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    q, k: [batch, heads, seq_len, head_dim]
    cos, sin: [1, 1, seq_len, head_dim]
    """
    # Simply apply
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
