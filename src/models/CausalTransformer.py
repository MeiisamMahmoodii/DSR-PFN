import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data.encoder import AdditiveEncoder
from src.models.rope import RotaryEmbedding, apply_rotary_pos_emb
from src.models.structural_refinement import (
    LayerwiseStructuralHead,
    GraphResidualStream,
    IterativeStructuralRefinement
)

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # x can be (B, N, D) or (B, S, N, D)
        return self.scale * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class SwiGLU(nn.Module):
    def __init__(self, d_model, expansion_factor=8/3): # Standard expansion
        super().__init__()
        h_dim = int(d_model * expansion_factor)
        self.w1 = nn.Linear(d_model, h_dim, bias=False) # Gate
        self.w2 = nn.Linear(d_model, h_dim, bias=False) # Value
        self.w3 = nn.Linear(h_dim, d_model, bias=False) # Output
    def forward(self, x):
        # The "Gate" mechanism controls information flow
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))

class TopologicalCausalHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        
        # Predicts "rank" score. 
        # High score = Downstream (Leaf), Low score = Upstream (Root)
        self.ordering_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (Batch, N, D) - Aggregated across S
        
        # 1. Content-based Edge Prediction (Standard Attention)
        Q = self.query(x)
        K = self.key(x)
        scale = x.shape[-1]**-0.5
        content_logits = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # 2. Ordering Constraint
        # Predict rank s for each node
        s = self.ordering_proj(x).squeeze(-1) # (Batch, N)
        
        # Compute pairwise difference: s_i - s_j
        # We want edge i -> j ONLY if i comes before j (rank_i < rank_j)
        # So we penalize if rank_i > rank_j
        s_i = s.unsqueeze(2) # (B, N, 1)
        s_j = s.unsqueeze(1) # (B, 1, N)
        
        # This acts as a "Soft Mask".
        # If (rank_j - rank_i) is positive, allow edge.
        # If (rank_j - rank_i) is negative (j is before i), suppress edge.
        ordering_bias = torch.tanh(s_j - s_i) * 5.0 # Scale for sharpness
        
        # Final Logits = Content + Ordering
        # This encourages the model to agree on an order AND the edges
        return content_logits + ordering_bias

class CausalMoELayer(nn.Module):
    """
    Causal Mixture of Experts (MoE).
    Each expert specializes in a different physical relationship (Linear, Non-linear, etc.)
    """
    def __init__(self, d_model, num_experts=4, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        
        # 1. Experts: Simple FFNs with varying nonlinearities
        # Expert 0: Pure Linear
        # Expert 1: GELU
        # Expert 2: SiLU (Swish)
        # Expert 3: Tanh (for saturated physics)
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU() if i != 0 else nn.Identity(),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout)
            ) for i in range(num_experts)
        ])
        
        # 2. Router: Predicts expert weights per sample/node
        self.router = nn.Sequential(
            nn.Linear(d_model, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x: (..., D)
        weights = self.router(x) # (..., num_experts)
        
        # Compute all expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1) # (..., D, num_experts)
        
        # Combine weighted experts
        out = torch.sum(expert_outputs * weights.unsqueeze(-2), dim=-1)
        return out
class RoPEAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        # Using standard MHA for simplicity and stability
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # Upgrade to Mixture of Experts (MoE) for physics mechanisms
        self.ffn = CausalMoELayer(d_model, num_experts=4, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rotary_emb=None, attn_mask=None):
        B_dims = x.shape[:-2] # (B,) or (B, S)
        N, D = x.shape[-2:]
        
        residual = x
        x = self.norm1(x)
        
        if attn_mask is not None:
            # attn_mask is (B, N, N)
            # If x is (B, S, N, D), we need to handle broadcasting or reshaping
            if x.dim() == 4:
                B, S, N, D = x.shape
                # Reshape for MHA: (B*S, N, D)
                x = x.view(B*S, N, D)
                # Expand mask: (B, N, N) -> (B*S, N, N)
                # Correct way for torch MHA: (num_heads * Batch, N, N)
                attn_mask = attn_mask.unsqueeze(1).expand(B, S, N, N).reshape(B*S, N, N)
                
                # Repeat for heads
                num_heads = self.self_attn.num_heads
                attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)
                
                out, _ = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)
                x = x + self.dropout(out)
                x = x.view(B, S, N, D) # Reshape back
            else:
                out, _ = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)
                x = residual + self.dropout(out)
        else:
            out, _ = self.self_attn(x, x, x, attn_mask=None, need_weights=False)
            x = residual + self.dropout(out)
        
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x

class CausalTransformer(nn.Module):
    """
    Unified In-Context Causal Transformer (ICCT)
    A single-pass backbone that jointly discovers causal structure and predicts interventional effects.
    """
    def __init__(self, num_nodes, d_model=256, nhead=8, num_layers=6, dropout=0.1, enable_dsr=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.enable_dsr = enable_dsr
        
        # 1. Additive Encoder (Efficiency Fix)
        # Encodes (X, X_int, target, mask) into a unified latent space
        self.encoder = AdditiveEncoder(num_nodes, d_model)
        
        # 2. Unified Backbone
        # Combined depth for joint structural/physical reasoning
        self.layers = nn.ModuleList([
            RoPEAttentionLayer(d_model, nhead, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 3. Discovery Head (The Scientist Output)
        # Predicts Adjacency Matrix for structural validation and loss
        self.dag_head = TopologicalCausalHead(d_model)
        
        # 3b. DSR Module (Optional: Deep Structural Refinement)
        if self.enable_dsr:
            self.dsr_module = IterativeStructuralRefinement(
                num_layers=num_layers,
                d_model=d_model,
                num_nodes=num_nodes
            )
        
        # 4. Physics Head (The Engineer Output)
        # Predicts Deltas (Interventional Effects)
        self.physics_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        
    def forward(self, base_samples, int_samples, target_row, int_mask, mcm_mask=None, return_dsr_graphs=False):
        # 1. Encode unified input
        # x: (B, S, N, D)
        x, _ = self.encoder(base_samples, int_samples, target_row, int_mask)
        
        # 2. Execute Unified Transformer Backbone
        # Structure and Physics are learned jointly in these shared representations
        B, S, N, D = x.shape
        x_flat = x.view(B*S, N, D)
        
        # Track hidden states for DSR if enabled
        hidden_states_sequence = []
        
        h = x_flat
        for layer in self.layers:
            h = layer(h)  # (B*S, N, D)
            
            # Store hidden states for DSR
            if self.enable_dsr:
                # Average over samples to get (B, N, D) for graph prediction
                h_for_graph = h.view(B, S, N, D).mean(dim=1)
                hidden_states_sequence.append(h_for_graph)
            
        # 3. Structural Projection (Discovery)
        # Aggregate across context samples S to find the global graph
        h_reshape = h.view(B, S, N, D)
        h_summary = h_reshape.mean(dim=1)  # (B, N, D)
        
        # Use DSR if enabled
        if self.enable_dsr and len(hidden_states_sequence) > 0:
            adj_logits, graph_sequence = self.dsr_module(
                hidden_states_sequence,
                return_all_graphs=return_dsr_graphs
            )
        else:
            adj_logits = self.dag_head(h_summary)  # (B, N, N)
            graph_sequence = None
        
        # 4. Physical Projection (Inference)
        # Predict interventional deltas for each node/sample
        deltas = self.physics_head(h).view(B, S, N)  # (B, S, N)
        
        # Return graph sequence for analysis if requested
        if return_dsr_graphs and graph_sequence is not None:
            return deltas, adj_logits, graph_sequence
        
        return deltas, adj_logits

