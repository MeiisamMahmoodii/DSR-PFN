"""
Deep Structural Refinement Components for DSR-PFN.

This module implements the core DSR mechanisms:
1. LayerwiseStructuralHead: Predicts graph structure at each transformer layer
2. GraphResidualStream: Propagates and refines graph estimates across layers
3. CausalPriorityAttention: Attention mechanism guided by predicted causal structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerwiseStructuralHead(nn.Module):
    """
    Lightweight structural predictor for each transformer layer.
    Predicts adjacency matrix from layer hidden states.
    
    This is called at EACH layer to progressively refine the graph estimate.
    """
    def __init__(self, d_model, num_nodes):
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_nodes
        
        # Query/Key projection for edge prediction
        self.query = nn.Linear(d_model, d_model // 4)  # Lightweight (1/4 size)
        self.key = nn.Linear(d_model, d_model // 4)
        
        # Optional: Topological ordering hint
        self.ordering_proj = nn.Linear(d_model, 1)
        
    def forward(self, h):
        """
        Args:
            h: Hidden states (B, N, D) or (B*S, N, D)
        Returns:
            adj_logits: (B, N, N) - predicted adjacency matrix logits
        """
        # Handle batched samples
        if h.dim() == 3 and h.shape[1] == self.num_nodes:
            # (B, N, D) case
            Q = self.query(h)  # (B, N, D/4)
            K = self.key(h)    # (B, N, D/4)
            
            # Compute edge scores: Q @ K^T
            scale = (self.d_model // 4) ** -0.5
            adj_logits = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, N, N)
            
            # Add topological ordering bias (optional enhancement)
            ordering_scores = self.ordering_proj(h).squeeze(-1)  # (B, N)
            # Create pairwise ordering bias: encourage i->j if order[i] < order[j]
            order_i = ordering_scores.unsqueeze(2)  # (B, N, 1)
            order_j = ordering_scores.unsqueeze(1)  # (B, 1, N)
            ordering_bias = torch.tanh(order_j - order_i) * 2.0
            
            adj_logits = adj_logits + ordering_bias
            
        else:
            # Handle other shapes gracefully
            adj_logits = None
            
        return adj_logits


class GraphResidualStream(nn.Module):
    """
    Maintains running graph estimate across layers with residual connections.
    
    Implements: G_ℓ = G_ℓ-1 + ΔG_ℓ
    
    Also applies temperature-based sharpening to increase confidence in later layers.
    """
    def __init__(self, num_layers, initial_temperature=1.0):
        super().__init__()
        self.num_layers = num_layers
        self.initial_temperature = initial_temperature
        
        # Learnable residual gate (how much to update vs keep previous)
        self.update_gates = nn.Parameter(torch.ones(num_layers) * 0.5)
        
    def forward(self, prev_graph, delta_graph, layer_idx):
        """
        Args:
            prev_graph: (B, N, N) - previous graph estimate
            delta_graph: (B, N, N) - new prediction from current layer
            layer_idx: int - current layer index
        Returns:
            refined_graph: (B, N, N) - updated graph estimate
        """
        # Learnable residual update
        gate = torch.sigmoid(self.update_gates[layer_idx])
        
        # Combine previous and new estimate
        refined_graph = prev_graph * (1 - gate) + delta_graph * gate
        
        # Temperature sharpening: later layers have lower temperature (sharper predictions)
        # Temperature decreases linearly from initial_temp to 0.1
        progress = layer_idx / max(self.num_layers - 1, 1)
        temperature = self.initial_temperature * (1 - progress * 0.9)
        temperature = max(temperature, 0.1)  # Minimum temperature
        
        # Apply temperature scaling
        refined_graph = refined_graph / temperature
        
        return refined_graph
    
    def initialize_graph(self, batch_size, num_nodes, device):
        """
        Initialize G_0 (prior graph estimate).
        We use a weak uniform prior (all edges equally likely but suppressed).
        """
        # Start with small negative bias (most edges unlikely)
        init_graph = torch.ones(batch_size, num_nodes, num_nodes, device=device) * -2.0
        
        # Zero out diagonal (no self-loops)
        for i in range(num_nodes):
            init_graph[:, i, i] = -100.0  # Strong suppression
            
        return init_graph


class CausalPriorityAttention(nn.Module):
    """
    Attention mechanism with soft-masking based on predicted causal structure.
    
    Encourages node i to attend more to its predicted causal parents.
    This physically grounds the transformer's reasoning process.
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        
        # Standard multi-head attention
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Learnable strength of causal bias
        self.causal_bias_strength = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x, graph_bias=None):
        """
        Args:
            x: (B, N, D) or (B*S, N, D) - input features
            graph_bias: (B, N, N) - predicted adjacency logits (optional)
        Returns:
            out: (B, N, D) - attention output
        """
        B, N, D = x.shape
        
        # Standard attention if no graph bias provided
        if graph_bias is None:
            out, _ = self.mha(x, x, x)
            return out
        
        # Create soft causal mask from graph predictions
        # graph_bias is adjacency logits: high value = edge likely exists
        # We want node j to attend to node i if edge i->j exists
        # So we use graph_bias^T as attention bias
        
        # Convert logits to probabilities
        graph_probs = torch.sigmoid(graph_bias)  # (B, N, N)
        
        # Transpose: graph_probs[i,j] = P(i->j), we want j to attend to i
        causal_mask = graph_probs.transpose(-2, -1)  # (B, N, N)
        
        # Apply learnable strength
        alpha = torch.sigmoid(self.causal_bias_strength)
        causal_mask = alpha * causal_mask + (1 - alpha) * 0.5  # Blend with uniform
        
        # Scale to attention bias range (additive bias in log-space)
        # High probability -> positive bias, low probability -> negative bias
        attention_bias = (causal_mask - 0.5) * 10.0  # Scale to reasonable range
        
        # Expand for multi-head attention
        # MHA expects: (B * num_heads, N, N) for additive bias
        attention_bias = attention_bias.unsqueeze(1).repeat(1, self.nhead, 1, 1)
        attention_bias = attention_bias.view(B * self.nhead, N, N)
        
        # Apply attention with causal bias
        out, _ = self.mha(x, x, x, attn_mask=attention_bias, need_weights=False)
        
        return out


class IterativeStructuralRefinement(nn.Module):
    """
    Complete DSR module that orchestrates iterative graph refinement.
    
    This wraps the transformer backbone and adds:
    - Layer-wise structural heads
    - Graph residual stream
    - Returns graph sequence for analysis
    """
    def __init__(self, num_layers, d_model, num_nodes):
        super().__init__()
        self.num_layers = num_layers
        
        # Create structural head for each layer
        self.structural_heads = nn.ModuleList([
            LayerwiseStructuralHead(d_model, num_nodes)
            for _ in range(num_layers)
        ])
        
        # Graph residual stream
        self.graph_stream = GraphResidualStream(num_layers, initial_temperature=2.0)
        
    def forward(self, hidden_states_sequence, return_all_graphs=False):
        """
        Args:
            hidden_states_sequence: List of (B, N, D) tensors, one per layer
            return_all_graphs: bool - whether to return all intermediate graphs
        Returns:
            final_graph: (B, N, N) - refined adjacency logits
            graph_sequence: List of (B, N, N) if return_all_graphs else None
        """
        batch_size = hidden_states_sequence[0].shape[0]
        num_nodes = hidden_states_sequence[0].shape[1]
        device = hidden_states_sequence[0].device
        
        # Initialize G_0
        graph = self.graph_stream.initialize_graph(batch_size, num_nodes, device)
        
        graph_sequence = [graph] if return_all_graphs else None
        
        # Iterative refinement
        for layer_idx, h in enumerate(hidden_states_sequence):
            # Predict delta graph at this layer
            delta_graph = self.structural_heads[layer_idx](h)
            
            if delta_graph is not None:
                # Update graph estimate
                graph = self.graph_stream(graph, delta_graph, layer_idx)
                
                if return_all_graphs:
                    graph_sequence.append(graph)
        
        return graph, graph_sequence
