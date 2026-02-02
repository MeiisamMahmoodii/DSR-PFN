import torch
import torch.nn as nn

# Removed: compute_h_loss logic as per Scientist-Engineer Refactor
# -------------------------------------------------------------
# New Upgrades for Phase 4 (Contrastive Learning)
def compute_h_loss(adj_matrix):
    """
    Differentiable acyclicity constraint (H-function).
    h(A) = tr(exp(A*A)) - d = 0 iff G is a DAG.
    Args:
        adj_matrix: (B, N, N) adjacency probabilities or logits.
    """
    B, N, _ = adj_matrix.shape
    device = adj_matrix.device
    
    # We use the Hadamard product (A*A) to ensure non-negativity
    # and use the simple matrix exponential trace property.
    A = torch.sigmoid(adj_matrix)
    
    # h(A) = tr(exp(A)) - N
    # We use a Taylor expansion or torch.matrix_exp for precision
    # Since N is typically < 100, matrix_exp is efficient enough.
    
    res = torch.matrix_exp(A * A)
    h = torch.stack([torch.trace(res[i]) for i in range(B)]) - N
    
    return h.mean()

def contrastive_invariance_loss(base_emb, int_emb, intervention_mask, dag_structure=None):
    """
    Minimizes latent distance between Base and Intervened states 
    for nodes that are NOT the intervention target (Invariance).
    
    Args:
        base_emb: (B, N, D)
        int_emb: (B, N, D)
        intervention_mask: (B, N) - 1.0 if intervened
        dag_structure: Optional (B, N, N) ground truth to exempt descendants.
                       If None, we use simple non-intervened invariance.
    """
    # Euclidean distance in latent space
    diff = torch.norm(base_emb - int_emb, dim=-1) # (B, N)
    
    # Mask: Nodes that are NOT intervened
    invariance_mask = (1.0 - intervention_mask)
    
    # Minimize change for non-intervened nodes (Soft constraint)
    loss = (diff * invariance_mask).mean()
    return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

def causal_loss_fn(pred_delta, true_delta, pred_adj, true_adj, 
                   lambda_delta=100.0, lambda_dag=0.0, lambda_h=0.0, lambda_l1=0.0,
                   loss_type='bce'): # Added loss_type
    

    
    # Robust Regression: Huber Loss with Scale Normalization
    # We normalize by the magnitude of the targets to prevent loss explosion on large deltas.
    with torch.no_grad():
        # Minimum scale of 1.0 to avoid magnification of small errors and division by zero
        scale = torch.mean(torch.abs(true_delta))
        scale = torch.clamp(scale, min=1.0) 
        
    loss_delta = nn.functional.huber_loss(pred_delta / scale, true_delta / scale, delta=1.0)
    
    # Removed: Loss clamping that was hiding real failures
    
    # Phase 5: Unified Learning (Structure Enabled)
    # 1. DAG Construction Loss
    loss_dag = torch.tensor(0.0, device=pred_adj.device, dtype=pred_adj.dtype)
    
    if loss_type == 'focal':
        # Focal Loss (Better for sparse graphs, focuses on hard examples)
        # alpha=0.25 (balance), gamma=2.0 (focus)
        fl = FocalLoss(alpha=0.25, gamma=2.0)
        loss_dag = fl(pred_adj, true_adj)
    else:
        # Standard BCE with Dynamic Imbalance Correction
        num_pos = true_adj.sum()
        num_total = true_adj.numel()
        num_neg = num_total - num_pos
        
        # ISSUE 17: Pos_weight Edge Cases (num_pos=0)
        if num_pos == 0:
            pos_weight = torch.tensor(1.0, device=pred_adj.device, dtype=pred_adj.dtype)
        else:
            pos_weight = num_neg / (num_pos + 1e-6)
            # ISSUE 5: Clamping to 20.0 was too restrictive for sparse graphs (e.g. 50 nodes, 1% density -> 100x ratio)
            # Relaxed constraint to [1.0, 100.0]
            pos_weight = torch.clamp(pos_weight, min=1.0, max=100.0)
        
        loss_dag = nn.functional.binary_cross_entropy_with_logits(
            pred_adj, 
            true_adj,
            pos_weight=pos_weight
        )
    
    # Removed: Loss clamping that was hiding real failures
    
    # 2. Acyclicity Loss (H-Score)
    # We need Probabilities for H-score (clamp to [0,1] approximation naturally via sigmoid)
    # But NO-TEARS usually operates on W^2, where W is adjacency weights.
    # Here we use sigmoid(logits).
    adj_prob = torch.sigmoid(pred_adj)
    
    # ISSUE 20: Device Mismatch / Dtype safety
    loss_h = torch.tensor(0.0, device=pred_adj.device, dtype=pred_adj.dtype)
    if lambda_h > 0:
        loss_h = compute_h_loss(pred_adj)
        
        # Removed: Loss clamping that was hiding real failures

    loss_l1 = torch.tensor(0.0, device=pred_adj.device, dtype=pred_adj.dtype)
    if lambda_l1 > 0:
        # User Feedback Fix: Naive L1(probs) targets 0 density (empty graph).
        # We change this to L1 Error (MAE) against True Adjacency.
        # This encourages sparsity ONLY where true_adj is 0, and encourages edges where true_adj is 1.
        # It acts as a linear complement to BCE.
        loss_l1 = nn.functional.l1_loss(adj_prob, true_adj)
        # Removed: Loss clamping that was hiding real failures
        
    total_loss = (loss_delta * lambda_delta) + (loss_dag * lambda_dag) + (loss_h * lambda_h) + (loss_l1 * lambda_l1)
    
    # Removed: Loss clamping that was hiding real failures
    
    return total_loss, {
        "delta": loss_delta.item() if loss_delta.item() == loss_delta.item() else 0.0, 
        "dag": loss_dag.item() if loss_dag.item() == loss_dag.item() else 0.0, 
        "h": loss_h.item() if isinstance(loss_h, torch.Tensor) else loss_h
    }

def mcm_loss_fn(pred_values, true_values, mask_indices):
    """
    Masked Causal Modeling Loss.
    """
    # Only calculate loss on masked tokens
    loss = nn.functional.mse_loss(pred_values[mask_indices], true_values[mask_indices])
    return loss
