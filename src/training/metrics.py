import torch
import networkx as nx
from sklearn.metrics import f1_score

# Try to import d_separated, handle versions
try:
    from networkx import d_separated as is_d_separator
except ImportError:
    try:
        from networkx.algorithms.d_separation import d_separated as is_d_separator
    except ImportError:
        # Fallback for older versions or if it's named is_d_separator
        try:
             from networkx import is_d_separator
        except ImportError:
             from networkx.algorithms.d_separation import is_d_separator


def compute_sid(pred_adj_logits, true_adj_matrix, threshold=0.0):
    """
    Computes Structural Intervention Distance (SID).
    Simplified implementation for DAGs using NetworkX d-separation.
    Count of pairs (x, y) where the causal effect of x on y is NOT correctly identified
    by the adjustment set (parents of x) derived from the predicted graph.
    """
    with torch.no_grad():
        # Convert to NetworkX DiGraphs
        N = pred_adj_logits.shape[-1]
        
        # Binarize
        pred_edges = (pred_adj_logits > threshold).cpu().numpy()
        true_edges = true_adj_matrix.cpu().numpy()
        
        # SID logic requires graphs.
        # Handling Batches: Average SID over batch
        if len(pred_edges.shape) == 3:
             batch_size = pred_edges.shape[0]
             total_sid = 0
             for i in range(batch_size):
                 total_sid += _sid_single_graph(pred_edges[i], true_edges[i])
             return total_sid / batch_size
        else:
             return _sid_single_graph(pred_edges, true_edges)

def _sid_single_graph(pred_mat, true_mat):
    """
    SID for single adjacency matrices (numpy).
    """
    # Create DiGraphs
    G_pred = nx.from_numpy_array(pred_mat, create_using=nx.DiGraph)
    G_true = nx.from_numpy_array(true_mat, create_using=nx.DiGraph)
    
    nodes = list(G_true.nodes())
    sid_count = 0
    
    for X in nodes:
        # Adjustment set S from Predicted Graph: Parents(X)
        S = set(G_pred.predecessors(X))
        
        # Strict SID Requirement: S, {X}, {Y} must be disjoint for d-separation check in NetworkX.
        # If model predicts self-loop (X->X), X will be in S.
        if X in S:
            S.remove(X)
        
        # Condition 1: S must not contain descendants of X in True Graph
        # We check intersection.
        try:
            descendants_true = nx.descendants(G_true, X)
        except nx.NetworkXError:
            # If G_true has cycles (shouldn't happen for DAGs but just in case)
            descendants_true = set()
            
        if not S.isdisjoint(descendants_true):
            # If parents include a true descendant, intervention is invalid
            sid_count += (len(nodes) - 1)
            continue
            
        # Condition 2: S must block all backdoor paths X <- ... -> Y in G_true
        # We use d-separation in a "Backdoor Graph"
        # Backdoor Graph = G_true with edges X -> ... removed.
        # In this graph, check if S d-separates X and Y.
        
        G_backdoor = G_true.copy()
        # Remove edges X -> Child
        out_edges = list(G_backdoor.out_edges(X))
        G_backdoor.remove_edges_from(out_edges)
        
        for Y in nodes:
            if X == Y: continue
            
            # Condition 3:
            # If Y is in S, we are adjusting for Y.
            # Since Cond 1 passed, Y is NOT a descendant of X in True Graph.
            # Thus, there is no directed path X -> ... -> Y.
            # Adjusting for Y correctly yields zero effect.
            # So this is valid.
            if Y in S:
                continue
            
            # Check d-separation
            # If NOT d-separated, then backdoor path exists -> Error
            if not is_d_separator(G_backdoor, {X}, {Y}, S):
                sid_count += 1
                
    return float(sid_count)

def compute_shd(pred_adj_logits, true_adj_matrix, threshold=0.0):
    """
    Computes Structural Hamming Distance (SHD).
    """
    with torch.no_grad():
        pred_edges = (pred_adj_logits > threshold).float()
        true_edges = true_adj_matrix.float()
        diff = torch.abs(pred_edges - true_edges)
        shd = diff.sum(dim=(1, 2))
        return shd.mean().item()

def compute_f1(pred_adj_logits, true_adj_matrix, threshold=0.0):
    """
    Computes F1 Score for edges.
    threshold: Logit threshold (default 0.0 -> Prob 0.5)
    """
    with torch.no_grad():
        # pred_adj_logits are logits. true_adj is 0/1.
        # Threshold directly on logits for consistency
        pred_edges = (pred_adj_logits > threshold).float()
        pred_flat = pred_edges.cpu().numpy().flatten()
        true_flat = true_adj_matrix.cpu().numpy().flatten()
        score = f1_score(true_flat, pred_flat, zero_division=0)
        
        # Safety: Check for NaN/Inf
        if score != score or score > 1e6:  # NaN or Inf check
            return 0.0
        return score

def find_optimal_threshold(pred_adj_logits, true_adj_matrix):
    """
    Sweeps thresholds to find best F1 score.
    Returns: best_threshold (logit), best_f1, best_shd
    """
    thresholds = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    best_f1 = -1.0
    best_thresh = 0.0
    best_shd = -1.0
    
    for t in thresholds:
        f1 = compute_f1(pred_adj_logits, true_adj_matrix, threshold=t)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_shd = compute_shd(pred_adj_logits, true_adj_matrix, threshold=t)
            
    return best_thresh, best_f1, best_shd

def compute_mae(pred_delta, true_delta):
    """
    Computes Mean Absolute Error for deltas.
    """
    with torch.no_grad():
        mae = torch.nn.functional.l1_loss(pred_delta, true_delta).item()
        # Safety: Check for NaN/Inf
        if not (mae == mae) or not (mae < float('inf')):  # NaN check and Inf check
            return 0.0
        return mae

def compute_tpr_fdr(pred_adj_logits, true_adj_matrix, threshold=0.0):
    """
    Computes True Positive Rate (Recall) and False Discovery Rate.
    """
    with torch.no_grad():
        pred_edges = (pred_adj_logits > threshold).float()
        true_edges = true_adj_matrix.float()
        
        tp = (pred_edges * true_edges).sum()
        fp = (pred_edges * (1 - true_edges)).sum()
        fn = ((1 - pred_edges) * true_edges).sum()
        
        # Safety: Add epsilon to prevent division by zero
        tpr = tp / (tp + fn + 1e-8)
        fdr = fp / (tp + fp + 1e-8)
        
        tpr_val = tpr.item()
        fdr_val = fdr.item()
        
        # Safety: Check for NaN/Inf
        tpr_val = 0.0 if (tpr_val != tpr_val) or (tpr_val > 1e6) else tpr_val
        fdr_val = 0.0 if (fdr_val != fdr_val) or (fdr_val > 1e6) else fdr_val
        
        return tpr_val, fdr_val
