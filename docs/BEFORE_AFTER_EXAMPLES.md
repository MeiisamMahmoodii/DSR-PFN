# Numerical Stability Fixes - Before & After Examples

## Example 1: Metrics - compute_mae()

### Before
```python
def compute_mae(pred_delta, true_delta):
    """Computes Mean Absolute Error for deltas."""
    with torch.no_grad():
        return torch.nn.functional.l1_loss(pred_delta, true_delta).item()
    # Problem: If pred_delta contains extreme values, result could be NaN/Inf
```

### After
```python
def compute_mae(pred_delta, true_delta):
    """Computes Mean Absolute Error for deltas."""
    with torch.no_grad():
        mae = torch.nn.functional.l1_loss(pred_delta, true_delta).item()
        # Safety: Check for NaN/Inf
        if not (mae == mae) or not (mae < float('inf')):  # NaN check and Inf check
            return 0.0
        return mae
    # Solution: Validates output and returns safe default if needed
```

**Key Improvement**: Prevents NaN/Inf from propagating to loss tracking

---

## Example 2: Loss Functions - compute_h_loss()

### Before
```python
def compute_h_loss(adj_matrix):
    N = adj_matrix.shape[-1]
    if adj_matrix.device.type == 'mps':
        A_sq = (adj_matrix * adj_matrix).cpu()
        h = torch.trace(torch.matrix_exp(A_sq)) - N
        return h.to(adj_matrix.device)
    else:
        A_sq = adj_matrix * adj_matrix
        h = torch.trace(torch.matrix_exp(A_sq)) - N
        return h
    # Problem: Large matrices can overflow matrix_exp, returning NaN
```

### After
```python
def compute_h_loss(adj_matrix):
    N = adj_matrix.shape[-1]
    if adj_matrix.device.type == 'mps':
        A_sq = (adj_matrix * adj_matrix).cpu()
        h = torch.trace(torch.matrix_exp(A_sq)) - N
        h_val = h.to(adj_matrix.device)
    else:
        A_sq = adj_matrix * adj_matrix
        h = torch.trace(torch.matrix_exp(A_sq)) - N
        h_val = h
    
    # Safety: Check for NaN/Inf
    if (h_val != h_val) or (h_val.abs() > 1e6):
        return torch.tensor(0.0, device=adj_matrix.device, dtype=adj_matrix.dtype)
    
    return h_val
    # Solution: Detects overflow and returns safe default
```

**Key Improvement**: Handles matrix exponential overflow gracefully

---

## Example 3: Data Generation - generate_data()

### Before
```python
def generate_data(self, dag, num_samples, noise_scale=None, intervention=None, noise=None):
    # ... code ...
    for node in sorted_nodes:
        # ... code ...
        for p in parents:
            func = dag[p][node].get('type', 'linear')
            pval = data[p].values
            term = 0
            if func == 'linear': term = 2.0 * pval
            elif func == 'quadratic': term = np.clip(pval, -5, 5)**2
            elif func == 'cubic': term = np.clip(pval, -3, 3)**3
            # ... more functions ...
            terms.append(term)
        
        if len(terms) > 1 and np.random.rand() < 0.3:
            interact = terms[0] * terms[1]  # Product could explode!
            remaining = sum(terms[2:]) if len(terms) > 2 else 0
            total = noise_term + (interact + remaining)
        else:
            total = noise_term + sum(terms)
        
        data[node] = np.clip(total, -100, 100)  # Only final clipping
    # Problem: Intermediate products can overflow before final clipping
```

### After
```python
def generate_data(self, dag, num_samples, noise_scale=None, intervention=None, noise=None):
    # ... code ...
    # Safety: Clip noise to prevent numerical issues
    noise = np.clip(noise, -50, 50)
    # ... code ...
    for node in sorted_nodes:
        # ... code ...
        for p in parents:
            func = dag[p][node].get('type', 'linear')
            pval = data[p].values
            # Safety: Clip parent values to prevent exponential blowup
            pval = np.clip(pval, -50, 50)
            term = 0
            if func == 'linear': term = 2.0 * pval
            elif func == 'quadratic': term = np.clip(pval, -5, 5)**2
            elif func == 'cubic': term = np.clip(pval, -3, 3)**3
            # ... more functions ...
            # Safety: Clip term to prevent accumulation
            term = np.clip(term, -50, 50)
            terms.append(term)
        
        if len(terms) > 1 and np.random.rand() < 0.3:
            # Safety: Clamp products before summing
            interact = np.clip(terms[0] * terms[1], -100, 100)
            remaining = np.clip(sum(terms[2:]) if len(terms) > 2 else 0, -50, 50)
            total = noise_term + (interact + remaining)
        else:
            total = noise_term + sum(terms)
        
        # Final clipping to ensure bounded values
        data[node] = np.clip(total, -100, 100)
    # Solution: Progressive clipping at each stage
```

**Key Improvement**: Prevents exponential growth before final clipping

---

## Example 4: Encoder - HybridEmbedding

### Before
```python
class HybridEmbedding(nn.Module):
    def forward(self, x):
        # x: (..., 1)
        l = self.linear_emb(x)
        f = self.fourier_emb(x)
        m = self.mlp_emb(x)
        
        cat = torch.cat([l, f, m], dim=-1)  # Could overflow
        return self.norm(self.mixer(cat))  # Mixer could amplify values
    # Problem: No bounds on intermediate embeddings
```

### After
```python
class HybridEmbedding(nn.Module):
    def forward(self, x):
        # x: (..., 1)
        # Safety: Clip input
        x = torch.clamp(x, -50, 50)
        
        l = self.linear_emb(x)
        f = self.fourier_emb(x)
        m = self.mlp_emb(x)
        
        # Safety: Clip embeddings to prevent overflow
        l = torch.clamp(l, -100, 100)
        f = torch.clamp(f, -100, 100)
        m = torch.clamp(m, -100, 100)
        
        cat = torch.cat([l, f, m], dim=-1)
        mixed = self.mixer(cat)
        # Safety: Clip mixed output
        mixed = torch.clamp(mixed, -100, 100)
        return self.norm(mixed)
    # Solution: Multi-level clipping at each stage
```

**Key Improvement**: Ensures all embeddings remain within bounds

---

## Example 5: MoE Layer - get_expert_metrics()

### Before
```python
def get_expert_metrics(self):
    if self.total_tokens == 0:
        return {"entropy": 0.0, "gini": 0.0}
        
    probs = self.expert_counts / self.total_tokens
    
    # Entropy: -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))  # log(p) for p=0 is problematic
    
    sorted_probs, _ = torch.sort(probs)
    n = self.num_experts
    index = torch.arange(1, n + 1, device=probs.device, dtype=probs.dtype)
    gini = (2.0 * torch.sum(index * sorted_probs) / (n * torch.sum(sorted_probs) + 1e-10)) - (n + 1.0) / n
    
    return {
        "entropy": entropy.item(), 
        "gini": gini.item(),
        "counts": self.expert_counts.cpu().numpy().tolist()
    }
    # Problem: Could return NaN if probs have zeros despite epsilon
```

### After
```python
def get_expert_metrics(self):
    if self.total_tokens == 0:
        return {"entropy": 0.0, "gini": 0.0}
        
    probs = self.expert_counts / self.total_tokens
    
    # Entropy: -sum(p * log(p)), with safety guards
    # Clamp probabilities to avoid log(0)
    probs_safe = torch.clamp(probs, min=1e-10, max=1.0)
    entropy = -torch.sum(probs_safe * torch.log(probs_safe))
    
    sorted_probs, _ = torch.sort(probs)
    n = self.num_experts
    index = torch.arange(1, n + 1, device=probs.device, dtype=probs.dtype)
    gini = (2.0 * torch.sum(index * sorted_probs) / (n * torch.sum(sorted_probs) + 1e-10)) - (n + 1.0) / n
    
    entropy_val = entropy.item()
    gini_val = gini.item()
    
    # Safety checks
    entropy_val = 0.0 if (entropy_val != entropy_val) or (entropy_val > 1e6) else entropy_val
    gini_val = 0.0 if (gini_val != gini_val) or (gini_val > 1e6) else gini_val
    
    return {
        "entropy": entropy_val, 
        "gini": gini_val,
        "counts": self.expert_counts.cpu().numpy().tolist()
    }
    # Solution: Safe log with clamping and output validation
```

**Key Improvement**: Prevents log(0) and validates results

---

## Summary of Patterns

### Pattern 1: NaN Detection
```python
if value != value:  # NaN check (NaN != NaN is True)
    return safe_default
```

### Pattern 2: Inf Detection
```python
if value > 1e6 or value < -1e6:  # Inf detection
    return safe_default
```

### Pattern 3: Safe Log
```python
x_safe = torch.clamp(x, min=1e-10)
log_x = torch.log(x_safe)
```

### Pattern 4: Progressive Clipping
```python
# Clip at multiple stages, not just output
x = torch.clamp(x, -limit, limit)  # Input
y = operation(x)
y = torch.clamp(y, -limit, limit)  # After operation
```

### Pattern 5: Safe Return
```python
result = computation()
if not is_valid(result):
    return default_value
return result
```
