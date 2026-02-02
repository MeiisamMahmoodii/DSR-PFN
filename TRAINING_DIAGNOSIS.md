# Training Diagnosis Report - Epoch 24

## Problem: Model Not Learning

### Symptoms
- **MAE**: Stuck at 1.82 (no improvement from epoch 1)
- **SHD**: 54.0 (random guessing - should be decreasing)
- **F1**: 0.000 (not detecting any edges)
- **HARD benchmark**: Exploding to 1.29e12 MAE (critical numerical instability)

### Root Causes Identified

#### 1. Scale Normalization Issue (CRITICAL)
**Location**: `src/training/loss.py` lines 76-81

```python
with torch.no_grad():
    scale = torch.mean(torch.abs(true_delta))
    scale = torch.clamp(scale, min=1.0)  # ← PROBLEM!
    
loss_delta = nn.functional.huber_loss(pred_delta / scale, true_delta / scale, delta=1.0)
```

**Problem**: 
- When `true_delta` is small (e.g., mean=0.5), scale is clamped to 1.0
- This DOUBLES the effective delta values: `0.5/1.0 = 0.5` instead of `0.5/0.5 = 1.0`
- Gradients become distorted and inconsistent across batches
- Model can't learn proper scaling

**Evidence**:
- MAE stuck at 1.82 despite 24 epochs
- No improvement in delta prediction

#### 2. Numerical Explosion on HARD Benchmark
**Symptom**: MAE = 1.29e12 on 50-node graphs

**Likely Causes**:
1. Model extrapolating poorly to larger graphs (trained on 20 nodes)
2. No gradient clipping to prevent exploding gradients
3. Scale normalization making large deltas worse

#### 3. Graph Structure Not Learning
**Symptom**: SHD = 54.0 (random), F1 = 0.0

**Possible Causes**:
1. `lambda_dag=10.0` might be too weak compared to `lambda_delta=100.0` (10:1 ratio)
2. BCE loss with pos_weight might not be working for sparse graphs
3. Model focusing only on delta prediction, ignoring structure

### Recommended Fixes

#### Fix 1: Remove Problematic Scale Normalization
```python
# BEFORE (BROKEN):
scale = torch.clamp(torch.mean(torch.abs(true_delta)), min=1.0)
loss_delta = huber_loss(pred_delta / scale, true_delta / scale)

# AFTER (FIXED):
loss_delta = huber_loss(pred_delta, true_delta)
# OR use adaptive scaling without clamping:
scale = torch.mean(torch.abs(true_delta)) + 1e-6  # No clamp!
loss_delta = huber_loss(pred_delta / scale, true_delta / scale)
```

#### Fix 2: Add Gradient Clipping
```python
# In main.py training loop:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### Fix 3: Increase Structure Learning Weight
```python
# Current: lambda_dag=10.0, lambda_delta=100.0 (ratio 1:10)
# Suggested: lambda_dag=50.0, lambda_delta=100.0 (ratio 1:2)
```

#### Fix 4: Warmup Lambda Delta
```python
# Current: Starts at 100.0, decreases with curriculum
# Problem: Too high initially, model ignores structure
# Suggested: Start at 10.0, increase to 100.0 over first 10 epochs
```

### Immediate Action Required

**Option A: Quick Fix (Recommended)**
1. Remove scale clamping in loss.py
2. Add gradient clipping
3. Restart training from scratch

**Option B: Hyperparameter Tuning**
1. Reduce lambda_delta to 10.0
2. Increase lambda_dag to 50.0
3. Add gradient clipping
4. Restart training

**Option C: Debug Mode**
1. Add logging to see actual delta values
2. Check if scale normalization is the issue
3. Monitor gradient norms

### Expected Results After Fix

| Epoch | MAE | SHD | F1 |
|-------|-----|-----|-----|
| 5 | 1.2-1.4 | 50-52 | 0.0 |
| 10 | 0.8-1.0 | 45-48 | 0.1-0.2 |
| 20 | 0.5-0.7 | 35-40 | 0.2-0.4 |
| 50 | 0.3-0.4 | 20-25 | 0.4-0.6 |

### Files to Modify

1. **`src/training/loss.py`** - Remove scale clamping
2. **`main.py`** - Add gradient clipping, adjust lambda values
3. **Restart training** - Current checkpoint is not salvageable

---

**Status**: CRITICAL - Model cannot learn with current configuration
**Priority**: HIGH - Fix immediately before continuing training
**Estimated Fix Time**: 15 minutes
**Estimated Retraining Time**: 10-20 hours to see good results
