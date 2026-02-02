# ISD-CP Project: Paper Summaries & Key Takeaways

## 📋 ONE-PAGE SUMMARIES

---

### 1. ATTENTION IS ALL YOU NEED (Vaswani et al., 2017)

**One-Liner**: Introduces transformer architecture with self-attention mechanism.

**Core Mechanism**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Key Insights**:
- Parallel processing of sequences (vs RNNs)
- Multi-head attention for diverse representation learning
- Positional encodings for sequence order
- Scaled dot-product prevents vanishing gradients

**ISD-CP Application**:
- Foundation architecture for CausalTransformer
- Custom attention with RoPE integration (not standard)
- All-to-all attention (no causal masking) because we're learning causality, not enforcing it

**Reading Time**: 20 minutes for abstract+intro+attention mechanism section

---

### 2. ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING (Su et al., 2021)

**One-Liner**: Replaces absolute positional embeddings with rotary encodings in complex space.

**Core Mechanism**:
```
q'_m = q rotate(mθ)
k'_n = k rotate(nθ)
Attention similarity depends on: (m-n)θ (RELATIVE distance)
```

**Key Insight**: Dot product of rotated vectors naturally encodes relative position without explicit encoding!

**Why Better Than Absolute Positions**:
- Absolute: Position(0)=1, Position(1)=[0,1,0,0,...] (one-hot)
- RoPE: Position(m) = rotation by mθ (smooth, continuous, mathematically elegant)

**ISD-CP Application**:
- Causal graphs have no inherent sequential order
- Two nodes far apart in token sequence may be close in the graph
- RoPE allows attention to learn: "Input X is value immediately following Node ID Y"

**Mathematical Detail** (if curious):
```python
# Rotation matrix in 2D
[cos(mθ)  -sin(mθ)] [x]
[sin(mθ)   cos(mθ)] [y]

# Complex number interpretation:
(x + iy) * e^(i*mθ) = rotated vector
```

**Reading Time**: 15 minutes (focus on mechanism, skip benchmarks)

---

### 3. DAGS WITH NO TEARS: CONTINUOUS OPTIMIZATION FOR STRUCTURE LEARNING (Zheng et al., 2018)

**One-Liner**: First to formulate causal discovery as differentiable optimization problem.

**Key Innovation - The h-function**:
```
Minimize: MSE(data, f(A)) + λ₁||A||₁ + λ₂h(A)
Where: h(A) = tr(e^(A⊙A)) - d = 0  [acyclicity constraint]
```

**Why Brilliant**:
- Before: Discrete search (slow, combinatorial)
- Now: Continuous optimization via gradient descent (fast, scalable)
- h-function = 0 iff no cycles (if A is DAG, then A⊙A has no nonzero products along cycles)

**How It Works**:
1. Matrix exponential e^(A⊙A) counts all paths in graph
2. Trace sums diagonal = number of self-paths
3. If no cycles, trace = dimension d
4. If cycles exist, trace > d

**ISD-CP Application**:
- Uses h-function for DAG constraint in `loss.py`
- Original NOTEARS uses augmented Lagrangian (complex dual optimization)
- ISD-CP simplifies: just weighted loss term λ_h * h(A)
- Trade-off: Less principled but simpler, trains better with neural networks

**Important Variants**:
- NOTEARS-MLP (Zheng et al., 2020): Allows MLPs instead of linear functions
- GraN-DAG: Similar h-function, gradient-based optimization
- GOLEM: Linear version, sparse penalties

**Reading Time**: 25 minutes (focus on h-function section)

---

### 4. CAUSALITY: MODELS, REASONING AND INFERENCE (Pearl, 2009)

**One-Liner**: Foundational book on causal reasoning. Every equation in ISD-CP traces back here.

**Must-Read Sections**:
- Chapter 1: Preliminaries & Notation
  - Causal graph definition
  - Do-operator notation: do(X=x)
  - Intervention vs observation
  
- Chapter 2: Graphical Models
  - DAGs, d-separation
  - Markov blanket
  - Confounders, mediators
  
- Chapter 3: Causal Models
  - Structural Causal Models (SCMs)
  - X_i = f_i(PA_i, U_i) [the exact form ISD-CP uses!]
  - Noise variables U

**Key Concepts**:
```
Observational: P(X)          [from data]
Interventional: P(X|do(Y=y)) [from experiments]

SCM equation: X = f(PA_X, U_X)
                 ↑       ↑     ↑
           parents   noise (exogenous)
```

**ISD-CP Application**:
- `SCMGenerator.py` implements exactly this SCM form
- Delta prediction learns intervention effects: Δ = P(X|do) - P(X|obs)
- Twin-world sampling: same noise Z in both observational and interventional

**Reading Time**: 3-4 hours (read carefully, skip proofs on first pass)

**Key Quote**: *"Causality cannot be derived from data alone; it requires causal assumptions."*

---

### 5. CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX (Jang et al., 2016)

**One-Liner**: Makes discrete sampling differentiable via continuous relaxation.

**The Problem**:
```
Want to sample from discrete distribution: categorical(π)
Problem: Sampling is not differentiable (can't backprop through argmax)
Solution: Use Gumbel-Softmax relaxation
```

**The Trick**:
```
Hard sampling (discrete):
y = one_hot(argmax(log(π) + g) / τ)  where g ~ Gumbel(0,1)

Soft sampling (continuous, differentiable):
ŷ = softmax((log(π) + g) / τ)         [gradient flows here]

Straight-through estimator:
y_hard - ŷ_soft.detach() + ŷ_soft    [use hard in forward, soft gradient in backward]
```

**Temperature τ**:
- τ → 0: Becomes one-hot (discrete)
- τ → ∞: Becomes uniform (soft)
- Annealing: Start at τ=1.0, decrease to 0.1 during training

**ISD-CP Application**:
- Expert routing in `CausalTransformer.py` line 122
- hard=True forces discrete expert selection (exactly one expert per token)
- Prevents "expert collapse" where all tokens prefer same expert
- τ fixed at 1.0 (no annealing) because ISD-CP wants stable specialization

**Variant: Concrete Distribution**:
- Same thing, different framing (Maddison et al., 2016)
- Maddison emphasizes: e^g where e ~ Exponential(1)
- Jang emphasizes: log space for numerical stability
- Both equivalent, Gumbel-Softmax more common name

**Reading Time**: 20 minutes (skip proofs, focus on mechanism)

---

### 6. OUTRAGEOUSLY LARGE NEURAL NETWORKS: THE SPARSELY-GATED MIX OF EXPERTS (Shazeer et al., 2017)

**One-Liner**: Scale networks by routing tokens to sparse subset of experts.

**Core Idea**:
```
Dense Layer: All tokens → All neurons
Sparse MoE:  Each token → One expert (out of K)

FFN_out = Σ(gating_weights[k] * expert_k(input))
```

**Standard Approach**:
1. **Top-k Gating**: softmax(gate(x)) selects top-k experts
2. **Load Balancing**: Auxiliary loss ensures all experts stay active
3. **Expert Capacity**: Limit tokens per expert to prevent imbalance

**ISD-CP's Hard MoE Variant**:
- No top-k → Hard Gumbel (select exactly 1 expert)
- No load balancing → Relies on Gumbel noise for diversity
- Vectorized execution: torch.einsum for parallel expert computation
- Why: Forces specialization (one expert per physics type)

**Expert Types in ISD-CP**:
- 8 experts total
- Each learns different "mechanism" (linear, quadratic, sine, sigmoid, etc.)
- Hard routing forces: "This token's physics belongs to Expert-3"

**Standard vs Hard Comparison**:
| Aspect | Standard MoE | Hard MoE (ISD-CP) |
|--------|-------------|------------------|
| Gating | Soft (weighted average) | Hard (discrete selection) |
| Load Balancing | Auxiliary loss | None (Gumbel noise) |
| Experts per token | k experts (blend) | 1 expert (exclusive) |
| Collapse risk | Lower (averaging) | Higher (mitigated by Gumbel) |
| Physics interpretation | Blended mechanisms | Dedicated mechanisms |

**Reading Time**: 25 minutes (focus on gating mechanism, skip GPU cluster details)

---

### 7. FOURIER FEATURES LET NETWORKS LEARN HIGH FREQUENCY FUNCTIONS (Tancik et al., 2020)

**One-Liner**: Transform scalar inputs into high-dimensional sinusoidal features to learn complex functions.

**The Problem**:
```
Neural networks learn low-frequency functions first (bias toward smoothness)
Difficulty: Learning sharp, oscillating functions (e.g., sin(10x))
```

**The Solution - Positional Encoding**:
```
γ(x) = [sin(2πb₁x), cos(2πb₁x), sin(2πb₂x), cos(2πb₂x), ...]
where b_i are frequencies (e.g., 2^0, 2^1, 2^2, ...)

Result: Network now easily learns sin(10x) because sin(10x) is already in feature space!
```

**Why It Works**:
- Fourier transform decomposes any function into sinusoids
- Pre-encoding with sinusoids = pre-computing Fourier basis
- Network only needs to learn linear combinations (much easier)

**ISD-CP Application**:
- Physics values are diverse: sin(x), linear(x), cubic(x), etc.
- Hybrid embedding combines:
  1. **Linear**: γ_linear(x) = Wx (captures magnitude)
  2. **Fourier**: γ_fourier(x) = [sin(2π*2^i*x), cos(...)] (captures oscillation)
  3. **MLP**: γ_mlp(x) = ReLU layers (captures non-linearity)
  
- Final: Concat all three embeddings

**Frequency Choice**:
```python
# Tancik et al. (2020) - Random from normal:
B = randn(m, d) * σ  [random basis]

# ISD-CP - Fixed powers of 2:
B = [2^0, 2^1, 2^2, ..., 2^7]  [deterministic, broader coverage]
```

**Reading Time**: 20 minutes (focus on mechanism, skip NeRF applications)

---

### 8. GLU VARIANTS IMPROVE TRANSFORMER (Shazeer, 2020)

**One-Liner**: Gated Linear Units outperform standard ReLU feedforward networks.

**Standard FFN**:
```
FFN(x) = ReLU(W₁x + b₁) → W₂ + b₂
         ↑      ↑
      activation projection
```

**GLU Variant - Swish**:
```
SwiGLU(x) = Swish(xW_gate) ⊙ (xW_val)
           where ⊙ is element-wise multiply
           and Swish(x) = x * sigmoid(x)
```

**Benefits**:
1. Gating learns which dimensions to pass through
2. Multiplicative interaction (not additive)
3. Better gradient flow during backprop
4. Slight computational overhead (2x params) but worth it

**ISD-CP Application**:
- Expert blocks use SwiGLU instead of standard FFN
- 8× expansion factor (d_model → 8*d_model → d_model)
- Helps experts specialize better

**Reading Time**: 10 minutes (focus on SwiGLU comparison, skip other GLU variants)

---

### 9. ELEMENTS OF CAUSAL INFERENCE (Peters, Janzing, Schölkopf, 2017)

**One-Liner**: Graduate-level book on learning from interventional data.

**Key Chapters for ISD-CP**:
- Chapter 1: Foundations of causality (D-separation, graphs)
- Chapter 2: Learning Graphical Models (SCMs, interventions)
- Chapter 3: Causal Discovery (when it's possible from data)
- Chapter 7: Learning with interventions (YOUR KEY CHAPTER!)

**Critical Concept - Interventional Data**:
```
Observational: X₁, X₂, ..., Xₙ ~ P(X)
Interventional: do(Xᵢ = c) makes Xᵢ deterministic, severs edges into Xᵢ

Effect: P(X | do(X₁=c)) ≠ P(X | X₁=c)  [intervention != observation]
```

**Why Twin-World Works** (Peters et al. foundation):
```
Observation:   P(X | Z)      [with noise Z]
Intervention:  P(X | do(Y=y)) where we set Y [with SAME noise Z]
Delta:         Δ = P(X|do) - P(X|obs) = f(Z, y) - f(Z, y₀)

Key insight: Same Z makes Δ cleaner!
```

**Reading Time**: 4-5 hours (read Chapter 2 and 7 carefully)

---

### 10. LEARNING SPARSE NONPARAMETRIC DAGS (Zheng et al., 2020)

**One-Liner**: NOTEARS extended to handle non-linear functions via MLPs.

**Key Innovation**:
```
NOTEARS (linear):   X = AX + noise
NOTEARS-MLP:        X = MLP(X, A) + noise
                         ↑    ↑
                    non-linear learnable
```

**Methodology**:
1. Each node gets its own MLP: f_i = MLP_i(PA_i)
2. Learn both structure (A) and functions (MLP weights)
3. Same h-function constraint for acyclicity
4. Trained on single dataset → optimizes graph for that data

**ISD-CP vs NOTEARS-MLP**:

| Aspect | NOTEARS-MLP | ISD-CP |
|--------|------------|--------|
| Training | Per-graph optimization | Amortized (train once) |
| Scalability | Cubic in node count | Linear (sequence length) |
| Generalization | Fixed to training data | Zero-shot to new data |
| Speed | ~seconds per graph | ~milliseconds per graph |
| Physics model | Single MLP | 8 specialized experts |

**Primary Baseline**: ISD-CP directly compares against NOTEARS-MLP in evaluation suite!

**Reading Time**: 30 minutes (focus on method, skip extensive experiments)

---

## 🧠 CONCEPTUAL DEPENDENCIES (What Must You Know First?)

```
START HERE
    ↓
Vaswani et al. (2017) - Attention
    ↓
Su et al. (2021) - RoPE
    ↓
Pearl (2009) - Causality [PARALLEL: read this while learning transformer concepts]
    ├→ Understand: SCM, do-operator, DAG
    │
    └→ Zheng et al. (2018) - NOTEARS & h-function
        ├→ Zheng et al. (2020) - NOTEARS-MLP [PRIMARY BASELINE]
        │
        └→ Understand: How continuous optimization discovers graphs

THEN EXPERT ARCHITECTURE:
    ├→ Jang et al. (2016) - Gumbel-Softmax
    ├→ Shazeer et al. (2017) - MoE
    └→ Shazeer (2020) - GLU/SwiGLU

THEN EMBEDDINGS:
    └→ Tancik et al. (2020) - Fourier Features

THEN OPTIMIZATION:
    ├→ Loshchilov & Hutter (2017) - AdamW
    ├→ Loshchilov & Hutter (2016) - SGDR
    └→ Bengio et al. (2009) - Curriculum Learning

THEN BASELINES & COMPARISON:
    ├→ Yu et al. (2019) - DAG-GNN
    ├→ Lachapelle et al. (2019) - GraN-DAG
    └→ Peters et al. (2017) - Elements of Causal Inference

ADVANCED (if interested):
    ├→ Modern papers (2022+)
    └→ Real-world benchmarks
```

---

## 💡 KEY INSIGHTS CHEAT SHEET

| Concept | Simple Explanation | ISD-CP Usage |
|---------|-------------------|------------|
| **Attention** | Every token attends to every other; weights learned | Core mechanism in transformer |
| **RoPE** | Rotate vectors by angle ∝ position | Encodes relative graph distances |
| **h-function** | tr(e^(A⊙A))=d detects cycles | Enforces DAG constraint |
| **Do-operator** | Intervention: set X=x, cut incoming edges | Generate synthetic interventional data |
| **Gumbel-Softmax** | Make discrete sampling differentiable | Select exactly 1 expert per token |
| **MoE** | Route tokens to sparse experts | 8 experts for different physics types |
| **SwiGLU** | Gated activation for better gradients | Improves expert block representation |
| **Fourier Features** | Sinusoidal basis functions | Learn oscillating physics (sin, cos) |
| **Twin-World** | Same noise in obs & interv data | Cleaner delta = f(interv) - f(obs) |
| **h-constraint** | Acyclicity: DAG property as loss term | Prevents cycles in discovered graph |

---

## 📝 POST-READING CHECKLIST

After reading each paper, you should be able to answer:

### Vaswani et al. (2017)
- [ ] What is the scaled dot-product attention formula?
- [ ] Why is the scaling by √d_k important?
- [ ] How do multi-head attention heads work?

### Su et al. (2021)
- [ ] How does RoPE encode relative positions?
- [ ] Why does ISD-CP use RoPE for graphs?
- [ ] What's the difference between absolute and rotary positions?

### Zheng et al. (2018)
- [ ] What is the h-function and why is it clever?
- [ ] How does tr(e^(A⊙A)) detect cycles?
- [ ] Why is this better than discrete search?

### Pearl (2009)
- [ ] What's the difference between do(X=x) and X=x?
- [ ] What is the SCM equation X_i = f_i(PA_i, U_i)?
- [ ] What are confounders, mediators, colliders?

### Jang et al. (2016)
- [ ] How does Gumbel-Softmax make discrete sampling differentiable?
- [ ] What does the temperature τ control?
- [ ] How does straight-through estimator work?

### Shazeer et al. (2017)
- [ ] How does MoE routing work?
- [ ] Why is load balancing important?
- [ ] What's the scaling advantage of sparse experts?

### Tancik et al. (2020)
- [ ] Why do Fourier features help learn high-frequency functions?
- [ ] What's the difference between Fourier and positional encodings?
- [ ] How does hybrid embedding combine multiple feature types?

### Shazeer (2020)
- [ ] What is SwiGLU and how does it differ from ReLU?
- [ ] Why is gating better than standard activations?
- [ ] What's the computational cost trade-off?

### Peters et al. (2017)
- [ ] How does interventional data differ from observational?
- [ ] Why can you discover causality with interventions?
- [ ] What is identifiability?

### Zheng et al. (2020)
- [ ] How does NOTEARS-MLP extend NOTEARS?
- [ ] What are the key differences from ISD-CP?
- [ ] Why does per-graph optimization limit scalability?

---

**Estimated Total Reading Time**: 40-50 hours (MUST READ papers)  
**Advanced Topics**: +20-30 hours (Better to Read)  
**Implementation Study**: +10-20 hours (code reading)

**Recommendation**: Read MUST READ papers first (1-2 weeks), then tackle BETTER TO READ papers as questions arise.

Generated: January 6, 2026
