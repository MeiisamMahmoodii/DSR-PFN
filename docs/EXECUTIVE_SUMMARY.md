# Executive Summary & Technical Report

> [!NOTE]
> **Status: PRODUCTION READY** (Session 3 System Hardening Complete)
> This document serves as the Master Technical Specification for the ISD-CP Phase 5 Architecture.

## 1. Project Overview
**Goal:** Build a "Universal Causal Simulator" that predicts how a complex system changes ($\Delta$) after an intervention ($do(X_i=v)$), *without* knowing the underlying causal graph beforehand.

**Philosophy (Physics-First):** "If the model can accurately predict the consequences of interventions across all possible scenarios, it must have implicitly understood the causal structure."

## 2. Full Model Architecture: The "ISD-CP" Transformer (Phase 5)
The **ISD-CP (Intervention-Structural-Delta Causal Predictor)** is a specialized Transformer designed to solve the *Inverse Problem* of causal discovery via the *Forward Problem* of effect prediction.

### A. Input Representation
We treat tabular data as a sequence of tokens, but with critical augmentations to capture the causal context:
1.  **Interleaved Encoding (Context Restoration)**:
    *   Instead of a simple vector, the input is a sequence of Token pairs: `(FeatureID, Value)`.
    *   **Effective State Construct**: The encoder constructs the effective state $X_{eff}$ by mixing the Observational Base ($X_{obs}$) and the Intervention Target ($X_{int}$) based on the mask:
        $$X_{eff} = X_{obs} \cdot (1 - M) + X_{int} \cdot M$$
    *   This ensures the model knows *what* the system looks like and *where* the force is applied.
2.  **Type Embeddings**: Tokens are tagged as `Observed`, `Intervened`, or `Masked`.

### B. Backbone (Deep Reasoning with Distributed MoE)
*   **RoPE Transformer**: We replace standard sinusoidal embeddings with **Rotary Positional Embeddings (RoPE)**. This allows the model to learn relative dependencies between variables (e.g., "Node $i$ affects Node $j$") rather than absolute slot positions.
*   **MoE in Every Layer**: Instead of a single MoE at the end, each transformer layer integrates **Mixture of Experts (8 experts per layer)** replacing the standard FFN. This provides:
    *   **Hierarchical specialization**: Early layers learn low-level patterns, deep layers learn high-level causal mechanisms
    *   **Massive capacity**: 12 layers × 8 experts = 96 specialized sub-networks
    *   **Efficient routing**: Hard Gumbel-Softmax ensures each token uses exactly 1 expert per layer
*   **Architecture**: 12 Layers, 8 Heads, 512 `d_model`, 8 experts per layer (2-layer SwiGLU each)
*   **Gradient Checkpointing**: Enabled to support large batches/depth on finite VRAM.

### C. The Two Heads
The backbone features feed into two distinct but coupled heads:

1.  **Physics Head (Simple Projection)**:
    *   **Goal**: Predict the scalar $\Delta$ (Change) for every node.
    *   **Mechanism**: Simple linear projection from token embeddings to delta predictions.
    *   **Why Simple?**: The heavy lifting is done by MoE layers throughout the backbone - each layer's experts already specialize in different functional forms (Linear, Sigmoid, Polynomial, etc.)
    *   **Output**: Continuous prediction of $\Delta$ via `nn.Linear(d_model, 1)`.

2.  **Structure Head (Learned Causal Mask)**:
    *   **Goal**: Explicitly recover the Adjacency Matrix $A$.
    *   **Mechanism**: Self-Attention Query/Key projection to predict edge probabilities.
    *   **Output**: $P(A_{ij} | X)$. This matrix is not just an output; it is *fed back* into the network (see Recurrence).

### D. Iterative Recurrent Refinement (The "3-Pass" Cycle)
The model does not just predict once. It "thinks" in loops:
1.  **Pass 1 (Guess)**: Standard forward pass. Rough guess of $\Delta$ and Structure $A_1$.
2.  **Pass 2 (Refine Structure)**: The predicted Structure $A_1$ is converted into a **Causal Attention Mask**. The model runs again, but attention is *biased* to follow the predicted graph. This generates a refined State and Structure $A_2$.
3.  **Pass 3 (Refine Physics)**: Using the high-fidelity graph $A_2$, the model makes its final precise $\Delta$ prediction.

---

## 3. Training Pipeline: From Zero to Hero

### A. Infinite Data Generator
*   **Dynamic SCMs**: We do not train on a fixed dataset. We generate **Synthetic Structural Causal Models (SCMs)** on the fly.
*   **Variety**: Random DAGs, random activation functions (GMM, Tanh, Poly), random noise distributions.
*   **Scale**: The model never sees the same graph twice. It must learn the *meta-algorithm* of causal inference.

### B. The "Twin World" Counterfactual Objective
To predict $\Delta$ accurately amidst noise, we use a Variance Reduction technique:
1.  Sample Noise $\epsilon$.
2.  Generate Base World $X_{base}$ using $\epsilon$.
3.  Generate Intervened World $X_{int}$ using *same* $\epsilon$ but $do(X_i=v)$.
4.  **Target**: $\Delta_{true} = X_{int} - X_{base}$.
*   This cancels out the exogenous noise, giving a clean signal for the causal effect.

### C. Curriculum Learning
The model is not overwhelmed initially. We scale complexity:
1.  **Level 1**: Small graphs (5-10 nodes), Linear only, High density.
2.  **Level 10**: Medium graphs (20 nodes), Mixed Non-Linearity.
3.  **Level 30**: Large graphs (50+ nodes), sparse connections, complex interactions.
*   **Adaptive Criteria**: Progression depends on Validation MAE/F1 stability.

### D. Distributed & Robust Optimization
*   **DDP (Distributed Data Parallel)**: Multi-GPU training with synchronized gradients and batch norm.
*   **Metric Sync**: Validation metrics and Curriculum decisions are explicitly synchronized (`all_reduce`) across ranks to prevent divergence.
*   **Optimization**:
    *   **AdamW** with `weight_decay=1e-4` (Standardized).
    *   **Gradient Clipping** at 10.0 (Deep model safety).
    *   **Scheduler**: Cosine Annealing with Warm Restarts, stepping *per-batch* for precise control.

---

## 4. System Hardening (Session 3 Status)
**Status: PRODUCTION READY**

We have resolved 20+ Critical Issues to ensure stability:
*   **Gradient Flow**: Fixed NaN handling (`requires_grad=True`), enabled Weight Decay, relaxed Gradient Clipping.
*   **Initialization**: Fixed MoE Router initialization (Broadcast from Master) to prevent expert collapse.
*   **Context Awareness**: Fixed Encoder to correctly ingest Intervention Values (solving context blindness).
*   **Structure Learning**: Re-enabled `lambda_dag=10.0` and fixed Dummy Adjacency returns.
*   **API Standardization**: Unified model signature to 5 returns across the entire codebase.

The system is now fully unified, type-safe, and theoretically sound, ready for the Phase 2/3 long-run training.

---

## 5. Technical Specification & Model Map

### A. Full Architecture Diagram
```mermaid
graph TD
    subgraph Data Input
        Obs["Obs Base X_obs"]
        Int["Int Target X_inv"]
        Mask["Int Mask M"]
    end

    subgraph Encoder ["Interleaved Encoder"]
        Mix["State Mixing"] -->Tokenizer["Tokenizer"]
        Tokenizer -->|"FeatID, Value, Type"| Emb["Embeddings"]
        Emb -->|"Interleave"| Tokens["Token Sequence (B, 2N, D)"]
    end

    subgraph Backbone ["RoPE Transformer (MoE-Enhanced)"]
        Rot["Rotary Pos Emb"] -->|"Relative Pos"| L1_in["Layer 1: Input"]
        L1_in -->|"x"| L1_norm1["LayerNorm"]
        L1_norm1 -->|"x_norm"| L1_attn["Self-Attention + RoPE"]
        L1_attn -->|"attn_out"| L1_add1["Add + Dropout"]
        L1_in -->|"residual"| L1_add1
        L1_add1 -->|"x'"| L1_norm2["LayerNorm"]
        L1_norm2 -->|"x_norm"| L1_moe["MoE Router (8 experts)"]
        L1_moe -->|"expert_out"| L1_add2["Add + Dropout"]
        L1_add1 -->|"residual"| L1_add2
        
        L1_add2 -->|"..."| L2["Layer 2: (Attn+Res) + (MoE+Res)"]
        L2 -->|"..."| L12["Layer 12: (Attn+Res) + (MoE+Res)"]
        L12 -->|"hidden_states, total_aux"| Feats["Output Features + Aux Loss"]
    end

    subgraph Recurrent Cycle ["3-Pass Refinement"]
        Feats -->|"Input"| Pass1["Pass 1: Initial Guess"]
        Pass1 -->|"Logits 1"| SHead1["Structure Head"]
        SHead1 -->|"Adj 1"| CausalMask1["Causal Mask Net"]
        CausalMask1 -->|"Bias"| Pass2["Pass 2: Structure Refinement"]
        Pass2 -->|"Logits 2"| SHead2["Structure Head"]
        SHead2 -->|"Adj 2"| CausalMask2["Causal Mask Net"]
        CausalMask2 -->|"Bias"| Pass3["Pass 3: Physics Refinement"]
    end

    subgraph Heads
        Pass3 -->|"Value Tokens"| Physics["Physics Head (Linear Projection)"]
        Pass3 -->|"Value Tokens"| Structure["Structure Head"]
        
        Physics -->|"d_model -> 1"| DeltaPred["Predicted Delta"]
        
        Structure -->|"Q @ K.T"| Logits["Adjacency Logits"]
    end

    Obs --> Mix
    Int --> Mix
    Mask --> Mix
    Tokens --> Rot
    
    DeltaPred -->|"L1/Focal"| LossPhysics["Physics Loss"]
    Logits -->|"BCE"| LossDAG["Structure Loss"]
```

### B. Detailed Component Specification (Start-to-Finish)

| Component | Specification | Function |
| :--- | :--- | :--- |
| **1. Input** | `(B, N)` Float32 | Base Sample ($X_{obs}$), Intervention Value ($X_{int}$), Type Mask ($M$). |
| **2. effective State** | `X_eff = X_obs*(1-M) + X_int*M` | Logic performed inside `InterleavedEncoder`. Ensures model sees the *post-intervention* start state. |
| **3. Encoder** | `InterleavedEncoder` | Converts scalars to vectors. Hybrid Embedding (Linear+Fourier+MLP). Interleaves `[ID, Value]` tokens. Output: `(B, 2N, D)`. |
| **4. Backbone** | `RoPETransformer` | 12 Layers, 8 Heads, `d_model=512`. Each layer: **Attention + MoE (8 experts)**. Uses **Rotary Positional Embeddings** to encode relative graph topology. |
| **5. Recurrence** | 3-Pass Loop | **Pass 1**: Unguided. **Pass 2**: Guided by $A_1$. **Pass 3**: Guided by $A_2$. Feedback loop via Attention Mask bias. |
| **6. Structure Head** | `LearnedCausalMask` | Project $X \to Q, K$. Compute $A = Q K^T$. Represents edges as attention compatibility. |
| **7. Physics Head** | `nn.Linear(d_model, 1)` | Simple projection from contextualized embeddings to delta predictions. Experts in backbone handle mechanism diversity. |
| **8. Outputs** | `Delta` (B, N), `Logits` (B, N, N) | `Delta`: Predicted change. `Logits`: Predicted graph structure. |
| **9. Loss** | `Focal + BCE + Aux` | `L_total = L1(Delta) + 10.0 * BCE(A) + 0.01 * Sparse(A) + 0.001 * Aux_MoE`. Aux loss aggregated across all 12 layers. |

### C. Pipeline Configuration (Session 3)
*   **Optimizer**: AdamW (`lr=1e-4`, `weight_decay=1e-4`).
*   **Safety**: Gradient Clipping (`10.0`), NaN Guards (`requires_grad=True`).
*   **DDP**: Full metric synchronization (`all_reduce`) on validation.
*   **Data**: Infinite Generator (Random DAGs, size 20-50).

> [!IMPORTANT]
> This architecture is fully implemented in `src/models/CausalTransformer.py` and is verified to handle tensor aliasing, recurrence context, and structure learning gradients as of Step 2760.

---

## 6. Micro-Architecture Deep Dive

### A. Interleaved Tokenizer & Hybrid Embedding
**File**: `src/data/encoder.py` (Class: `InterleavedEncoder`)

1.  **Input**: Scalar values $x \in \mathbb{R}$ for each of $N$ variables.
2.  **Hybrid Value Embedding:**
    *   **Linear**: $E_{lin} = W_{lin} x$ (Captures Magnitude)
    *   **Fourier**: $E_{four} = W_{four} [\sin(\omega x), \cos(\omega x)]$ (Captures Periodicity/High-Freq)
    *   **MLP**: $E_{mlp} = MLP(x)$ (Captures Non-linear distortions)
    *   **Mix**: $E_{val} = \text{LayerNorm}(Linear(E_{lin} \oplus E_{four} \oplus E_{mlp}))$
3.  **Token Interleaving**:
    *   For each variable $i$, we generate TWO tokens:
        *   **Identity Token**: Embedding of the Variable ID ($i$). "Who am I?"
        *   **Value Token**: Embedding of the Value ($x_i$) + Type Embedding (Obs/Int/Mask). "What is my state?"
    *   Sequence: $[ID_1, Val_1, ID_2, Val_2, \dots]$
    *   Length: $2N$.
4.  **Output**: Sequence $(B, 2N, D)$.

### B. Rotary Positional Embeddings (RoPE)
**File**: `src/models/rope.py` & `CausalTransformer.py` (Class: `RoPEAttn`)

1.  **Concept**: Instead of adding a vector $P_pos$ to inputs, we **rotate** the Query ($Q$) and Key ($K$) vectors in the complex plane.
2.  **Logic**:
    *   For token at position $m$ and dimension $d$: $\theta_d = 10000^{-2d/D}$.
    *   Rotate: $R_{\Theta, m}^{d} x$.
    *   Property: $Q_m^T K_n$ depends only on relative distance $(m-n)$.
3.  **Why?**: This is crucial for causal graphs because the "absolute position" (Column 1 vs Column 20) is arbitrary. RoPE allows the model to learn that "Node $A$ is the parent of Node $B$" regardless of where they appear in the sequence.
4.  **Output**: Rotated $Q, K$ for Attention computation.

### C. Transformer Layer with Residual Connections
**File**: `src/models/CausalTransformer.py` (Class: `RoPEAttentionLayer`)

Each transformer layer has **two residual blocks** (Pre-LayerNorm with residual connections):

**Block 1 (Attention + Residual)**:
$$x' = x + \text{Dropout}(\text{Attention}(\text{LayerNorm}(x)))$$

**Block 2 (MoE + Residual)**:
$$x_{out} = x' + \text{Dropout}(\text{MoE}(\text{LayerNorm}(x')))$$

**Why Residuals?**
- Enable deep networks (gradient flow through 12 layers without vanishing gradients)
- Preserve low-level information (position embeddings, input features) while allowing high-level reasoning
- Stabilize training with MoE (prevents expert collapse, distributes information)

**Implementation Details**:
- **Pre-norm**: LayerNorm applied *before* the sub-layer (more stable for deep networks)
- **Dropout**: Applied to sub-layer outputs to prevent co-adaptation
- **Residual scale**: Identity mapping ensures gradient magnitude is preserved

### D. Mixture of Experts (MoE) in Transformer Layers
**File**: `src/models/CausalTransformer.py` (Class: `RoPEAttentionLayer`, `MoELayer`)

1.  **Architecture Change**: MoE replaces the standard FFN in **every transformer layer** (not just at the end).
2.  **Per-Layer Structure**: Each of 12 layers contains:
    *   **Attention Block**: RoPE-enhanced self-attention with residual
    *   **MoE Block**: 8 experts (replacing standard MLP) with residual
3.  **Expert Design**:
    *   Type: **Vectorized SwiGLU** (2 layers per expert for efficiency)
    *   Math: $Out = (SiLU(x W_g) \odot (x W_v)) W_o$
    *   Routing: **Hard Gumbel-Softmax** - each token goes to exactly 1 expert
4.  **Benefits**:
    *   **Early Layers**: Experts specialize in low-level feature detection (linear vs nonlinear, monotonic vs non-monotonic)
    *   **Middle Layers**: Experts handle different graph topologies (sparse vs dense, chain vs fork structures)
    *   **Deep Layers**: Experts specialize in causal mechanisms (additive, multiplicative, threshold effects)
5.  **Auxiliary Loss**: Aggregated across all 12 layers to ensure balanced expert usage. $\mathcal{L}_{aux} = \sum_{l=1}^{12} \sum_{e=1}^{8} (Count_{l,e} - N/8)^2$
6.  **Output**: Each layer contributes aux loss; transformer returns `(hidden_states, total_aux_loss)`

**Physics Head Simplification**: With MoE in every layer, the final physics head is simplified to `nn.Linear(d_model, 1)` - a simple projection. The complexity and specialization are handled throughout the backbone.
