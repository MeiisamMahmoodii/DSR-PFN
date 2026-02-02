# The Annotated Codebase: From Data to Decision
**A Block-by-Block Explanation of the ISD-CP Architecture**

---

## 1. Data Generation (`src/data/SCMGenerator.py`)

This module creates the synthetic universe. It defines the "Physics" our model must learn.

### `generate_dag`
*   **Purpose**: Creates the causal graph skeleton.
*   **Logic**:
    1.  Creates $N$ nodes.
    2.  Shuffles them to define a "Topological Order" (ensures acyclicity).
    3.  Iterates through pairs $(i, j)$ and adds edge with probability $p$.

### `edge_parameters`
*   **Purpose**: Assigns a "Mechanism" to each edge.
*   **Logic**: Randomly selects one of 16 function types (Linear, Sin, Sigmoid, Step, etc.). This ensures the model cannot just learn linear regression.

### `generate_data`
*   **Purpose**: Simulates the universe to generate data $X$.
*   **Logic**:
    1.  Starts with global noise $\epsilon \sim N(0, \sigma)$.
    2.  Iterates in topological order.
    3.  For each node, calculates parents' effect using the assigned mechanism.
    4.  **Clipping**: Ensures values don't explode to Infinity (Critical for stability).

### `generate_pipeline` (The "Twin World" Engine)
*   **Purpose**: Generates matched pairs $(X, X_{int})$ for training.
*   **Innovation**:
    1.  Generates global noise $\epsilon$.
    2.  Simulates Base World $X(\epsilon)$.
    3.  Simulates Intervened World $X_{int}(\epsilon, do(I))$.
    4.  Because $\epsilon$ is identical, $\Delta = X_{int} - X$ is purely causal (variance reduced).

---

## 2. The Model (`src/models/CausalTransformer.py`)

The brain of the operation.

### `VectorizedSwiGLUResBlock`
*   **Purpose**: The "Neuron" of the experts.
*   **Logic**: Uses a SwiGLU activation (Silu * Linear) which is state-of-the-art for LLMs. Vectorized to run all experts in parallel using `torch.einsum`.

### `MoELayer` (The "Muscles")
*   **Purpose**: Choosing the right physical law.
*   **Logic**:
    1.  **Router**: A linear layer predicts which expert is best for the current token.
    2.  **Gumbel Softmax (Hard)**: Forces a discrete choice (One-Hot), making experts specialized.
    3.  **Auxiliary Loss**: Penalizes the router if it only uses one expert (prevents collapse).

### `LearnedCausalMask` (The "Eye")
*   **Purpose**: Seeing the Causal Graph.
*   **Logic**: Predicts $N \times N$ adjacency logits. These are transposed and added to the Attention Matrix. This forces the Transformer to "look" at parents when predicting children.

### `CausalTransformer.forward` (The "Refinement Loop")
*   **Purpose**: Thinking Fast and Slow.
*   **Logic**:
    *   **Pass 1**: Quick guess using `_forward_pass`.
    *   **Pass 2**: New mask is generated from Pass 1's logits. Model "re-thinks" with this mask.
    *   **Pass 3**: Final high-precision prediction.

---

## 3. Training Logic (`main.py` & `loss.py`)

### `causal_loss_fn` (`src/training/loss.py`)
*   **Purpose**: The Scorecard.
*   **Components**:
    1.  **Delta Loss (Huber)**: Error in predicting $\Delta$. (The "Physics" Grade).
    2.  **DAG Loss (BCE)**: Error in predicting edges. (The "Structure" Grade).
    3.  **Acyclicity (H-Loss)**: Penalty for cycles (Matrix Exponential).
    4.  **Sparsity (L1)**: Penalty for too many edges.

### `CurriculumManager` (`src/training/curriculum.py`)
*   **Purpose**: The Teacher.
*   **Logic**:
    *   Starts with **Level 1** (10 nodes, simple physics).
    *   Monitors Validation MAE.
    *   If MAE < Threshold for 5 epochs -> **Level Up**.
    *   Increases generic parameters (N, Density, Intervention Scale).

---

## 4. Execution Flow (`main.py`)

1.  **Setup**: Initialize DDP (Distributed Data Parallel) for multi-GPU.
2.  **Data**: Create a generator that spawns infinite random graphs.
3.  **Loop**:
    *   Generate Batch ($X, X_{int}, \Delta$).
    *   Model Forward (3 Passes).
    *   Compute Loss (Physics + Structure + Load Balancing).
    *   Backpropagate.
    *   **Snippet**: `loss += args.lambda_aux_moe * aux_safe` (Ensures brain health).
4.  **Validation**:
    *   Every epoch, test on Fixed Validation Set (Easy, Medium, Hard).
    *   Log metrics to CSV.
    *   Check for Level Up.
