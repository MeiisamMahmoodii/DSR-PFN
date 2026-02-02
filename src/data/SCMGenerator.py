import numpy as np
import pandas as pd
import networkx as nx
import torch

from typing import List, Optional
from scipy.special import expit

class SCMGenerator:
    r"""
    Structural Causal Model (SCM) Generator for Synthetic Data.

    Generates random DAGs and associated data based on a variety of physical mechanisms
    (Linear, Polynomial, Step, Sinusoidal, etc.).

    Key Features:
    1.  **Twin World Generation**: 
        - Generates observational data ($X$) and interventional data ($X | do(I)$) using the SAME noise vector ($\epsilon$).
        - This allows for precise calculation of the causal effect $\Delta = X_{int} - X_{base}$ by canceling out noise variance.
    2.  **Diverse Mechanisms**:
        - Supports 16 different edge functions (linear, exp, sigmoid, rational, etc.) to test model generalization.
    3.  **Adaptive Scaling**:
        - Intervention magnitudes are scaled relative to the variable's natural standard deviation ($\sigma$).
    """
    def __init__(
        self,
        num_nodes: int = 10,
        edge_prob: float = 0.2,
        noise_scale: float = 1.0,
        num_samples_per_intervention: int = 100,
        intervention_prob: float = 0.3,
        intervention_values: Optional[List[float]] = None,
        seed: Optional[int] = None,
        complexity: float = 0.2,
        mechanism_set: str = 'simple'  # 'simple', 'medium', 'complex'
    ):
        """
        Initialize the SCM Generator.

        Args:
            num_nodes (int): Number of variables in the graph.
            edge_prob (float): Probability of an edge existing between two nodes (sparsity).
            noise_scale (float): Standard deviation of the additive Gaussian noise.
            num_samples_per_intervention (int): Number of samples to generate per intervention setting.
            intervention_prob (float): Probability of creating an intervention target (used in pipeline).
            intervention_values (List[float]): Raw values for interventions (superseded by adaptive scaling).
            seed (int): Random seed for reproducibility.
            complexity (float): Percentage of non-linear mechanisms.
            mechanism_set (str): Control mechanism complexity - 'simple', 'medium', or 'complex'.
                - 'simple': Linear and quadratic only (easy to learn)
                - 'medium': Adds sigmoid, abs, sqrt (moderate complexity)
                - 'complex': All 16 mechanisms including log, exp, trig (full complexity)
        """
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.noise_scale = noise_scale
        self.num_samples_per_intervention = num_samples_per_intervention
        self.intervention_prob = intervention_prob
        self.complexity = complexity
        self.mechanism_set = mechanism_set
        if intervention_values is None:
            self.intervention_values = [5.0, 8.0, 10.0]
        else:
            self.intervention_values = list(intervention_values)
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        # Define mechanism pools for progressive complexity
        self.mechanism_pools = {
            'simple': [1, 9],  # linear, quadratic
            'medium': [1, 9, 11, 13, 8],  # + sigmoid, abs, sqrt
            'complex': list(range(1, 17))  # all 16 mechanisms
        }

    def generate_dag(self, num_nodes: Optional[int] = None, edge_prob: Optional[float] = None, seed: Optional[int] = None):
        # Only reset seed if a specific seed is passed for THIS call. 
        # Do not use self.seed here, as it would force the same DAG every time.
        if seed is not None: 
            np.random.seed(seed)
            
        if num_nodes is None: num_nodes = self.num_nodes
        if edge_prob is None: edge_prob = self.edge_prob

        dag = nx.DiGraph()
        dag.add_nodes_from(range(num_nodes))
        topo_order = np.arange(num_nodes)
        np.random.shuffle(topo_order)
        position_to_node = {i: topo_order[i] for i in range(num_nodes)}

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.rand() < edge_prob:
                    parent = position_to_node[i]
                    child = position_to_node[j]
                    dag.add_edge(parent, child)
        return dag

    def edge_parameters(self, dag, complexity=0.2, mechanism_set=None):
        """
        Assign functional forms to edges based on mechanism_set and complexity.
        
        Args:
            dag: NetworkX graph
            complexity: Probability of using non-linear mechanism (within the chosen set)
            mechanism_set: Override instance mechanism_set if provided
        """
        if mechanism_set is None:
            mechanism_set = self.mechanism_set
            
        # Get the allowed mechanism pool
        mechanism_pool = self.mechanism_pools.get(mechanism_set, self.mechanism_pools['simple'])
        
        for u, v in dag.edges():
            # complexity 0.0 -> always use first mechanism (linear)
            # complexity 1.0 -> sample uniformly from pool
            if np.random.rand() > complexity:
                eq = mechanism_pool[0]  # Default to linear
            else:
                eq = np.random.choice(mechanism_pool)
            
            # Assign type based on equation ID
            if eq == 1: dag[u][v]['type'] = "linear"
            elif eq == 2: dag[u][v]['type'] = "negative linear"
            elif eq == 3: dag[u][v]['type'] = "sin"
            elif eq == 4: dag[u][v]['type'] = "cos"
            elif eq == 5: dag[u][v]['type'] = "tan"
            elif eq == 6: dag[u][v]['type'] = "log"
            elif eq == 7: dag[u][v]['type'] = "exp"
            elif eq == 8: dag[u][v]['type'] = "sqrt"
            elif eq == 9: dag[u][v]['type'] = "quadratic"
            elif eq == 10: dag[u][v]['type'] = "cubic"
            elif eq == 11: dag[u][v]['type'] = "sigmoid"
            elif eq == 12: dag[u][v]['type'] = "step"
            elif eq == 13: dag[u][v]['type'] = "abs"
            elif eq == 14: dag[u][v]['type'] = "poly"
            elif eq == 15: dag[u][v]['type'] = "sawtooth"
            elif eq == 16: dag[u][v]['type'] = "rational"
        return dag

    def sample_noise(self, size, noise_type='normal'):
        """
        Sample noise from different distributions for Domain Randomization (Sim-to-Real).
        """
        if noise_type == 'normal':
            return np.random.normal(scale=self.noise_scale, size=size)
        elif noise_type == 'laplace':
            return np.random.laplace(scale=self.noise_scale, size=size)
        elif noise_type == 'gumbel':
            return np.random.gumbel(scale=self.noise_scale, size=size)
        elif noise_type == 'cauchy':
            # Cauchy has undefined variance. scale=1.0 is standard.
            # Clip rigorously to avoid extreme outliers
            raw_cauchy = np.random.standard_cauchy(size=size) * self.noise_scale
            return np.clip(raw_cauchy, -20.0, 20.0)
        elif noise_type == 'uniform':
            return np.random.uniform(low=-self.noise_scale*1.73, high=self.noise_scale*1.73, size=size)
        else:
            return np.random.normal(scale=self.noise_scale, size=size)

    def generate_data(self, dag, num_samples, noise_scale=None, intervention=None, noise=None, noise_type='normal'):
        if noise_scale is None: noise_scale = self.noise_scale
        nodes = list(dag.nodes())
        
        # Twin World Logic: Use provided noise if available
        if noise is None:
            noise = self.sample_noise(size=(num_samples, len(nodes)), noise_type=noise_type)
        
        # Safety: Clip noise to prevent numerical issues
        # Removed: noise = np.clip(noise, -50, 50)
        
        data = pd.DataFrame(noise, columns=nodes)
        
        try:
            sorted_nodes = list(nx.topological_sort(dag))
        except:
            return data, noise # Cycle fallback

        for node in sorted_nodes:
            if intervention and node in intervention:
                data[node] = intervention[node]
                continue
            
            parents = list(dag.predecessors(node))
            if not parents: continue
            
            noise_term = data[node].values.copy() 
            
            terms = []
            for p in parents:
                func = dag[p][node].get('type', 'linear')
                pval = data[p].values
                
                term = 0
                if func == 'linear': term = 0.3 * pval
                elif func == 'negative linear': term = -0.3 * pval
                elif func == 'sin': term = 0.5 * np.sin(pval)
                elif func == 'cos': term = 0.5 * np.cos(pval)
                elif func == 'tan': term = 0.5 * np.tanh(pval)
                elif func == 'quadratic': term = 0.01 * (pval**2) * np.sign(pval) # Further Dampened (0.1 -> 0.01)
                elif func == 'cubic': term = 0.001 * (pval**3) # Heavily Dampened (0.05 -> 0.001) for deep stability
                elif func == 'sigmoid': term = expit(pval) - 0.5
                elif func == 'step': term = 0.5 * (pval > 0).astype(float)
                elif func == 'abs': term = 0.3 * np.abs(pval)
                elif func == 'log': term = 0.2 * np.log(np.abs(pval) + 1.0)
                elif func == 'exp': term = 0.1 * (np.exp(np.clip(pval, -1, 1)) - 1.0) # Tightened internal clip
                elif func == 'sqrt': term = 0.3 * np.sqrt(np.abs(pval))
                elif func == 'poly': 
                    term = 0.01 * (pval**2) + 0.1 * pval # Dampened poly
                elif func == 'sawtooth': 
                    term = 0.3 * (2.0 * (pval - np.floor(pval)) - 1.0)
                elif func == 'rational': 
                    term = 0.5 * pval / (1.0 + np.abs(pval))
                else: term = 0.2 * pval
                
                terms.append(term)
            
            if len(terms) > 1 and np.random.rand() < 0.3:
                # Dampened Interaction
                interact = 0.1 * terms[0] * terms[1]
                remaining = sum(terms[2:]) if len(terms) > 2 else 0
                total = noise_term + (interact + remaining)
            else:
                total = noise_term + sum(terms)
            
            data[node] = total
        return data, noise

    def generate_pipeline(self, num_nodes=None, edge_prob=None, num_samples_base=100, num_samples_per_intervention=100, intervention_prob=None, as_torch=True, use_twin_world=True, intervention_scale=1.0, random_noise_type=True, complexity=0.2):
        if num_nodes is None: num_nodes = self.num_nodes
        dag = self.generate_dag(num_nodes, edge_prob)
        dag = self.edge_parameters(dag, complexity=complexity)
        
        # 1. Generate Global Noise (The "World State")
        # We generate enough noise for the interventions. 
        # Note: Ideally baseline and interventions share the SAME noise rows.
        # Let's assume we want N samples where row i is "Person i".
        # We observe Person i normally, AND we observe Person i under intervention.
        nodes = list(dag.nodes())
        noise_dim = len(nodes)
        
        # Generate noise for the largest batch we might need
        # Actually, let's just generate distinct noise for base? 
        # NO! The whole point of Twin World is variance reduction.
        # We want: Delta = (Mechanisms(Noise) + Noise) - (Mechanisms_Int(Noise) + Noise)
        # So we MUST reuse the noise.
        
        # Let's generate a fixed set of noise vectors to be used across all scenarios
        # shape: (num_samples, num_vars)
        # Fix: Ensure noise is large enough for both base and intervention batches
        max_samples = max(num_samples_base, num_samples_per_intervention)
        
        # Domain Randomization: Pick a noise type for this Universe
        if random_noise_type:
            noise_type = np.random.choice(['normal', 'laplace', 'gumbel', 'cauchy', 'uniform'])
        else:
            noise_type = 'normal'
            
        global_noise = self.sample_noise(size=(max_samples, noise_dim), noise_type=noise_type)
        
        # Base Data (Observational) using Global Noise
        # Note: If num_samples_base != num_samples_per_int, we have a mismatch.
        # Now we strictly slice the noise to match the requested base samples.
        df_base, _ = self.generate_data(dag, num_samples_base, noise=global_noise[:num_samples_base], noise_type=noise_type)
        
        # Interventions
        prob = intervention_prob if intervention_prob else self.intervention_prob
        num_targets = max(1, int(len(nodes) * prob))
        targets = np.random.choice(nodes, size=num_targets, replace=False)
        
        all_dfs = [df_base]
        all_masks = [np.zeros((num_samples_per_intervention, len(nodes)))]
        
        # Standard Deviation for Adaptive Scaling
        base_stds = df_base.std()
        
        for t in targets:
            # Adaptive Intervention: Based on Sigma of the observational distribution
            # This ensures interventions are relative to the variable's natural scale.
            sigma = float(base_stds[t]) if hasattr(base_stds, '__getitem__') else float(base_stds)
            if sigma < 1e-4: sigma = 1.0 # Safety fallback for constant/zero-var nodes
            
            # Intervene at ±2σ, scaled by the curriculum difficulty (intervention_scale)
            # The prompt requested: [-2*sigma, +2*sigma] (dynamic)
            # We will use exactly that.
            coeffs = [-2.0, 2.0]
            loop_values = [c * sigma * intervention_scale for c in coeffs]
            for val in loop_values:
                # Intervened Data using SAME Global Noise (Twin World) OR Random Noise (Ablation)
                noise_for_int = global_noise[:num_samples_per_intervention]
                if not use_twin_world:
                    # Generate fresh noise for this intervention
                    noise_for_int = self.sample_noise(size=(num_samples_per_intervention, noise_dim), noise_type=noise_type)
                    
                df_int, _ = self.generate_data(dag, num_samples_per_intervention, intervention={t: val}, noise=noise_for_int, noise_type=noise_type)
                
                mask = np.zeros((num_samples_per_intervention, len(nodes)))
                mask[:, t] = 1.0
                all_dfs.append(df_int)
                all_masks.append(mask)
        
        # ---------------------------------------------------------
        # No Normalization (Raw Physics)
        # ---------------------------------------------------------
        df_base_raw = df_base
        all_dfs_raw = all_dfs
            
        if as_torch:
            base_tensor = torch.tensor(df_base_raw.values, dtype=torch.float32)
        else:
            base_tensor = df_base_raw
            
        result = {
            "dag": dag,
            "base_tensor": base_tensor, 
            "all_dfs": all_dfs_raw, 
            "all_masks": all_masks,
            "stats": {"mean": 0.0, "std": 1.0} # Identity stats
        }
        return result
