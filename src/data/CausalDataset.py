import torch
import numpy as np
import networkx as nx
from torch.utils.data import IterableDataset

class CausalDataset(IterableDataset):
    def __init__(self, generator, num_nodes_range=(5, 10), samples_per_graph=100, edge_prob=0.3, intervention_prob=0.5, infinite=True, validation_graphs=32, reuse_factor=1, use_twin_world=True, intervention_scale_range=(1.0, 1.0)):
        self.generator = generator
        self.num_nodes_range = num_nodes_range
        self.samples_per_graph = samples_per_graph
        self.edge_prob = edge_prob
        self.intervention_prob = intervention_prob
        self.infinite = infinite
        self.validation_graphs = validation_graphs
        self.reuse_factor = reuse_factor
        self.use_twin_world = use_twin_world
        self.intervention_scale_range = intervention_scale_range
    
    def __iter__(self):
        # Validation Cache Mechanism
        if not self.infinite and hasattr(self, 'val_cache') and self.val_cache:
            # Yield from cache if available
            for batch in self.val_cache:
                yield batch
            return

        graphs_generated = 0
        validation_buffer = []

        while True:
            if not self.infinite and graphs_generated >= self.validation_graphs:
                # If validation mode, save cache and break
                if not self.infinite:
                    self.val_cache = validation_buffer
                break
            
            # Sample Random Intervention Scale
            scale = np.random.uniform(self.intervention_scale_range[0], self.intervention_scale_range[1])
            
            n = np.random.randint(self.num_nodes_range[0], self.num_nodes_range[1] + 1)
            res = self.generator.generate_pipeline(
                num_nodes=n,
                edge_prob=self.edge_prob,
                num_samples_base=self.samples_per_graph,
                num_samples_per_intervention=self.samples_per_graph,
                intervention_prob=self.intervention_prob,
                as_torch=True,
                use_twin_world=self.use_twin_world,
                intervention_scale=scale
            )
            
            graphs_generated += 1
            
            # --- BATCH OPTIMIZATION: Yield chunks directly ---
            # Instead of yielding 1 row at a time, we yield the whole interaction block.
            # But the DataLoader expects individual samples (or requires a custom collate that handles batches).
            # To be compatible with standard DataLoader, we usually yield samples.
            # BUT the user specifically asked for "Yield pre-batched tensors".
            # This implies `collate_fn` needs to handle List[Batch] -> Batch.
            # Wait, `collate_fn_pad` currently handles List[Dict].
            # If we yield a Batch (Dict of stacks), collate_fn needs to merge them.
            # A simple way: Yield the whole (samples_per_graph) stack as ONE item?
            # No, standard torch DataLoader adds a batch dimension.
            # If we yield a list, collate gets List[List].
            
            # SOLUTION: We actually want to avoid the Python Loop overhead.
            # We can yield the dictionary containing the full tensors.
            # The DataLoader batch_size should be 1 (or small), and `collate` should just concat these.
            # OR, we stick to yielding samples but minimize transformation overhead.
            
            # User Issue: "Yields single rows in a loop; collate_fn_pad then re-stacks them."
            # Correct Fix: Yield the entire graph's data as ONE "Sample" (which is actually a mini-batch).
            # Then set DataLoader batch_size=1, and the collate_fn just returns the item (or stacks if batch_size>1).
            
            # However, `samples_per_graph` is 64. If we yield 64 samples at once, that's a batch.
            # Let's yield the tensors directly.
            
            adj = torch.tensor(nx.to_numpy_array(res['dag']), dtype=torch.float32)
            base_tensor = res['base_tensor']
            
            # Pre-calculate interactions
            # Flatten all interactions into one big tensor for this graph
            all_int_tensors = []
            all_masks = []
            all_indices = []
            all_targets = []
            all_deltas = []
            
            for i in range(1, len(res['all_dfs'])):
                int_tensor = torch.tensor(res['all_dfs'][i].values, dtype=torch.float32)
                int_mask = torch.tensor(res['all_masks'][i][0], dtype=torch.float32)
                int_node_idx = torch.argmax(int_mask)
                
                # Twin World Matching
                # Base uses same noise, so row j matches row j
                # We can process the whole block at once
                target_block = base_tensor # (N_samples, N_vars)
                delta_block = int_tensor - target_block
                
                all_int_tensors.append(int_tensor)
                all_masks.append(int_mask.unsqueeze(0).expand(int_tensor.shape[0], -1)) # Expand mask
                all_indices.append(int_node_idx.unsqueeze(0).expand(int_tensor.shape[0])) # Expand idx
                # ISSUE 18: Tensor Reference Aliasing
                # base_tensor is reused. If modified, history corrupted. Clone it.
                all_targets.append(target_block.clone())
                all_deltas.append(delta_block)
            
            # Stack everything for this graph
            # Total samples = Interventions * Samples_Per_Int
            # This yields data for ONE graph.
            # We yield this as a "Batch".
            
            # Reuse logic
            for _ in range(self.reuse_factor):
                for i in range(len(all_int_tensors)):
                     # Yielding Interaction Block (Batch of `samples_per_graph`)
                     # This is much faster than row-by-row
                     
                     batch_item = {
                        "base_samples": all_targets[i],
                        "int_samples": all_int_tensors[i], 
                        "target_row": all_targets[i],
                        "int_mask": all_masks[i],
                        "int_node_idx": all_indices[i],
                        "delta": all_deltas[i],
                        "adj": adj
                     }
                     
                     yield batch_item
                     
                     if not self.infinite:
                         validation_buffer.append(batch_item)
