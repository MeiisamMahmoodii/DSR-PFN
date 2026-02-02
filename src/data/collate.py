import torch
import torch.nn.functional as F



def collate_context_batch(batch):
    """
    Pads and stacks a batch of graph chunks, PRESERVING the context dimension.
    Input: List of Dicts (Each dict contains tensors for ONE graph, shape (S, N))
    Output: Collated Dict (Stacked tensors, shape (B, S, Max_N))
    """
    # 1. Find Max Nodes in this batch
    max_nodes = 0
    for item in batch:
        n = item['base_samples'].shape[1]
        if n > max_nodes:
            max_nodes = n
            
    # 2. Pad & Collect
    new_batch = {
        'base_samples': [],
        'int_samples': [],
        'target_row': [],
        'int_mask': [],
        'delta': [],
        'adj': [],
        'int_node_idx': [],
        'padding_mask': []
    }
    
    for item in batch:
        # item is a chunk of S samples from one graph
        S, N = item['base_samples'].shape
        diff = max_nodes - N
        
        # Pad features: (S, N) -> (S, Max_N)
        pad_config_2d = (0, diff) 
        
        new_batch['base_samples'].append(F.pad(item['base_samples'], pad_config_2d))
        new_batch['int_samples'].append(F.pad(item['int_samples'], pad_config_2d))
        new_batch['target_row'].append(F.pad(item['target_row'], pad_config_2d))
        new_batch['delta'].append(F.pad(item['delta'], pad_config_2d))
        new_batch['int_mask'].append(F.pad(item['int_mask'], pad_config_2d))
        new_batch['int_node_idx'].append(item['int_node_idx']) # (S,)
        
        # ADJ: (N, N) -> (Max_N, Max_N)
        adj_padded = F.pad(item['adj'], (0, diff, 0, diff))
        new_batch['adj'].append(adj_padded)
        
        # PADDING MASK: (Max_N,)
        # True where padded (indices [N, Max_N))
        mask = torch.zeros(max_nodes, dtype=torch.bool)
        if diff > 0:
            mask[N:] = True
        new_batch['padding_mask'].append(mask)

    # 3. Stack (B, S, Max_N)
    collated = {}
    for k in ['base_samples', 'int_samples', 'target_row', 'delta', 'int_mask', 'int_node_idx']:
        collated[k] = torch.stack(new_batch[k], dim=0)
    
    collated['adj'] = torch.stack(new_batch['adj'], dim=0)
    collated['padding_mask'] = torch.stack(new_batch['padding_mask'], dim=0) # (B, Max_N)
            
    return collated
