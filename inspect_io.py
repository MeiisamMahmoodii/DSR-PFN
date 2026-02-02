import torch
from src.models.CausalTransformer import CausalTransformer
from src.data.SCMGenerator import SCMGenerator
from src.data.CausalDataset import CausalDataset
from src.data.collate import collate_context_batch
from torch.utils.data import DataLoader

def main():
    print("--- ICCT Input/Output Inspection ---")
    
    # 1. Setup Generator & Dataset
    gen = SCMGenerator(num_nodes=5, complexity=0.2)
    dataset = CausalDataset(
        gen, 
        num_nodes_range=(5, 5), 
        samples_per_graph=4, 
        infinite=False, 
        validation_graphs=1
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_context_batch)
    
    # 2. Setup Model
    model = CausalTransformer(num_nodes=5, d_model=64, nhead=4, num_layers_scientist=2)
    model.eval()
    
    # 3. Get a batch
    batch = next(iter(loader))
    
    base = batch['base_samples']   # (B, S, N)
    int_s = batch['int_samples']   # (B, S, N)
    target = batch['target_row']   # (B, S, N)
    mask = batch['int_mask']       # (B, N)
    true_delta = batch['delta']    # (B, S, N)
    true_adj = batch['adj']        # (B, N, N)

    print(f"\n[INPUT SHAPES]")
    print(f"Base Samples: {base.shape}  (Batch, Samples, Nodes)")
    print(f"Int. Samples: {int_s.shape}")
    print(f"Target Row:   {target.shape}")
    print(f"Int. Mask:    {mask.shape}     (Batch, Nodes)")
    
    print("\n[SAMPLE VALUES (First Graph, First Sample)]")
    print(f"Base Trace: {base[0, 0].tolist()}")
    print(f"Int. Mask:  {mask[0].tolist()}")
    
    # 4. Forward Pass
    with torch.no_grad():
        deltas, adj_logits = model(base, int_s, target, mask)
        
    print(f"\n[OUTPUT SHAPES]")
    print(f"Pred Deltas: {deltas.shape} (Batch, Samples, Nodes)")
    print(f"Adj Logits:  {adj_logits.shape} (Batch, Nodes, Nodes)")
    
    print("\n[SAMPLE OUTPUTS]")
    print(f"Pred Delta (Node 0): {deltas[0, :, 0].mean().item():.4f}")
    print(f"True Delta (Node 0): {true_delta[0, :, 0].mean().item():.4f}")
    print(f"Mean H-Loss Logic:   {torch.sigmoid(adj_logits[0]).mean().item():.4f} (Avg Edge Prob)")

if __name__ == "__main__":
    main()
